#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include <memory>
#include <mutex>
using namespace tensorflow;

REGISTER_OP("PilotPairScores")
 .Input("received: complex64").Input("factors: complex64")
 .Input("whitened: complex64").Input("logdet: float").Input("phi: float")
 .Input("cells: int64").Input("symbols: complex64").Input("energies: int64")
 .Input("labels: int64").Input("enabled: bool").Attr("single_user: bool = false")
 .Output("scores: float").Output("fit: float")
 .SetShapeFn([](shape_inference::InferenceContext* c) {
   bool single; TF_RETURN_IF_ERROR(c->GetAttr("single_user",&single));
   auto m=c->Dim(c->input(6),0),q=c->Dim(c->input(8),1);
   shape_inference::DimensionHandle pairs,bits,size;
   TF_RETURN_IF_ERROR(c->Multiply(m,single?c->MakeDim(1):m,&pairs));
   TF_RETURN_IF_ERROR(c->Multiply(q,single?1:2,&bits));
   TF_RETURN_IF_ERROR(c->Add(pairs,bits,&size));
   c->set_output(0,c->MakeShape({c->Dim(c->input(0),0),c->Dim(c->input(0),1),size}));
   c->set_output(1,c->MakeShape({c->Dim(c->input(0),0),c->Dim(c->input(0),1)}));
   return OkStatus();
 });

Status InitializeScores(int*);
Status ScoresWorkspace(bool,int,int,int,int,int,size_t*,bool*);
Status LaunchScores(cudaStream_t,bool,int,int,int,int,int,int,int,size_t,bool,const Tensor* const*,Tensor*,Tensor*);

struct ScoreStream {
 std::unique_ptr<se::Stream> stream;
 int device=-1,shared_limit=0;
 std::mutex mutex;
 Status status;
 explicit ScoreStream(se::StreamExecutor* executor) {
   if(cudaGetDevice(&device)!=cudaSuccess){status=errors::Internal("Cannot identify physical-score device");return;}
   status=InitializeScores(&shared_limit);
   if(!status.ok())return;
   stream=std::make_unique<se::Stream>(executor);stream->Init();
   if(!stream->ok())status=errors::Internal("Cannot create physical-score stream");
 }
 ~ScoreStream(){
   int previous=-1;cudaGetDevice(&previous);
   if(device>=0)cudaSetDevice(device);
   stream.reset();
   if(previous>=0 && previous!=device)cudaSetDevice(previous);
 }
};

struct ScoreCall {
 std::vector<Tensor> inputs,outputs;
 std::shared_ptr<ScoreStream> execution;
};

class PairScoresOp : public AsyncOpKernel {
 bool single_;
 std::mutex initialization_;
 std::shared_ptr<ScoreStream> execution_;
 public:
 explicit PairScoresOp(OpKernelConstruction* c):AsyncOpKernel(c){
   OP_REQUIRES_OK(c,c->GetAttr("single_user",&single_));
 }
 void ComputeAsync(OpKernelContext* c,DoneCallback done) override {
   const Tensor* in[10];for(int i=0;i<10;++i)in[i]=&c->input(i);
   OP_REQUIRES_ASYNC(c,in[0]->dims()==3 && in[1]->dims()==5 && in[2]->dims()==5 &&
     in[3]->dims()==3 && in[4]->dims()==3 && in[5]->dims()==1 && in[6]->dims()==1 &&
     in[7]->dims()==1 && in[8]->dims()==2 && in[9]->dims()==2,
     errors::InvalidArgument("Invalid physical-score ranks"),done);
   int b=in[0]->dim_size(0),n=in[0]->dim_size(1),r=in[0]->dim_size(2),cells=in[1]->dim_size(1),
       groups=in[1]->dim_size(2),m=in[6]->dim_size(0),q=in[8]->dim_size(1);
   OP_REQUIRES_ASYNC(c,b>0 && n>0 && r>0 && cells>0 && groups>0 && m>1 && q>0 && q<16 && m==(1<<q),
     errors::InvalidArgument("Invalid physical-score dimensions"),done);
   OP_REQUIRES_ASYNC(c,in[1]->shape()==TensorShape({b,cells,groups,r,r}) &&
     in[2]->shape()==TensorShape({b,cells,groups,r,6}) && in[3]->shape()==TensorShape({b,cells,groups}) &&
     in[4]->shape()==TensorShape({n,2,3}) && in[5]->NumElements()==n && in[7]->NumElements()==m &&
     in[8]->shape()==TensorShape({m,q}) && in[9]->shape()==TensorShape({b,cells}),
     errors::InvalidArgument("Inconsistent physical-score shapes"),done);
   auto call=std::make_shared<ScoreCall>();
   for(int i=0;i<10;++i)call->inputs.push_back(*in[i]);
   auto main=c->op_device_context()->stream();
   {
     std::lock_guard<std::mutex> lock(initialization_);
     if(!execution_)execution_=std::make_shared<ScoreStream>(main->parent());
     call->execution=execution_;
   }
   auto execution=call->execution;
   OP_REQUIRES_OK_ASYNC(c,execution->status,done);
   size_t bytes;bool cache_vectors;
   OP_REQUIRES_OK_ASYNC(c,ScoresWorkspace(single_,r,groups,m,q,execution->shared_limit,&bytes,&cache_vectors),done);
   Tensor *output,*fit;
   OP_REQUIRES_OK_ASYNC(c,c->allocate_output(0,TensorShape({b,n,m*(single_?1:m)+(single_?q:2*q)}),&output),done);
   OP_REQUIRES_OK_ASYNC(c,c->allocate_output(1,TensorShape({b,n}),&fit),done);
   call->outputs={*output,*fit};
   {
     std::lock_guard<std::mutex> lock(execution->mutex);
     execution->stream->ThenWaitFor(main);
     auto stream=reinterpret_cast<cudaStream_t>(execution->stream->platform_specific_handle().stream);
     auto status=LaunchScores(stream,single_,b,n,r,cells,groups,m,q,bytes,cache_vectors,in,output,fit);
     c->device()->tensorflow_accelerator_device_info()->event_mgr->ThenExecute(execution->stream.get(),
       [call,c,done,status]{
         if(!status.ok())c->SetStatus(status);
         else if(c->cancellation_manager() && c->cancellation_manager()->IsCancelled())
           c->SetStatus(errors::Cancelled("Physical symbol scoring cancelled"));
         done();
       });
   }
 }
};
REGISTER_KERNEL_BUILDER(Name("PilotPairScores").Device(DEVICE_GPU),PairScoresOp);
