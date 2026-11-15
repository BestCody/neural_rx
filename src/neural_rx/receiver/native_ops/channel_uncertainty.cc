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

REGISTER_OP("PilotChannelUncertainty")
 .Input("coefficients: complex64").Input("beta: complex64").Input("weights: float")
 .Input("innovation: complex64").Input("h0: complex64").Input("q0: float")
 .Input("qj: float").Input("p0: float").Input("reference: float").Input("floor: float")
 .Input("active: bool").Input("layout: float")
 .Input("kernel: float").Input("bias: float").Input("output_kernel: float").Input("output_bias: float")
 .Output("covariance: complex64")
 .SetShapeFn([](shape_inference::InferenceContext* c){
   auto s=c->input(0);c->set_output(0,c->MakeShape({c->Dim(s,0),c->Dim(s,1),c->Dim(s,2),c->Dim(s,3),c->Dim(s,3)}));
   return OkStatus();
 });

Status UncertaintyLaunch(cudaStream_t,int,int,int,int,int,int,const Tensor* const*,Tensor*);

struct UncertaintyStream {
 std::unique_ptr<se::Stream> stream;
 int device=-1;
 std::mutex mutex;
 Status status;
 explicit UncertaintyStream(se::StreamExecutor* executor){
   if(cudaGetDevice(&device)!=cudaSuccess){status=errors::Internal("Cannot identify uncertainty device");return;}
   stream=std::make_unique<se::Stream>(executor);stream->Init();
   if(!stream->ok())status=errors::Internal("Cannot create uncertainty stream");
 }
 ~UncertaintyStream(){int previous=-1;cudaGetDevice(&previous);if(device>=0)cudaSetDevice(device);
   stream.reset();if(previous>=0 && previous!=device)cudaSetDevice(previous);}
};

struct UncertaintyCall {
 std::vector<Tensor> inputs;
 Tensor output;
 std::shared_ptr<UncertaintyStream> execution;
};

class ChannelUncertaintyOp:public AsyncOpKernel {
 std::mutex initialization_;
 std::shared_ptr<UncertaintyStream> execution_;
 public:
 explicit ChannelUncertaintyOp(OpKernelConstruction* c):AsyncOpKernel(c){}
 void ComputeAsync(OpKernelContext* c,DoneCallback done) override {
   OP_REQUIRES_ASYNC(c,c->input(0).dims()==5 && c->input(1).dims()==5 && c->input(12).dims()==2,
     errors::InvalidArgument("Invalid channel uncertainty ranks"),done);
   int b=c->input(0).dim_size(0),u=c->input(0).dim_size(1),p=c->input(0).dim_size(2),r=c->input(0).dim_size(3);
   int j=c->input(1).dim_size(3),h=c->input(12).dim_size(1),n=b*u*p;
   OP_REQUIRES_ASYNC(c,b>0 && u>0 && p>0 && r>0 && j>0 && h>0 &&
     c->input(0).shape()==TensorShape({b,u,p,r,3}) && c->input(1).shape()==TensorShape({b,u,p,j,3}) &&
     c->input(2).shape()==TensorShape({b,u,p,j}) && c->input(3).shape()==TensorShape({b,u,p,j,r}) &&
     c->input(4).shape()==TensorShape({b,u,p,r}) && c->input(5).shape()==TensorShape({b,u,p}) &&
     c->input(6).shape()==TensorShape({b,u,p,j}) && c->input(7).shape()==TensorShape({b,u,p}) &&
     c->input(8).shape()==TensorShape({b,u,p}) && c->input(9).shape()==TensorShape({b,u,p}) &&
     c->input(10).shape()==TensorShape({b,u}) && c->input(11).shape()==TensorShape({u,p,3,3}) &&
     c->input(12).shape()==TensorShape({9,h}) && c->input(13).shape()==TensorShape({h}) &&
     c->input(14).shape()==TensorShape({h,3}) && c->input(15).shape()==TensorShape({3}),
     errors::InvalidArgument("Inconsistent channel uncertainty shapes"),done);
   Tensor* output;OP_REQUIRES_OK_ASYNC(c,c->allocate_output(0,TensorShape({b,u,p,r,r}),&output),done);
   auto call=std::make_shared<UncertaintyCall>();for(int i=0;i<16;++i)call->inputs.push_back(c->input(i));
   call->output=*output;
   auto main=c->op_device_context()->stream();
   {std::lock_guard<std::mutex> lock(initialization_);
     if(!execution_)execution_=std::make_shared<UncertaintyStream>(main->parent());call->execution=execution_;}
   auto execution=call->execution;Status status;
   {
     std::lock_guard<std::mutex> lock(execution->mutex);
     OP_REQUIRES_OK_ASYNC(c,execution->status,done);
     execution->stream->ThenWaitFor(main);
     auto stream=reinterpret_cast<cudaStream_t>(execution->stream->platform_specific_handle().stream);
     const Tensor* in[16];for(int i=0;i<16;++i)in[i]=&call->inputs[i];
     status=UncertaintyLaunch(stream,n,u*p,p,j,r,h,in,output);
     if(!status.ok())execution->status=status;
     c->device()->tensorflow_accelerator_device_info()->event_mgr->ThenExecute(execution->stream.get(),[call,c,done,status]{
       if(!status.ok())c->SetStatus(status);
       else if(c->cancellation_manager() && c->cancellation_manager()->IsCancelled())
         c->SetStatus(errors::Cancelled("Channel uncertainty cancelled"));
       done();
     });
   }
 }
};
REGISTER_KERNEL_BUILDER(Name("PilotChannelUncertainty").Device(DEVICE_GPU),ChannelUncertaintyOp);
