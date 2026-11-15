#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include <mutex>
#include <memory>
#include <cmath>
using namespace tensorflow;

REGISTER_OP("PilotPairPrepare")
 .Input("coefficients: complex64").Input("covariance: complex64")
 .Input("noise: float").Input("background: complex64").Input("energies: float")
 .Attr("symbol_energy: float")
 .Output("factor: complex64").Output("white: complex64").Output("logdet: float")
 .SetShapeFn([](shape_inference::InferenceContext* c) {
   auto s=c->input(0); auto g=c->Dim(c->input(4),0);
   shape_inference::DimensionHandle groups;
   TF_RETURN_IF_ERROR(c->Multiply(g,g,&groups));
   c->set_output(0,c->MakeShape({c->Dim(s,0),c->Dim(s,1),groups,c->Dim(s,3),c->Dim(s,3)}));
   c->set_output(1,c->MakeShape({c->Dim(s,0),c->Dim(s,1),groups,c->Dim(s,3),6}));
   c->set_output(2,c->MakeShape({c->Dim(s,0),c->Dim(s,1),groups}));
   return OkStatus();
 });

Status FusedPreparation(cudaStream_t,int,int,int,int,float,const Tensor* const*,Tensor* const*,Tensor*);

struct PreparationStream {
 std::unique_ptr<se::Stream> stream;
 int device=-1;
 std::mutex mutex;
 Status status;
 explicit PreparationStream(se::StreamExecutor* executor) {
   if(cudaGetDevice(&device)!=cudaSuccess){status=errors::Internal("Cannot identify preparation device");return;}
   stream=std::make_unique<se::Stream>(executor); stream->Init();
   if(!stream->ok()) {status=errors::Internal("Cannot create preparation stream");return;}
 }
 ~PreparationStream() {
   int previous=-1;
   cudaGetDevice(&previous);
   if(device>=0)cudaSetDevice(device);
   stream.reset();
   if(previous>=0 && previous!=device)cudaSetDevice(previous);
 }
};

struct PreparationCall {
 std::vector<Tensor> inputs;
 Tensor status,host_status;
 std::shared_ptr<PreparationStream> execution;
};

class PairPrepareOp : public AsyncOpKernel {
 float energy_;
 std::mutex initialization_;
 std::shared_ptr<PreparationStream> execution_;
 public:
 explicit PairPrepareOp(OpKernelConstruction* c):AsyncOpKernel(c) {
   OP_REQUIRES_OK(c,c->GetAttr("symbol_energy",&energy_));
   OP_REQUIRES(c,std::isfinite(energy_) && energy_>0,errors::InvalidArgument("Invalid symbol energy"));
 }
 void ComputeAsync(OpKernelContext* c,DoneCallback done) override {
   const auto& coefficients=c->input(0); const auto& covariance=c->input(1);
   OP_REQUIRES_ASYNC(c,coefficients.dims()==5 && covariance.dims()==5 && c->input(2).dims()==1 &&
     c->input(4).dims()==1,errors::InvalidArgument("Invalid preparation ranks"),done);
   int b=coefficients.dim_size(0),j=coefficients.dim_size(1),r=coefficients.dim_size(3),e=c->input(4).NumElements();
   OP_REQUIRES_ASYNC(c,b>0 && j>0 && r>0 && e>0 && coefficients.shape()==TensorShape({b,j,2,r,3}) &&
     covariance.shape()==TensorShape({b,j,2,r,r}) && c->input(2).NumElements()==b &&
     (c->input(3).NumElements()==0 || c->input(3).shape()==TensorShape({b,j,r,r})),
     errors::InvalidArgument("Inconsistent preparation shapes"),done);
   int groups=e*e,count=b*j*groups;
   Tensor* out[3];
   OP_REQUIRES_OK_ASYNC(c,c->allocate_output(0,TensorShape({b,j,groups,r,r}),&out[0]),done);
   OP_REQUIRES_OK_ASYNC(c,c->allocate_output(1,TensorShape({b,j,groups,r,6}),&out[1]),done);
   OP_REQUIRES_OK_ASYNC(c,c->allocate_output(2,TensorShape({b,j,groups}),&out[2]),done);
   auto call=std::make_shared<PreparationCall>();
   for(int i=0;i<5;++i)call->inputs.push_back(c->input(i));
   OP_REQUIRES_OK_ASYNC(c,c->allocate_temp(DT_INT32,TensorShape({}),&call->status),done);
   AllocatorAttributes host;host.set_on_host(true);host.set_gpu_compatible(true);
   OP_REQUIRES_OK_ASYNC(c,c->allocate_temp(DT_INT32,TensorShape({}),&call->host_status,host),done);
   auto main=c->op_device_context()->stream();
   {
     std::lock_guard<std::mutex> lock(initialization_);
     if(!execution_)execution_=std::make_shared<PreparationStream>(main->parent());
     call->execution=execution_;
   }
   auto execution=call->execution;
   Status status;
   {
     std::lock_guard<std::mutex> lock(execution->mutex);
     OP_REQUIRES_OK_ASYNC(c,execution->status,done);
     execution->stream->ThenWaitFor(main);
     auto stream=reinterpret_cast<cudaStream_t>(execution->stream->platform_specific_handle().stream);
     const Tensor* in[5];for(int i=0;i<5;++i)in[i]=&call->inputs[i];
     status=FusedPreparation(stream,b,j,r,e,energy_,in,out,&call->status);
     if(status.ok() && cudaMemcpyAsync(call->host_status.flat<int32>().data(),call->status.flat<int32>().data(),
                                     sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)
       status=errors::Internal("Preparation status copy failed");
     if(!status.ok())execution->status=status;
     c->device()->tensorflow_accelerator_device_info()->event_mgr->ThenExecute(execution->stream.get(),
       [call,c,done,status] {
         if(!status.ok())c->SetStatus(status);
         else if(call->host_status.scalar<int32>()()!=0)
           c->SetStatus(errors::InvalidArgument("Channel covariance preparation is not finite or positive definite"));
         else if(c->cancellation_manager() && c->cancellation_manager()->IsCancelled())
           c->SetStatus(errors::Cancelled("Channel covariance preparation cancelled"));
         done();
       });
   }
 }
};
REGISTER_KERNEL_BUILDER(Name("PilotPairPrepare").Device(DEVICE_GPU),PairPrepareOp);
