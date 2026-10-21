#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
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

Status AssemblePreparation(cudaStream_t,int,int,int,int,float,const Tensor* const*,Tensor* const*,Tensor*,Tensor*,Tensor*);
Status FinishPreparation(cudaStream_t,int,int,Tensor*,Tensor*,Tensor*,Tensor*);
Status CheckPreparation(cudaStream_t,int,int,Tensor*,Tensor*);

struct PreparationStream {
 std::unique_ptr<se::Stream> stream;
 cublasHandle_t blas=nullptr;
 cusolverDnHandle_t solver=nullptr;
 int device=-1;
 std::mutex mutex;
 Status status;
 explicit PreparationStream(se::StreamExecutor* executor) {
   if(cudaGetDevice(&device)!=cudaSuccess){status=errors::Internal("Cannot identify preparation device");return;}
   stream=std::make_unique<se::Stream>(executor); stream->Init();
   if(!stream->ok()) {status=errors::Internal("Cannot create preparation stream");return;}
   auto raw=reinterpret_cast<cudaStream_t>(stream->platform_specific_handle().stream);
   if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS || cusolverDnCreate(&solver)!=CUSOLVER_STATUS_SUCCESS ||
      cublasSetStream(blas,raw)!=CUBLAS_STATUS_SUCCESS || cusolverDnSetStream(solver,raw)!=CUSOLVER_STATUS_SUCCESS)
     status=errors::Internal("Cannot initialize preparation solvers");
 }
 ~PreparationStream() {
   int previous=-1;
   cudaGetDevice(&previous);
   if(device>=0)cudaSetDevice(device);
   if(blas)cublasDestroy(blas);
   if(solver)cusolverDnDestroy(solver);
   stream.reset();
   if(previous>=0 && previous!=device)cudaSetDevice(previous);
 }
};

struct PreparationCall {
 std::vector<Tensor> inputs;
 Tensor pointers,info,status,host_status;
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
   OP_REQUIRES_OK_ASYNC(c,c->allocate_temp(DT_INT64,TensorShape({2,count}),&call->pointers),done);
   OP_REQUIRES_OK_ASYNC(c,c->allocate_temp(DT_INT32,TensorShape({count}),&call->info),done);
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
     status=AssemblePreparation(stream,b,j,r,e,energy_,in,out,&call->pointers,&call->info,&call->status);
     auto matrices=reinterpret_cast<cuComplex**>(call->pointers.flat<int64_t>().data());
     auto rhs=matrices+count;
     if(status.ok() && cusolverDnCpotrfBatched(execution->solver,CUBLAS_FILL_MODE_UPPER,r,matrices,r,
                                             call->info.flat<int32>().data(),count)!=CUSOLVER_STATUS_SUCCESS)
       status=errors::Internal("Preparation Cholesky launch failed");
     if(status.ok())status=FinishPreparation(stream,count,r,out[0],out[2],&call->info,&call->status);
     cuComplex alpha=make_cuComplex(1.f,0.f);
     if(status.ok()){
       cublasStatus_t solved=CUBLAS_STATUS_SUCCESS;
       if(count>1 && r<=128*count)
         solved=cublasCtrsmBatched(execution->blas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_UPPER,
             CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,6,r,&alpha,matrices,r,rhs,6,count);
       else for(int i=0;i<count && solved==CUBLAS_STATUS_SUCCESS;++i)
         solved=cublasCtrsm(execution->blas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_UPPER,
             CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,6,r,&alpha,
             reinterpret_cast<cuComplex*>(out[0]->flat<complex64>().data())+i*r*r,r,
             reinterpret_cast<cuComplex*>(out[1]->flat<complex64>().data())+i*r*6,6);
       if(solved!=CUBLAS_STATUS_SUCCESS)status=errors::Internal("Preparation triangular solve launch failed");
     }
     if(status.ok())status=CheckPreparation(stream,count,r,out[1],&call->status);
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
