#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include <algorithm>
#include <memory>
#include <vector>
using namespace tensorflow;

REGISTER_OP("PilotMemoryAppend")
 .Input("observations: resource").Input("timestamps: resource").Input("ports: resource")
 .Input("record: float").Input("head: int64").Input("owners: int64")
 .Input("written: bool").Input("step: int64").Output("complete: int64")
 .SetIsStateful()
 .SetShapeFn([](shape_inference::InferenceContext* c){c->set_output(0,c->Scalar());return OkStatus();});

Status AppendPilotMemory(OpKernelContext*,int,int,int,int,int,const Tensor* const*,Tensor* const*,Tensor*);

class PilotMemoryAppendOp : public OpKernel {
 public:
 explicit PilotMemoryAppendOp(OpKernelConstruction* c):OpKernel(c){}
 void Compute(OpKernelContext* c) override {
   core::RefCountPtr<Var> vars[3];
   for(int i=0;i<3;++i)OP_REQUIRES_OK(c,LookupResource(c,HandleFromInput(c,i),&vars[i]));
   OP_REQUIRES(c,vars[0].get()!=vars[1].get() && vars[0].get()!=vars[2].get() && vars[1].get()!=vars[2].get(),
               errors::InvalidArgument("Memory resources must be distinct"));
   std::vector<mutex*> order{vars[0]->mu(),vars[1]->mu(),vars[2]->mu()};
   std::sort(order.begin(),order.end(),std::less<mutex*>());
   std::vector<std::unique_ptr<mutex_lock>> locks;
   for(auto m:order)locks.emplace_back(std::make_unique<mutex_lock>(*m));
   auto* observations=vars[0]->tensor();auto* timestamps=vars[1]->tensor();auto* ports=vars[2]->tensor();
   const auto& record=c->input(3);const auto& head=c->input(4);
   OP_REQUIRES(c,observations->dtype()==DT_FLOAT && timestamps->dtype()==DT_INT64 && ports->dtype()==DT_INT64 &&
     observations->dims()==5 && record.dims()==4 && head.dims()==2 && TensorShapeUtils::IsScalar(c->input(7).shape()),
     errors::InvalidArgument("Invalid memory append types or ranks"));
   int b=observations->dim_size(0),capacity=observations->dim_size(1),slots=observations->dim_size(2);
   int cells=observations->dim_size(3),components=observations->dim_size(4),users=record.dim_size(1);
   OP_REQUIRES(c,b>0 && capacity>0 && slots>1 && cells>0 && components>0 && users>0 &&
     record.shape()==TensorShape({b,users,cells,components}) && head.shape()==TensorShape({b,capacity}) &&
     c->input(5).shape()==head.shape() && c->input(6).shape()==head.shape() &&
     timestamps->shape()==TensorShape({b,capacity,slots}) && ports->shape()==timestamps->shape(),
     errors::InvalidArgument("Inconsistent memory append shapes"));
   Tensor* complete;OP_REQUIRES_OK(c,c->allocate_output(0,TensorShape({}),&complete));
   OP_REQUIRES_OK(c,(PrepareToUpdateVariable<Eigen::GpuDevice,float>(c,observations,vars[0]->copy_on_read_mode.load())));
   OP_REQUIRES_OK(c,(PrepareToUpdateVariable<Eigen::GpuDevice,int64_t>(c,timestamps,vars[1]->copy_on_read_mode.load())));
   OP_REQUIRES_OK(c,(PrepareToUpdateVariable<Eigen::GpuDevice,int64_t>(c,ports,vars[2]->copy_on_read_mode.load())));
   const Tensor* in[5];for(int i=0;i<5;++i)in[i]=&c->input(3+i);
   Tensor* out[3]={observations,timestamps,ports};
   OP_REQUIRES_OK(c,AppendPilotMemory(c,b,capacity,slots,users,cells*components,in,out,complete));
 }
};
REGISTER_KERNEL_BUILDER(Name("PilotMemoryAppend").Device(DEVICE_GPU)
 .HostMemory("observations").HostMemory("timestamps").HostMemory("ports"),PilotMemoryAppendOp);
