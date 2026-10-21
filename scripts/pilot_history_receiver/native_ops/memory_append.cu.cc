#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
using namespace tensorflow;

namespace tensorflow { namespace functor {
template <> void DenseUpdate<Eigen::GpuDevice,float,ASSIGN>::operator()(
 const Eigen::GpuDevice& device,TTypes<float>::Flat destination,TTypes<float>::ConstFlat source){
 destination.device(device)=source;
}
template <> void DenseUpdate<Eigen::GpuDevice,int64_t,ASSIGN>::operator()(
 const Eigen::GpuDevice& device,TTypes<int64_t>::Flat destination,TTypes<int64_t>::ConstFlat source){
 destination.device(device)=source;
}
}}

__global__ void AppendRecords(int capacity,int slots,int users,int size,const float* record,
 const int64_t* head,const int64_t* owners,const bool* written,const int64_t* step,
 float* observations,int64_t* timestamps,int64_t* ports,int64_t* complete){
 int address=blockIdx.x,batch=address/capacity;
 if(written[address]){
   int64_t slot=int64_t(address)*slots+head[address],owner=owners[address];
   for(int i=threadIdx.x;i<size;i+=blockDim.x)
     observations[slot*size+i]=record[(int64_t(batch)*users+owner)*size+i];
   if(threadIdx.x==0){timestamps[slot]=*step;ports[slot]=owner;}
 }
 if(address==0 && threadIdx.x==0)*complete=*step+1;
}

Status AppendPilotMemory(OpKernelContext* c,int b,int capacity,int slots,int users,int size,
 const Tensor* const* in,Tensor* const* out,Tensor* complete){
 return GpuLaunchKernel(AppendRecords,b*capacity,256,0,c->eigen_device<Eigen::GpuDevice>().stream(),
   capacity,slots,users,size,in[0]->flat<float>().data(),in[1]->flat<int64_t>().data(),
   in[2]->flat<int64_t>().data(),in[3]->flat<bool>().data(),in[4]->flat<int64_t>().data(),
   out[0]->flat<float>().data(),out[1]->flat<int64_t>().data(),out[2]->flat<int64_t>().data(),complete->flat<int64_t>().data());
}
