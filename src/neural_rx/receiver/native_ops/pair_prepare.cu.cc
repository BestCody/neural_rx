#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include <cuda_runtime.h>
#include <cmath>
using namespace tensorflow;
using C=float2;
struct PreparationSum {__host__ __device__ float operator()(float a,float b) const{return a+b;}};

__device__ float squared(C z){return __fadd_rn(__fmul_rn(z.x,z.x),__fmul_rn(z.y,z.y));}
__device__ float scaled_sum(float a,float x,float b,float y,float energy){
 return __fmul_rn(energy,__fadd_rn(__fmul_rn(a,x),__fmul_rn(b,y)));
}
__global__ void Assemble(int j,int r,int e,float energy,const C* coefficients,const C* covariance,
 const float* noise,const C* background,const float* energies,C* factor,C* white,C** pointers,int* status) {
 int matrix=blockIdx.x,cell=matrix/(e*e),group=matrix%(e*e),batch=cell/j;
 __shared__ float diagonal;__shared__ int bad;
 __shared__ gpuprim::WarpReduce<float>::TempStorage reduction;
 if(threadIdx.x<32){
   float sum=0.f;
   for(int a=threadIdx.x;a<r;a+=32)sum=__fadd_rn(sum,__fadd_rn(squared(coefficients[(cell*2*r+a)*3]),
                                                                 squared(coefficients[((cell*2+1)*r+a)*3])));
   sum=gpuprim::WarpReduce<float>(reduction).Reduce(sum,PreparationSum(),min(r,32));
   if(threadIdx.x==0){
   float reference=fmaxf(__fadd_rn(__fdiv_rn(sum,float(r)),noise[batch]),1e-30f);
   diagonal=fmaxf(noise[batch],__fmul_rn(8*r*1.1920928955078125e-7f,reference));
   bad=!(isfinite(reference) && isfinite(diagonal) && isfinite(noise[batch]));
   pointers[matrix]=factor+matrix*r*r;
   pointers[gridDim.x+matrix]=white+matrix*r*6;
   }
 }
 __syncthreads();
 for(int i=threadIdx.x;i<r*r;i+=blockDim.x){
   int row=i/r,col=i%r;C a=covariance[cell*2*r*r+i],b=covariance[(cell*2+1)*r*r+i];
   C value=make_float2(scaled_sum(energies[group/e],a.x,energies[group%e],b.x,energy),
                      scaled_sum(energies[group/e],a.y,energies[group%e],b.y,energy));
   value.x=__fadd_rn(value.x,row==col?diagonal:0.f);
   if(background){value.x=__fadd_rn(value.x,background[cell*r*r+i].x);value.y=__fadd_rn(value.y,background[cell*r*r+i].y);}
   if(!isfinite(value.x)||!isfinite(value.y))atomicExch(&bad,1);
   factor[matrix*r*r+i]=col<=row?value:make_float2(0,0);
 }
 for(int i=threadIdx.x;i<r*6;i+=blockDim.x){
   int a=i/6,k=i%6;C value=coefficients[((cell*2+k/3)*r+a)*3+k%3];
   value=make_float2(__fmul_rn(sqrtf(energy),value.x),__fmul_rn(sqrtf(energy),value.y));
   if(!isfinite(value.x)||!isfinite(value.y))atomicExch(&bad,1);
   white[matrix*r*6+i]=value;
 }
 __syncthreads();
 if(bad){
   if(threadIdx.x==0)atomicExch(status,1);
   for(int i=threadIdx.x;i<r*r;i+=blockDim.x)factor[matrix*r*r+i]=make_float2(i/r==i%r?1.f:0.f,0.f);
   for(int i=threadIdx.x;i<r*6;i+=blockDim.x)white[matrix*r*6+i]=make_float2(0,0);
 }
}
__global__ void Finish(int r,C* factor,float* logdet,const int* info,int* status){
 int matrix=blockIdx.x;__shared__ int bad;
 __shared__ gpuprim::WarpReduce<float>::TempStorage reduction;
 if(threadIdx.x<32){float sum=0.f;
   for(int a=threadIdx.x;a<r;a+=32)sum+=logf(factor[(matrix*r+a)*r+a].x);
   sum=gpuprim::WarpReduce<float>(reduction).Reduce(sum,PreparationSum(),min(r,32));
   if(threadIdx.x==0){bad=info[matrix]!=0;logdet[matrix]=2.f*sum;if(!isfinite(logdet[matrix]))bad=1;}
 }
 __syncthreads();
 for(int i=threadIdx.x;i<r*r;i+=blockDim.x){C value=factor[matrix*r*r+i];
   if(i%r<=i/r && (!isfinite(value.x)||!isfinite(value.y)))atomicExch(&bad,1);
 }
 __syncthreads();
 for(int i=threadIdx.x;i<r*r;i+=blockDim.x){
   if(bad)factor[matrix*r*r+i]=make_float2(i/r==i%r?1.f:0.f,0.f);
   else if(i%r>i/r)factor[matrix*r*r+i]=make_float2(0,0);
 }
 if(bad && threadIdx.x==0)atomicExch(status,1);
}
__global__ void Check(int size,const C* white,int* status){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x)
   if(!isfinite(white[i].x)||!isfinite(white[i].y))atomicExch(status,1);
}
Status AssemblePreparation(cudaStream_t stream,int b,int j,int r,int e,float energy,const Tensor* const* in,
 Tensor* const* out,Tensor* pointers,Tensor* info,Tensor* status){
 if(cudaMemsetAsync(status->flat<int32>().data(),0,sizeof(int),stream)!=cudaSuccess)
   return errors::Internal("Cannot clear preparation status");
 return GpuLaunchKernel(Assemble,b*j*e*e,128,0,stream,j,r,e,energy,
   reinterpret_cast<const C*>(in[0]->flat<complex64>().data()),reinterpret_cast<const C*>(in[1]->flat<complex64>().data()),
   in[2]->flat<float>().data(),in[3]->NumElements()?reinterpret_cast<const C*>(in[3]->flat<complex64>().data()):nullptr,
   in[4]->flat<float>().data(),reinterpret_cast<C*>(out[0]->flat<complex64>().data()),
   reinterpret_cast<C*>(out[1]->flat<complex64>().data()),reinterpret_cast<C**>(pointers->flat<int64_t>().data()),status->flat<int32>().data());
}
Status FinishPreparation(cudaStream_t stream,int count,int r,Tensor* factor,Tensor* logdet,Tensor* info,Tensor* status){
 return GpuLaunchKernel(Finish,count,128,0,stream,r,reinterpret_cast<C*>(factor->flat<complex64>().data()),
   logdet->flat<float>().data(),info->flat<int32>().data(),status->flat<int32>().data());
}
Status CheckPreparation(cudaStream_t stream,int count,int r,Tensor* white,Tensor* status){
 int size=count*r*6;
 return GpuLaunchKernel(Check,(size+255)/256,256,0,stream,size,reinterpret_cast<const C*>(white->flat<complex64>().data()),status->flat<int32>().data());
}
