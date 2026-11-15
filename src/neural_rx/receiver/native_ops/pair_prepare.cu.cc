#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda_runtime.h>
#include <cmath>
using namespace tensorflow;
using C=float2;

__device__ float squared(C z){return __fadd_rn(__fmul_rn(z.x,z.x),__fmul_rn(z.y,z.y));}
__device__ float scaled_sum(float a,float x,float b,float y,float energy){
 return __fmul_rn(energy,__fadd_rn(__fmul_rn(a,x),__fmul_rn(b,y)));
}
// The whole preparation in one kernel, one thread per matrix (r <= FUSED_ANTENNAS): assemble C(x) for one cell and energy
// pair, factorize C = L L^H (L lower and row-major, as the detector reads it), take the log-determinant, and whiten the
// six coefficient vectors with L^{-1}. A matrix that is not finite or not positive definite sets the status flag.
#define FUSED_ANTENNAS 8
__device__ C fused_mul(C a,C b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ C fused_mul_conj(C a,C b){return make_float2(a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y);}  // a*conj(b)
__global__ void FusedPrepare(int count,int j,int r,int e,float energy,const C* coefficients,const C* covariance,
 const float* noise,const C* background,const float* energies,C* factor,C* white,float* logdet,int* status){
 int matrix=blockIdx.x*blockDim.x+threadIdx.x;
 if(matrix>=count)return;
 int cell=matrix/(e*e),group=matrix%(e*e),batch=cell/j;
 float sum=0.f;
 for(int a=0;a<r;++a)sum=__fadd_rn(sum,__fadd_rn(squared(coefficients[(cell*2*r+a)*3]),
                                                 squared(coefficients[((cell*2+1)*r+a)*3])));
 float reference=fmaxf(__fadd_rn(__fdiv_rn(sum,float(r)),noise[batch]),1e-30f);
 float diagonal=fmaxf(noise[batch],__fmul_rn(8*r*1.1920928955078125e-7f,reference));
 bool bad=!(isfinite(reference) && isfinite(diagonal) && isfinite(noise[batch]));
 C L[FUSED_ANTENNAS][FUSED_ANTENNAS];
 for(int row=0;row<r;++row)for(int col=0;col<=row;++col){
   int i=row*r+col;C a=covariance[cell*2*r*r+i],b=covariance[(cell*2+1)*r*r+i];
   C value=make_float2(scaled_sum(energies[group/e],a.x,energies[group%e],b.x,energy),
                      scaled_sum(energies[group/e],a.y,energies[group%e],b.y,energy));
   value.x=__fadd_rn(value.x,row==col?diagonal:0.f);
   if(background){value.x=__fadd_rn(value.x,background[cell*r*r+i].x);value.y=__fadd_rn(value.y,background[cell*r*r+i].y);}
   if(!isfinite(value.x)||!isfinite(value.y))bad=true;
   L[row][col]=value;
 }
 float log_determinant=0.f;
 for(int i=0;i<r && !bad;++i){
   float d=L[i][i].x;
   for(int k=0;k<i;++k)d-=squared(L[i][k]);
   if(!(d>0.f) || !isfinite(d)){bad=true;break;}
   float s=sqrtf(d);L[i][i]=make_float2(s,0.f);log_determinant+=logf(s);
   for(int row=i+1;row<r;++row){
     C v=L[row][i];
     for(int k=0;k<i;++k){C t=fused_mul_conj(L[row][k],L[i][k]);v.x-=t.x;v.y-=t.y;}
     L[row][i]=make_float2(v.x/s,v.y/s);
   }
 }
 C* f=factor+static_cast<size_t>(matrix)*r*r;
 C* w=white+static_cast<size_t>(matrix)*r*6;
 if(!bad){
   float root=sqrtf(energy);
   for(int k=0;k<6 && !bad;++k){
     C x[FUSED_ANTENNAS];
     for(int a=0;a<r;++a){
       C value=coefficients[((cell*2+k/3)*r+a)*3+k%3];
       value=make_float2(__fmul_rn(root,value.x),__fmul_rn(root,value.y));
       for(int col=0;col<a;++col){C t=fused_mul(L[a][col],x[col]);value.x-=t.x;value.y-=t.y;}
       x[a]=make_float2(value.x/L[a][a].x,value.y/L[a][a].x);
       if(!isfinite(x[a].x)||!isfinite(x[a].y))bad=true;
     }
     for(int a=0;a<r;++a)w[a*6+k]=x[a];
   }
 }
 if(bad){
   atomicExch(status,1);
   for(int i=0;i<r*r;++i)f[i]=make_float2(i/r==i%r?1.f:0.f,0.f);
   for(int i=0;i<r*6;++i)w[i]=make_float2(0,0);
   logdet[matrix]=0.f;
   return;
 }
 for(int row=0;row<r;++row)for(int col=0;col<r;++col)f[row*r+col]=col<=row?L[row][col]:make_float2(0,0);
 logdet[matrix]=2.f*log_determinant;
}
Status FusedPreparation(cudaStream_t stream,int b,int j,int r,int e,float energy,const Tensor* const* in,
 Tensor* const* out,Tensor* status){
 if(r>FUSED_ANTENNAS)return errors::InvalidArgument("Fused preparation supports at most 8 antennas");
 if(cudaMemsetAsync(status->flat<int32>().data(),0,sizeof(int),stream)!=cudaSuccess)
   return errors::Internal("Cannot clear preparation status");
 int count=b*j*e*e;
 return GpuLaunchKernel(FusedPrepare,(count+127)/128,128,0,stream,count,j,r,e,energy,
   reinterpret_cast<const C*>(in[0]->flat<complex64>().data()),reinterpret_cast<const C*>(in[1]->flat<complex64>().data()),
   in[2]->flat<float>().data(),in[3]->NumElements()?reinterpret_cast<const C*>(in[3]->flat<complex64>().data()):nullptr,
   in[4]->flat<float>().data(),reinterpret_cast<C*>(out[0]->flat<complex64>().data()),
   reinterpret_cast<C*>(out[1]->flat<complex64>().data()),out[2]->flat<float>().data(),status->flat<int32>().data());
}
