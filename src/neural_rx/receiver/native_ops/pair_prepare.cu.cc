#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
using namespace tensorflow;
using C=float2;

__device__ float squared(C z){return __fadd_rn(__fmul_rn(z.x,z.x),__fmul_rn(z.y,z.y));}
__device__ float scaled_sum(float a,float x,float b,float y,float energy){
 return __fmul_rn(energy,__fadd_rn(__fmul_rn(a,x),__fmul_rn(b,y)));
}
// The whole preparation in one kernel, one warp per cell: for each of the cell's e*e energy pairs, assemble C(x),
// factorize C = L L^H (L lower and row-major, as the detector reads it), take the log-determinant, and whiten the six
// coefficient vectors with L^{-1}. The matrices of a cell are spread over the warp's lanes, `chunk` matrices at a time
// (as many as the shared memory holds); each matrix keeps its sequential arithmetic. A matrix that is not finite or
// not positive definite sets the status flag.
// Shared memory (sized at launch): L [chunk, r, r] and whitened vectors [chunk, 6, r] complex; log-determinants and
// flags [chunk].
__device__ C fused_mul(C a,C b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ C fused_mul_conj(C a,C b){return make_float2(a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y);}  // a*conj(b)
__global__ void FusedPrepare(int chunk,int j,int r,int e,float energy,const C* coefficients,const C* covariance,
 const float* noise,const C* background,const float* energies,C* factor,C* white,float* logdet,int* status){
 extern __shared__ float memory[];
 int cell=blockIdx.x,lane=threadIdx.x,groups=e*e,batch=cell/j;
 C* L=reinterpret_cast<C*>(memory);
 C* X=L+chunk*r*r;
 float* logs=reinterpret_cast<float*>(X+chunk*6*r);
 int* flags=reinterpret_cast<int*>(logs+chunk);
 float sum=0.f;
 for(int a=0;a<r;++a)sum=__fadd_rn(sum,__fadd_rn(squared(coefficients[(cell*2*r+a)*3]),
                                                 squared(coefficients[((cell*2+1)*r+a)*3])));
 float reference=fmaxf(__fadd_rn(__fdiv_rn(sum,float(r)),noise[batch]),1e-30f);
 float diagonal=fmaxf(noise[batch],__fmul_rn(8*r*1.1920928955078125e-7f,reference));
 int cell_bad=!(isfinite(reference) && isfinite(diagonal) && isfinite(noise[batch]));
 float root=sqrtf(energy);
 for(int base=0;base<groups;base+=chunk){
   int count=min(chunk,groups-base);
   for(int g=lane;g<count;g+=32)flags[g]=cell_bad;
   __syncwarp();
   for(int i=lane;i<count*r*r;i+=32){
     int local=i/(r*r),g=base+local,row=i/r%r,col=i%r;
     if(col>row)continue;
     int index=row*r+col;C a=covariance[cell*2*r*r+index],b=covariance[(cell*2+1)*r*r+index];
     C value=make_float2(scaled_sum(energies[g/e],a.x,energies[g%e],b.x,energy),
                        scaled_sum(energies[g/e],a.y,energies[g%e],b.y,energy));
     value.x=__fadd_rn(value.x,row==col?diagonal:0.f);
     if(background){value.x=__fadd_rn(value.x,background[cell*r*r+index].x);value.y=__fadd_rn(value.y,background[cell*r*r+index].y);}
     if(!isfinite(value.x)||!isfinite(value.y))atomicOr(flags+local,1);
     L[i]=value;
   }
   __syncwarp();
   for(int g=lane;g<count;g+=32){
     C* M=L+g*r*r;bool bad=flags[g];float log_determinant=0.f;
     for(int i=0;i<r && !bad;++i){
       float d=M[i*r+i].x;
       for(int k=0;k<i;++k)d-=squared(M[i*r+k]);
       if(!(d>0.f) || !isfinite(d)){bad=true;break;}
       float s=sqrtf(d);M[i*r+i]=make_float2(s,0.f);log_determinant+=logf(s);
       for(int row=i+1;row<r;++row){
         C v=M[row*r+i];
         for(int k=0;k<i;++k){C t=fused_mul_conj(M[row*r+k],M[i*r+k]);v.x-=t.x;v.y-=t.y;}
         M[row*r+i]=make_float2(v.x/s,v.y/s);
       }
     }
     if(bad)flags[g]=1;
     logs[g]=log_determinant;
   }
   __syncwarp();
   for(int i=lane;i<count*6;i+=32){
     int g=i/6,k=i%6;
     if(flags[g])continue;
     const C* M=L+g*r*r;C* x=X+i*r;bool bad=false;
     for(int a=0;a<r;++a){
       C value=coefficients[((cell*2+k/3)*r+a)*3+k%3];
       value=make_float2(__fmul_rn(root,value.x),__fmul_rn(root,value.y));
       for(int col=0;col<a;++col){C t=fused_mul(M[a*r+col],x[col]);value.x-=t.x;value.y-=t.y;}
       x[a]=make_float2(value.x/M[a*r+a].x,value.y/M[a*r+a].x);
       if(!isfinite(x[a].x)||!isfinite(x[a].y))bad=true;
     }
     if(bad)atomicOr(flags+g,1);
   }
   __syncwarp();
   size_t first=static_cast<size_t>(cell)*groups+base;
   for(int i=lane;i<count*r*r;i+=32){
     int g=i/(r*r),row=i/r%r,col=i%r;
     factor[first*r*r+i]=flags[g]?make_float2(row==col?1.f:0.f,0.f):(col<=row?L[i]:make_float2(0,0));
   }
   for(int i=lane;i<count*r*6;i+=32){
     int g=i/(r*6),a=i/6%r,k=i%6;
     white[first*r*6+i]=flags[g]?make_float2(0,0):X[(g*6+k)*r+a];
   }
   for(int g=lane;g<count;g+=32){
     logdet[first+g]=flags[g]?0.f:2.f*logs[g];
     if(flags[g])atomicExch(status,1);
   }
   __syncwarp();
 }
}
Status FusedPreparation(cudaStream_t stream,int b,int j,int r,int e,float energy,const Tensor* const* in,
 Tensor* const* out,Tensor* status){
 size_t matrix=sizeof(C)*(r*r+6*r)+sizeof(float)+sizeof(int);
 int limit=0,device=0;
 if(cudaGetDevice(&device)!=cudaSuccess ||
    cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlock,device)!=cudaSuccess)
   return errors::Internal("Cannot query the preparation workspace");
 // Matrices of a cell processed at a time: all of them if they fit into the shared memory, at most one per lane.
 int chunk=static_cast<int>(std::min<size_t>(std::min(e*e,32),static_cast<size_t>(limit)/matrix));
 if(chunk<1)return errors::InvalidArgument("The antenna workspace exceeds this GPU's shared-memory capacity");
 if(cudaMemsetAsync(status->flat<int32>().data(),0,sizeof(int),stream)!=cudaSuccess)
   return errors::Internal("Cannot clear preparation status");
 return GpuLaunchKernel(FusedPrepare,b*j,32,chunk*matrix,stream,chunk,j,r,e,energy,
   reinterpret_cast<const C*>(in[0]->flat<complex64>().data()),reinterpret_cast<const C*>(in[1]->flat<complex64>().data()),
   in[2]->flat<float>().data(),in[3]->NumElements()?reinterpret_cast<const C*>(in[3]->flat<complex64>().data()):nullptr,
   in[4]->flat<float>().data(),reinterpret_cast<C*>(out[0]->flat<complex64>().data()),
   reinterpret_cast<C*>(out[1]->flat<complex64>().data()),out[2]->flat<float>().data(),status->flat<int32>().data());
}
