#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda_runtime.h>
#include <cmath>
using namespace tensorflow;
using C=float2;
__device__ float channel_sum(float value){
 for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
 return value;
}
__device__ float cpow(C x){return __fmul_rn(x.x,x.x)+__fmul_rn(x.y,x.y);}
__device__ float channel_sqrt(float x){float y;asm("sqrt.approx.f32 %0, %1;":"=f"(y):"f"(x));return y;}
__device__ float channel_rsqrt(float x){float y;asm("rsqrt.approx.f32 %0, %1;":"=f"(y):"f"(x));return y;}
__device__ float softplus(float x){
 const float threshold=-13.942384719848633f;
 return x>-threshold?x:(x<threshold?expf(x):log1pf(expf(x)));
}

__global__ void PackUncertainty(int layout_cells,int j,int r,const C* beta,const float* weights,
 const C* innovation,const float* layout,C* bt,C* factor,C* weighted,C* conjugated){
 int cell=blockIdx.x;
 for(int i=threadIdx.x;i<3*j;i+=blockDim.x)bt[(cell*3+i/j)*j+i%j]=beta[(cell*j+i%j)*3+i/j];
 for(int i=threadIdx.x;i<9;i+=blockDim.x)factor[cell*9+i]=make_float2(layout[(cell%layout_cells)*9+i],0.f);
 for(int i=threadIdx.x;i<j*r;i+=blockDim.x){C h=innovation[cell*j*r+i];float w=weights[cell*j+i/r];
   weighted[cell*j*r+i]=make_float2(w*h.x,w*h.y);conjugated[cell*j*r+i]=make_float2(h.x,-h.y);}
}

__global__ void UncertaintyStats(int j,int r,const C* beta,const float* weights,const float* q0,
 const float* qj,const float* floor,const C* factor,const C* transformed,C* observed,float* statistics){
 int cell=blockIdx.x,lane=threadIdx.x;
 __shared__ C d0[3];__shared__ float nominal;
 if(lane<3){
   float x=0.f,y=0.f;for(int t=0;t<j;++t){C v=beta[(cell*j+t)*3+lane];x+=v.x;y+=v.y;}
   d0[lane]=make_float2((lane==0?1.f:0.f)-x,-y);
 }
 __syncwarp();
 if(lane==0){float value=0.f;for(int i=0;i<3;++i){C z=make_float2(0.f,0.f);
     for(int k=0;k<3;++k){float f=factor[cell*9+i*3+k].x;z.x+=__fmul_rn(f,d0[k].x);z.y+=__fmul_rn(f,d0[k].y);}
     value+=cpow(z);}
   nominal=q0[cell]*value;}
 float contribution=0.f,sum=0.f,squares=0.f;
 for(int t=lane;t<j;t+=32){float value=0.f;for(int k=0;k<3;++k)value+=cpow(transformed[(cell*3+k)*j+t]);
   contribution+=__fmul_rn(qj[cell*j+t],value);float w=weights[cell*j+t];sum+=w;squares+=__fmul_rn(w,w);}
 contribution=channel_sum(contribution);
 if(lane==0){nominal+=contribution;statistics[cell*3]=nominal;}
 sum=channel_sum(sum);
 if(lane==0)statistics[cell*3+1]=sum;
 squares=channel_sum(squares);
 if(lane==0)statistics[cell*3+2]=squares;
 __syncwarp();
 for(int i=lane;i<r*r;i+=32)observed[cell*r*r+i]=make_float2(i/r==i%r?nominal+floor[cell]:0.f,0.f);
}

__global__ void UncertaintyFeatureValues(int n,int r,const C* coefficients,const C* h0,
 const float* q0,const float* p0,const float* reference,const float* floor,const C* observed,
 const float* statistics,float* features){
 int cell=blockIdx.x,lane=threadIdx.x,rows=n*r;
 __shared__ float off_fraction;
 float energy=0.f,diag_energy=0.f,ref=reference[cell];
 for(int i=lane;i<r*r;i+=32){C v=observed[cell*r*r+i];v.x/=ref;v.y/=ref;energy+=cpow(v);}
 for(int i=lane;i<r;i+=32){float v=observed[(cell*r+i)*r+i].x/ref;diag_energy+=__fmul_rn(v,v);}
 energy=channel_sum(energy);
 diag_energy=channel_sum(diag_energy);
 if(lane==0)off_fraction=fmaxf(energy-diag_energy,0.f)/fmaxf(energy,1.4210854715202004e-14f);
 __syncwarp();
 float nominal=statistics[cell*3],sum=statistics[cell*3+1],squares=statistics[cell*3+2];
 float effective=squares==0.f?0.f:sum*sum/squares;
 for(int a=lane;a<r;a+=32){int row=cell*r+a;
   float values[9]={p0[cell]/ref,nominal/ref,q0[cell]/ref,cpow(h0[row])/ref,
     ((observed[(cell*r+a)*r+a].x-nominal)-floor[cell])/ref,
     (cpow(coefficients[row*3+1])+cpow(coefficients[row*3+2]))/ref,sum,log1pf(effective),off_fraction};
   for(int k=0;k<9;++k)features[k*rows+row]=values[k];
 }
}

__global__ void UncertaintyRelu(int size,int h,const float* bias,float* hidden){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x){
   float value=hidden[i]+bias[i%h];hidden[i]=isnan(value)?value:fmaxf(value,0.f);
 }
}

__global__ void UncertaintyOutput(int n,int p,int r,const float* prediction,const float* bias,
 const float* reference,const float* floor,const bool* active,const C* observed,const float* statistics,
 float* scale,C* covariance){
 int cell=blockIdx.x,lane=threadIdx.x,rows=n*r;
 __shared__ float rho;
 float sum=0.f;
 for(int a=lane;a<r;a+=32){int row=cell*r+a;float x=prediction[row]+bias[0],y=prediction[rows+row]+bias[1];
   float variance=__fmul_rn(softplus(x),statistics[cell*3])+__fmul_rn(softplus(y),reference[cell])+floor[cell];
   scale[row]=channel_sqrt(variance);sum+=prediction[2*rows+row]+bias[2];}
 sum=channel_sum(sum);
 if(lane==0)rho=1.f/(1.f+expf(-sum/float(r)));
 __syncwarp();
 for(int i=lane;i<r*r;i+=32){int a=i/r,b=i%r;C value=observed[cell*r*r+i];
   float norm=channel_rsqrt(observed[(cell*r+a)*r+a].x)*channel_rsqrt(observed[(cell*r+b)*r+b].x);
   C matrix=make_float2(__fmul_rn(rho,value.x*norm)+(a==b?1.f-rho:0.f),rho*(value.y*norm));
   float multiplier=scale[cell*r+a]*scale[cell*r+b];
   covariance[cell*r*r+i]=active[cell/p]?make_float2(matrix.x*multiplier,matrix.y*multiplier):make_float2(0.f,0.f);
 }
}

Status UncertaintyPack(cudaStream_t stream,int n,int layout_cells,int j,int r,const Tensor* const* in,Tensor* s){
 return GpuLaunchKernel(PackUncertainty,n,128,0,stream,layout_cells,j,r,
   reinterpret_cast<const C*>(in[1]->flat<complex64>().data()),in[2]->flat<float>().data(),
   reinterpret_cast<const C*>(in[3]->flat<complex64>().data()),in[11]->flat<float>().data(),
   reinterpret_cast<C*>(s[0].flat<complex64>().data()),reinterpret_cast<C*>(s[1].flat<complex64>().data()),
   reinterpret_cast<C*>(s[3].flat<complex64>().data()),reinterpret_cast<C*>(s[4].flat<complex64>().data()));
}
Status UncertaintyStatistics(cudaStream_t stream,int n,int j,int r,const Tensor* const* in,Tensor* s){
 return GpuLaunchKernel(UncertaintyStats,n,32,0,stream,j,r,reinterpret_cast<const C*>(in[1]->flat<complex64>().data()),
   in[2]->flat<float>().data(),in[5]->flat<float>().data(),in[6]->flat<float>().data(),in[9]->flat<float>().data(),
   reinterpret_cast<const C*>(s[1].flat<complex64>().data()),reinterpret_cast<const C*>(s[2].flat<complex64>().data()),
   reinterpret_cast<C*>(s[5].flat<complex64>().data()),s[6].flat<float>().data());
}
Status UncertaintyFeatures(cudaStream_t stream,int n,int r,const Tensor* const* in,Tensor* s){
 return GpuLaunchKernel(UncertaintyFeatureValues,n,32,0,stream,n,r,
   reinterpret_cast<const C*>(in[0]->flat<complex64>().data()),reinterpret_cast<const C*>(in[4]->flat<complex64>().data()),
   in[5]->flat<float>().data(),in[7]->flat<float>().data(),in[8]->flat<float>().data(),in[9]->flat<float>().data(),
   reinterpret_cast<const C*>(s[5].flat<complex64>().data()),s[6].flat<float>().data(),s[7].flat<float>().data());
}
Status UncertaintyHidden(cudaStream_t stream,int rows,int h,const Tensor* const* in,Tensor* s){
 return GpuLaunchKernel(UncertaintyRelu,(rows*h+255)/256,256,0,stream,rows*h,h,
   in[13]->flat<float>().data(),s[8].flat<float>().data());
}
Status UncertaintyFinish(cudaStream_t stream,int n,int p,int r,const Tensor* const* in,Tensor* s,Tensor* out){
 return GpuLaunchKernel(UncertaintyOutput,n,32,0,stream,n,p,r,s[9].flat<float>().data(),in[15]->flat<float>().data(),
   in[8]->flat<float>().data(),in[9]->flat<float>().data(),in[10]->flat<bool>().data(),
   reinterpret_cast<const C*>(s[5].flat<complex64>().data()),s[6].flat<float>().data(),s[10].flat<float>().data(),
   reinterpret_cast<C*>(out->flat<complex64>().data()));
}
