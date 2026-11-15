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

// The error covariance of one pilot cell per block (one warp): nominal error variance of the filter output, sample
// covariance of the innovations, the error network (9 features, one hidden layer), and the calibrated covariance.
// Shared memory (sized at launch): observed [r, r] and innovations [j, r] complex; features [r, 9], hidden [r, h],
// prediction [r, 3], scale [r].
__global__ void ChannelUncertainty(int layout_cells,int p,int j,int r,int h,const C* coefficients,const C* beta,
 const float* weights,const C* innovation,const C* h0,const float* q0,const float* qj,const float* p0,
 const float* reference,const float* floor,const bool* active,const float* layout,const float* kernel,
 const float* bias,const float* output_kernel,const float* output_bias,C* covariance){
 extern __shared__ float memory[];
 C* observed=reinterpret_cast<C*>(memory);
 C* local=observed+r*r;
 float* features=memory+2*r*r+2*j*r;float* hidden=features+9*r;float* prediction=hidden+r*h;float* scale=prediction+3*r;
 __shared__ C d0[3];__shared__ float statistics[3];__shared__ float off_fraction;__shared__ float rho;
 int cell=blockIdx.x,lane=threadIdx.x;
 const float* factor=layout+(cell%layout_cells)*9;
 for(int i=lane;i<j*r;i+=32)local[i]=innovation[cell*j*r+i];
 for(int k=0;k<3;++k){
   float x=0.f,y=0.f;for(int t=lane;t<j;t+=32){C v=beta[(cell*j+t)*3+k];x+=v.x;y+=v.y;}
   x=channel_sum(x);y=channel_sum(y);
   if(lane==0)d0[k]=make_float2((k==0?1.f:0.f)-x,-y);
 }
 __syncwarp();
 float direct=0.f;
 if(lane==0){float value=0.f;for(int i=0;i<3;++i){C z=make_float2(0.f,0.f);
     for(int k=0;k<3;++k){float f=factor[i*3+k];z.x+=__fmul_rn(f,d0[k].x);z.y+=__fmul_rn(f,d0[k].y);}
     value+=cpow(z);}
   direct=q0[cell]*value;}
 float contribution=0.f,sum=0.f,squares=0.f;
 for(int t=lane;t<j;t+=32){float value=0.f;
   for(int k=0;k<3;++k){C z=make_float2(0.f,0.f);
     for(int i=0;i<3;++i){C v=beta[(cell*j+t)*3+i];float f=factor[k*3+i];z.x+=f*v.x;z.y+=f*v.y;}
     value+=cpow(z);}
   contribution+=__fmul_rn(qj[cell*j+t],value);float w=weights[cell*j+t];sum+=w;squares+=__fmul_rn(w,w);}
 contribution=channel_sum(contribution);sum=channel_sum(sum);squares=channel_sum(squares);
 if(lane==0){statistics[0]=direct+contribution;statistics[1]=sum;statistics[2]=squares;}
 __syncwarp();
 float nominal=statistics[0];sum=statistics[1];squares=statistics[2];
 for(int i=lane;i<r*r;i+=32){int a=i/r,b=i%r;C value=make_float2(0.f,0.f);
   for(int t=0;t<j;++t){C x=local[t*r+a],y=local[t*r+b];float w=weights[cell*j+t];
     x=make_float2(w*x.x,w*x.y);value.x+=x.x*y.x+x.y*y.y;value.y+=x.y*y.x-x.x*y.y;}
   if(a==b)value.x+=nominal+floor[cell];
   observed[i]=value;}
 __syncwarp();
 float energy=0.f,diag_energy=0.f,ref=reference[cell];
 for(int i=lane;i<r*r;i+=32){C v=observed[i];v.x/=ref;v.y/=ref;energy+=cpow(v);}
 for(int i=lane;i<r;i+=32){float v=observed[i*r+i].x/ref;diag_energy+=__fmul_rn(v,v);}
 energy=channel_sum(energy);
 diag_energy=channel_sum(diag_energy);
 if(lane==0)off_fraction=fmaxf(energy-diag_energy,0.f)/fmaxf(energy,1.4210854715202004e-14f);
 __syncwarp();
 float effective=squares==0.f?0.f:sum*sum/squares;
 for(int a=lane;a<r;a+=32){int row=cell*r+a;
   float values[9]={p0[cell]/ref,nominal/ref,q0[cell]/ref,cpow(h0[row])/ref,
     ((observed[a*r+a].x-nominal)-floor[cell])/ref,
     (cpow(coefficients[row*3+1])+cpow(coefficients[row*3+2]))/ref,sum,log1pf(effective),off_fraction};
   for(int k=0;k<9;++k)features[a*9+k]=values[k];
 }
 __syncwarp();
 for(int i=lane;i<r*h;i+=32){int a=i/h,unit=i%h;float value=0.f;
   for(int k=0;k<9;++k)value+=kernel[k*h+unit]*features[a*9+k];
   value+=bias[unit];hidden[i]=isnan(value)?value:fmaxf(value,0.f);}
 __syncwarp();
 for(int i=lane;i<r*3;i+=32){int a=i/3,o=i%3;float value=0.f;
   for(int unit=0;unit<h;++unit)value+=hidden[a*h+unit]*output_kernel[unit*3+o];
   prediction[i]=value;}
 __syncwarp();
 float total=0.f;
 for(int a=lane;a<r;a+=32){float x=prediction[a*3]+output_bias[0],y=prediction[a*3+1]+output_bias[1];
   float variance=__fmul_rn(softplus(x),nominal)+__fmul_rn(softplus(y),ref)+floor[cell];
   scale[a]=channel_sqrt(variance);total+=prediction[a*3+2]+output_bias[2];}
 total=channel_sum(total);
 if(lane==0)rho=1.f/(1.f+expf(-total/float(r)));
 __syncwarp();
 for(int i=lane;i<r*r;i+=32){int a=i/r,b=i%r;C value=observed[i];
   float norm=channel_rsqrt(observed[a*r+a].x)*channel_rsqrt(observed[b*r+b].x);
   C matrix=make_float2(__fmul_rn(rho,value.x*norm)+(a==b?1.f-rho:0.f),rho*(value.y*norm));
   float multiplier=scale[a]*scale[b];
   covariance[cell*r*r+i]=active[cell/p]?make_float2(matrix.x*multiplier,matrix.y*multiplier):make_float2(0.f,0.f);
 }
}

Status UncertaintyLaunch(cudaStream_t stream,int n,int layout_cells,int p,int j,int r,int h,const Tensor* const* in,
 Tensor* out){
 size_t bytes=sizeof(float)*(2*r*r+2*j*r+9*r+r*h+3*r+r);
 int limit=0,device=0;
 if(cudaGetDevice(&device)!=cudaSuccess ||
    cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlock,device)!=cudaSuccess)
   return errors::Internal("Cannot query the uncertainty workspace");
 if(bytes>static_cast<size_t>(limit))
   return errors::InvalidArgument("The antenna count or error network exceeds this GPU's shared-memory capacity");
 auto z=[&](int i){return reinterpret_cast<const C*>(in[i]->flat<complex64>().data());};
 auto f=[&](int i){return in[i]->flat<float>().data();};
 return GpuLaunchKernel(ChannelUncertainty,n,32,bytes,stream,layout_cells,p,j,r,h,z(0),z(1),f(2),z(3),z(4),f(5),f(6),
   f(7),f(8),f(9),in[10]->flat<bool>().data(),f(11),f(12),f(13),f(14),f(15),
   reinterpret_cast<C*>(out->flat<complex64>().data()));
}
