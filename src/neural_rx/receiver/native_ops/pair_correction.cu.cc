#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <cstdlib>
#include <string>
using namespace tensorflow;
using C = float2;

__device__ C add(C a,C b){return make_float2(a.x+b.x,a.y+b.y);}
__device__ C sub(C a,C b){return make_float2(a.x-b.x,a.y-b.y);}
__device__ C mul(C a,C b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ C scale(C a,float s){return make_float2(a.x*s,a.y*s);}
__device__ C conj(C a){return make_float2(a.x,-a.y);}
__device__ float norm(C a){return a.x*a.x+a.y*a.y;}
__device__ float sign(float x){return (x>0.f)-(x<0.f);}
__device__ unsigned group_mask(int width){return (0xffffffffu>>(32-width))<<((threadIdx.x%32)/width*width);}
__device__ float group_max(float value,int width){
 unsigned mask=group_mask(width);
 for(int d=width/2;d;d>>=1)value=fmaxf(value,__shfl_down_sync(mask,value,d,width));
 return __shfl_sync(mask,value,0,width);
}
__device__ float group_sum(float value,int width){
 unsigned mask=group_mask(width);
 for(int d=width/2;d;d>>=1)value+=__shfl_down_sync(mask,value,d,width);
 return __shfl_sync(mask,value,0,width);
}
__device__ float score_at(const float* score,int user,int symbol,int other,int stride){
 return user==0?score[symbol*stride+other]:score[other*stride+symbol];
}
__device__ float own_prior(int symbol,int target,int q,const int64_t* labels,const float* evidence){
 float value=0.f;
 for(int k=0;k<q;++k)if(k!=target)value+=labels[symbol*q+k]?fminf(evidence[k],0.f):-fmaxf(evidence[k],0.f);
 return value;
}
__device__ void symbol_sums(bool single,bool conditioned,int m,
 const float* score,const float* priors,float* rows){
 int divisor=single?1:m,width=min(m,32),lane=threadIdx.x%width;
 int count=single?m:2*m;
 for(int index=threadIdx.x/width;index<count;index+=blockDim.x/width){
   int user=index/m,symbol=index%m,others=user?m:divisor;
   float maximum=-INFINITY;
   for(int other=lane;other<others;other+=width){
     float value=score_at(score,user,symbol,other,divisor+1);
     if(conditioned && !single)value+=priors[(1-user)*m+other];
     maximum=fmaxf(maximum,value);
   }
   maximum=group_max(maximum,width);
   float sum=0.f;
   for(int other=lane;other<others;other+=width){
     float value=score_at(score,user,symbol,other,divisor+1);
     if(conditioned && !single)value+=priors[(1-user)*m+other];
     sum+=expf(value-maximum);
   }
   sum=group_sum(sum,width);
   if(lane==0)rows[index]=maximum+logf(sum);
 }
}
__device__ void bit_sums(bool conditioned,int m,int q,int kbits,const int64_t* labels,
 const float* rows,const float* evidence,float* norms,float* physical){
 int width=min(m,32),lane=threadIdx.x%width;
 for(int j=threadIdx.x/width;j<kbits;j+=blockDim.x/width){
   int user=j/q,target=j%q;
   float maxima[2]={-INFINITY,-INFINITY};
   for(int symbol=lane;symbol<m;symbol+=width){
     int value=labels[symbol*q+target];
     float score=rows[user*m+symbol];
     if(conditioned)score+=own_prior(symbol,target,q,labels,evidence+user*q);
     maxima[value]=fmaxf(maxima[value],score);
   }
   for(int v=0;v<2;++v)maxima[v]=group_max(maxima[v],width);
   float sums[2]={0.f,0.f};
   for(int symbol=lane;symbol<m;symbol+=width){
     int value=labels[symbol*q+target];
     float score=rows[user*m+symbol];
     if(conditioned)score+=own_prior(symbol,target,q,labels,evidence+user*q);
     sums[value]+=expf(score-maxima[value]);
   }
   for(int v=0;v<2;++v)sums[v]=group_sum(sums[v],width);
   if(lane==0){
     int offset=4*j+(conditioned?2:0);
     norms[offset]=maxima[0]+logf(sums[0]);
     norms[offset+1]=maxima[1]+logf(sums[1]);
     if(!conditioned)physical[j]=norms[offset+1]-norms[offset];
   }
 }
}
__device__ C residual(int pair,int g,int row,int r,int divisor,const C* symbols,
 const C* white,const float* phi,const C* yw,const C* local){
 C a=make_float2(0,0),b=a;
 if(local){a=local[(g*r+row)*2];b=local[(g*r+row)*2+1];}
 else for(int k=0;k<3;++k){a=add(a,scale(white[(g*r+row)*6+k],phi[k]));b=add(b,scale(white[(g*r+row)*6+3+k],phi[3+k]));}
 return sub(sub(yw[g*r+row],mul(a,symbols[pair/divisor])),mul(b,symbols[pair%divisor]));
}

size_t ScalarWorkspace(bool backward,bool single,int m,int q,int hidden){
 size_t divisor=single?1:m,kbits=single?q:2*q,nsymbols=single?m:2*m;
 size_t count=m*(divisor+1)+9*kbits+3*nsymbols+hidden+10;
 if(backward)count+=m*divisor+3*kbits+2*nsymbols+kbits*m+7;
 if(backward)count+=m*divisor+11*kbits+2*kbits*hidden+10*hidden+1;
 return (count+1)&~size_t(1);
}

__device__ void physical_scores(bool single,int p,int r,int groups,int m,int q,bool cache_vectors,
 const C* y,const C* factor,const C* white,const float* ld,const float* phi,const C* symbols,
 const int64_t* energies,const int64_t* labels,float* score,float* rows,float* norms,float* physical,C* yw,C* local,
 float* residual_energy,float* fit_scratch){
 int divisor=single?1:m,pairs=m*divisor,kbits=single?q:2*q;
 int energy_count=static_cast<int>(sqrtf(static_cast<float>(groups))+.5f);
 for(int g=threadIdx.x;g<groups;g+=blockDim.x){
   for(int row=0;row<r;++row){
     C value=y[p*r+row];
     for(int col=0;col<row;++col)value=sub(value,mul(factor[(g*r+row)*r+col],yw[g*r+col]));
     yw[g*r+row]=scale(value,1.f/factor[(g*r+row)*r+row].x);
   }
 }
 if(cache_vectors)for(int i=threadIdx.x;i<groups*r*2;i+=blockDim.x){
   int row=i/2,user=i%2;C value=make_float2(0,0);
   for(int k=0;k<3;++k)value=add(value,scale(white[row*6+user*3+k],phi[user*3+k]));
   local[i]=value;
 }
 __syncthreads();
 float minimum=INFINITY;
 for(int pair=threadIdx.x;pair<pairs;pair+=blockDim.x){
   int a=pair/divisor,beta=pair%divisor;
   int g=energies[a]*energy_count+energies[beta];
   float value=-ld[g],energy=0.f;
   for(int row=0;row<r;++row){
     float term=norm(residual(pair,g,row,r,divisor,symbols,white,phi,yw,cache_vectors?local:nullptr));
     value-=term;energy+=term;
   }
   score[a*(divisor+1)+beta]=value;
   if(residual_energy)residual_energy[pair]=energy;
   minimum=fminf(minimum,energy);
 }
 minimum=-group_max(-minimum,32);
 if(threadIdx.x%32==0)fit_scratch[threadIdx.x/32]=minimum;
 __syncthreads();
 if(threadIdx.x==0){
   float value=INFINITY;
   for(int i=0;i<blockDim.x/32;++i)value=fminf(value,fit_scratch[i]);
   fit_scratch[8]=value;
   if(residual_energy){
     int ties=0;for(int i=0;i<pairs;++i)ties+=residual_energy[i]==value;
     fit_scratch[9]=static_cast<float>(ties);
   }
 }
 symbol_sums(single,false,m,score,nullptr,rows);
 __syncthreads();
 bit_sums(false,m,q,kbits,labels,rows,nullptr,norms,physical);
 __syncthreads();
}

// Only the gradient kernel Correction<true> is instantiated; the forward pass runs DetectorWarp below.
template<bool BACKWARD>
__global__ void Correction(bool single,bool update,float limit,int n,int r,int cells,int groups,int m,int q,
 int max_positions,int code_size,bool cache_vectors,int hidden,
 const C* y,const C* factors,const C* whitened,const float* logdet,const float* evidence,
 const float* all_phi,const int64_t* cell_ids,const int64_t* positions,const C* symbols,const int64_t* energies,const int64_t* labels,
 const bool* enabled,const int64_t* bit_indices,const float* bit_signs,const bool* wanted,
 const float* effective,const float* calibration,const float* strength,const float* active,const float* cached,
 const float* fit_cache,const float* blend_kernel,const float* blend_bias,const float* blend_output,const float* blend_output_bias,
 const float* incoming,float* output,
 C* gy,C* gf,C* gw,float* gd,float* ge,float* gl,float* gc,float* gs,float* gb){
 extern __shared__ float memory[];
 int divisor=single?1:m,pairs=m*divisor,kbits=single?q:2*q,nsymbols=single?m:2*m;
 float* cursor=memory;
 float* score=cursor;cursor+=m*(divisor+1);
 float* norms=cursor;cursor+=4*kbits;
 float* e=cursor;cursor+=kbits;
 float* base=cursor;cursor+=kbits;
 float* calibrated=cursor;cursor+=kbits;
 float* gain=cursor;cursor+=kbits;
 float* physical=cursor;cursor+=kbits;
 float* priors=cursor;cursor+=nsymbols;
 float* rows0=cursor;cursor+=nsymbols;
 float* rows=cursor;cursor+=nsymbols;
 float* ds=cursor;cursor+=BACKWARD?pairs:0;
 float* inc=cursor;cursor+=BACKWARD?kbits:0;
 float* dcal=cursor;cursor+=BACKWARD?kbits:0;
 float* dp=cursor;cursor+=BACKWARD?kbits:0;
 float* drows=cursor;cursor+=BACKWARD?nsymbols:0;
 float* drows0=cursor;cursor+=BACKWARD?nsymbols:0;
 float* target_gradient=cursor;cursor+=BACKWARD?kbits*m:0;
 float* calibration_sum=cursor;cursor+=BACKWARD?7:0;
 float* gate_features=cursor;cursor+=BACKWARD?7*kbits:0;
 float* gate_psi=cursor;cursor+=BACKWARD?kbits*hidden:0;
 float* gate_head=cursor;cursor+=BACKWARD?kbits*hidden:0;
 float* gate_dz=cursor;cursor+=BACKWARD?kbits:0;
 float* input_seed=cursor;cursor+=BACKWARD?kbits:0;
 float* strength_seed=cursor;cursor+=BACKWARD?kbits:0;
 float* fit_seed=cursor;cursor+=BACKWARD?kbits:0;
 float* residual_energy=BACKWARD?cursor:nullptr;cursor+=BACKWARD?pairs:0;
 float* blend_sum=cursor;cursor+=BACKWARD?10*hidden+1:0;
 float* prefix=cursor;cursor+=hidden;
 float* fit_scratch=cursor;cursor+=10;
 C* yw=BACKWARD?reinterpret_cast<C*>(memory+((cursor-memory+1)&~1)):nullptr;
 C* gyw=BACKWARD?yw+groups*r:nullptr;
 C* dyg=BACKWARD?gyw+groups*r:nullptr;
 C* local=BACKWARD?dyg+groups*r:nullptr;
 C* aw=BACKWARD?local+(cache_vectors?2*groups*r:0):nullptr;
 C* af=BACKWARD?aw+groups*r*6:nullptr;
 float* ad=BACKWARD?reinterpret_cast<float*>(af+groups*r*r):nullptr;
 int b=BACKWARD?blockIdx.x/cells:blockIdx.x/n;
 int c=BACKWARD?blockIdx.x%cells:cell_ids[blockIdx.x%n];
 float strength_sum=0.f;
 if(!enabled[b*cells+c]){
   if(BACKWARD){
     for(int i=threadIdx.x;i<groups*r*6;i+=blockDim.x)gw[(b*cells+c)*groups*r*6+i]=make_float2(0,0);
     for(int i=threadIdx.x;i<groups*r*r;i+=blockDim.x)gf[(b*cells+c)*groups*r*r+i]=make_float2(0,0);
     for(int i=threadIdx.x;i<groups;i+=blockDim.x)gd[(b*cells+c)*groups+i]=0;
     for(int i=threadIdx.x;i<7;i+=blockDim.x)gc[(b*cells+c)*7+i]=0.f;
     for(int i=threadIdx.x;i<10*hidden+1;i+=blockDim.x)gb[(b*cells+c)*(10*hidden+1)+i]=0.f;
     if(threadIdx.x==0)gs[b*cells+c]=0.f;
     for(int j=0;j<max_positions;++j){int p=positions[c*max_positions+j];if(p<0)break;
       for(int i=threadIdx.x;i<r;i+=blockDim.x)gy[(b*n+p)*r+i]=make_float2(0,0);
     }
   }
   return;
 }
 const C* factor=factors+(b*cells+c)*groups*r*r;
 const C* white=whitened+(b*cells+c)*groups*r*6;
 const float* ld=logdet+(b*cells+c)*groups;
 int energy_count=static_cast<int>(sqrtf(static_cast<float>(groups))+.5f);
 if(BACKWARD){
   for(int i=threadIdx.x;i<groups*r*6;i+=blockDim.x)aw[i]=make_float2(0,0);
   for(int i=threadIdx.x;i<groups*r*r;i+=blockDim.x)af[i]=make_float2(0,0);
   for(int i=threadIdx.x;i<groups;i+=blockDim.x)ad[i]=0.f;
   for(int i=threadIdx.x;i<7;i+=blockDim.x)calibration_sum[i]=0.f;
   for(int i=threadIdx.x;i<10*hidden+1;i+=blockDim.x)blend_sum[i]=0.f;
 }
 int steps=BACKWARD?max_positions:1;
 for(int step=0;step<steps;++step){
   int p=BACKWARD?positions[c*max_positions+step]:blockIdx.x%n;
   if(p<0)break;
   const float* phi=all_phi+p*6;
   for(int j=threadIdx.x;j<kbits;j+=blockDim.x){
     int table=(j/q*n+p)*q+j%q;
     int ix=(b*2+j/q)*code_size+bit_indices[table];
     float value=update?__fmul_rn(__fsub_rn(evidence[ix],effective[ix]),active[b*2+j/q]):evidence[ix];
     e[j]=bit_signs[table]*value;base[j]=bit_signs[table]*effective[ix];
   }
   if(BACKWARD){
     physical_scores(single,p,r,groups,m,q,cache_vectors,y+b*n*r,factor,white,ld,phi,symbols,energies,labels,
                     score,rows0,norms,physical,yw,local,residual_energy,fit_scratch);
   }else{
     const float* values=cached+static_cast<size_t>(b*n+p)*(pairs+kbits);
     for(int pair=threadIdx.x;pair<pairs;pair+=blockDim.x)score[(pair/divisor)*(divisor+1)+pair%divisor]=values[pair];
     for(int j=threadIdx.x;j<kbits;j+=blockDim.x)physical[j]=values[pairs+j];
     __syncthreads();
   }
   float rho=BACKWARD?fit_scratch[8]/r:fit_cache[b*n+p];
   float fit=rho/(1.f+rho);
   for(int h=threadIdx.x;h<hidden;h+=blockDim.x)prefix[h]=blend_bias[h]+blend_kernel[7*hidden+h]*fit;
   for(int j=threadIdx.x;j<kbits;j+=blockDim.x){
     float re=e[j]/(limit+fabsf(e[j])),rl=base[j]/(limit+fabsf(base[j])),rp=physical[j]/(limit+fabsf(physical[j]));
     float features[6]={fabsf(re),fabsf(rl),fabsf(rp),re*rl,re*rp,rl*rp};
     float z=calibration[6];for(int k=0;k<6;++k)z+=calibration[k]*features[k];
     gain[j]=2.f/(1.f+expf(-z));calibrated[j]=gain[j]*e[j];
   }
   __syncthreads();
   for(int index=threadIdx.x;index<nsymbols;index+=blockDim.x){
     int user=index/m,symbol=index%m;float value=0.f;
     for(int k=0;k<q;++k){float e=calibrated[user*q+k];value+=labels[symbol*q+k]?fminf(e,0.f):-fmaxf(e,0.f);}
     priors[index]=value;
   }
   __syncthreads();
   symbol_sums(single,true,m,score,priors,rows);
   __syncthreads();
   bit_sums(true,m,q,kbits,labels,rows,calibrated,norms,physical);
   __syncthreads();
   int gate_width=1;while(gate_width<min(hidden,32))gate_width*=2;
   int gate_lane=threadIdx.x%gate_width;
   unsigned gate_mask=group_mask(gate_width);
   for(int j=threadIdx.x/gate_width;j<kbits;j+=blockDim.x/gate_width){
     int table=(j/q*n+p)*q+j%q,ix=(b*2+j/q)*code_size+bit_indices[table];
     bool owned=wanted[(b*2+j/q)*n+p];
     float delta=0.f,u=0.f,b0=0.f,b1=0.f,inverse=0.f,x0=0.f,x1=0.f,xu=0.f;
     float features[7]={0.f,0.f,0.f,0.f,0.f,0.f,0.f};
     if(gate_lane==0){
       delta=norms[4*j+3]-norms[4*j+2]-physical[j];
       u=strength[0]*delta;b0=base[j]+u;b1=physical[j]+u;
       inverse=1.f/(limit+fabsf(b0)+fabsf(b1)+fabsf(u));
       x0=b0*inverse;x1=b1*inverse;xu=u*inverse;
       features[0]=fabsf(x0);features[1]=fabsf(x1);features[2]=x0*x1;
       features[3]=fabsf(x1-x0);features[4]=fabsf(xu);features[5]=x0*xu;features[6]=x1*xu;
       if(BACKWARD)for(int k=0;k<7;++k)gate_features[j*7+k]=features[k];
     }
     for(int k=0;k<7;++k)features[k]=__shfl_sync(gate_mask,features[k],0,gate_width);
     float z=0.f;
     for(int h=gate_lane;h<hidden;h+=gate_width){
       float value=prefix[h];for(int k=0;k<7;++k)value+=features[k]*blend_kernel[k*hidden+h];
       float activation=fmaxf(value,0.f);z+=activation*blend_output[h];
       if(BACKWARD){gate_head[j*hidden+h]=activation;gate_psi[j*hidden+h]=value>0.f?blend_output[h]:0.f;}
     }
     z=group_sum(z,gate_width)+blend_output_bias[0];
     float a=1.f/(1.f+expf(-z));
     if(!BACKWARD){
       if(gate_lane==0 && owned){
         float correction=__fmul_rn(strength[0],__fmul_rn(bit_signs[table]*delta,active[b*2+j/q]));
         float replacement=__fmul_rn(a,__fsub_rn(bit_signs[table]*physical[j],effective[ix]));
         output[ix]=update?__fadd_rn(__fadd_rn(effective[ix],correction),replacement):__fadd_rn(correction,replacement);
       }
     }else{
       float upstream=0.f,dz=0.f;
       if(gate_lane==0){
         upstream=owned?bit_signs[table]*incoming[ix]:0.f;
         dz=upstream*(physical[j]-base[j])*a*(1.f-a);
       }
       dz=__shfl_sync(gate_mask,dz,0,gate_width);
       for(int h=gate_lane;h<hidden;h+=gate_width){gate_head[j*hidden+h]*=dz;gate_psi[j*hidden+h]*=dz;}
       float df[8];
       for(int k=0;k<8;++k){
         float value=0.f;
         for(int h=gate_lane;h<hidden;h+=gate_width)value+=gate_psi[j*hidden+h]*blend_kernel[k*hidden+h];
         df[k]=group_sum(value,gate_width);
       }
       if(gate_lane==0){
         float dx0=df[0]*sign(x0)+df[2]*x1-df[3]*sign(x1-x0)+df[5]*xu;
         float dx1=df[1]*sign(x1)+df[2]*x0+df[3]*sign(x1-x0)+df[6]*xu;
         float dxu=df[4]*sign(xu)+df[5]*x0+df[6]*x1;
         float dt=-(dx0*x0+dx1*x1+dxu*xu)*inverse;
         float db0=dx0*inverse+dt*sign(b0),db1=dx1*inverse+dt*sign(b1),du=dxu*inverse+dt*sign(u);
         float w=upstream+db0+db1+du;
         inc[j]=strength[0]*w;dp[j]=upstream*a+db1-inc[j];
         input_seed[j]=-upstream*a+db0;strength_seed[j]=delta*w;
         fit_seed[j]=df[7]/((1.f+rho)*(1.f+rho));gate_dz[j]=dz;
       }
     }
   }
   if(BACKWARD){
     __syncthreads();
     if(threadIdx.x==0){
       float value=0.f;
       for(int j=0;j<kbits;++j){strength_sum+=strength_seed[j];value+=fit_seed[j];}
       fit_scratch[0]=value/(r*fit_scratch[9]);
     }
     for(int k=threadIdx.x;k<10*hidden+1;k+=blockDim.x){
       float value=0.f;
       for(int j=0;j<kbits;++j){
         if(k<7*hidden)value+=gate_features[j*7+k/hidden]*gate_psi[j*hidden+k%hidden];
         else if(k<9*hidden)value+=gate_psi[j*hidden+k%hidden];
         else if(k<10*hidden)value+=gate_head[j*hidden+k%hidden];
         else value+=gate_dz[j];
       }
       if(k>=7*hidden && k<8*hidden)value*=fit;
       blend_sum[k]+=value;
     }
     __syncthreads();
     for(int pair=threadIdx.x;pair<pairs;pair+=blockDim.x)
       residual_energy[pair]=residual_energy[pair]==fit_scratch[8]?fit_scratch[0]:0.f;
     for(int index=threadIdx.x;index<kbits*m;index+=blockDim.x){
       int j=index/m,symbol=index%m,user=j/q,v=labels[symbol*q+j%q];
       float z=rows[user*m+symbol]+own_prior(symbol,j%q,q,labels,calibrated+user*q);
       target_gradient[index]=inc[j]*(2*v-1)*expf(z-norms[4*j+2+v]);
     }
     __syncthreads();
     for(int index=threadIdx.x;index<nsymbols;index+=blockDim.x){
       int user=index/m,symbol=index%m;float value=0.f;
       for(int j=0;j<q;++j)value+=target_gradient[(user*q+j)*m+symbol];
       drows[index]=value;
     }
     __syncthreads();
     for(int index=threadIdx.x;index<nsymbols;index+=blockDim.x){
       int user=index/m,symbol=index%m;float value=0.f;
       if(!single)for(int other=0;other<m;++other){
         float z=score_at(score,user,symbol,other,divisor+1)+priors[index];
         value+=drows[(1-user)*m+other]*expf(z-rows[(1-user)*m+other]);
       }
       drows0[index]=value;
     }
     __syncthreads();
     int width=min(m,32),lane=threadIdx.x%width;
     for(int j=threadIdx.x/width;j<kbits;j+=blockDim.x/width){
       int user=j/q,target=j%q;float value=0.f;
       for(int symbol=lane;symbol<m;symbol+=width){
         float coefficient=labels[symbol*q+target]-(calibrated[j]>0.f);
         value+=coefficient*drows0[user*m+symbol];
         for(int k=0;k<q;++k)if(k!=target)value+=coefficient*target_gradient[(user*q+k)*m+symbol];
       }
       value=group_sum(value,width);
       if(lane==0)dcal[j]=value;
     }
     __syncthreads();
     for(int j=threadIdx.x;j<kbits;j+=blockDim.x){
       float de=limit+fabsf(e[j]),dl=limit+fabsf(base[j]),df=limit+fabsf(physical[j]);
       float re=e[j]/de,rl=base[j]/dl,rp=physical[j]/df;
       float dz=dcal[j]*e[j]*gain[j]*(1.f-.5f*gain[j]);
       float gradient_e=dcal[j]*gain[j]+dz*(calibration[0]*sign(re)+calibration[3]*rl+calibration[4]*rp)*limit/(de*de);
       float gradient_l=dz*(calibration[1]*sign(rl)+calibration[3]*re+calibration[5]*rp)*limit/(dl*dl);
       dp[j]+=dz*(calibration[2]*sign(rp)+calibration[4]*re+calibration[5]*rl)*limit/(df*df);
       int table=(j/q*n+p)*q+j%q,ix=(b*2+j/q)*code_size+bit_indices[table];
       ge[ix]=bit_signs[table]*gradient_e;gl[ix]=bit_signs[table]*(input_seed[j]+gradient_l);
     }
     for(int k=threadIdx.x;k<7;k+=blockDim.x){
       float value=0.f;
       for(int j=0;j<kbits;++j){
         float re=e[j]/(limit+fabsf(e[j])),rl=base[j]/(limit+fabsf(base[j])),rp=physical[j]/(limit+fabsf(physical[j]));
         float features[7]={fabsf(re),fabsf(rl),fabsf(rp),re*rl,re*rp,rl*rp,1.f};
         value+=dcal[j]*e[j]*gain[j]*(1.f-.5f*gain[j])*features[k];
       }
       calibration_sum[k]+=value;
     }
     __syncthreads();
     for(int index=threadIdx.x;index<nsymbols;index+=blockDim.x){
       int user=index/m,symbol=index%m;float value=0.f;
       for(int k=0;k<q;++k){
         int j=user*q+k,v=labels[symbol*q+k];
         value+=dp[j]*(2*v-1)*expf(rows0[index]-norms[4*j+v]);
       }
       drows0[index]=value;
     }
     __syncthreads();
     for(int pair=threadIdx.x;pair<pairs;pair+=blockDim.x){
       int a=pair/divisor,beta=pair%divisor;float s=score[a*(divisor+1)+beta];
       float value=drows[a]*expf(s+(single?0.f:priors[m+beta])-rows[a])+drows0[a]*expf(s-rows0[a]);
       if(!single)value+=drows[m+beta]*expf(s+priors[a]-rows[m+beta])+drows0[m+beta]*expf(s-rows0[m+beta]);
       ds[pair]=value;
     }
     __syncthreads();
     for(int i=threadIdx.x;i<groups*r*7;i+=blockDim.x){
       int g=i/(r*7),row=(i/7)%r,component=i%7;C value=make_float2(0,0);
       for(int pair=0;pair<pairs;++pair)if(energies[pair/divisor]*energy_count+energies[pair%divisor]==g){
         C derivative=scale(residual(pair,g,row,r,divisor,symbols,white,phi,yw,cache_vectors?local:nullptr),2.f*(residual_energy[pair]-ds[pair]));
         if(component<6){
           C symbol=symbols[component<3?pair/divisor:pair%divisor];
           derivative=scale(mul(derivative,conj(symbol)),-phi[component]);
         }
         value=add(value,derivative);
       }
       if(component<6)aw[(g*r+row)*6+component]=add(aw[(g*r+row)*6+component],value);
       else gyw[g*r+row]=value;
     }
     for(int g=threadIdx.x;g<groups;g+=blockDim.x){
       float value=0.f;
       for(int pair=0;pair<pairs;++pair)if(energies[pair/divisor]*energy_count+energies[pair%divisor]==g)value-=ds[pair];
       ad[g]+=value;
     }
     __syncthreads();
     for(int g=threadIdx.x;g<groups;g+=blockDim.x){
       for(int row=r-1;row>=0;--row){
         C value=gyw[g*r+row];
         for(int col=row+1;col<r;++col)value=sub(value,mul(conj(factor[(g*r+col)*r+row]),dyg[g*r+col]));
         dyg[g*r+row]=scale(value,1.f/factor[(g*r+row)*r+row].x);
       }
     }
     __syncthreads();
     for(int i=threadIdx.x;i<groups*r*r;i+=blockDim.x){
       int g=i/(r*r),row=(i/r)%r,col=i%r;
       if(col<=row){C value=mul(dyg[g*r+row],conj(yw[g*r+col]));if(row==col)value.y=0;af[i]=sub(af[i],value);}
     }
     for(int row=threadIdx.x;row<r;row+=blockDim.x){
       C value=make_float2(0,0);for(int g=0;g<groups;++g)value=add(value,dyg[g*r+row]);
       gy[(b*n+p)*r+row]=value;
     }
     __syncthreads();
   }
 }
 if(BACKWARD){
   for(int i=threadIdx.x;i<groups*r*6;i+=blockDim.x)gw[(b*cells+c)*groups*r*6+i]=aw[i];
   for(int i=threadIdx.x;i<groups*r*r;i+=blockDim.x)gf[(b*cells+c)*groups*r*r+i]=af[i];
   for(int i=threadIdx.x;i<groups;i+=blockDim.x)gd[(b*cells+c)*groups+i]=ad[i];
   for(int i=threadIdx.x;i<7;i+=blockDim.x)gc[(b*cells+c)*7+i]=calibration_sum[i];
   for(int i=threadIdx.x;i<10*hidden+1;i+=blockDim.x)gb[(b*cells+c)*(10*hidden+1)+i]=blend_sum[i];
   if(threadIdx.x==0)gs[b*cells+c]=strength_sum;
 }
}

Status InitializeCorrection(bool backward,int* limit){
 static std::mutex mutex;
 std::lock_guard<std::mutex> lock(mutex);
 int device;
 auto status=cudaGetDevice(&device);
 if(status==cudaSuccess)status=cudaDeviceGetAttribute(limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);
 if(status==cudaSuccess && backward)
   status=cudaFuncSetAttribute(Correction<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,*limit);
 if(status!=cudaSuccess)return errors::Internal("Cannot configure correction workspace: ",cudaGetErrorString(status));
 return OkStatus();
}



// ---- The whole forward detector in one kernel with one warp per resource element (one or two UEs, up to 16-QAM,
// r <= 4, at most 9 energy groups): symbol-pair scores, their LLRs without decoder feedback, and the correction from
// the decoder's evidence. The constellation is a run-time parameter. All tables use the maximal sizes (WARP_SYMBOLS,
// WARP_BITS) and are padded so that unused entries drop out by themselves (scores of -inf, evidence and labels of
// zero); no loop over symbols or bits tests its bounds. The sums follow the shuffle trees of group_sum, so the results
// equal those of physical_scores and Correction, which the gradient kernel uses. NRX_IDD_MAXLOG=1 replaces
// log-sum-exp by maxima.
#define WARP_RES 4
#define WARP_SYMBOLS 16
#define WARP_BITS 4
#define WARP_HIDDEN 32
#define WARP_ANTENNAS 4
#define WARP_GROUPS 9
__device__ float tree_sum(float* v,int width){for(int d=width/2;d;d>>=1)for(int i=0;i<d;++i)v[i]+=v[i+d];return v[0];}
__device__ C warp_residual(int a,int beta,int g,int row,int r,const C* symbols,const C* yw,const C* local){
 return sub(sub(yw[g*r+row],mul(local[(g*r+row)*2],symbols[a])),mul(local[(g*r+row)*2+1],symbols[beta]));
}
__device__ float warp_min(float v){for(int o=16;o;o>>=1)v=fminf(v,__shfl_xor_sync(0xffffffffu,v,o));return v;}
__device__ float padded_symbol_sum(bool maxlog,int user,int symbol,const float* score,const float* priors){
 float v[WARP_SYMBOLS],maximum=-INFINITY;
 for(int other=0;other<WARP_SYMBOLS;++other){
   float value=user==0?score[symbol*WARP_SYMBOLS+other]:score[other*WARP_SYMBOLS+symbol];
   if(priors)value+=priors[(1-user)*WARP_SYMBOLS+other];
   v[other]=value;maximum=fmaxf(maximum,value);
 }
 for(int other=0;other<WARP_SYMBOLS;++other)v[other]=expf(v[other]-maximum);
 float sum=tree_sum(v,WARP_SYMBOLS);
 return maxlog?maximum:maximum+logf(sum);
}
__device__ float padded_prior(int target,unsigned code,const float* evidence){
 float value=0.f;
 for(int k=0;k<WARP_BITS;++k)if(k!=target)value+=(code>>k)&1?fminf(evidence[k],0.f):-fmaxf(evidence[k],0.f);
 return value;
}
// LLRs of the kbits bits of a resource element, lane j returning bit j. metric [2*WARP_BITS][WARP_SYMBOLS] (shared)
// holds the padded symbol metrics of every bit's UE and is overwritten; maxima is shared scratch. The exponentials
// of all bits are spread over the warp's 32 lanes; every bit's sum keeps the order of tree_sum.
__device__ float padded_bit_llrs(bool maxlog,int lane,int kbits,int q,const unsigned* codes,float* metric,float* maxima){
 const int user=lane>=q,target=lane-user*q;
 if(lane<kbits){
   float m0=-INFINITY,m1=-INFINITY;
   for(int symbol=0;symbol<WARP_SYMBOLS;++symbol){
     float value=metric[lane*WARP_SYMBOLS+symbol];
     if((codes[symbol]>>target)&1)m1=fmaxf(m1,value);else m0=fmaxf(m0,value);
   }
   maxima[lane*2]=m0;maxima[lane*2+1]=m1;
 }
 __syncwarp();
 if(!maxlog)for(int item=lane;item<2*WARP_BITS*WARP_SYMBOLS;item+=32){
   int bit=item/WARP_SYMBOLS,symbol=item%WARP_SYMBOLS;
   if(bit<kbits){
     int one=(codes[symbol]>>(bit>=q?bit-q:bit))&1;
     metric[item]=expf(metric[item]-maxima[bit*2+one]);
   }
 }
 __syncwarp();
 if(lane>=kbits)return 0.f;
 float m0=maxima[lane*2],m1=maxima[lane*2+1];
 if(maxlog)return m1-m0;
 float s0[WARP_SYMBOLS],s1[WARP_SYMBOLS];
 for(int symbol=0;symbol<WARP_SYMBOLS;++symbol){
   bool one=(codes[symbol]>>target)&1;float value=metric[lane*WARP_SYMBOLS+symbol];
   s0[symbol]=one?0.f:value;s1[symbol]=one?value:0.f;
 }
 float n0=m0+logf(tree_sum(s0,WARP_SYMBOLS)),n1=m1+logf(tree_sum(s1,WARP_SYMBOLS));
 return n1-n0;
}
__global__ void DetectorWarp(bool single,bool update,float limit,int total,int n,int r,int cells,int groups,int code_size,
 int hidden,int m,int q,bool maxlog,
 const C* y,const C* factors,const C* whitened,const float* logdet,const float* all_phi,const int64_t* cell_ids,
 const C* symbols,const int64_t* energies,const int64_t* labels,const bool* enabled,
 const float* evidence,const int64_t* bit_indices,const float* bit_signs,const bool* wanted,const float* effective,
 const float* calibration,const float* strength,const float* active,const float* blend_kernel,const float* blend_bias,
 const float* blend_output,const float* blend_output_bias,float* output){
 const int divisor=single?1:m,pairs=m*divisor,kbits=single?q:2*q,shift=single?0:q;
 int warp=threadIdx.x/32,lane=threadIdx.x%32,index=blockIdx.x*WARP_RES+warp;
 if(index>=total)return;
 int b=index/n,p=index%n,c=cell_ids[p];
 if(!enabled[b*cells+c])return;
 __shared__ C yw_all[WARP_RES][WARP_GROUPS*WARP_ANTENNAS];
 __shared__ C local_all[WARP_RES][2*WARP_GROUPS*WARP_ANTENNAS];
 __shared__ float score_all[WARP_RES][WARP_SYMBOLS*WARP_SYMBOLS];
 __shared__ float rows_all[WARP_RES][2*WARP_SYMBOLS];
 __shared__ float priors_all[WARP_RES][2*WARP_SYMBOLS];
 __shared__ float calibrated_all[WARP_RES][2*WARP_BITS];
 __shared__ unsigned codes_all[WARP_RES][WARP_SYMBOLS];
 __shared__ float metric_all[WARP_RES][2*WARP_BITS*WARP_SYMBOLS];
 __shared__ float maxima_all[WARP_RES][4*WARP_BITS];
 C* yw=yw_all[warp];C* local=local_all[warp];float* score=score_all[warp];float* rows=rows_all[warp];
 float* priors=priors_all[warp];float* calibrated=calibrated_all[warp];
 unsigned* codes=codes_all[warp];float* metric=metric_all[warp];float* maxima=maxima_all[warp];
 // lane -> (UE, symbol) for the symbol sums, and lane -> (UE, bit) for the bit LLRs
 const int user=lane/WARP_SYMBOLS,symbol=lane%WARP_SYMBOLS;
 const bool symbol_lane=symbol<m && (user==0 || !single);
 const int bit_user=lane>=q,target=lane-bit_user*q;
 if(lane<WARP_SYMBOLS){
   unsigned code=0;
   if(lane<m)for(int k=0;k<WARP_BITS;++k)if(k<q && labels[lane*q+k])code|=1u<<k;
   codes[lane]=code;
 }
 if(lane<2*WARP_BITS)calibrated[lane]=0.f;
 if(pairs<WARP_SYMBOLS*WARP_SYMBOLS)for(int i=lane;i<WARP_SYMBOLS*WARP_SYMBOLS;i+=32)score[i]=-INFINITY;
 const C* factor=factors+static_cast<size_t>(b*cells+c)*groups*r*r;
 const C* white=whitened+static_cast<size_t>(b*cells+c)*groups*r*6;
 const float* ld=logdet+(b*cells+c)*groups;
 const float* phi=all_phi+p*6;
 const C* received=y+static_cast<size_t>(b)*n*r;
 if(lane<groups){
   int g=lane;
   for(int row=0;row<r;++row){
     C value=received[p*r+row];
     for(int col=0;col<row;++col)value=sub(value,mul(factor[(g*r+row)*r+col],yw[g*r+col]));
     yw[g*r+row]=scale(value,1.f/factor[(g*r+row)*r+row].x);
   }
 }
 for(int i=lane;i<groups*r*2;i+=32){
   int row=i/2,ue=i%2;C value=make_float2(0,0);
   for(int k=0;k<3;++k)value=add(value,scale(white[row*6+ue*3+k],phi[ue*3+k]));
   local[i]=value;
 }
 __syncwarp();
 int energy_count=static_cast<int>(sqrtf(static_cast<float>(groups))+.5f);
 float minimum=INFINITY;
 for(int pair=lane;pair<pairs;pair+=32){
   int a=pair>>shift,beta=pair&(divisor-1);
   int g=energies[a]*energy_count+energies[beta];
   float value=-ld[g],energy=0.f;
   for(int row=0;row<r;++row){
     float term=norm(warp_residual(a,beta,g,row,r,symbols,yw,local));
     value-=term;energy+=term;
   }
   score[a*WARP_SYMBOLS+beta]=value;
   minimum=fminf(minimum,energy);
 }
 minimum=warp_min(minimum);
 __syncwarp();
 // LLRs without decoder feedback
 rows[lane]=symbol_lane?padded_symbol_sum(maxlog,user,symbol,score,nullptr):-INFINITY;
 __syncwarp();
 for(int item=lane;item<2*WARP_BITS*WARP_SYMBOLS;item+=32){
   int bit=item/WARP_SYMBOLS;
   if(bit<kbits)metric[item]=rows[(bit>=q)*WARP_SYMBOLS+item%WARP_SYMBOLS];
 }
 __syncwarp();
 const float physical=padded_bit_llrs(maxlog,lane,kbits,q,codes,metric,maxima);
 // Calibrated evidence of the decoder
 float base=0.f;int table=0,ix=0;
 if(lane<kbits){
   table=(bit_user*n+p)*q+target;ix=(b*2+bit_user)*code_size+bit_indices[table];
   float value=update?__fmul_rn(__fsub_rn(evidence[ix],effective[ix]),active[b*2+bit_user]):evidence[ix];
   float e=bit_signs[table]*value;base=bit_signs[table]*effective[ix];
   float re=e/(limit+fabsf(e)),rl=base/(limit+fabsf(base)),rp=physical/(limit+fabsf(physical));
   float features[6]={fabsf(re),fabsf(rl),fabsf(rp),re*rl,re*rp,rl*rp};
   float z=calibration[6];for(int k=0;k<6;++k)z+=calibration[k]*features[k];
   float gain=2.f/(1.f+expf(-z));calibrated[bit_user*WARP_BITS+target]=gain*e;
 }
 __syncwarp();
 {
   float value=0.f;
   for(int k=0;k<WARP_BITS;++k){
     float v=calibrated[user*WARP_BITS+k];value+=(codes[symbol]>>k)&1?fminf(v,0.f):-fmaxf(v,0.f);
   }
   priors[lane]=symbol_lane?value:0.f;
 }
 __syncwarp();
 rows[lane]=symbol_lane?padded_symbol_sum(maxlog,user,symbol,score,single?nullptr:priors):-INFINITY;
 __syncwarp();
 for(int item=lane;item<2*WARP_BITS*WARP_SYMBOLS;item+=32){
   int bit=item/WARP_SYMBOLS,other=item%WARP_SYMBOLS;
   if(bit<kbits){
     int ue=bit>=q;
     metric[item]=rows[ue*WARP_SYMBOLS+other]+padded_prior(bit-ue*q,codes[other],calibrated+ue*WARP_BITS);
   }
 }
 __syncwarp();
 float llr=padded_bit_llrs(maxlog,lane,kbits,q,codes,metric,maxima);
 if(lane<kbits){
   float delta=llr-physical;
   float rho=minimum/r,fit=rho/(1.f+rho);
   float u=strength[0]*delta,b0=base+u,b1=physical+u;
   float inverse=1.f/(limit+fabsf(b0)+fabsf(b1)+fabsf(u));
   float x0=b0*inverse,x1=b1*inverse,xu=u*inverse;
   float features[7]={fabsf(x0),fabsf(x1),x0*x1,fabsf(x1-x0),fabsf(xu),x0*xu,x1*xu};
   int gate_width=1;while(gate_width<min(hidden,WARP_HIDDEN))gate_width*=2;
   float units[WARP_HIDDEN];
   for(int h=0;h<gate_width;++h)units[h]=0.f;
   for(int h=0;h<hidden;++h){
     float value=blend_bias[h]+blend_kernel[7*hidden+h]*fit;
     for(int k=0;k<7;++k)value+=features[k]*blend_kernel[k*hidden+h];
     units[h]=fmaxf(value,0.f)*blend_output[h];
   }
   float z=tree_sum(units,gate_width)+blend_output_bias[0];
   float a=1.f/(1.f+expf(-z));
   if(wanted[(b*2+bit_user)*n+p]){
     float correction=__fmul_rn(strength[0],__fmul_rn(bit_signs[table]*delta,active[b*2+bit_user]));
     float replacement=__fmul_rn(a,__fsub_rn(bit_signs[table]*physical,effective[ix]));
     output[ix]=update?__fadd_rn(__fadd_rn(effective[ix],correction),replacement):__fadd_rn(correction,replacement);
   }
 }
}
bool MaxLog(){const char* v=std::getenv("NRX_IDD_MAXLOG");return v && v[0]=='1';}

__global__ void InitializeCorrected(int size,const float* base,const float* strength,float* output){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x)
   output[i]=__fadd_rn(base[i],__fmul_rn(strength[0],0.f));
}

void LaunchCorrection(OpKernelContext* c,bool backward,bool single,bool update,float llr_limit,int b,int n,int r,
 int cells,int groups,int m,int q,int t,int code_size,int limit,const Tensor* const* in,Tensor* const* out){
 int hidden=in[19]->dim_size(1);
 size_t bytes=sizeof(float)*ScalarWorkspace(backward,single,m,q,hidden);
 if(backward)bytes+=sizeof(C)*3*groups*r;
 if(backward)bytes+=sizeof(C)*groups*r*(6+r)+sizeof(float)*groups;
 OP_REQUIRES(c,bytes<=static_cast<size_t>(limit),errors::InvalidArgument("The constellation/antenna workspace exceeds this GPU's shared-memory capacity"));
 size_t vectors=sizeof(C)*2*groups*r;
 bool cache_vectors=backward && bytes+vectors<=static_cast<size_t>(limit);
 if(cache_vectors)bytes+=vectors;
 auto stream=c->eigen_device<Eigen::GpuDevice>().stream();
 auto y=reinterpret_cast<const C*>(in[0]->flat<complex64>().data());
 auto factor=reinterpret_cast<const C*>(in[1]->flat<complex64>().data());
 auto white=reinterpret_cast<const C*>(in[2]->flat<complex64>().data());
 auto symbols=reinterpret_cast<const C*>(in[8]->flat<complex64>().data());
 if(backward){
   for(int i:{4,5})OP_REQUIRES(c,cudaMemsetAsync(out[i]->flat<float>().data(),0,out[i]->NumElements()*sizeof(float),stream)==cudaSuccess,errors::Internal("Cannot clear correction input gradients"));
   OP_REQUIRES_OK(c,GpuLaunchKernel(Correction<true>,b*cells,256,bytes,stream,single,update,llr_limit,n,r,cells,groups,m,q,t,code_size,cache_vectors,hidden,
     y,factor,white,in[3]->flat<float>().data(),in[4]->flat<float>().data(),in[5]->flat<float>().data(),
     in[6]->flat<int64_t>().data(),in[7]->flat<int64_t>().data(),symbols,in[9]->flat<int64_t>().data(),in[10]->flat<int64_t>().data(),
     in[11]->flat<bool>().data(),in[12]->flat<int64_t>().data(),in[13]->flat<float>().data(),in[14]->flat<bool>().data(),
     in[15]->flat<float>().data(),in[16]->flat<float>().data(),in[17]->flat<float>().data(),in[18]->flat<float>().data(),
     static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),in[19]->flat<float>().data(),
     in[20]->flat<float>().data(),in[21]->flat<float>().data(),in[22]->flat<float>().data(),
     in[23]->flat<float>().data(),static_cast<float*>(nullptr),
     reinterpret_cast<C*>(out[0]->flat<complex64>().data()),reinterpret_cast<C*>(out[1]->flat<complex64>().data()),
     reinterpret_cast<C*>(out[2]->flat<complex64>().data()),out[3]->flat<float>().data(),out[4]->flat<float>().data(),
     out[5]->flat<float>().data(),out[6]->flat<float>().data(),out[7]->flat<float>().data(),out[8]->flat<float>().data()));
 }else{
   if(update){
     int size=out[0]->NumElements();
     OP_REQUIRES_OK(c,GpuLaunchKernel(InitializeCorrected,(size+255)/256,256,0,stream,size,
       in[15]->flat<float>().data(),in[17]->flat<float>().data(),out[0]->flat<float>().data()));
   } else OP_REQUIRES(c,cudaMemsetAsync(out[0]->flat<float>().data(),0,out[0]->NumElements()*sizeof(float),stream)==cudaSuccess,errors::Internal("Cannot clear correction padding"));
   OP_REQUIRES(c,m<=WARP_SYMBOLS && q<=WARP_BITS && r<=WARP_ANTENNAS && groups<=WARP_GROUPS && hidden<=WARP_HIDDEN,
     errors::Unimplemented("The forward detector supports up to 16-QAM and four antennas"));
   OP_REQUIRES_OK(c,GpuLaunchKernel(DetectorWarp,(b*n+WARP_RES-1)/WARP_RES,32*WARP_RES,0,stream,single,update,llr_limit,b*n,n,r,
     cells,groups,code_size,hidden,m,q,MaxLog(),y,factor,white,in[3]->flat<float>().data(),in[5]->flat<float>().data(),
     in[6]->flat<int64_t>().data(),symbols,in[9]->flat<int64_t>().data(),in[10]->flat<int64_t>().data(),
     in[11]->flat<bool>().data(),in[4]->flat<float>().data(),in[12]->flat<int64_t>().data(),in[13]->flat<float>().data(),
     in[14]->flat<bool>().data(),in[15]->flat<float>().data(),in[16]->flat<float>().data(),in[17]->flat<float>().data(),
     in[18]->flat<float>().data(),in[19]->flat<float>().data(),in[20]->flat<float>().data(),in[21]->flat<float>().data(),
     in[22]->flat<float>().data(),out[0]->flat<float>().data()));
 }
}
