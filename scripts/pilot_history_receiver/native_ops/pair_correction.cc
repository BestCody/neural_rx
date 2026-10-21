#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <mutex>
#include <cmath>
using namespace tensorflow;

#define INPUTS \
 .Input("received: complex64").Input("factors: complex64") \
 .Input("whitened: complex64").Input("logdet: float").Input("evidence: float") \
 .Input("phi: float").Input("cells: int64").Input("positions: int64") \
 .Input("symbols: complex64").Input("energies: int64").Input("labels: int64").Input("enabled: bool") \
 .Input("bit_indices: int64").Input("bit_signs: float").Input("wanted: bool") \
 .Input("effective_input: float").Input("calibration: float") \
 .Input("strength: float").Input("active: float").Input("physical_scores: float") \
 .Input("fit: float").Input("blend_kernel: float").Input("blend_bias: float") \
 .Input("blend_output: float").Input("blend_output_bias: float")

REGISTER_OP("PilotPairCorrection") INPUTS .Attr("single_user: bool = false").Attr("llr_limit: float").Attr("apply_update: bool = false").Output("delta: float")
 .SetShapeFn([](shape_inference::InferenceContext* c) {c->set_output(0,c->input(4)); return OkStatus();});
REGISTER_OP("PilotPairCorrectionGrad") INPUTS .Attr("single_user: bool = false").Attr("llr_limit: float").Attr("apply_update: bool = false").Input("incoming: float")
 .Output("received_gradient: complex64").Output("factor_gradient: complex64")
 .Output("white_gradient: complex64").Output("logdet_gradient: float")
 .Output("evidence_gradient: float").Output("input_gradient: float").Output("calibration_gradient: float").Output("strength_gradient: float")
 .Output("blend_gradient: float")
 .SetShapeFn([](shape_inference::InferenceContext* c) {
   for(int i=0;i<5;++i)c->set_output(i,c->input(i));
   c->set_output(5,c->input(15));
   c->set_output(6,c->MakeShape({c->Dim(c->input(1),0),c->Dim(c->input(1),1),7}));
   c->set_output(7,c->MakeShape({c->Dim(c->input(1),0),c->Dim(c->input(1),1)}));
   shape_inference::DimensionHandle count;
   TF_RETURN_IF_ERROR(c->Multiply(c->Dim(c->input(21),1),10,&count));
   TF_RETURN_IF_ERROR(c->Add(count,1,&count));
   c->set_output(8,c->MakeShape({c->Dim(c->input(1),0),c->Dim(c->input(1),1),count}));
   return OkStatus();});

Status InitializeCorrection(bool,int*);
void LaunchCorrection(OpKernelContext*, bool, bool, bool, float, int, int, int, int, int, int, int, int,int,int,
                      const Tensor* const*, Tensor* const*);

class PairCorrectionOp : public OpKernel {
 bool gradient_, single_, update_;
 float llr_limit_;
 std::once_flag initialized_;
 int shared_limit_=0;
 Status initialization_status_;
 public:
 explicit PairCorrectionOp(OpKernelConstruction* c):OpKernel(c),gradient_(c->def().op()=="PilotPairCorrectionGrad"){
   OP_REQUIRES_OK(c,c->GetAttr("single_user",&single_));
   OP_REQUIRES_OK(c,c->GetAttr("llr_limit",&llr_limit_));
   OP_REQUIRES_OK(c,c->GetAttr("apply_update",&update_));
   OP_REQUIRES(c,std::isfinite(llr_limit_) && llr_limit_>0,errors::InvalidArgument("Invalid decoder LLR limit"));
 }
 void Compute(OpKernelContext* c) override {
   const Tensor* in[26]; for(int i=0;i<c->num_inputs();++i)in[i]=&c->input(i);
   OP_REQUIRES(c,in[0]->dims()==3 && in[1]->dims()==5 && in[2]->dims()==5 && in[3]->dims()==3 &&
     in[4]->dims()==4 && in[5]->dims()==3 && in[6]->dims()==1 && in[7]->dims()==2 &&
     in[8]->dims()==1 && in[9]->dims()==1 && in[10]->dims()==2 && in[11]->dims()==2 &&
     in[12]->dims()==3 && in[13]->dims()==3 && in[14]->dims()==3 && in[21]->dims()==2,
     errors::InvalidArgument("Invalid pair-correction ranks"));
   int b=in[0]->dim_size(0),n=in[0]->dim_size(1),r=in[0]->dim_size(2),cells=in[1]->dim_size(1),
       groups=in[1]->dim_size(2),m=in[8]->dim_size(0),q=in[10]->dim_size(1),t=in[7]->dim_size(1),
       code_size=in[4]->dim_size(2)*in[4]->dim_size(3);
   int hidden=in[21]->dim_size(1);
   OP_REQUIRES(c,b>0 && n>0 && r>0 && cells>0 && groups>0 && m>1 && q>0 && q<16 && m==(1<<q) && hidden>0,
     errors::InvalidArgument("Invalid pair-correction dimensions"));
   OP_REQUIRES(c,in[1]->shape()==TensorShape({b,cells,groups,r,r}) &&
     in[2]->shape()==TensorShape({b,cells,groups,r,6}) && in[3]->shape()==TensorShape({b,cells,groups}) &&
     in[4]->dim_size(0)==b && in[4]->dim_size(1)==2 && code_size>=n*q && in[5]->shape()==TensorShape({n,2,3}) &&
     in[6]->dim_size(0)==n && in[7]->dim_size(0)==cells && in[9]->dim_size(0)==m &&
     in[10]->shape()==TensorShape({m,q}) && in[11]->shape()==TensorShape({b,cells}) &&
     in[12]->shape()==TensorShape({2,n,q}) && in[13]->shape()==TensorShape({2,n,q}) &&
     in[14]->shape()==TensorShape({b,2,n}) && in[15]->shape()==in[4]->shape() &&
     in[16]->shape()==TensorShape({7}) && in[17]->dims()==0 && in[18]->shape()==TensorShape({b,2}) &&
     in[19]->shape()==TensorShape({b,n,m*(single_?1:m)+(single_?q:2*q)}) &&
     in[20]->shape()==TensorShape({b,n}) && in[21]->shape()==TensorShape({8,hidden}) &&
     in[22]->shape()==TensorShape({hidden}) && in[23]->shape()==TensorShape({hidden}) && in[24]->dims()==0,
     errors::InvalidArgument("Inconsistent pair-correction shapes"));
   Tensor* out[9];
   if(gradient_) {
     OP_REQUIRES(c,in[25]->shape()==in[4]->shape(),errors::InvalidArgument("Invalid correction gradient"));
     for(int i=0;i<5;++i)OP_REQUIRES_OK(c,c->allocate_output(i,in[i]->shape(),&out[i]));
     OP_REQUIRES_OK(c,c->allocate_output(5,in[15]->shape(),&out[5]));
     OP_REQUIRES_OK(c,c->allocate_output(6,TensorShape({b,cells,7}),&out[6]));
     OP_REQUIRES_OK(c,c->allocate_output(7,TensorShape({b,cells}),&out[7]));
     OP_REQUIRES_OK(c,c->allocate_output(8,TensorShape({b,cells,10*hidden+1}),&out[8]));
   } else OP_REQUIRES_OK(c,c->allocate_output(0,in[4]->shape(),&out[0]));
   std::call_once(initialized_,[&]{initialization_status_=InitializeCorrection(gradient_,&shared_limit_);});
   OP_REQUIRES_OK(c,initialization_status_);
   LaunchCorrection(c,gradient_,single_,update_,llr_limit_,b,n,r,cells,groups,m,q,t,code_size,shared_limit_,in,out);
 }
};
REGISTER_KERNEL_BUILDER(Name("PilotPairCorrection").Device(DEVICE_GPU),PairCorrectionOp);
REGISTER_KERNEL_BUILDER(Name("PilotPairCorrectionGrad").Device(DEVICE_GPU),PairCorrectionOp);
