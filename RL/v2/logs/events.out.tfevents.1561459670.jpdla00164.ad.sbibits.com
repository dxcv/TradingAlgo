       ЃK"	  ѕ~DзAbrain.Event:2Ѕ\ ч      w1уВ	АХѕ~DзA"ѓЭ
n
PlaceholderPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
g
truncated_normal/shapeConst*
valueB"   P   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
_output_shapes

:P*
seed2 *

seed *
T0*
dtype0

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes

:P
m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
_output_shapes

:P*
T0
|
Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:P*
	container *
shape
:P
Є
Variable/AssignAssignVariabletruncated_normal*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:P
R
ConstConst*
valueBP*
з#<*
dtype0*
_output_shapes
:P
v

Variable_1
VariableV2*
dtype0*
_output_shapes
:P*
	container *
shape:P*
shared_name 

Variable_1/AssignAssign
Variable_1Const*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:P
i
truncated_normal_1/shapeConst*
valueB"P   (   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*
_output_shapes

:P(*
seed2 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:P(
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes

:P(
~

Variable_2
VariableV2*
shape
:P(*
shared_name *
dtype0*
_output_shapes

:P(*
	container 
Ќ
Variable_2/AssignAssign
Variable_2truncated_normal_1*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(*
use_locking(*
T0
o
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:P(
T
Const_1Const*
valueB(*
з#<*
dtype0*
_output_shapes
:(
v

Variable_3
VariableV2*
_output_shapes
:(*
	container *
shape:(*
shared_name *
dtype0

Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:(
i
truncated_normal_2/shapeConst*
valueB"(      *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
_output_shapes

:(*
seed2 *

seed *
T0

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes

:(*
T0
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:(
~

Variable_4
VariableV2*
shared_name *
dtype0*
_output_shapes

:(*
	container *
shape
:(
Ќ
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:(
T
Const_2Const*
_output_shapes
:*
valueB*
з#<*
dtype0
v

Variable_5
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0

Variable_5/AssignAssign
Variable_5Const_2*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:*
T0*
_class
loc:@Variable_5
i
truncated_normal_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes

:*
T0
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:
~

Variable_6
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
Ќ
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:*
T0*
_class
loc:@Variable_6
T
Const_3Const*
valueB*
з#<*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 

Variable_7/AssignAssign
Variable_7Const_3*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
P
summaries/RankConst*
_output_shapes
: *
value	B :*
dtype0
W
summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
W
summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
summaries/rangeRangesummaries/range/startsummaries/Ranksummaries/range/delta*
_output_shapes
:*

Tidx0
v
summaries/MeanMeanVariable_7/readsummaries/range*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
z
summaries/mean/layer2/bias/tagsConst*+
value"B  Bsummaries/mean/layer2/bias*
dtype0*
_output_shapes
: 
}
summaries/mean/layer2/biasScalarSummarysummaries/mean/layer2/bias/tagssummaries/Mean*
_output_shapes
: *
T0
W

stddev/subSubVariable_7/readsummaries/Mean*
_output_shapes
:*
T0
H
stddev/SquareSquare
stddev/sub*
T0*
_output_shapes
:
V
stddev/ConstConst*
_output_shapes
:*
valueB: *
dtype0
l

stddev/SumSumstddev/Squarestddev/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
@
stddev/SqrtSqrt
stddev/Sum*
T0*
_output_shapes
: 
j
sttdev/layer2/bias/tagsConst*#
valueB Bsttdev/layer2/bias*
dtype0*
_output_shapes
: 
j
sttdev/layer2/biasScalarSummarysttdev/layer2/bias/tagsstddev/Sqrt*
T0*
_output_shapes
: 
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
V
rangeRangerange/startRankrange/delta*
_output_shapes
:*

Tidx0
`
MaxMaxVariable_7/readrange*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
d
max/layer2/bias/tagsConst* 
valueB Bmax/layer2/bias*
dtype0*
_output_shapes
: 
\
max/layer2/biasScalarSummarymax/layer2/bias/tagsMax*
_output_shapes
: *
T0
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
O
range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
^
range_1Rangerange_1/startRank_1range_1/delta*
_output_shapes
:*

Tidx0
b
MinMinVariable_7/readrange_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
d
min/layer2/bias/tagsConst*
_output_shapes
: * 
valueB Bmin/layer2/bias*
dtype0
\
min/layer2/biasScalarSummarymin/layer2/bias/tagsMin*
T0*
_output_shapes
: 
[
layer2/bias/tagConst*
valueB Blayer2/bias*
dtype0*
_output_shapes
: 
b
layer2/biasHistogramSummarylayer2/bias/tagVariable_7/read*
T0*
_output_shapes
: 

MatMulMatMulPlaceholderVariable/read*
T0*'
_output_shapes
:џџџџџџџџџP*
transpose_a( *
transpose_b( 
U
addAddMatMulVariable_1/read*'
_output_shapes
:џџџџџџџџџP*
T0
C
ReluReluadd*'
_output_shapes
:џџџџџџџџџP*
T0

MatMul_1MatMulReluVariable_2/read*
T0*'
_output_shapes
:џџџџџџџџџ(*
transpose_a( *
transpose_b( 
Y
add_1AddMatMul_1Variable_3/read*'
_output_shapes
:џџџџџџџџџ(*
T0
G
Relu_1Reluadd_1*'
_output_shapes
:џџџџџџџџџ(*
T0

MatMul_2MatMulRelu_1Variable_4/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Y
add_2AddMatMul_2Variable_5/read*'
_output_shapes
:џџџџџџџџџ*
T0
G
Relu_2Reluadd_2*'
_output_shapes
:џџџџџџџџџ*
T0

MatMul_3MatMulRelu_2Variable_6/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Y
add_3AddMatMul_3Variable_7/read*'
_output_shapes
:џџџџџџџџџ*
T0
K
SoftmaxSoftmaxadd_3*'
_output_shapes
:џџџџџџџџџ*
T0

9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_1*
T0*'
_output_shapes
:џџџџџџџџџ
k
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
*softmax_cross_entropy_with_logits_sg/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
,softmax_cross_entropy_with_logits_sg/Shape_1ShapeSoftmax*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Љ
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 

0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
і
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ж
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeSoftmax+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ѕ
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
­
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
 
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
ќ
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
ь
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
э
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ћ
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*

axis *
N*
_output_shapes
:*
T0
њ
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
Щ
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:

MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_4*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
Ј
Merge/MergeSummaryMergeSummarysummaries/mean/layer2/biassttdev/layer2/biasmax/layer2/biasmin/layer2/biaslayer2/biasloss*
N*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

gradients/Mean_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
Ї
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0
ю
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
с
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Е
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Й
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
і
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
Т
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
п
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
х
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ё
gradients/Softmax_grad/mulMulCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeSoftmax*'
_output_shapes
:џџџџџџџџџ*
T0
w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
К
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
Д
gradients/Softmax_grad/subSubCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapegradients/Softmax_grad/Sum*
T0*'
_output_shapes
:џџџџџџџџџ
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:џџџџџџџџџ
b
gradients/add_3_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
gradients/add_3_grad/SumSumgradients/Softmax_grad/mul_1*gradients/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
­
gradients/add_3_grad/Sum_1Sumgradients/Softmax_grad/mul_1,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
т
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
л
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
Р
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
А
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/add_3_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
ь
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
щ
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:

gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*'
_output_shapes
:џџџџџџџџџ*
T0
b
gradients/add_2_grad/ShapeShapeMatMul_2*
_output_shapes
:*
T0*
out_type0
f
gradients/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
т
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
л
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes
:
Р
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
T0*'
_output_shapes
:џџџџџџџџџ(*
transpose_a( *
transpose_b(
А
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a(
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
ь
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ(*
T0
щ
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
_output_shapes

:(*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1

gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*'
_output_shapes
:џџџџџџџџџ(*
T0
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ(
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:(*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
т
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ(
л
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:(
Р
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*'
_output_shapes
:џџџџџџџџџP*
transpose_a( *
transpose_b(*
T0
Ў
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:P(*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ь
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџP
щ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes

:P(*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1

gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:џџџџџџџџџP
^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
T0*
out_type0
d
gradients/add_grad/Shape_1Const*
valueB:P*
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџP
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:P
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
к
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџP
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:P
К
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Б
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes

:P*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ф
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
с
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:P*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
{
beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable*
dtype0

beta1_power
VariableV2*
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?*
_class
loc:@Variable

beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container *
shape: 
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 

Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
valueBP*    *
dtype0*
_output_shapes

:P

Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape
:P*
dtype0*
_output_shapes

:P
Н
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P*
use_locking(
s
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes

:P

!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*
valueBP*    *
dtype0*
_output_shapes

:P
 
Variable/Adam_1
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *
_class
loc:@Variable*
	container *
shape
:P
У
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes

:P

!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
valueBP*    *
dtype0*
_output_shapes
:P

Variable_1/Adam
VariableV2*
_output_shapes
:P*
shared_name *
_class
loc:@Variable_1*
	container *
shape:P*
dtype0
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:P

#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
valueBP*    *
dtype0*
_output_shapes
:P

Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:P*
dtype0*
_output_shapes
:P
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:P*
use_locking(*
T0*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
_output_shapes
:P*
T0
Ё
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Variable_2*
valueB"P   (   

'Variable_2/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
п
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_2*

index_type0*
_output_shapes

:P(
Ђ
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:P(*
dtype0*
_output_shapes

:P(
Х
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
y
Variable_2/Adam/readIdentityVariable_2/Adam*
_output_shapes

:P(*
T0*
_class
loc:@Variable_2
Ѓ
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*
valueB"P   (   *
dtype0*
_output_shapes
:

)Variable_2/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
х
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_2*

index_type0*
_output_shapes

:P(
Є
Variable_2/Adam_1
VariableV2*
dtype0*
_output_shapes

:P(*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:P(
Ы
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_output_shapes

:P(*
T0*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB(*    *
dtype0*
_output_shapes
:(

Variable_3/Adam
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *
_class
loc:@Variable_3*
	container *
shape:(
С
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:(

#Variable_3/Adam_1/Initializer/zerosConst*
_output_shapes
:(*
_class
loc:@Variable_3*
valueB(*    *
dtype0

Variable_3/Adam_1
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *
_class
loc:@Variable_3*
	container *
shape:(
Ч
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:(

!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB(*    *
dtype0*
_output_shapes

:(
Ђ
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name 
Х
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
y
Variable_4/Adam/readIdentityVariable_4/Adam*
_output_shapes

:(*
T0*
_class
loc:@Variable_4

#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB(*    *
dtype0*
_output_shapes

:(
Є
Variable_4/Adam_1
VariableV2*
dtype0*
_output_shapes

:(*
shared_name *
_class
loc:@Variable_4*
	container *
shape
:(
Ы
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
_output_shapes

:(*
T0

!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB*    *
dtype0*
_output_shapes
:

Variable_5/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_5*
	container 
С
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes
:

#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB*    *
dtype0*
_output_shapes
:

Variable_5/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_5*
	container *
shape:
Ч
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:

!Variable_6/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*
_class
loc:@Variable_6*
valueB*    
Ђ
Variable_6/Adam
VariableV2*
_class
loc:@Variable_6*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Х
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes

:

#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB*    *
dtype0*
_output_shapes

:
Є
Variable_6/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@Variable_6*
	container *
shape
:
Ы
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:

!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes
:

Variable_7/Adam
VariableV2*
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:

#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes
:

Variable_7/Adam_1
VariableV2*
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
в
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes

:P*
use_locking( 
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes
:P
о
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:P(
з
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes
:(*
use_locking( 
о
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes

:(*
use_locking( 
з
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes
:
о
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( *
_output_shapes

:
з
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( *
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
Р
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0*
output_type0	
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
`
CastCastEqual*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
[
Mean_1MeanCastConst_5*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
№
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
і
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*Љ
valueBBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power

save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_powerbeta2_power*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*Љ
valueBBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
Љ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
Ї
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
Љ
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P*
use_locking(
Ђ
save/Assign_3Assign
Variable_1save/RestoreV2:3*
_output_shapes
:P*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
Ї
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
Љ
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
_output_shapes
:P*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
І
save/Assign_6Assign
Variable_2save/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
Ћ
save/Assign_7AssignVariable_2/Adamsave/RestoreV2:7*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(*
use_locking(*
T0
­
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2:8*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(*
use_locking(
Ђ
save/Assign_9Assign
Variable_3save/RestoreV2:9*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0
Љ
save/Assign_10AssignVariable_3/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
Ћ
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
Ј
save/Assign_12Assign
Variable_4save/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
­
save/Assign_13AssignVariable_4/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
Џ
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2:14*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
Є
save/Assign_15Assign
Variable_5save/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
Љ
save/Assign_16AssignVariable_5/Adamsave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
Ћ
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
Ј
save/Assign_18Assign
Variable_6save/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
­
save/Assign_19AssignVariable_6/Adamsave/RestoreV2:19*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6
Џ
save/Assign_20AssignVariable_6/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
Є
save/Assign_21Assign
Variable_7save/RestoreV2:21*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
Љ
save/Assign_22AssignVariable_7/Adamsave/RestoreV2:22*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ћ
save/Assign_23AssignVariable_7/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:

save/Assign_24Assignbeta1_powersave/RestoreV2:24*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(

save/Assign_25Assignbeta2_powersave/RestoreV2:25*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
Ц
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"Ю[v     uри1	ЪОѕ~DзAJ
и"Г"
:
Add
x"T
y"T
z"T"
Ttype:
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
2
StopGradient

input"T
output"T"	
Ttype
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.1-0-g6612da8951'ѓЭ
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
g
truncated_normal/shapeConst*
valueB"   P   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*
seed2 *
_output_shapes

:P*

seed 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
_output_shapes

:P*
T0
m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:P
|
Variable
VariableV2*
	container *
_output_shapes

:P*
shape
:P*
shared_name *
dtype0
Є
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:P
R
ConstConst*
valueBP*
з#<*
dtype0*
_output_shapes
:P
v

Variable_1
VariableV2*
shape:P*
shared_name *
dtype0*
	container *
_output_shapes
:P

Variable_1/AssignAssign
Variable_1Const*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:P
i
truncated_normal_1/shapeConst*
valueB"P   (   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_1/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes

:P(

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:P(
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes

:P(
~

Variable_2
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:P(*
shape
:P(
Ќ
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2
o
Variable_2/readIdentity
Variable_2*
_output_shapes

:P(*
T0*
_class
loc:@Variable_2
T
Const_1Const*
valueB(*
з#<*
dtype0*
_output_shapes
:(
v

Variable_3
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:(*
shape:(

Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:(
i
truncated_normal_2/shapeConst*
valueB"(      *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*
seed2 *
_output_shapes

:(*

seed 

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes

:(*
T0
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:(
~

Variable_4
VariableV2*
shape
:(*
shared_name *
dtype0*
	container *
_output_shapes

:(
Ќ
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
o
Variable_4/readIdentity
Variable_4*
_output_shapes

:(*
T0*
_class
loc:@Variable_4
T
Const_2Const*
valueB*
з#<*
dtype0*
_output_shapes
:
v

Variable_5
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 

Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_5
k
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes
:
i
truncated_normal_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes

:

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes

:*
T0
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes

:*
T0
~

Variable_6
VariableV2*
dtype0*
	container *
_output_shapes

:*
shape
:*
shared_name 
Ќ
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:
T
Const_3Const*
_output_shapes
:*
valueB*
з#<*
dtype0
v

Variable_7
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 

Variable_7/AssignAssign
Variable_7Const_3*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes
:*
T0
P
summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
W
summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
W
summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
summaries/rangeRangesummaries/range/startsummaries/Ranksummaries/range/delta*

Tidx0*
_output_shapes
:
v
summaries/MeanMeanVariable_7/readsummaries/range*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
z
summaries/mean/layer2/bias/tagsConst*+
value"B  Bsummaries/mean/layer2/bias*
dtype0*
_output_shapes
: 
}
summaries/mean/layer2/biasScalarSummarysummaries/mean/layer2/bias/tagssummaries/Mean*
_output_shapes
: *
T0
W

stddev/subSubVariable_7/readsummaries/Mean*
_output_shapes
:*
T0
H
stddev/SquareSquare
stddev/sub*
T0*
_output_shapes
:
V
stddev/ConstConst*
valueB: *
dtype0*
_output_shapes
:
l

stddev/SumSumstddev/Squarestddev/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
@
stddev/SqrtSqrt
stddev/Sum*
_output_shapes
: *
T0
j
sttdev/layer2/bias/tagsConst*#
valueB Bsttdev/layer2/bias*
dtype0*
_output_shapes
: 
j
sttdev/layer2/biasScalarSummarysttdev/layer2/bias/tagsstddev/Sqrt*
_output_shapes
: *
T0
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
M
range/startConst*
_output_shapes
: *
value	B : *
dtype0
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
V
rangeRangerange/startRankrange/delta*
_output_shapes
:*

Tidx0
`
MaxMaxVariable_7/readrange*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
max/layer2/bias/tagsConst* 
valueB Bmax/layer2/bias*
dtype0*
_output_shapes
: 
\
max/layer2/biasScalarSummarymax/layer2/bias/tagsMax*
T0*
_output_shapes
: 
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
O
range_1/startConst*
_output_shapes
: *
value	B : *
dtype0
O
range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
^
range_1Rangerange_1/startRank_1range_1/delta*

Tidx0*
_output_shapes
:
b
MinMinVariable_7/readrange_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
min/layer2/bias/tagsConst* 
valueB Bmin/layer2/bias*
dtype0*
_output_shapes
: 
\
min/layer2/biasScalarSummarymin/layer2/bias/tagsMin*
T0*
_output_shapes
: 
[
layer2/bias/tagConst*
valueB Blayer2/bias*
dtype0*
_output_shapes
: 
b
layer2/biasHistogramSummarylayer2/bias/tagVariable_7/read*
T0*
_output_shapes
: 

MatMulMatMulPlaceholderVariable/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџP*
transpose_b( 
U
addAddMatMulVariable_1/read*
T0*'
_output_shapes
:џџџџџџџџџP
C
ReluReluadd*
T0*'
_output_shapes
:џџџџџџџџџP

MatMul_1MatMulReluVariable_2/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ(*
transpose_b( 
Y
add_1AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:џџџџџџџџџ(
G
Relu_1Reluadd_1*'
_output_shapes
:џџџџџџџџџ(*
T0

MatMul_2MatMulRelu_1Variable_4/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Y
add_2AddMatMul_2Variable_5/read*'
_output_shapes
:џџџџџџџџџ*
T0
G
Relu_2Reluadd_2*'
_output_shapes
:џџџџџџџџџ*
T0

MatMul_3MatMulRelu_2Variable_6/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Y
add_3AddMatMul_3Variable_7/read*'
_output_shapes
:џџџџџџџџџ*
T0
K
SoftmaxSoftmaxadd_3*
T0*'
_output_shapes
:џџџџџџџџџ

9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_1*
T0*'
_output_shapes
:џџџџџџџџџ
k
)softmax_cross_entropy_with_logits_sg/RankConst*
_output_shapes
: *
value	B :*
dtype0
q
*softmax_cross_entropy_with_logits_sg/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
s
,softmax_cross_entropy_with_logits_sg/Shape_1ShapeSoftmax*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
Љ
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 

0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
і
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
T0*
Index0*
_output_shapes
:

4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 

+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
Ж
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeSoftmax+softmax_cross_entropy_with_logits_sg/concat*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ѕ
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
­
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
 
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ќ
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
T0*
Index0*
_output_shapes
:

6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0

-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
ь
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
э
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ћ
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
њ
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
T0*
Index0*
_output_shapes
:
Щ
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:

MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_4*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
Ј
Merge/MergeSummaryMergeSummarysummaries/mean/layer2/biassttdev/layer2/biasmax/layer2/biasmin/layer2/biaslayer2/biasloss*
N*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

gradients/Mean_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
ю
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
с
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Е
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Й
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0

@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
і
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Т
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
п
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
х
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1

Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ё
gradients/Softmax_grad/mulMulCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeSoftmax*
T0*'
_output_shapes
:џџџџџџџџџ
w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
К
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:џџџџџџџџџ
Д
gradients/Softmax_grad/subSubCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapegradients/Softmax_grad/Sum*
T0*'
_output_shapes
:џџџџџџџџџ
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*'
_output_shapes
:џџџџџџџџџ*
T0
b
gradients/add_3_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Љ
gradients/add_3_grad/SumSumgradients/Softmax_grad/mul_1*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
­
gradients/add_3_grad/Sum_1Sumgradients/Softmax_grad/mul_1,gradients/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
т
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
л
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
Р
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
А
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/add_3_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
ь
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
щ
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:

gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*
T0*'
_output_shapes
:џџџџџџџџџ
b
gradients/add_2_grad/ShapeShapeMatMul_2*
T0*
out_type0*
_output_shapes
:
f
gradients/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
т
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
л
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes
:*
T0
Р
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ(*
transpose_b(
А
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
ь
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ(
щ
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:(

gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0*'
_output_shapes
:џџџџџџџџџ(
b
gradients/add_1_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_1_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ(*
T0*
Tshape0
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:(
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
т
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ(*
T0
л
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
:(*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
Р
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџP*
transpose_b(*
T0
Ў
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:P(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ь
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџP*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
щ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:P(

gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:џџџџџџџџџP*
T0
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:P*
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџP
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:P
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
к
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџP
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:P*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
К
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
Б
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:P*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ф
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
с
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:P
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container *
shape: 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *wО?

beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable

Variable/Adam/Initializer/zerosConst*
valueBP*    *
_class
loc:@Variable*
dtype0*
_output_shapes

:P

Variable/Adam
VariableV2*
_class
loc:@Variable*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
Н
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
s
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes

:P

!Variable/Adam_1/Initializer/zerosConst*
valueBP*    *
_class
loc:@Variable*
dtype0*
_output_shapes

:P
 
Variable/Adam_1
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *
_class
loc:@Variable*
	container *
shape
:P
У
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes

:P

!Variable_1/Adam/Initializer/zerosConst*
valueBP*    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:P

Variable_1/Adam
VariableV2*
dtype0*
_output_shapes
:P*
shared_name *
_class
loc:@Variable_1*
	container *
shape:P
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:P

#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:P*
valueBP*    *
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
shape:P*
dtype0*
_output_shapes
:P*
shared_name *
_class
loc:@Variable_1*
	container 
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P*
use_locking(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:P
Ё
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"P   (   *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:

'Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
п
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*
_output_shapes

:P(
Ђ
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:P(*
dtype0*
_output_shapes

:P(
Х
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(*
use_locking(
y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*
_output_shapes

:P(
Ѓ
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"P   (   *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:

)Variable_2/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_2
х
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
_output_shapes

:P(*
T0*

index_type0*
_class
loc:@Variable_2
Є
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*
	container *
shape
:P(*
dtype0*
_output_shapes

:P(*
shared_name 
Ы
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:P(

!Variable_3/Adam/Initializer/zerosConst*
valueB(*    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:(

Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name 
С
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:(

#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:(*
valueB(*    *
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape:(*
dtype0*
_output_shapes
:(
Ч
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:(

!Variable_4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:(*
valueB(*    *
_class
loc:@Variable_4
Ђ
Variable_4/Adam
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *
_class
loc:@Variable_4*
	container 
Х
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0
y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*
_output_shapes

:(

#Variable_4/Adam_1/Initializer/zerosConst*
_output_shapes

:(*
valueB(*    *
_class
loc:@Variable_4*
dtype0
Є
Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape
:(*
dtype0*
_output_shapes

:(
Ы
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_output_shapes

:(*
T0*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_5*
dtype0

Variable_5/Adam
VariableV2*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0*
_output_shapes
:
С
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes
:

#Variable_5/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

Variable_5/Adam_1
VariableV2*
_class
loc:@Variable_5*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ч
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:

!Variable_6/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_6*
dtype0*
_output_shapes

:
Ђ
Variable_6/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@Variable_6*
	container 
Х
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes

:

#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *
_class
loc:@Variable_6
Є
Variable_6/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@Variable_6*
	container *
shape
:
Ы
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_output_shapes

:*
T0*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_7*
	container 
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:

#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes
:
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
в
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:P*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( 
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:P*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( 
о
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:P(*
use_locking( *
T0
з
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes
:(
о
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
_output_shapes

:(*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( 
з
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes
:
о
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( *
_output_shapes

:
з
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_7*
use_nesterov( *
_output_shapes
:*
use_locking( 

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
Р
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
x
ArgMaxArgMaxSoftmaxArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
`
CastCastEqual*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ*

SrcT0

Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_5*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
№
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
і
save/SaveV2/tensor_namesConst*Љ
valueBBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_powerbeta2_power*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const

save/RestoreV2/tensor_namesConst"/device:CPU:0*Љ
valueBBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
Љ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
Ї
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
Љ
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable
Ђ
save/Assign_3Assign
Variable_1save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
Ї
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
_output_shapes
:P*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
Љ
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
І
save/Assign_6Assign
Variable_2save/RestoreV2:6*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
Ћ
save/Assign_7AssignVariable_2/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
­
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2:8*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
Ђ
save/Assign_9Assign
Variable_3save/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
Љ
save/Assign_10AssignVariable_3/Adamsave/RestoreV2:10*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0
Ћ
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2:11*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(
Ј
save/Assign_12Assign
Variable_4save/RestoreV2:12*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4
­
save/Assign_13AssignVariable_4/Adamsave/RestoreV2:13*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0
Џ
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2:14*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(*
use_locking(
Є
save/Assign_15Assign
Variable_5save/RestoreV2:15*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(
Љ
save/Assign_16AssignVariable_5/Adamsave/RestoreV2:16*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(
Ћ
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
Ј
save/Assign_18Assign
Variable_6save/RestoreV2:18*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
­
save/Assign_19AssignVariable_6/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
Џ
save/Assign_20AssignVariable_6/Adam_1save/RestoreV2:20*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
Є
save/Assign_21Assign
Variable_7save/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
Љ
save/Assign_22AssignVariable_7/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
Ћ
save/Assign_23AssignVariable_7/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:

save/Assign_24Assignbeta1_powersave/RestoreV2:24*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(

save/Assign_25Assignbeta2_powersave/RestoreV2:25*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
Ц
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""
	summariess
q
summaries/mean/layer2/bias:0
sttdev/layer2/bias:0
max/layer2/bias:0
min/layer2/bias:0
layer2/bias:0
loss:0"е
trainable_variablesНК
D

Variable:0Variable/AssignVariable/read:02truncated_normal:08
?
Variable_1:0Variable_1/AssignVariable_1/read:02Const:08
L
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:08
A
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_1:08
L
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:08
A
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:08
L
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:08
A
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:08"
train_op

Adam"Ч
	variablesЙЖ
D

Variable:0Variable/AssignVariable/read:02truncated_normal:08
?
Variable_1:0Variable_1/AssignVariable_1/read:02Const:08
L
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:08
A
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_1:08
L
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:08
A
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:08
L
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:08
A
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0w"ыј