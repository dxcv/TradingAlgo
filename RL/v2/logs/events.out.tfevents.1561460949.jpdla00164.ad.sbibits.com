       �K"	  @5�D�Abrain.Event:2g5E �      w1�	�~5�D�A"��
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
p
Placeholder_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
g
truncated_normal/shapeConst*
valueB"   P   *
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
 *  �?*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
_output_shapes

:P*
seed2 *

seed *
T0

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
_output_shapes

:P*
T0
m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:P
|
Variable
VariableV2*
_output_shapes

:P*
	container *
shape
:P*
shared_name *
dtype0
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
i
Variable/readIdentityVariable*
_output_shapes

:P*
T0*
_class
loc:@Variable
R
ConstConst*
_output_shapes
:P*
valueBP*
�#<*
dtype0
v

Variable_1
VariableV2*
shape:P*
shared_name *
dtype0*
_output_shapes
:P*
	container 
�
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
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
dtype0*
_output_shapes
:*
valueB"P   (   
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
 *  �?*
dtype0*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
_output_shapes

:P(*
seed2 *

seed *
T0
�
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
VariableV2*
dtype0*
_output_shapes

:P(*
	container *
shape
:P(*
shared_name 
�
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
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:P(
T
Const_1Const*
_output_shapes
:(*
valueB(*
�#<*
dtype0
v

Variable_3
VariableV2*
shared_name *
dtype0*
_output_shapes
:(*
	container *
shape:(
�
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
Variable_3*
_output_shapes
:(*
T0*
_class
loc:@Variable_3
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
 *  �?*
dtype0*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*
_output_shapes

:(*
seed2 *

seed 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes

:(
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
�
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
Const_2Const*
dtype0*
_output_shapes
:*
valueB*
�#<
v

Variable_5
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
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
truncated_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
_output_shapes

:*
seed2 *

seed *
T0
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes

:
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:
~

Variable_6
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:*
T0*
_class
loc:@Variable_6
T
Const_3Const*
dtype0*
_output_shapes
:*
valueB*
�#<
v

Variable_7
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
P
summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
W
summaries/range/startConst*
_output_shapes
: *
value	B : *
dtype0
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
: *
	keep_dims( *

Tidx0
z
summaries/mean/layer2/bias/tagsConst*
dtype0*
_output_shapes
: *+
value"B  Bsummaries/mean/layer2/bias
}
summaries/mean/layer2/biasScalarSummarysummaries/mean/layer2/bias/tagssummaries/Mean*
T0*
_output_shapes
: 
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

stddev/SumSumstddev/Squarestddev/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
@
stddev/SqrtSqrt
stddev/Sum*
_output_shapes
: *
T0
j
sttdev/layer2/bias/tagsConst*
_output_shapes
: *#
valueB Bsttdev/layer2/bias*
dtype0
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
rangeRangerange/startRankrange/delta*

Tidx0*
_output_shapes
:
`
MaxMaxVariable_7/readrange*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
d
max/layer2/bias/tagsConst*
dtype0*
_output_shapes
: * 
valueB Bmax/layer2/bias
\
max/layer2/biasScalarSummarymax/layer2/bias/tagsMax*
T0*
_output_shapes
: 
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
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
min/layer2/bias/tagsConst*
dtype0*
_output_shapes
: * 
valueB Bmin/layer2/bias
\
min/layer2/biasScalarSummarymin/layer2/bias/tagsMin*
_output_shapes
: *
T0
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
�
MatMulMatMulPlaceholderVariable/read*
transpose_b( *
T0*'
_output_shapes
:���������P*
transpose_a( 
U
addAddMatMulVariable_1/read*
T0*'
_output_shapes
:���������P
C
ReluReluadd*
T0*'
_output_shapes
:���������P
�
MatMul_1MatMulReluVariable_2/read*
T0*'
_output_shapes
:���������(*
transpose_a( *
transpose_b( 
Y
add_1AddMatMul_1Variable_3/read*'
_output_shapes
:���������(*
T0
G
Relu_1Reluadd_1*'
_output_shapes
:���������(*
T0
�
MatMul_2MatMulRelu_1Variable_4/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
Y
add_2AddMatMul_2Variable_5/read*'
_output_shapes
:���������*
T0
G
Relu_2Reluadd_2*'
_output_shapes
:���������*
T0
�
MatMul_3MatMulRelu_2Variable_6/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
Y
add_3AddMatMul_3Variable_7/read*'
_output_shapes
:���������*
T0
K
SoftmaxSoftmaxadd_3*'
_output_shapes
:���������*
T0
�
9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_1*
T0*'
_output_shapes
:���������
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
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
s
,softmax_cross_entropy_with_logits_sg/Shape_1ShapeSoftmax*
_output_shapes
:*
T0*
out_type0
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
�
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
�
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
�
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:
�
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeSoftmax+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:������������������
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
�
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
�
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
�
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*

axis *
N*
_output_shapes
:*
T0
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
�
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
�
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
_output_shapes
: *
T0
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
_output_shapes
:*
T0*

axis *
N
�
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
Tshape0*#
_output_shapes
:���������*
T0
Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
�
MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
�
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
 *  �?*
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
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
out_type0*
_output_shapes
:*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
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
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
�
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0
�
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:������������������
�
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:������������������
�
Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*0
_output_shapes
:������������������*
T0
�
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
�
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:������������������
�
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:������������������
�
Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:
�
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Softmax_grad/mulMulCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeSoftmax*
T0*'
_output_shapes
:���������
w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
gradients/Softmax_grad/subSubCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapegradients/Softmax_grad/Sum*
T0*'
_output_shapes
:���������
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*'
_output_shapes
:���������*
T0
b
gradients/add_3_grad/ShapeShapeMatMul_3*
_output_shapes
:*
T0*
out_type0
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_3_grad/SumSumgradients/Softmax_grad/mul_1*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_3_grad/Sum_1Sumgradients/Softmax_grad/mul_1,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_3_grad/Reshape*'
_output_shapes
:���������*
T0
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
�
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/add_3_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*'
_output_shapes
:���������
�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:
�
gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*
T0*'
_output_shapes
:���������
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
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes
:*
T0
�
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*'
_output_shapes
:���������(*
transpose_a( 
�
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*'
_output_shapes
:���������(
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:(
�
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0*'
_output_shapes
:���������(
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:(
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������(
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:(*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:���������(
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:(
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*'
_output_shapes
:���������P*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:P(*
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������P*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:P(
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:���������P*
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
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������P
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:P*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������P
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:P*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes

:P*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:P*
T0
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*
_class
loc:@Variable*
dtype0
�
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
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
valueBP*    *
dtype0*
_output_shapes

:P
�
Variable/Adam
VariableV2*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name *
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
s
Variable/Adam/readIdentityVariable/Adam*
_output_shapes

:P*
T0*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*
valueBP*    *
dtype0*
_output_shapes

:P
�
Variable/Adam_1
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *
_class
loc:@Variable*
	container *
shape
:P
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes

:P
�
!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
valueBP*    *
dtype0*
_output_shapes
:P
�
Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
	container *
shape:P*
dtype0*
_output_shapes
:P*
shared_name 
�
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
�
#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
valueBP*    *
dtype0*
_output_shapes
:P
�
Variable_1/Adam_1
VariableV2*
dtype0*
_output_shapes
:P*
shared_name *
_class
loc:@Variable_1*
	container *
shape:P
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:P
�
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*
valueB"P   (   *
dtype0*
_output_shapes
:
�
'Variable_2/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_2*

index_type0*
_output_shapes

:P(
�
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
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*
_output_shapes

:P(
�
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*
valueB"P   (   *
dtype0*
_output_shapes
:
�
)Variable_2/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
_output_shapes

:P(*
T0*
_class
loc:@Variable_2*

index_type0
�
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
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(*
use_locking(
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:P(
�
!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB(*    *
dtype0*
_output_shapes
:(
�
Variable_3/Adam
VariableV2*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name *
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_output_shapes
:(*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:(*
T0*
_class
loc:@Variable_3
�
#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB(*    *
dtype0*
_output_shapes
:(
�
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
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
_output_shapes
:(*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB(*    *
dtype0*
_output_shapes

:(
�
Variable_4/Adam
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
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*
_output_shapes

:(
�
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB(*    *
dtype0*
_output_shapes

:(
�
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
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*
_output_shapes

:(
�
!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_5/Adam
VariableV2*
_output_shapes
:*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0
�
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
�
#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB*    *
dtype0*
_output_shapes
:
�
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
�
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
�
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB*    *
dtype0*
_output_shapes

:
�
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
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6
y
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
_output_shapes

:*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB*    *
dtype0*
_output_shapes

:
�
Variable_6/Adam_1
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
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:
�
!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_7/Adam
VariableV2*
_output_shapes
:*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
_output_shapes
:*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_7*
	container *
shape:
�
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
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *��8
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
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:P*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( 
�
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
�
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
�
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
�
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
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( 
�
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
�
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
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( 
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
`
CastCastEqual*
Truncate( *#
_output_shapes
:���������*

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
�
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�BVariableBVariable/AdamBVariable/Adam_1B
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
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
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
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BVariableBVariable/AdamBVariable/Adam_1B
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
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::
�
save/AssignAssignVariablesave/RestoreV2*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
�
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
�
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable
�
save/Assign_3Assign
Variable_1save/RestoreV2:3*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P*
use_locking(*
T0
�
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
validate_shape(*
_output_shapes
:P*
use_locking(*
T0*
_class
loc:@Variable_1
�
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
�
save/Assign_6Assign
Variable_2save/RestoreV2:6*
validate_shape(*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2
�
save/Assign_7AssignVariable_2/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
�
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
�
save/Assign_9Assign
Variable_3save/RestoreV2:9*
_output_shapes
:(*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
�
save/Assign_10AssignVariable_3/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
�
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
�
save/Assign_12Assign
Variable_4save/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(
�
save/Assign_13AssignVariable_4/Adamsave/RestoreV2:13*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(*
use_locking(
�
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2:14*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(*
use_locking(
�
save/Assign_15Assign
Variable_5save/RestoreV2:15*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_16AssignVariable_5/Adamsave/RestoreV2:16*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
�
save/Assign_18Assign
Variable_6save/RestoreV2:18*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_19AssignVariable_6/Adamsave/RestoreV2:19*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_20AssignVariable_6/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assign
Variable_7save/RestoreV2:21*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_7
�
save/Assign_22AssignVariable_7/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
�
save/Assign_23AssignVariable_7/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assignbeta1_powersave/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
�
save/Assign_25Assignbeta2_powersave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"��3     u��1	��5�D�AJ��
�"�"
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
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
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
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
�
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
�
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

2	�
�
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
�
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
2	�
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
�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.1-0-g6612da8951'��
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
p
Placeholder_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
g
truncated_normal/shapeConst*
valueB"   P   *
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
 *  �?*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes

:P

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
_output_shapes

:P*
T0
m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:P
|
Variable
VariableV2*
dtype0*
	container *
_output_shapes

:P*
shape
:P*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
i
Variable/readIdentityVariable*
_class
loc:@Variable*
_output_shapes

:P*
T0
R
ConstConst*
valueBP*
�#<*
dtype0*
_output_shapes
:P
v

Variable_1
VariableV2*
dtype0*
	container *
_output_shapes
:P*
shape:P*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
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
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *
_output_shapes

:P(*

seed *
T0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:P(
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
_output_shapes

:P(*
T0
~

Variable_2
VariableV2*
shape
:P(*
shared_name *
dtype0*
	container *
_output_shapes

:P(
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
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
�#<*
dtype0*
_output_shapes
:(
v

Variable_3
VariableV2*
dtype0*
	container *
_output_shapes
:(*
shape:(*
shared_name 
�
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
Variable_3*
_class
loc:@Variable_3*
_output_shapes
:(*
T0
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
 *  �?*
dtype0*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes

:(
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes

:(
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:(
~

Variable_4
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:(*
shape
:(
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:(
T
Const_2Const*
valueB*
�#<*
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
�
Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
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
truncated_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*
seed2 *
_output_shapes

:*

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes

:
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:
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
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:
T
Const_3Const*
valueB*
�#<*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
Variable_7/AssignAssign
Variable_7Const_3*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
T0*
_class
loc:@Variable_7
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
summaries/rangeRangesummaries/range/startsummaries/Ranksummaries/range/delta*
_output_shapes
:*

Tidx0
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
summaries/mean/layer2/biasScalarSummarysummaries/mean/layer2/bias/tagssummaries/Mean*
T0*
_output_shapes
: 
W

stddev/subSubVariable_7/readsummaries/Mean*
T0*
_output_shapes
:
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

stddev/SumSumstddev/Squarestddev/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
@
stddev/SqrtSqrt
stddev/Sum*
_output_shapes
: *
T0
j
sttdev/layer2/bias/tagsConst*
_output_shapes
: *#
valueB Bsttdev/layer2/bias*
dtype0
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
MinMinVariable_7/readrange_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
layer2/biasHistogramSummarylayer2/bias/tagVariable_7/read*
_output_shapes
: *
T0
�
MatMulMatMulPlaceholderVariable/read*
T0*
transpose_a( *'
_output_shapes
:���������P*
transpose_b( 
U
addAddMatMulVariable_1/read*'
_output_shapes
:���������P*
T0
C
ReluReluadd*'
_output_shapes
:���������P*
T0
�
MatMul_1MatMulReluVariable_2/read*
transpose_a( *'
_output_shapes
:���������(*
transpose_b( *
T0
Y
add_1AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:���������(
G
Relu_1Reluadd_1*'
_output_shapes
:���������(*
T0
�
MatMul_2MatMulRelu_1Variable_4/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
Y
add_2AddMatMul_2Variable_5/read*
T0*'
_output_shapes
:���������
G
Relu_2Reluadd_2*
T0*'
_output_shapes
:���������
�
MatMul_3MatMulRelu_2Variable_6/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
Y
add_3AddMatMul_3Variable_7/read*'
_output_shapes
:���������*
T0
K
SoftmaxSoftmaxadd_3*'
_output_shapes
:���������*
T0
�
9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_1*'
_output_shapes
:���������*
T0
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
�
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
_output_shapes
:*
T0*

axis *
N
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
T0*
Index0*
_output_shapes
:
�
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeSoftmax+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:������������������
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
_output_shapes
:*
T0*
out_type0
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
�
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
�
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
_output_shapes
:*
T0*

axis *
N
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
T0*
Index0*
_output_shapes
:
�
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
_output_shapes
: *
T0
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*

axis *
N*
_output_shapes
:*
T0
�
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
_output_shapes
:*
T0*
Index0
�
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
�
MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
�
Merge/MergeSummaryMergeSummarysummaries/mean/layer2/biassttdev/layer2/biasmax/layer2/biasmin/layer2/biaslayer2/biasloss*
N*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
�
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
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
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
�
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
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
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
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
�
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:������������������*
T0
�
Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:������������������*
T0
�
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:������������������
�
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:������������������
�
Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:������������������
�
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
�
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:������������������
�
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:������������������*
T0
�
Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:
�
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
gradients/Softmax_grad/mulMulCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeSoftmax*
T0*'
_output_shapes
:���������
w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
gradients/Softmax_grad/subSubCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapegradients/Softmax_grad/Sum*'
_output_shapes
:���������*
T0
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:���������
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
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_3_grad/SumSumgradients/Softmax_grad/mul_1*gradients/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_3_grad/Sum_1Sumgradients/Softmax_grad/mul_1,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
�
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b(*
T0
�
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*'
_output_shapes
:���������
�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:
�
gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*'
_output_shapes
:���������*
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
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
�
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *'
_output_shapes
:���������(*
transpose_b(*
T0
�
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*'
_output_shapes
:���������(
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
_output_shapes

:(*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1
�
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0*'
_output_shapes
:���������(
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
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������(*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:(
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:���������(
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:(*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������P
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:P(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:���������P
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:P(
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:���������P*
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
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������P
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:P*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������P
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:P
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:P*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:P
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
_output_shapes
: *
T0
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
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
�
Variable/Adam/Initializer/zerosConst*
valueBP*    *
_class
loc:@Variable*
dtype0*
_output_shapes

:P
�
Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape
:P*
dtype0*
_output_shapes

:P
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0*
_class
loc:@Variable
s
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
_output_shapes

:P*
T0
�
!Variable/Adam_1/Initializer/zerosConst*
valueBP*    *
_class
loc:@Variable*
dtype0*
_output_shapes

:P
�
Variable/Adam_1
VariableV2*
_class
loc:@Variable*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes

:P
�
!Variable_1/Adam/Initializer/zerosConst*
valueBP*    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:P
�
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
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P*
use_locking(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:P*
T0*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes
:P*
valueBP*    *
_class
loc:@Variable_1*
dtype0
�
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
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P*
use_locking(*
T0
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:P*
T0*
_class
loc:@Variable_1
�
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"P   (   *
_class
loc:@Variable_2
�
'Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
�
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*
_output_shapes

:P(
�
Variable_2/Adam
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
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2
y
Variable_2/Adam/readIdentityVariable_2/Adam*
_output_shapes

:P(*
T0*
_class
loc:@Variable_2
�
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"P   (   *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:
�
)Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
�
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*
_output_shapes

:P(
�
Variable_2/Adam_1
VariableV2*
shape
:P(*
dtype0*
_output_shapes

:P(*
shared_name *
_class
loc:@Variable_2*
	container 
�
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
_class
loc:@Variable_2*
_output_shapes

:P(*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
valueB(*    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:(
�
Variable_3/Adam
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
_output_shapes
:(*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:(*
valueB(*    *
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name 
�
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
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:(*
valueB(*    *
_class
loc:@Variable_4
�
Variable_4/Adam
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
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4
y
Variable_4/Adam/readIdentityVariable_4/Adam*
_output_shapes

:(*
T0*
_class
loc:@Variable_4
�
#Variable_4/Adam_1/Initializer/zerosConst*
_output_shapes

:(*
valueB(*    *
_class
loc:@Variable_4*
dtype0
�
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
�
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
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_5
�
Variable_5/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_5*
	container *
shape:
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
_output_shapes
:*
T0
�
#Variable_5/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:
�
Variable_5/Adam_1
VariableV2*
_output_shapes
:*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0
�
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
�
!Variable_6/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    *
_class
loc:@Variable_6*
dtype0
�
Variable_6/Adam
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape
:*
dtype0*
_output_shapes

:
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(
y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes

:
�
#Variable_6/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_6*
dtype0*
_output_shapes

:
�
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
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_output_shapes

:*
T0*
_class
loc:@Variable_6
�
!Variable_7/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_7*
dtype0
�
Variable_7/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_7*
	container *
shape:
�
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
�
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:
�
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
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
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
 *��8*
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

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:P*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( 
�
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
�
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
�
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
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes

:(
�
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
�
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
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
x
ArgMaxArgMaxSoftmaxArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
�
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
`
CastCastEqual*

DstT0*#
_output_shapes
:���������*

SrcT0
*
Truncate( 
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
�
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
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
�
save/SaveV2/tensor_namesConst*�
value�B�BVariableBVariable/AdamBVariable/Adam_1B
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
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
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
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BVariableBVariable/AdamBVariable/Adam_1B
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
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
�
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P*
use_locking(
�
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:P
�
save/Assign_3Assign
Variable_1save/RestoreV2:3*
_output_shapes
:P*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
�
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
�
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:P
�
save/Assign_6Assign
Variable_2save/RestoreV2:6*
validate_shape(*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2
�
save/Assign_7AssignVariable_2/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:P(
�
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2:8*
_output_shapes

:P(*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
�
save/Assign_9Assign
Variable_3save/RestoreV2:9*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0
�
save/Assign_10AssignVariable_3/Adamsave/RestoreV2:10*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(*
use_locking(
�
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:(
�
save/Assign_12Assign
Variable_4save/RestoreV2:12*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0
�
save/Assign_13AssignVariable_4/Adamsave/RestoreV2:13*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
�
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2:14*
_output_shapes

:(*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
�
save/Assign_15Assign
Variable_5save/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:
�
save/Assign_16AssignVariable_5/Adamsave/RestoreV2:16*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2:17*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_18Assign
Variable_6save/RestoreV2:18*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
�
save/Assign_19AssignVariable_6/Adamsave/RestoreV2:19*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_20AssignVariable_6/Adam_1save/RestoreV2:20*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_21Assign
Variable_7save/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
�
save/Assign_22AssignVariable_7/Adamsave/RestoreV2:22*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(
�
save/Assign_23AssignVariable_7/Adam_1save/RestoreV2:23*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_24Assignbeta1_powersave/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
�
save/Assign_25Assignbeta2_powersave/RestoreV2:25*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""�
	summariess
q
summaries/mean/layer2/bias:0
sttdev/layer2/bias:0
max/layer2/bias:0
min/layer2/bias:0
layer2/bias:0
loss:0"�
trainable_variables��
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
Adam"�
	variables��
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
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0$��       �{�	#
�5�D�A�*�
!
summaries/mean/layer2/bias�#<

sttdev/layer2/bias{��1

max/layer2/bias
�#<

min/layer2/bias�#<
p
layer2/bias*a	   ��z�?   @�z�?      @!   �Q��?)�/�)�3?2���J�\�?-Ա�L�?�������:              @        

lossz+?�E5>�       �{�	gi�5�D�A�*�
!
summaries/mean/layer2/bias�=$<

sttdev/layer2/bias_	�8

max/layer2/bias�%<

min/layer2/bias�"<
p
layer2/bias*a	    �S�?   `ޡ�?      @!   М˞?)�)[�"�3?2���J�\�?-Ա�L�?�������:              @        

loss�:?U���       �{�	ͤ�5�D�A�*�
!
summaries/mean/layer2/bias�$<

sttdev/layer2/bias���8

max/layer2/biasy�%<

min/layer2/bias�A#<
p
layer2/bias*a	   �:h�?    o��?      @!   P�؞?)��s��3?2���J�\�?-Ա�L�?�������:              @        

lossE?`����       �{�	^�5�D�A�	*�
!
summaries/mean/layer2/bias�$<

sttdev/layer2/bias��9

max/layer2/bias��&<

min/layer2/bias!�#<
p
layer2/bias*a	    �x�?   �҄?      @!   pB�?)P��z��3?2���J�\�?-Ա�L�?�������:              @        

loss�+?g�a��       �{�	��5�D�A�*�
!
summaries/mean/layer2/bias��$<

sttdev/layer2/bias(�R9

max/layer2/biasay'<

min/layer2/bias,�"<
p
layer2/bias*a	   �_�?    ,�?      @!   P��?)�H�R�3?2���J�\�?-Ա�L�?�������:              @        

lossx?�)��       �{�	J͢5�D�A�*�
!
summaries/mean/layer2/biasO%<

sttdev/layer2/bias��9

max/layer2/bias�A(<

min/layer2/biasv3"<
p
layer2/bias*a	   �nF�?   �?�?      @!   ���?)I=k��3?2���J�\�?-Ա�L�?�������:              @        

loss{+?B[�	�       �{�	���5�D�A�*�
!
summaries/mean/layer2/biasY%<

sttdev/layer2/biasb��9

max/layer2/bias�(<

min/layer2/bias��!<
p
layer2/bias*a	   `�2�?   ���?      @!   ����?)ж�9��3?2���J�\�?-Ա�L�?�������:              @        

loss06?�m��       �{�	�®5�D�A�*�
!
summaries/mean/layer2/bias�9%<

sttdev/layer2/bias�R�9

max/layer2/bias�m)<

min/layer2/bias,� <
p
layer2/bias*a	   �e�?   ��-�?      @!   `���?)@��B� 4?2���J�\�?-Ա�L�?�������:              @        

loss2D?O�h�       �{�	���5�D�A�*�
!
summaries/mean/layer2/bias�A%<

sttdev/layer2/bias�V�9

max/layer2/bias�x)<

min/layer2/bias<� <
p
layer2/bias*a	   �'�?   @/�?      @!    Y��?)�E��4?2���J�\�?-Ա�L�?�������:              @        

loss�z?���2�       �{�	̬�5�D�A�*�
!
summaries/mean/layer2/bias3L%<

sttdev/layer2/biasA�9

max/layer2/biasy�(<

min/layer2/biasK� <
p
layer2/bias*a	   `i�?    o�?      @!   �I��?)�`��_4?2���J�\�?-Ա�L�?�������:              @        

loss�3?;T�9�       �{�	��5�D�A�*�
!
summaries/mean/layer2/bias�U%<

sttdev/layer2/bias�ߺ9

max/layer2/biasu,(<

min/layer2/bias� <
p
layer2/bias*a	   ` �?   ���?      @!     �?)`j���4?2���J�\�?-Ա�L�?�������:              @        

lossR0?��eL�       �{�	�!�5�D�A�*�
!
summaries/mean/layer2/bias�b%<

sttdev/layer2/bias!��9

max/layer2/bias��'<

min/layer2/bias�!<
p
layer2/bias*a	   ��"�?   @���?      @!   ���?)�~�Z
4?2���J�\�?-Ա�L�?�������:              @        

loss-�?�%p��       �{�	���5�D�A� *�
!
summaries/mean/layer2/biasdm%<

sttdev/layer2/bias���9

max/layer2/bias�(<

min/layer2/biasG<!<
p
layer2/bias*a	   ��'�?   ���?      @!   Ђ�?)P����4?2���J�\�?-Ա�L�?�������:              @        

loss��?��.h�       �{�	�	�5�D�A�"*�
!
summaries/mean/layer2/bias"q%<

sttdev/layer2/bias�9

max/layer2/biasr@(<

min/layer2/bias�I!<
p
layer2/bias*a	   �?)�?   @�?      @!   p6�?)�p���4?2���J�\�?-Ա�L�?�������:              @        

loss��?@S �       �{�	�F�5�D�A�$*�
!
summaries/mean/layer2/bias�j%<

sttdev/layer2/bias��9

max/layer2/biasb!(<

min/layer2/biasw!<
p
layer2/bias*a	   �.#�?   @,�?      @!   @�?)�.��Z4?2���J�\�?-Ա�L�?�������:              @        

loss�?���2�       �{�	���5�D�A�'*�
!
summaries/mean/layer2/bias�d%<

sttdev/layer2/bias9O�9

max/layer2/bias� (<

min/layer2/biasD� <
p
layer2/bias*a	   �H�?     �?      @!    ��?) �/��
4?2���J�\�?-Ա�L�?�������:              @        

lossU�?��z�       �{�	H5�5�D�A�)*�
!
summaries/mean/layer2/biasa%<

sttdev/layer2/bias]*�9

max/layer2/biase�'<

min/layer2/biasBK!<
p
layer2/bias*a	   @h)�?   ����?      @!   �2�?)ЎX�	4?2���J�\�?-Ա�L�?�������:              @        

loss%"?���,�       �{�	Yk�5�D�A�,*�
!
summaries/mean/layer2/bias�Z%<

sttdev/layer2/bias�U�9

max/layer2/bias6�'<

min/layer2/bias}!<
p
layer2/bias*a	   @�/�?   �&�?      @!   ��?)�b 4?2���J�\�?-Ա�L�?�������:              @        

loss�z?�n���       �{�	��5�D�A�/*�
!
summaries/mean/layer2/bias�O%<

sttdev/layer2/biasOI�9

max/layer2/bias�N'<

min/layer2/bias�	"<
p
layer2/bias*a	    <A�?   ���?      @!   P���?)�V�n�4?2���J�\�?-Ա�L�?�������:              @        

loss^S?�a��       �{�	jM�5�D�A�1*�
!
summaries/mean/layer2/bias�T%<

sttdev/layer2/biasC�y9

max/layer2/bias+x'<

min/layer2/bias8"<
p
layer2/bias*a	   � G�?   `�?      @!   ����?)� �<$4?2���J�\�?-Ա�L�?�������:              @        

loss��?��E��       �{�	���5�D�A�3*�
!
summaries/mean/layer2/bias�Y%<

sttdev/layer2/bias]uv9

max/layer2/biasН'<

min/layer2/biasQT"<
p
layer2/bias*a	    �J�?    ��?      @!   �� �?)��$�I4?2���J�\�?-Ա�L�?�������:              @        

lossmF?aHXR�       �{�	B�5�D�A�6*�
!
summaries/mean/layer2/bias_%<

sttdev/layer2/bias$��9

max/layer2/bias��'<

min/layer2/biasn;"<
p
layer2/bias*a	   �mG�?   �z��?      @!   ���?)����4?2���J�\�?-Ա�L�?�������:              @        

lossz?���>�       �{�	s6�D�A�8*�
!
summaries/mean/layer2/bias9q%<

sttdev/layer2/bias_�9

max/layer2/bias�|(<

min/layer2/bias�!<
p
layer2/bias*a	   �|6�?   @��?      @!   �:�?)�YxҒ4?2���J�\�?-Ա�L�?�������:              @        

lossqF?W���       �{�	�	6�D�A�:*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias�J�9

max/layer2/bias�$)<

min/layer2/biast!<
p
layer2/bias*a	   ��#�?   ��$�?      @!   0��?)P;dƹ4?2���J�\�?-Ա�L�?�������:              @        

loss�\?SB5��       �{�	��6�D�A�=*�
!
summaries/mean/layer2/biasӏ%<

sttdev/layer2/biask	�9

max/layer2/biascw)<

min/layer2/bias�~ <
p
layer2/bias*a	   ���?   `�.�?      @!   ��
�?) K�n&4?2���J�\�?-Ա�L�?�������:              @        

loss%�?�ɓ�       �{�	E�6�D�A�@*�
!
summaries/mean/layer2/bias,�%<

sttdev/layer2/bias�G�9

max/layer2/biasC�)<

min/layer2/bias��<
p
layer2/bias*a	   `?��?   `�;�?      @!   @x�?)`,��4?2���J�\�?-Ա�L�?�������:              @        

loss��?Ñ y      :�(n	��6�D�A�B*�
!
summaries/mean/layer2/biasǩ%<

sttdev/layer2/bias�':

max/layer2/bias�v*<

min/layer2/biasZ<
�
layer2/bias*q	   `C�?    �N�?      @!   `��?) l���4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss��?�{�      :�(n	e= 6�D�A�D*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias8��9

max/layer2/bias�{*<

min/layer2/biasgz<
�
layer2/bias*q	   �L�?    rO�?      @!   @R�?)�E+�4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss
?Kv�;�       �{�	}�&6�D�A�G*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/biasg�9

max/layer2/bias�E*<

min/layer2/bias��<
p
layer2/bias*a	    x�?   ��H�?      @!   ��?)�{B��4?2���J�\�?-Ա�L�?�������:              @        

loss�|?fI��       �{�	�.,6�D�A�I*�
!
summaries/mean/layer2/bias��%<

sttdev/layer2/bias���9

max/layer2/bias��)<

min/layer2/biasW�<
p
layer2/bias*a	   �
�?   `�8�?      @!   ���?)0�y�j4?2���J�\�?-Ա�L�?�������:              @        

loss~?-aF��       �{�	�\16�D�A�L*�
!
summaries/mean/layer2/bias��%<

sttdev/layer2/bias���9

max/layer2/biasg�(<

min/layer2/bias��<
p
layer2/bias*a	    ��?   ���?      @!   @t�?) �G�4?2���J�\�?-Ա�L�?�������:              @        

loss�]?�����       �{�	H86�D�A�O*�
!
summaries/mean/layer2/biasy�%<

sttdev/layer2/bias��9

max/layer2/bias�(<

min/layer2/bias-�<
p
layer2/bias*a	   ���?    c�?      @!   �v�?)���)4?2���J�\�?-Ա�L�?�������:              @        

lossݖ?mm���       �{�	L=6�D�A�Q*�
!
summaries/mean/layer2/bias��%<

sttdev/layer2/bias�?�9

max/layer2/biase�(<

min/layer2/biask�<
p
layer2/bias*a	   `���?   �l�?      @!    �?) 9)�4?2���J�\�?-Ա�L�?�������:              @        

loss}f?�!w��       �{�	�B6�D�A�S*�
!
summaries/mean/layer2/biasE�%<

sttdev/layer2/bias��9

max/layer2/biasC�(<

min/layer2/bias��<
p
layer2/bias*a	   �u��?   `��?      @!   �l�?)й�{�4?2���J�\�?-Ա�L�?�������:              @        

loss��?ȃb��       �{�	�0I6�D�A�V*�
!
summaries/mean/layer2/biaso�%<

sttdev/layer2/biasZP :

max/layer2/bias�V)<

min/layer2/biasa<
p
layer2/bias*a	    �?   `�*�?      @!   �4�?)�킮!4?2���J�\�?-Ա�L�?�������:              @        

loss��?�E�       �{�	�dN6�D�A�X*�
!
summaries/mean/layer2/biasǘ%<

sttdev/layer2/bias4I:

max/layer2/bias%�)<

min/layer2/bias�<
p
layer2/bias*a	   �؃?   ��0�?      @!   `��?)���Q4?2���J�\�?-Ա�L�?�������:              @        

lossn?����       �{�	8�S6�D�A�[*�
!
summaries/mean/layer2/biass�%<

sttdev/layer2/biasZ�:

max/layer2/biasio)<

min/layer2/bias@<
p
layer2/bias*a	   �ȃ?    �-�?      @!   ���?)���4?2���J�\�?-Ա�L�?�������:              @        

loss�.?0�Ѫ�       �{�	:TZ6�D�A�^*�
!
summaries/mean/layer2/bias��%<

sttdev/layer2/biasS�:

max/layer2/biasЊ)<

min/layer2/bias�<
p
layer2/bias*a	    ���?    Z1�?      @!    �
�?)  �:�4?2���J�\�?-Ա�L�?�������:              @        

loss6s?%�h�       �{�	$�_6�D�A�`*�
!
summaries/mean/layer2/biasl�%<

sttdev/layer2/biasm�:

max/layer2/bias(*<

min/layer2/biasӆ<
p
layer2/bias*a	   `ڰ�?    E�?      @!   PT
�?)�`��4?2���J�\�?-Ա�L�?�������:              @        

loss!}?+M=�      :�(n	��d6�D�A�b*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias�&:

max/layer2/bias�*<

min/layer2/bias�<
�
layer2/bias*q	   �Ԣ�?   ��V�?      @!   0�	�?) ���4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

lossB??*�p      :�(n	R�k6�D�A�e*�
!
summaries/mean/layer2/bias7�%<

sttdev/layer2/bias��):

max/layer2/bias�/+<

min/layer2/bias�<
�
layer2/bias*q	    ���?   ��e�?      @!    ��?)`����4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss�?�f�      :�(n	K�p6�D�A�g*�
!
summaries/mean/layer2/bias<|%<

sttdev/layer2/bias,4%:

max/layer2/biasXb+<

min/layer2/bias�Q<
�
layer2/bias*q	   `;��?    Kl�?      @!   @K�?)�	l4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss�K?�%      :�(n	��u6�D�A�i*�
!
summaries/mean/layer2/bias�s%<

sttdev/layer2/biasv:

max/layer2/bias�q+<

min/layer2/bias��<
�
layer2/bias*q	    ���?   �5n�?      @!    ��?)����4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

lossh�?Kʛ      :�(n	��|6�D�A�l*�
!
summaries/mean/layer2/bias�i%<

sttdev/layer2/bias�:

max/layer2/bias�y+<

min/layer2/bias[<
�
layer2/bias*q	   `�Ã?   �>o�?      @!   ���?)0p[��4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss��?4vJ�      :�(n	�Ӂ6�D�A�o*�
!
summaries/mean/layer2/bias�`%<

sttdev/layer2/biasc�:

max/layer2/bias}�+<

min/layer2/bias�<
�
layer2/bias*q	   `}Ѓ?   �/p�?      @!   P-�?)��Tx4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

lossh-?���      :�(n	e�6�D�A�q*�
!
summaries/mean/layer2/biasW%<

sttdev/layer2/bias��:

max/layer2/bias��+<

min/layer2/biasF�<
�
layer2/bias*q	   �hރ?    6r�?      @!   `S �?)`0���
4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss
�?d��      :�(n	^��6�D�A�t*�
!
summaries/mean/layer2/biasMG%<

sttdev/layer2/bias��:

max/layer2/bias�p+<

min/layer2/biasl�<
�
layer2/bias*q	   ����?   `n�?      @!   p^��?)�ٲJ4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss�8?
v'+      :�(n	>�6�D�A�v*�
!
summaries/mean/layer2/bias96%<

sttdev/layer2/bias8o�9

max/layer2/bias_+<

min/layer2/biasǊ <
�
layer2/bias*q	   �X�?   ��a�?      @!   �*��?)`!��c4?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss5�?���      :�(n	K'�6�D�A�x*�
!
summaries/mean/layer2/bias*%<

sttdev/layer2/bias���9

max/layer2/biasz�*<

min/layer2/bias�4!<
�
layer2/bias*q	   `�&�?   @�V�?      @!   P���?)�OY���3?2 ���J�\�?-Ա�L�?eiS�m�?�������:                @      �?        

loss��?/G.	�       �{�	�^�6�D�A�{*�
!
summaries/mean/layer2/bias�%%<

sttdev/layer2/bias���9

max/layer2/bias�c*<

min/layer2/biasՁ!<
p
layer2/bias*a	   �:0�?   �uL�?      @!   p��?)0��`��3?2���J�\�?-Ա�L�?�������:              @        

loss��?Y$S�       �{�	c!�6�D�A�~*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias�K�9

max/layer2/biasW*<

min/layer2/bias�"<
p
layer2/bias*a	    xA�?   �*C�?      @!    (��?) 
�U��3?2���J�\�?-Ա�L�?�������:              @        

loss�i?��i�       b�D�	V�6�D�A��*�
!
summaries/mean/layer2/bias=%<

sttdev/layer2/bias�9

max/layer2/bias��)<

min/layer2/biaslF"<
p
layer2/bias*a	   ��H�?   @�:�?      @!   P[��?)P��p�3?2���J�\�?-Ա�L�?�������:              @        

loss{?&
�:�       b�D�	���6�D�Aɂ*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias3�9

max/layer2/biasg�)<

min/layer2/bias%Z"<
p
layer2/bias*a	   �DK�?   �3�?      @!   p��?)0ΦT��3?2���J�\�?-Ա�L�?�������:              @        

loss��?d�J��       b�D�	M<�6�D�AÅ*�
!
summaries/mean/layer2/bias%<

sttdev/layer2/bias��9

max/layer2/bias�`)<

min/layer2/bias�"<
p
layer2/bias*a	   �S�?   �,�?      @!   @0�?)�ꦍ�3?2���J�\�?-Ա�L�?�������:              @        

lossb�?�-���       b�D�	�x�6�D�A�*�
!
summaries/mean/layer2/bias�
%<

sttdev/layer2/bias/�9

max/layer2/bias�-)<

min/layer2/bias��"<
p
layer2/bias*a	   �]�?   ��%�?      @!   ��?)@|���3?2���J�\�?-Ա�L�?�������:              @        

loss�?<�       b�D�	���6�D�A��*�
!
summaries/mean/layer2/biasU%<

sttdev/layer2/bias
��9

max/layer2/bias��(<

min/layer2/bias?�"<
p
layer2/bias*a	   �GY�?    ��?      @!     �?) VHK��3?2���J�\�?-Ա�L�?�������:              @        

lossܓ?��b�       b�D�	��6�D�A��*�
!
summaries/mean/layer2/bias{%<

sttdev/layer2/bias��9

max/layer2/bias*�(<

min/layer2/bias�"<
p
layer2/bias*a	   `~R�?   @��?      @!    ��?)`�w�T�3?2���J�\�?-Ա�L�?�������:              @        

loss�x?��;�       b�D�	�6�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias0�9

max/layer2/bias<�(<

min/layer2/bias{k"<
p
layer2/bias*a	   `oM�?   ��?      @!   P��?)к����3?2���J�\�?-Ա�L�?�������:              @        

loss�M?"1B��       b�D�	DU�6�D�A��*�
!
summaries/mean/layer2/bias%<

sttdev/layer2/bias豐9

max/layer2/bias΍(<

min/layer2/bias�F"<
p
layer2/bias*a	   ��H�?   ���?      @!   ��?)�����3?2���J�\�?-Ա�L�?�������:              @        

loss[`?DQ k�       b�D�	�%�6�D�A��*�
!
summaries/mean/layer2/biasO%<

sttdev/layer2/biasrX�9

max/layer2/bias�n(<

min/layer2/bias�3"<
p
layer2/bias*a	   �{F�?   ���?      @!   ��?)P0����3?2���J�\�?-Ա�L�?�������:              @        

loss��?�|�3�       b�D�	V�6�D�Aז*�
!
summaries/mean/layer2/bias%<

sttdev/layer2/biasُ9

max/layer2/bias2R(<

min/layer2/bias��!<
p
layer2/bias*a	    �>�?   @F
�?      @!   ���?)��U��3?2���J�\�?-Ա�L�?�������:              @        

loss;�? ����       b�D�	���6�D�A��*�
!
summaries/mean/layer2/bias^%<

sttdev/layer2/biasc�9

max/layer2/biasv8(<

min/layer2/biasZ�!<
p
layer2/bias*a	   @K<�?   ��?      @!   ���?)��ޟ�3?2���J�\�?-Ա�L�?�������:              @        

lossu�?��~��       b�D�	�C�6�D�A�*�
!
summaries/mean/layer2/bias%<

sttdev/layer2/bias��9

max/layer2/bias!(<

min/layer2/biash�!<
p
layer2/bias*a	    9�?   �#�?      @!   p��?)�� ���3?2���J�\�?-Ա�L�?�������:              @        

lossxm?V	�1�       b�D�	
y�6�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias�9

max/layer2/bias�(<

min/layer2/bias��!<
p
layer2/bias*a	   �08�?   �}�?      @!   `��?)���,��3?2���J�\�?-Ա�L�?�������:              @        

loss'�?$���       b�D�	\��6�D�A��*�
!
summaries/mean/layer2/bias)%<

sttdev/layer2/biasQ�9

max/layer2/bias��'<

min/layer2/biasн!<
p
layer2/bias*a	    �7�?   ���?      @!   ���?) }-���3?2���J�\�?-Ա�L�?�������:              @        

loss&�?F��       b�D�	`T�6�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/biasnʋ9

max/layer2/biasF�'<

min/layer2/bias��!<
p
layer2/bias*a	    �8�?   ����?      @!   �\�?)��W�6�3?2���J�\�?-Ա�L�?�������:              @        

lossh�?t���       b�D�	Y��6�D�A˥*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias@x�9

max/layer2/biasw�'<

min/layer2/bias��!<
p
layer2/bias*a	   ��=�?   ����?      @!   ���?)0��f�3?2���J�\�?-Ա�L�?�������:              @        

loss�	?�?���       b�D�	��7�D�A�*�
!
summaries/mean/layer2/bias`
%<

sttdev/layer2/bias&y}9

max/layer2/bias!�'<

min/layer2/bias0"<
p
layer2/bias*a	    F�?    $��?      @!    ��?) �� -�3?2���J�\�?-Ա�L�?�������:              @        

loss�\?�X���       b�D�	�y7�D�A�*�
!
summaries/mean/layer2/bias[	%<

sttdev/layer2/bias��y9

max/layer2/bias�'<

min/layer2/bias7"<
p
layer2/bias*a	   ��F�?   @���?      @!   ��?)���3?2���J�\�?-Ա�L�?�������:              @        

lossp�?�!}e�       b�D�	�7�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias�>x9

max/layer2/bias��'<

min/layer2/bias�5"<
p
layer2/bias*a	   ��F�?   @���?      @!   P��?)۳	��3?2���J�\�?-Ա�L�?�������:              @        

lossK?{j���       b�D�	��7�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/biasC=x9

max/layer2/biasl�'<

min/layer2/bias�,"<
p
layer2/bias*a	   `�E�?   ���?      @!   p��?)�Y����3?2���J�\�?-Ա�L�?�������:              @        

loss�G?���T�       b�D�	/�7�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias�t9

max/layer2/bias�'<

min/layer2/biasW9"<
p
layer2/bias*a	   �*G�?    C�?      @!   �m�?)`��p�3?2���J�\�?-Ա�L�?�������:              @        

loss�?YVƛ�       b�D�	��!7�D�A��*�
!
summaries/mean/layer2/biasa%<

sttdev/layer2/biasKes9

max/layer2/bias��'<

min/layer2/bias�5"<
p
layer2/bias*a	   ��F�?   ��?      @!   Pb�?)���.`�3?2���J�\�?-Ա�L�?�������:              @        

lossIT?��>��       b�D�	�'7�D�A߶*�
!
summaries/mean/layer2/bias�	%<

sttdev/layer2/bias�}9

max/layer2/biasG�'<

min/layer2/bias�!<
p
layer2/bias*a	    �?�?   �h��?      @!   ���?) W�3?2���J�\�?-Ա�L�?�������:              @        

loss��?Y�̃�       b�D�	�V,7�D�A��*�
!
summaries/mean/layer2/biasW	%<

sttdev/layer2/bias
2~9

max/layer2/bias�p'<

min/layer2/bias(�!<
p
layer2/bias*a	    e>�?   ��?      @!   0��?)�5�5��3?2���J�\�?-Ա�L�?�������:              @        

loss�9?����       b�D�	�37�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias+��9

max/layer2/bias�m'<

min/layer2/bias�!<
p
layer2/bias*a	   ��6�?   `��?      @!   0_�?)��F}��3?2���J�\�?-Ա�L�?�������:              @        

loss�?����       b�D�	B887�D�A��*�
!
summaries/mean/layer2/biasD%<

sttdev/layer2/bias��9

max/layer2/bias=�'<

min/layer2/biasR]!<
p
layer2/bias*a	   @�+�?   ���?      @!   ���?)�D�&��3?2���J�\�?-Ա�L�?�������:              @        

lossv?0����       b�D�	}s=7�D�A��*�
!
summaries/mean/layer2/bias�%<

sttdev/layer2/bias6C�9

max/layer2/bias�'<

min/layer2/bias�!<
p
layer2/bias*a	    � �?   @ ��?      @!   ����?)P�@~�3?2���J�\�?-Ա�L�?�������:              @        

loss�>?W�       b�D�	�)D7�D�A��*�
!
summaries/mean/layer2/bias�$%<

sttdev/layer2/biasۧ�9

max/layer2/bias)�'<

min/layer2/biask� <
p
layer2/bias*a	   `m�?    ��?      @!   ����?)�D���3?2���J�\�?-Ա�L�?�������:              @        

loss�?�m