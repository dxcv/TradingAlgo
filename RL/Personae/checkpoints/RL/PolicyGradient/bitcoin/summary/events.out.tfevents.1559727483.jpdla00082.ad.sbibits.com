       ЃK"	  Р^у=зAbrain.Event:2'ЩDГ      ЁG	№ни^у=зA"Зц
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
Є
/nn/dense/kernel/Initializer/random_normal/shapeConst*
valueB"   @   *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
:

.nn/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@nn/dense/kernel

0nn/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *o:*"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
: 
њ
>nn/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/nn/dense/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*"
_class
loc:@nn/dense/kernel
ѓ
-nn/dense/kernel/Initializer/random_normal/mulMul>nn/dense/kernel/Initializer/random_normal/RandomStandardNormal0nn/dense/kernel/Initializer/random_normal/stddev*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
м
)nn/dense/kernel/Initializer/random_normalAdd-nn/dense/kernel/Initializer/random_normal/mul.nn/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:@*
T0*"
_class
loc:@nn/dense/kernel
Ї
nn/dense/kernel
VariableV2*"
_class
loc:@nn/dense/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
в
nn/dense/kernel/AssignAssignnn/dense/kernel)nn/dense/kernel/Initializer/random_normal*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
~
nn/dense/kernel/readIdentitynn/dense/kernel*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@

nn/dense/bias/Initializer/ConstConst*
valueB@*ЭЬЬ=* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@

nn/dense/bias
VariableV2*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
О
nn/dense/bias/AssignAssignnn/dense/biasnn/dense/bias/Initializer/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias
t
nn/dense/bias/readIdentitynn/dense/bias*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@

nn/dense/MatMulMatMulPlaceholder_2nn/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 

nn/dense/BiasAddBiasAddnn/dense/MatMulnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
Y
nn/dense/ReluRelunn/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
Ј
1nn/dense_1/kernel/Initializer/random_normal/shapeConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:

0nn/dense_1/kernel/Initializer/random_normal/meanConst*
valueB
 *    *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 

2nn/dense_1/kernel/Initializer/random_normal/stddevConst*
valueB
 *o:*$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 

@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:@@*

seed *
T0*$
_class
loc:@nn/dense_1/kernel*
seed2 
ћ
/nn/dense_1/kernel/Initializer/random_normal/mulMul@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_1/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
ф
+nn/dense_1/kernel/Initializer/random_normalAdd/nn/dense_1/kernel/Initializer/random_normal/mul0nn/dense_1/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Ћ
nn/dense_1/kernel
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *$
_class
loc:@nn/dense_1/kernel*
	container 
к
nn/dense_1/kernel/AssignAssignnn/dense_1/kernel+nn/dense_1/kernel/Initializer/random_normal*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@

nn/dense_1/kernel/readIdentitynn/dense_1/kernel*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@

!nn/dense_1/bias/Initializer/ConstConst*
valueB@*ЭЬЬ=*"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@

nn/dense_1/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container *
shape:@
Ц
nn/dense_1/bias/AssignAssignnn/dense_1/bias!nn/dense_1/bias/Initializer/Const*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
z
nn/dense_1/bias/readIdentitynn/dense_1/bias*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@

nn/dense_1/MatMulMatMulnn/dense/Relunn/dense_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 

nn/dense_1/BiasAddBiasAddnn/dense_1/MatMulnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
]
nn/dense_1/ReluRelunn/dense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
Ј
1nn/dense_2/kernel/Initializer/random_normal/shapeConst*
valueB"@      *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
:

0nn/dense_2/kernel/Initializer/random_normal/meanConst*
valueB
 *    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
: 

2nn/dense_2/kernel/Initializer/random_normal/stddevConst*
valueB
 *o:*$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
: 

@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_2/kernel/Initializer/random_normal/shape*
T0*$
_class
loc:@nn/dense_2/kernel*
seed2 *
dtype0*
_output_shapes

:@*

seed 
ћ
/nn/dense_2/kernel/Initializer/random_normal/mulMul@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_2/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
ф
+nn/dense_2/kernel/Initializer/random_normalAdd/nn/dense_2/kernel/Initializer/random_normal/mul0nn/dense_2/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ћ
nn/dense_2/kernel
VariableV2*
shared_name *$
_class
loc:@nn/dense_2/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
к
nn/dense_2/kernel/AssignAssignnn/dense_2/kernel+nn/dense_2/kernel/Initializer/random_normal*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(

nn/dense_2/kernel/readIdentitynn/dense_2/kernel*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@

!nn/dense_2/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:

nn/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias*
	container *
shape:
Ц
nn/dense_2/bias/AssignAssignnn/dense_2/bias!nn/dense_2/bias/Initializer/Const*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
z
nn/dense_2/bias/readIdentitynn/dense_2/bias*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:

nn/dense_2/MatMulMatMulnn/dense_1/Relunn/dense_2/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

nn/dense_2/BiasAddBiasAddnn/dense_2/MatMulnn/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
[

nn/SoftmaxSoftmaxnn/dense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
y
.loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
ф
Lloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnn/dense_2/BiasAddPlaceholder*
T0*6
_output_shapes$
":џџџџџџџџџ:џџџџџџџџџ*
Tlabels0

loss/mulMulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsPlaceholder_1*
T0*#
_output_shapes
:џџџџџџџџџ
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/mul
loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
^
train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Ј
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/mul*
T0*
out_type0*
_output_shapes
:
Й
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
n
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/mul*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
З
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Л
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ѓ
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Ё
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 

#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Љ
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
Џ
#train/gradients/loss/mul_grad/ShapeShapeLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
r
%train/gradients/loss/mul_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
е
3train/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/mul_grad/Shape%train/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

!train/gradients/loss/mul_grad/MulMul&train/gradients/loss/Mean_grad/truedivPlaceholder_1*#
_output_shapes
:џџџџџџџџџ*
T0
Р
!train/gradients/loss/mul_grad/SumSum!train/gradients/loss/mul_grad/Mul3train/gradients/loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
%train/gradients/loss/mul_grad/ReshapeReshape!train/gradients/loss/mul_grad/Sum#train/gradients/loss/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Ю
#train/gradients/loss/mul_grad/Mul_1MulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&train/gradients/loss/Mean_grad/truediv*
T0*#
_output_shapes
:џџџџџџџџџ
Ц
#train/gradients/loss/mul_grad/Sum_1Sum#train/gradients/loss/mul_grad/Mul_15train/gradients/loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
К
'train/gradients/loss/mul_grad/Reshape_1Reshape#train/gradients/loss/mul_grad/Sum_1%train/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

.train/gradients/loss/mul_grad/tuple/group_depsNoOp&^train/gradients/loss/mul_grad/Reshape(^train/gradients/loss/mul_grad/Reshape_1

6train/gradients/loss/mul_grad/tuple/control_dependencyIdentity%train/gradients/loss/mul_grad/Reshape/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/mul_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

8train/gradients/loss/mul_grad/tuple/control_dependency_1Identity'train/gradients/loss/mul_grad/Reshape_1/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/mul_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
Љ
train/gradients/zeros_like	ZerosLikeNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:џџџџџџџџџ*
T0
Н
qtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:џџџџџџџџџ*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
Л
ptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
т
ltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims6train/gradients/loss/mul_grad/tuple/control_dependencyptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ*

Tdim0
џ
etrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsqtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:џџџџџџџџџ
х
3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:
о
8train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_depsNoOpf^train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul4^train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad

@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*x
_classn
ljloc:@train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul
Ћ
Btrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
щ
-train/gradients/nn/dense_2/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencynn/dense_2/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
л
/train/gradients/nn/dense_2/MatMul_grad/MatMul_1MatMulnn/dense_1/Relu@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
Ё
7train/gradients/nn/dense_2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_2/MatMul_grad/MatMul0^train/gradients/nn/dense_2/MatMul_grad/MatMul_1
Ј
?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_2/MatMul_grad/MatMul8^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
Ѕ
Atrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_2/MatMul_grad/MatMul_18^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
Н
-train/gradients/nn/dense_1/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencynn/dense_1/Relu*'
_output_shapes
:џџџџџџџџџ@*
T0
­
3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/nn/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
І
8train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad.^train/gradients/nn/dense_1/Relu_grad/ReluGrad
Њ
@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/Relu_grad/ReluGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/Relu_grad/ReluGrad
Ћ
Btrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
щ
-train/gradients/nn/dense_1/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencynn/dense_1/kernel/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
й
/train/gradients/nn/dense_1/MatMul_grad/MatMul_1MatMulnn/dense/Relu@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
Ё
7train/gradients/nn/dense_1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_1/MatMul_grad/MatMul0^train/gradients/nn/dense_1/MatMul_grad/MatMul_1
Ј
?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/MatMul_grad/MatMul8^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
Ѕ
Atrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_1/MatMul_grad/MatMul_18^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
Й
+train/gradients/nn/dense/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencynn/dense/Relu*
T0*'
_output_shapes
:џџџџџџџџџ@
Љ
1train/gradients/nn/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+train/gradients/nn/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
 
6train/gradients/nn/dense/BiasAdd_grad/tuple/group_depsNoOp2^train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad,^train/gradients/nn/dense/Relu_grad/ReluGrad
Ђ
>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/Relu_grad/ReluGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/Relu_grad/ReluGrad*'
_output_shapes
:џџџџџџџџџ@
Ѓ
@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1Identity1train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
у
+train/gradients/nn/dense/MatMul_grad/MatMulMatMul>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencynn/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
е
-train/gradients/nn/dense/MatMul_grad/MatMul_1MatMulPlaceholder_2>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(

5train/gradients/nn/dense/MatMul_grad/tuple/group_depsNoOp,^train/gradients/nn/dense/MatMul_grad/MatMul.^train/gradients/nn/dense/MatMul_grad/MatMul_1
 
=train/gradients/nn/dense/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/MatMul_grad/MatMul6^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ

?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/nn/dense/MatMul_grad/MatMul_16^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense/MatMul_grad/MatMul_1*
_output_shapes

:@

train/beta1_power/initial_valueConst*
valueB
 *fff?* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
: 

train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape: 
Т
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
x
train/beta1_power/readIdentitytrain/beta1_power*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 

train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?* 
_class
loc:@nn/dense/bias

train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape: 
Т
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
x
train/beta2_power/readIdentitytrain/beta2_power*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
Ѕ
,train/nn/dense/kernel/Adam/Initializer/zerosConst*"
_class
loc:@nn/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
В
train/nn/dense/kernel/Adam
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *"
_class
loc:@nn/dense/kernel*
	container 
ы
!train/nn/dense/kernel/Adam/AssignAssigntrain/nn/dense/kernel/Adam,train/nn/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel

train/nn/dense/kernel/Adam/readIdentitytrain/nn/dense/kernel/Adam*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Ї
.train/nn/dense/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*"
_class
loc:@nn/dense/kernel*
valueB@*    
Д
train/nn/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *"
_class
loc:@nn/dense/kernel*
	container *
shape
:@
ё
#train/nn/dense/kernel/Adam_1/AssignAssigntrain/nn/dense/kernel/Adam_1.train/nn/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel

!train/nn/dense/kernel/Adam_1/readIdentitytrain/nn/dense/kernel/Adam_1*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@

*train/nn/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@nn/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
І
train/nn/dense/bias/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@nn/dense/bias*
	container 
п
train/nn/dense/bias/Adam/AssignAssigntrain/nn/dense/bias/Adam*train/nn/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@

train/nn/dense/bias/Adam/readIdentitytrain/nn/dense/bias/Adam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@

,train/nn/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@nn/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ј
train/nn/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape:@
х
!train/nn/dense/bias/Adam_1/AssignAssigntrain/nn/dense/bias/Adam_1,train/nn/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@

train/nn/dense/bias/Adam_1/readIdentitytrain/nn/dense/bias/Adam_1*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
Е
>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:

4train/nn/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.train/nn/dense_1/kernel/Adam/Initializer/zerosFill>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor4train/nn/dense_1/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel*

index_type0
Ж
train/nn/dense_1/kernel/Adam
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *$
_class
loc:@nn/dense_1/kernel
ѓ
#train/nn/dense_1/kernel/Adam/AssignAssigntrain/nn/dense_1/kernel/Adam.train/nn/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@

!train/nn/dense_1/kernel/Adam/readIdentitytrain/nn/dense_1/kernel/Adam*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
З
@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   
Ё
6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0train/nn/dense_1/kernel/Adam_1/Initializer/zerosFill@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*$
_class
loc:@nn/dense_1/kernel*

index_type0*
_output_shapes

:@@
И
train/nn/dense_1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *$
_class
loc:@nn/dense_1/kernel*
	container *
shape
:@@
љ
%train/nn/dense_1/kernel/Adam_1/AssignAssigntrain/nn/dense_1/kernel/Adam_10train/nn/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel

#train/nn/dense_1/kernel/Adam_1/readIdentitytrain/nn/dense_1/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@

,train/nn/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@nn/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Њ
train/nn/dense_1/bias/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias
ч
!train/nn/dense_1/bias/Adam/AssignAssigntrain/nn/dense_1/bias/Adam,train/nn/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@

train/nn/dense_1/bias/Adam/readIdentitytrain/nn/dense_1/bias/Adam*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@

.train/nn/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*"
_class
loc:@nn/dense_1/bias*
valueB@*    
Ќ
train/nn/dense_1/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
э
#train/nn/dense_1/bias/Adam_1/AssignAssigntrain/nn/dense_1/bias/Adam_1.train/nn/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@

!train/nn/dense_1/bias/Adam_1/readIdentitytrain/nn/dense_1/bias/Adam_1*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Љ
.train/nn/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@nn/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
Ж
train/nn/dense_2/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@nn/dense_2/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
ѓ
#train/nn/dense_2/kernel/Adam/AssignAssigntrain/nn/dense_2/kernel/Adam.train/nn/dense_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(

!train/nn/dense_2/kernel/Adam/readIdentitytrain/nn/dense_2/kernel/Adam*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
Ћ
0train/nn/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@nn/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
И
train/nn/dense_2/kernel/Adam_1
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *$
_class
loc:@nn/dense_2/kernel
љ
%train/nn/dense_2/kernel/Adam_1/AssignAssigntrain/nn/dense_2/kernel/Adam_10train/nn/dense_2/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(

#train/nn/dense_2/kernel/Adam_1/readIdentitytrain/nn/dense_2/kernel/Adam_1*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel

,train/nn/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@nn/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Њ
train/nn/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias*
	container *
shape:
ч
!train/nn/dense_2/bias/Adam/AssignAssigntrain/nn/dense_2/bias/Adam,train/nn/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias

train/nn/dense_2/bias/Adam/readIdentitytrain/nn/dense_2/bias/Adam*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:

.train/nn/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@nn/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Ќ
train/nn/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias*
	container *
shape:
э
#train/nn/dense_2/bias/Adam_1/AssignAssigntrain/nn/dense_2/bias/Adam_1.train/nn/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias

!train/nn/dense_2/bias/Adam_1/readIdentitytrain/nn/dense_2/bias/Adam_1*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *ІФ;*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wО?
W
train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
К
+train/Adam/update_nn/dense/kernel/ApplyAdam	ApplyAdamnn/dense/kerneltrain/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*"
_class
loc:@nn/dense/kernel
­
)train/Adam/update_nn/dense/bias/ApplyAdam	ApplyAdamnn/dense/biastrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0* 
_class
loc:@nn/dense/bias
Ц
-train/Adam/update_nn/dense_1/kernel/ApplyAdam	ApplyAdamnn/dense_1/kerneltrain/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@nn/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
Й
+train/Adam/update_nn/dense_1/bias/ApplyAdam	ApplyAdamnn/dense_1/biastrain/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense_1/bias*
use_nesterov( *
_output_shapes
:@
Ц
-train/Adam/update_nn/dense_2/kernel/ApplyAdam	ApplyAdamnn/dense_2/kerneltrain/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@nn/dense_2/kernel*
use_nesterov( *
_output_shapes

:@
Й
+train/Adam/update_nn/dense_2/bias/ApplyAdam	ApplyAdamnn/dense_2/biastrain/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense_2/bias*
use_nesterov( *
_output_shapes
:

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@nn/dense/bias
Њ
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@nn/dense/bias
Ў
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
в

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam

initNoOp^nn/dense/bias/Assign^nn/dense/kernel/Assign^nn/dense_1/bias/Assign^nn/dense_1/kernel/Assign^nn/dense_2/bias/Assign^nn/dense_2/kernel/Assign^train/beta1_power/Assign^train/beta2_power/Assign ^train/nn/dense/bias/Adam/Assign"^train/nn/dense/bias/Adam_1/Assign"^train/nn/dense/kernel/Adam/Assign$^train/nn/dense/kernel/Adam_1/Assign"^train/nn/dense_1/bias/Adam/Assign$^train/nn/dense_1/bias/Adam_1/Assign$^train/nn/dense_1/kernel/Adam/Assign&^train/nn/dense_1/kernel/Adam_1/Assign"^train/nn/dense_2/bias/Adam/Assign$^train/nn/dense_2/bias/Adam_1/Assign$^train/nn/dense_2/kernel/Adam/Assign&^train/nn/dense_2/kernel/Adam_1/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
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
а
save/SaveV2/tensor_namesConst*
valueљBіBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesnn/dense/biasnn/dense/kernelnn/dense_1/biasnn/dense_1/kernelnn/dense_2/biasnn/dense_2/kerneltrain/beta1_powertrain/beta2_powertrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1*"
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
т
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueљBіBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
ў
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Є
save/AssignAssignnn/dense/biassave/RestoreV2*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
А
save/Assign_1Assignnn/dense/kernelsave/RestoreV2:1*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
Ќ
save/Assign_2Assignnn/dense_1/biassave/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
Д
save/Assign_3Assignnn/dense_1/kernelsave/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ќ
save/Assign_4Assignnn/dense_2/biassave/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
Д
save/Assign_5Assignnn/dense_2/kernelsave/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Ј
save/Assign_6Assigntrain/beta1_powersave/RestoreV2:6*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ј
save/Assign_7Assigntrain/beta2_powersave/RestoreV2:7*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Г
save/Assign_8Assigntrain/nn/dense/bias/Adamsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
Е
save/Assign_9Assigntrain/nn/dense/bias/Adam_1save/RestoreV2:9*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
Н
save/Assign_10Assigntrain/nn/dense/kernel/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
П
save/Assign_11Assigntrain/nn/dense/kernel/Adam_1save/RestoreV2:11*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
Й
save/Assign_12Assigntrain/nn/dense_1/bias/Adamsave/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
Л
save/Assign_13Assigntrain/nn/dense_1/bias/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
С
save/Assign_14Assigntrain/nn/dense_1/kernel/Adamsave/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
У
save/Assign_15Assigntrain/nn/dense_1/kernel/Adam_1save/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Й
save/Assign_16Assigntrain/nn/dense_2/bias/Adamsave/RestoreV2:16*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Л
save/Assign_17Assigntrain/nn/dense_2/bias/Adam_1save/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
С
save/Assign_18Assigntrain/nn/dense_2/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
У
save/Assign_19Assigntrain/nn/dense_2/kernel/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
р
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"ЌJ)ѓHЮ      кЊЮ	ЃЮй^у=зAJЛ
Џ
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
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
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
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
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
Ttype*1.13.12b'v1.13.1-0-g6612da8951'Зц
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
Є
/nn/dense/kernel/Initializer/random_normal/shapeConst*"
_class
loc:@nn/dense/kernel*
valueB"   @   *
dtype0*
_output_shapes
:

.nn/dense/kernel/Initializer/random_normal/meanConst*"
_class
loc:@nn/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0nn/dense/kernel/Initializer/random_normal/stddevConst*"
_class
loc:@nn/dense/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
њ
>nn/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/nn/dense/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*"
_class
loc:@nn/dense/kernel
ѓ
-nn/dense/kernel/Initializer/random_normal/mulMul>nn/dense/kernel/Initializer/random_normal/RandomStandardNormal0nn/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:@*
T0*"
_class
loc:@nn/dense/kernel
м
)nn/dense/kernel/Initializer/random_normalAdd-nn/dense/kernel/Initializer/random_normal/mul.nn/dense/kernel/Initializer/random_normal/mean*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Ї
nn/dense/kernel
VariableV2*
shared_name *"
_class
loc:@nn/dense/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
в
nn/dense/kernel/AssignAssignnn/dense/kernel)nn/dense/kernel/Initializer/random_normal*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
~
nn/dense/kernel/readIdentitynn/dense/kernel*
_output_shapes

:@*
T0*"
_class
loc:@nn/dense/kernel

nn/dense/bias/Initializer/ConstConst* 
_class
loc:@nn/dense/bias*
valueB@*ЭЬЬ=*
dtype0*
_output_shapes
:@

nn/dense/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape:@
О
nn/dense/bias/AssignAssignnn/dense/biasnn/dense/bias/Initializer/Const*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
t
nn/dense/bias/readIdentitynn/dense/bias*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@

nn/dense/MatMulMatMulPlaceholder_2nn/dense/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ@*
transpose_b( *
T0

nn/dense/BiasAddBiasAddnn/dense/MatMulnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
Y
nn/dense/ReluRelunn/dense/BiasAdd*'
_output_shapes
:џџџџџџџџџ@*
T0
Ј
1nn/dense_1/kernel/Initializer/random_normal/shapeConst*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:

0nn/dense_1/kernel/Initializer/random_normal/meanConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

2nn/dense_1/kernel/Initializer/random_normal/stddevConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 

@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:@@*

seed *
T0*$
_class
loc:@nn/dense_1/kernel*
seed2 
ћ
/nn/dense_1/kernel/Initializer/random_normal/mulMul@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_1/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
ф
+nn/dense_1/kernel/Initializer/random_normalAdd/nn/dense_1/kernel/Initializer/random_normal/mul0nn/dense_1/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Ћ
nn/dense_1/kernel
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *$
_class
loc:@nn/dense_1/kernel*
	container *
shape
:@@
к
nn/dense_1/kernel/AssignAssignnn/dense_1/kernel+nn/dense_1/kernel/Initializer/random_normal*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@

nn/dense_1/kernel/readIdentitynn/dense_1/kernel*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@

!nn/dense_1/bias/Initializer/ConstConst*"
_class
loc:@nn/dense_1/bias*
valueB@*ЭЬЬ=*
dtype0*
_output_shapes
:@

nn/dense_1/bias
VariableV2*"
_class
loc:@nn/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ц
nn/dense_1/bias/AssignAssignnn/dense_1/bias!nn/dense_1/bias/Initializer/Const*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
z
nn/dense_1/bias/readIdentitynn/dense_1/bias*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@

nn/dense_1/MatMulMatMulnn/dense/Relunn/dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ@*
transpose_b( 

nn/dense_1/BiasAddBiasAddnn/dense_1/MatMulnn/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@*
T0
]
nn/dense_1/ReluRelunn/dense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
Ј
1nn/dense_2/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*$
_class
loc:@nn/dense_2/kernel*
valueB"@      

0nn/dense_2/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *$
_class
loc:@nn/dense_2/kernel*
valueB
 *    

2nn/dense_2/kernel/Initializer/random_normal/stddevConst*$
_class
loc:@nn/dense_2/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 

@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_2/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*$
_class
loc:@nn/dense_2/kernel
ћ
/nn/dense_2/kernel/Initializer/random_normal/mulMul@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_2/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
ф
+nn/dense_2/kernel/Initializer/random_normalAdd/nn/dense_2/kernel/Initializer/random_normal/mul0nn/dense_2/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ћ
nn/dense_2/kernel
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *$
_class
loc:@nn/dense_2/kernel*
	container 
к
nn/dense_2/kernel/AssignAssignnn/dense_2/kernel+nn/dense_2/kernel/Initializer/random_normal*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(

nn/dense_2/kernel/readIdentitynn/dense_2/kernel*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel

!nn/dense_2/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*"
_class
loc:@nn/dense_2/bias*
valueB*ЭЬЬ=

nn/dense_2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias
Ц
nn/dense_2/bias/AssignAssignnn/dense_2/bias!nn/dense_2/bias/Initializer/Const*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
z
nn/dense_2/bias/readIdentitynn/dense_2/bias*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:

nn/dense_2/MatMulMatMulnn/dense_1/Relunn/dense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 

nn/dense_2/BiasAddBiasAddnn/dense_2/MatMulnn/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
[

nn/SoftmaxSoftmaxnn/dense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
y
.loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
ф
Lloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnn/dense_2/BiasAddPlaceholder*
T0*
Tlabels0*6
_output_shapes$
":џџџџџџџџџ:џџџџџџџџџ

loss/mulMulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsPlaceholder_1*#
_output_shapes
:џџџџџџџџџ*
T0
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/mul
loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ј
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/mul*
T0*
out_type0*
_output_shapes
:
Й
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
n
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/mul*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
З
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
p
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Л
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ѓ
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Ё
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 

#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
Љ
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
Џ
#train/gradients/loss/mul_grad/ShapeShapeLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
r
%train/gradients/loss/mul_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
е
3train/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/mul_grad/Shape%train/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

!train/gradients/loss/mul_grad/MulMul&train/gradients/loss/Mean_grad/truedivPlaceholder_1*
T0*#
_output_shapes
:џџџџџџџџџ
Р
!train/gradients/loss/mul_grad/SumSum!train/gradients/loss/mul_grad/Mul3train/gradients/loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Д
%train/gradients/loss/mul_grad/ReshapeReshape!train/gradients/loss/mul_grad/Sum#train/gradients/loss/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Ю
#train/gradients/loss/mul_grad/Mul_1MulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&train/gradients/loss/Mean_grad/truediv*#
_output_shapes
:џџџџџџџџџ*
T0
Ц
#train/gradients/loss/mul_grad/Sum_1Sum#train/gradients/loss/mul_grad/Mul_15train/gradients/loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
К
'train/gradients/loss/mul_grad/Reshape_1Reshape#train/gradients/loss/mul_grad/Sum_1%train/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

.train/gradients/loss/mul_grad/tuple/group_depsNoOp&^train/gradients/loss/mul_grad/Reshape(^train/gradients/loss/mul_grad/Reshape_1

6train/gradients/loss/mul_grad/tuple/control_dependencyIdentity%train/gradients/loss/mul_grad/Reshape/^train/gradients/loss/mul_grad/tuple/group_deps*#
_output_shapes
:џџџџџџџџџ*
T0*8
_class.
,*loc:@train/gradients/loss/mul_grad/Reshape

8train/gradients/loss/mul_grad/tuple/control_dependency_1Identity'train/gradients/loss/mul_grad/Reshape_1/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/mul_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
Љ
train/gradients/zeros_like	ZerosLikeNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:џџџџџџџџџ*
T0
Н
qtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:џџџџџџџџџ*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
Л
ptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
т
ltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims6train/gradients/loss/mul_grad/tuple/control_dependencyptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
џ
etrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsqtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:џџџџџџџџџ
х
3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:
о
8train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_depsNoOpf^train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul4^train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad

@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*x
_classn
ljloc:@train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*'
_output_shapes
:џџџџџџџџџ
Ћ
Btrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
щ
-train/gradients/nn/dense_2/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencynn/dense_2/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ@*
transpose_b(*
T0
л
/train/gradients/nn/dense_2/MatMul_grad/MatMul_1MatMulnn/dense_1/Relu@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@*
transpose_b( 
Ё
7train/gradients/nn/dense_2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_2/MatMul_grad/MatMul0^train/gradients/nn/dense_2/MatMul_grad/MatMul_1
Ј
?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_2/MatMul_grad/MatMul8^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*@
_class6
42loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul
Ѕ
Atrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_2/MatMul_grad/MatMul_18^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*B
_class8
64loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul_1
Н
-train/gradients/nn/dense_1/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencynn/dense_1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ@
­
3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/nn/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
І
8train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad.^train/gradients/nn/dense_1/Relu_grad/ReluGrad
Њ
@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/Relu_grad/ReluGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:џџџџџџџџџ@
Ћ
Btrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*F
_class<
:8loc:@train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad
щ
-train/gradients/nn/dense_1/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencynn/dense_1/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ@
й
/train/gradients/nn/dense_1/MatMul_grad/MatMul_1MatMulnn/dense/Relu@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:@@
Ё
7train/gradients/nn/dense_1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_1/MatMul_grad/MatMul0^train/gradients/nn/dense_1/MatMul_grad/MatMul_1
Ј
?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/MatMul_grad/MatMul8^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul
Ѕ
Atrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_1/MatMul_grad/MatMul_18^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
Й
+train/gradients/nn/dense/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencynn/dense/Relu*'
_output_shapes
:џџџџџџџџџ@*
T0
Љ
1train/gradients/nn/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+train/gradients/nn/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
 
6train/gradients/nn/dense/BiasAdd_grad/tuple/group_depsNoOp2^train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad,^train/gradients/nn/dense/Relu_grad/ReluGrad
Ђ
>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/Relu_grad/ReluGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/Relu_grad/ReluGrad*'
_output_shapes
:џџџџџџџџџ@
Ѓ
@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1Identity1train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
у
+train/gradients/nn/dense/MatMul_grad/MatMulMatMul>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencynn/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
е
-train/gradients/nn/dense/MatMul_grad/MatMul_1MatMulPlaceholder_2>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@*
transpose_b( *
T0

5train/gradients/nn/dense/MatMul_grad/tuple/group_depsNoOp,^train/gradients/nn/dense/MatMul_grad/MatMul.^train/gradients/nn/dense/MatMul_grad/MatMul_1
 
=train/gradients/nn/dense/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/MatMul_grad/MatMul6^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*>
_class4
20loc:@train/gradients/nn/dense/MatMul_grad/MatMul

?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/nn/dense/MatMul_grad/MatMul_16^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense/MatMul_grad/MatMul_1*
_output_shapes

:@

train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@nn/dense/bias*
valueB
 *fff?

train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape: 
Т
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
x
train/beta1_power/readIdentitytrain/beta1_power*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 

train/beta2_power/initial_valueConst* 
_class
loc:@nn/dense/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 

train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape: 
Т
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
x
train/beta2_power/readIdentitytrain/beta2_power*
_output_shapes
: *
T0* 
_class
loc:@nn/dense/bias
Ѕ
,train/nn/dense/kernel/Adam/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes

:@
В
train/nn/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *"
_class
loc:@nn/dense/kernel*
	container *
shape
:@
ы
!train/nn/dense/kernel/Adam/AssignAssigntrain/nn/dense/kernel/Adam,train/nn/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@

train/nn/dense/kernel/Adam/readIdentitytrain/nn/dense/kernel/Adam*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Ї
.train/nn/dense/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes

:@
Д
train/nn/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *"
_class
loc:@nn/dense/kernel*
	container *
shape
:@
ё
#train/nn/dense/kernel/Adam_1/AssignAssigntrain/nn/dense/kernel/Adam_1.train/nn/dense/kernel/Adam_1/Initializer/zeros*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(

!train/nn/dense/kernel/Adam_1/readIdentitytrain/nn/dense/kernel/Adam_1*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@

*train/nn/dense/bias/Adam/Initializer/zerosConst*
valueB@*    * 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
І
train/nn/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape:@
п
train/nn/dense/bias/Adam/AssignAssigntrain/nn/dense/bias/Adam*train/nn/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(

train/nn/dense/bias/Adam/readIdentitytrain/nn/dense/bias/Adam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@

,train/nn/dense/bias/Adam_1/Initializer/zerosConst*
valueB@*    * 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
Ј
train/nn/dense/bias/Adam_1
VariableV2* 
_class
loc:@nn/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
х
!train/nn/dense/bias/Adam_1/AssignAssigntrain/nn/dense/bias/Adam_1,train/nn/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias

train/nn/dense/bias/Adam_1/readIdentitytrain/nn/dense/bias/Adam_1*
_output_shapes
:@*
T0* 
_class
loc:@nn/dense/bias
Е
>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:

4train/nn/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 

.train/nn/dense_1/kernel/Adam/Initializer/zerosFill>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor4train/nn/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Ж
train/nn/dense_1/kernel/Adam
VariableV2*$
_class
loc:@nn/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
ѓ
#train/nn/dense_1/kernel/Adam/AssignAssigntrain/nn/dense_1/kernel/Adam.train/nn/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel

!train/nn/dense_1/kernel/Adam/readIdentitytrain/nn/dense_1/kernel/Adam*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
З
@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:
Ё
6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 

0train/nn/dense_1/kernel/Adam_1/Initializer/zerosFill@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:@@*
T0*

index_type0*$
_class
loc:@nn/dense_1/kernel
И
train/nn/dense_1/kernel/Adam_1
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *$
_class
loc:@nn/dense_1/kernel
љ
%train/nn/dense_1/kernel/Adam_1/AssignAssigntrain/nn/dense_1/kernel/Adam_10train/nn/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel

#train/nn/dense_1/kernel/Adam_1/readIdentitytrain/nn/dense_1/kernel/Adam_1*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel

,train/nn/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@
Њ
train/nn/dense_1/bias/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container 
ч
!train/nn/dense_1/bias/Adam/AssignAssigntrain/nn/dense_1/bias/Adam,train/nn/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias

train/nn/dense_1/bias/Adam/readIdentitytrain/nn/dense_1/bias/Adam*
_output_shapes
:@*
T0*"
_class
loc:@nn/dense_1/bias

.train/nn/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *"
_class
loc:@nn/dense_1/bias
Ќ
train/nn/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@nn/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
э
#train/nn/dense_1/bias/Adam_1/AssignAssigntrain/nn/dense_1/bias/Adam_1.train/nn/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias

!train/nn/dense_1/bias/Adam_1/readIdentitytrain/nn/dense_1/bias/Adam_1*
_output_shapes
:@*
T0*"
_class
loc:@nn/dense_1/bias
Љ
.train/nn/dense_2/kernel/Adam/Initializer/zerosConst*
valueB@*    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes

:@
Ж
train/nn/dense_2/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *$
_class
loc:@nn/dense_2/kernel*
	container *
shape
:@
ѓ
#train/nn/dense_2/kernel/Adam/AssignAssigntrain/nn/dense_2/kernel/Adam.train/nn/dense_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(

!train/nn/dense_2/kernel/Adam/readIdentitytrain/nn/dense_2/kernel/Adam*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ћ
0train/nn/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes

:@
И
train/nn/dense_2/kernel/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *$
_class
loc:@nn/dense_2/kernel*
	container 
љ
%train/nn/dense_2/kernel/Adam_1/AssignAssigntrain/nn/dense_2/kernel/Adam_10train/nn/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@

#train/nn/dense_2/kernel/Adam_1/readIdentitytrain/nn/dense_2/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@

,train/nn/dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
Њ
train/nn/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias*
	container *
shape:
ч
!train/nn/dense_2/bias/Adam/AssignAssigntrain/nn/dense_2/bias/Adam,train/nn/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:

train/nn/dense_2/bias/Adam/readIdentitytrain/nn/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@nn/dense_2/bias

.train/nn/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
Ќ
train/nn/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias*
	container *
shape:
э
#train/nn/dense_2/bias/Adam_1/AssignAssigntrain/nn/dense_2/bias/Adam_1.train/nn/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:

!train/nn/dense_2/bias/Adam_1/readIdentitytrain/nn/dense_2/bias/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@nn/dense_2/bias
]
train/Adam/learning_rateConst*
valueB
 *ІФ;*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wО?
W
train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
К
+train/Adam/update_nn/dense/kernel/ApplyAdam	ApplyAdamnn/dense/kerneltrain/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense/kernel*
use_nesterov( *
_output_shapes

:@
­
)train/Adam/update_nn/dense/bias/ApplyAdam	ApplyAdamnn/dense/biastrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1*
T0* 
_class
loc:@nn/dense/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( 
Ц
-train/Adam/update_nn/dense_1/kernel/ApplyAdam	ApplyAdamnn/dense_1/kerneltrain/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@nn/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@
Й
+train/Adam/update_nn/dense_1/bias/ApplyAdam	ApplyAdamnn/dense_1/biastrain/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*"
_class
loc:@nn/dense_1/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( 
Ц
-train/Adam/update_nn/dense_2/kernel/ApplyAdam	ApplyAdamnn/dense_2/kerneltrain/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@nn/dense_2/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
Й
+train/Adam/update_nn/dense_2/bias/ApplyAdam	ApplyAdamnn/dense_2/biastrain/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense_2/bias*
use_nesterov( *
_output_shapes
:

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
Њ
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@nn/dense/bias
Ў
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@nn/dense/bias
в

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam

initNoOp^nn/dense/bias/Assign^nn/dense/kernel/Assign^nn/dense_1/bias/Assign^nn/dense_1/kernel/Assign^nn/dense_2/bias/Assign^nn/dense_2/kernel/Assign^train/beta1_power/Assign^train/beta2_power/Assign ^train/nn/dense/bias/Adam/Assign"^train/nn/dense/bias/Adam_1/Assign"^train/nn/dense/kernel/Adam/Assign$^train/nn/dense/kernel/Adam_1/Assign"^train/nn/dense_1/bias/Adam/Assign$^train/nn/dense_1/bias/Adam_1/Assign$^train/nn/dense_1/kernel/Adam/Assign&^train/nn/dense_1/kernel/Adam_1/Assign"^train/nn/dense_2/bias/Adam/Assign$^train/nn/dense_2/bias/Adam_1/Assign$^train/nn/dense_2/kernel/Adam/Assign&^train/nn/dense_2/kernel/Adam_1/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
а
save/SaveV2/tensor_namesConst*
valueљBіBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesnn/dense/biasnn/dense/kernelnn/dense_1/biasnn/dense_1/kernelnn/dense_2/biasnn/dense_2/kerneltrain/beta1_powertrain/beta2_powertrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1*"
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
т
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueљBіBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
ў
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Є
save/AssignAssignnn/dense/biassave/RestoreV2*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
А
save/Assign_1Assignnn/dense/kernelsave/RestoreV2:1*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
Ќ
save/Assign_2Assignnn/dense_1/biassave/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
Д
save/Assign_3Assignnn/dense_1/kernelsave/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ќ
save/Assign_4Assignnn/dense_2/biassave/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
Д
save/Assign_5Assignnn/dense_2/kernelsave/RestoreV2:5*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
Ј
save/Assign_6Assigntrain/beta1_powersave/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
Ј
save/Assign_7Assigntrain/beta2_powersave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
Г
save/Assign_8Assigntrain/nn/dense/bias/Adamsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
Е
save/Assign_9Assigntrain/nn/dense/bias/Adam_1save/RestoreV2:9*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Н
save/Assign_10Assigntrain/nn/dense/kernel/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
П
save/Assign_11Assigntrain/nn/dense/kernel/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
Й
save/Assign_12Assigntrain/nn/dense_1/bias/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
Л
save/Assign_13Assigntrain/nn/dense_1/bias/Adam_1save/RestoreV2:13*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
С
save/Assign_14Assigntrain/nn/dense_1/kernel/Adamsave/RestoreV2:14*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
У
save/Assign_15Assigntrain/nn/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
Й
save/Assign_16Assigntrain/nn/dense_2/bias/Adamsave/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
Л
save/Assign_17Assigntrain/nn/dense_2/bias/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
С
save/Assign_18Assigntrain/nn/dense_2/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
У
save/Assign_19Assigntrain/nn/dense_2/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
р
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""У
trainable_variablesЋЈ
r
nn/dense/kernel:0nn/dense/kernel/Assignnn/dense/kernel/read:02+nn/dense/kernel/Initializer/random_normal:08
b
nn/dense/bias:0nn/dense/bias/Assignnn/dense/bias/read:02!nn/dense/bias/Initializer/Const:08
z
nn/dense_1/kernel:0nn/dense_1/kernel/Assignnn/dense_1/kernel/read:02-nn/dense_1/kernel/Initializer/random_normal:08
j
nn/dense_1/bias:0nn/dense_1/bias/Assignnn/dense_1/bias/read:02#nn/dense_1/bias/Initializer/Const:08
z
nn/dense_2/kernel:0nn/dense_2/kernel/Assignnn/dense_2/kernel/read:02-nn/dense_2/kernel/Initializer/random_normal:08
j
nn/dense_2/bias:0nn/dense_2/bias/Assignnn/dense_2/bias/read:02#nn/dense_2/bias/Initializer/Const:08"
train_op


train/Adam"щ
	variablesли
r
nn/dense/kernel:0nn/dense/kernel/Assignnn/dense/kernel/read:02+nn/dense/kernel/Initializer/random_normal:08
b
nn/dense/bias:0nn/dense/bias/Assignnn/dense/bias/read:02!nn/dense/bias/Initializer/Const:08
z
nn/dense_1/kernel:0nn/dense_1/kernel/Assignnn/dense_1/kernel/read:02-nn/dense_1/kernel/Initializer/random_normal:08
j
nn/dense_1/bias:0nn/dense_1/bias/Assignnn/dense_1/bias/read:02#nn/dense_1/bias/Initializer/Const:08
z
nn/dense_2/kernel:0nn/dense_2/kernel/Assignnn/dense_2/kernel/read:02-nn/dense_2/kernel/Initializer/random_normal:08
j
nn/dense_2/bias:0nn/dense_2/bias/Assignnn/dense_2/bias/read:02#nn/dense_2/bias/Initializer/Const:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0

train/nn/dense/kernel/Adam:0!train/nn/dense/kernel/Adam/Assign!train/nn/dense/kernel/Adam/read:02.train/nn/dense/kernel/Adam/Initializer/zeros:0

train/nn/dense/kernel/Adam_1:0#train/nn/dense/kernel/Adam_1/Assign#train/nn/dense/kernel/Adam_1/read:020train/nn/dense/kernel/Adam_1/Initializer/zeros:0

train/nn/dense/bias/Adam:0train/nn/dense/bias/Adam/Assigntrain/nn/dense/bias/Adam/read:02,train/nn/dense/bias/Adam/Initializer/zeros:0

train/nn/dense/bias/Adam_1:0!train/nn/dense/bias/Adam_1/Assign!train/nn/dense/bias/Adam_1/read:02.train/nn/dense/bias/Adam_1/Initializer/zeros:0

train/nn/dense_1/kernel/Adam:0#train/nn/dense_1/kernel/Adam/Assign#train/nn/dense_1/kernel/Adam/read:020train/nn/dense_1/kernel/Adam/Initializer/zeros:0
Є
 train/nn/dense_1/kernel/Adam_1:0%train/nn/dense_1/kernel/Adam_1/Assign%train/nn/dense_1/kernel/Adam_1/read:022train/nn/dense_1/kernel/Adam_1/Initializer/zeros:0

train/nn/dense_1/bias/Adam:0!train/nn/dense_1/bias/Adam/Assign!train/nn/dense_1/bias/Adam/read:02.train/nn/dense_1/bias/Adam/Initializer/zeros:0

train/nn/dense_1/bias/Adam_1:0#train/nn/dense_1/bias/Adam_1/Assign#train/nn/dense_1/bias/Adam_1/read:020train/nn/dense_1/bias/Adam_1/Initializer/zeros:0

train/nn/dense_2/kernel/Adam:0#train/nn/dense_2/kernel/Adam/Assign#train/nn/dense_2/kernel/Adam/read:020train/nn/dense_2/kernel/Adam/Initializer/zeros:0
Є
 train/nn/dense_2/kernel/Adam_1:0%train/nn/dense_2/kernel/Adam_1/Assign%train/nn/dense_2/kernel/Adam_1/read:022train/nn/dense_2/kernel/Adam_1/Initializer/zeros:0

train/nn/dense_2/bias/Adam:0!train/nn/dense_2/bias/Adam/Assign!train/nn/dense_2/bias/Adam/read:02.train/nn/dense_2/bias/Adam/Initializer/zeros:0

train/nn/dense_2/bias/Adam_1:0#train/nn/dense_2/bias/Adam_1/Assign#train/nn/dense_2/bias/Adam_1/read:020train/nn/dense_2/bias/Adam_1/Initializer/zeros:0{Џ№