       БK"	  ђчС=ОAbrain.Event:2(g ЎD│      ЋАGЋ	l┐ечС=ОA"иТ
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
ц
/nn/dense/kernel/Initializer/random_normal/shapeConst*
valueB"   @   *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
:
Ќ
.nn/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
: 
Ў
0nn/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *oЃ:*"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
: 
Щ
>nn/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/nn/dense/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*"
_class
loc:@nn/dense/kernel
з
-nn/dense/kernel/Initializer/random_normal/mulMul>nn/dense/kernel/Initializer/random_normal/RandomStandardNormal0nn/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:@*
T0*"
_class
loc:@nn/dense/kernel
▄
)nn/dense/kernel/Initializer/random_normalAdd-nn/dense/kernel/Initializer/random_normal/mul.nn/dense/kernel/Initializer/random_normal/mean*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Д
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
м
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
ј
nn/dense/bias/Initializer/ConstConst*
valueB@*═╠╠=* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
Џ
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
Й
nn/dense/bias/AssignAssignnn/dense/biasnn/dense/bias/Initializer/Const*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
nn/dense/bias/readIdentitynn/dense/bias*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
ќ
nn/dense/MatMulMatMulPlaceholder_2nn/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         @*
transpose_a( 
Ѕ
nn/dense/BiasAddBiasAddnn/dense/MatMulnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
Y
nn/dense/ReluRelunn/dense/BiasAdd*
T0*'
_output_shapes
:         @
е
1nn/dense_1/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel
Џ
0nn/dense_1/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@nn/dense_1/kernel
Ю
2nn/dense_1/kernel/Initializer/random_normal/stddevConst*
valueB
 *oЃ:*$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 
ђ
@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_1/kernel/Initializer/random_normal/shape*

seed *
T0*$
_class
loc:@nn/dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:@@
ч
/nn/dense_1/kernel/Initializer/random_normal/mulMul@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_1/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
С
+nn/dense_1/kernel/Initializer/random_normalAdd/nn/dense_1/kernel/Initializer/random_normal/mul0nn/dense_1/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Ф
nn/dense_1/kernel
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
┌
nn/dense_1/kernel/AssignAssignnn/dense_1/kernel+nn/dense_1/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel
ё
nn/dense_1/kernel/readIdentitynn/dense_1/kernel*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel
њ
!nn/dense_1/bias/Initializer/ConstConst*
valueB@*═╠╠=*"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@
Ъ
nn/dense_1/bias
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias
к
nn/dense_1/bias/AssignAssignnn/dense_1/bias!nn/dense_1/bias/Initializer/Const*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
z
nn/dense_1/bias/readIdentitynn/dense_1/bias*
_output_shapes
:@*
T0*"
_class
loc:@nn/dense_1/bias
џ
nn/dense_1/MatMulMatMulnn/dense/Relunn/dense_1/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
Ј
nn/dense_1/BiasAddBiasAddnn/dense_1/MatMulnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
]
nn/dense_1/ReluRelunn/dense_1/BiasAdd*
T0*'
_output_shapes
:         @
е
1nn/dense_2/kernel/Initializer/random_normal/shapeConst*
valueB"@      *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
:
Џ
0nn/dense_2/kernel/Initializer/random_normal/meanConst*
valueB
 *    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
: 
Ю
2nn/dense_2/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *oЃ:*$
_class
loc:@nn/dense_2/kernel
ђ
@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_2/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:@*

seed *
T0*$
_class
loc:@nn/dense_2/kernel*
seed2 
ч
/nn/dense_2/kernel/Initializer/random_normal/mulMul@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_2/kernel/Initializer/random_normal/stddev*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
С
+nn/dense_2/kernel/Initializer/random_normalAdd/nn/dense_2/kernel/Initializer/random_normal/mul0nn/dense_2/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ф
nn/dense_2/kernel
VariableV2*$
_class
loc:@nn/dense_2/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
┌
nn/dense_2/kernel/AssignAssignnn/dense_2/kernel+nn/dense_2/kernel/Initializer/random_normal*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
ё
nn/dense_2/kernel/readIdentitynn/dense_2/kernel*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
њ
!nn/dense_2/bias/Initializer/ConstConst*
valueB*═╠╠=*"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
Ъ
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
к
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
ю
nn/dense_2/MatMulMatMulnn/dense_1/Relunn/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
Ј
nn/dense_2/BiasAddBiasAddnn/dense_2/MatMulnn/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
[

nn/SoftmaxSoftmaxnn/dense_2/BiasAdd*
T0*'
_output_shapes
:         
y
.loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
С
Lloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnn/dense_2/BiasAddPlaceholder*6
_output_shapes$
":         :         *
Tlabels0*
T0
џ
loss/mulMulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsPlaceholder_1*#
_output_shapes
:         *
T0
T

loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
e
	loss/MeanMeanloss/mul
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
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
 *  ђ?*
dtype0*
_output_shapes
: 
Ђ
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
е
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
╣
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
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
и
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
╗
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Б
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
ћ
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Е
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:         
»
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
Н
3train/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/mul_grad/Shape%train/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ї
!train/gradients/loss/mul_grad/MulMul&train/gradients/loss/Mean_grad/truedivPlaceholder_1*
T0*#
_output_shapes
:         
└
!train/gradients/loss/mul_grad/SumSum!train/gradients/loss/mul_grad/Mul3train/gradients/loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┤
%train/gradients/loss/mul_grad/ReshapeReshape!train/gradients/loss/mul_grad/Sum#train/gradients/loss/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
╬
#train/gradients/loss/mul_grad/Mul_1MulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&train/gradients/loss/Mean_grad/truediv*#
_output_shapes
:         *
T0
к
#train/gradients/loss/mul_grad/Sum_1Sum#train/gradients/loss/mul_grad/Mul_15train/gradients/loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
║
'train/gradients/loss/mul_grad/Reshape_1Reshape#train/gradients/loss/mul_grad/Sum_1%train/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
ѕ
.train/gradients/loss/mul_grad/tuple/group_depsNoOp&^train/gradients/loss/mul_grad/Reshape(^train/gradients/loss/mul_grad/Reshape_1
ѓ
6train/gradients/loss/mul_grad/tuple/control_dependencyIdentity%train/gradients/loss/mul_grad/Reshape/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/mul_grad/Reshape*#
_output_shapes
:         
ѕ
8train/gradients/loss/mul_grad/tuple/control_dependency_1Identity'train/gradients/loss/mul_grad/Reshape_1/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/mul_grad/Reshape_1*#
_output_shapes
:         
Е
train/gradients/zeros_like	ZerosLikeNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         *
T0
й
qtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:         *┤
messageеЦCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
╗
ptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
Р
ltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims6train/gradients/loss/mul_grad/tuple/control_dependencyptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:         *

Tdim0*
T0
 
etrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsqtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:         
т
3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:
я
8train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_depsNoOpf^train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul4^train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
џ
@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*x
_classn
ljloc:@train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*'
_output_shapes
:         
Ф
Btrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ж
-train/gradients/nn/dense_2/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencynn/dense_2/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
█
/train/gradients/nn/dense_2/MatMul_grad/MatMul_1MatMulnn/dense_1/Relu@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
А
7train/gradients/nn/dense_2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_2/MatMul_grad/MatMul0^train/gradients/nn/dense_2/MatMul_grad/MatMul_1
е
?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_2/MatMul_grad/MatMul8^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*@
_class6
42loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul
Ц
Atrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_2/MatMul_grad/MatMul_18^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
й
-train/gradients/nn/dense_1/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencynn/dense_1/Relu*
T0*'
_output_shapes
:         @
Г
3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/nn/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
д
8train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad.^train/gradients/nn/dense_1/Relu_grad/ReluGrad
ф
@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/Relu_grad/ReluGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:         @
Ф
Btrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
ж
-train/gradients/nn/dense_1/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencynn/dense_1/kernel/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b(*
T0
┘
/train/gradients/nn/dense_1/MatMul_grad/MatMul_1MatMulnn/dense/Relu@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
А
7train/gradients/nn/dense_1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_1/MatMul_grad/MatMul0^train/gradients/nn/dense_1/MatMul_grad/MatMul_1
е
?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/MatMul_grad/MatMul8^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul*'
_output_shapes
:         @
Ц
Atrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_1/MatMul_grad/MatMul_18^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
╣
+train/gradients/nn/dense/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencynn/dense/Relu*'
_output_shapes
:         @*
T0
Е
1train/gradients/nn/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+train/gradients/nn/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
а
6train/gradients/nn/dense/BiasAdd_grad/tuple/group_depsNoOp2^train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad,^train/gradients/nn/dense/Relu_grad/ReluGrad
б
>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/Relu_grad/ReluGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*>
_class4
20loc:@train/gradients/nn/dense/Relu_grad/ReluGrad
Б
@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1Identity1train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
с
+train/gradients/nn/dense/MatMul_grad/MatMulMatMul>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencynn/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
Н
-train/gradients/nn/dense/MatMul_grad/MatMul_1MatMulPlaceholder_2>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Џ
5train/gradients/nn/dense/MatMul_grad/tuple/group_depsNoOp,^train/gradients/nn/dense/MatMul_grad/MatMul.^train/gradients/nn/dense/MatMul_grad/MatMul_1
а
=train/gradients/nn/dense/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/MatMul_grad/MatMul6^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*>
_class4
20loc:@train/gradients/nn/dense/MatMul_grad/MatMul
Ю
?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/nn/dense/MatMul_grad/MatMul_16^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense/MatMul_grad/MatMul_1*
_output_shapes

:@
є
train/beta1_power/initial_valueConst*
valueB
 *fff?* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
: 
Ќ
train/beta1_power
VariableV2* 
_class
loc:@nn/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
┬
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
x
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0* 
_class
loc:@nn/dense/bias
є
train/beta2_power/initial_valueConst*
valueB
 *wЙ?* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
: 
Ќ
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
┬
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@nn/dense/bias
x
train/beta2_power/readIdentitytrain/beta2_power*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
Ц
,train/nn/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*"
_class
loc:@nn/dense/kernel*
valueB@*    
▓
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
в
!train/nn/dense/kernel/Adam/AssignAssigntrain/nn/dense/kernel/Adam,train/nn/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
ћ
train/nn/dense/kernel/Adam/readIdentitytrain/nn/dense/kernel/Adam*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Д
.train/nn/dense/kernel/Adam_1/Initializer/zerosConst*"
_class
loc:@nn/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
┤
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
ы
#train/nn/dense/kernel/Adam_1/AssignAssigntrain/nn/dense/kernel/Adam_1.train/nn/dense/kernel/Adam_1/Initializer/zeros*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
ў
!train/nn/dense/kernel/Adam_1/readIdentitytrain/nn/dense/kernel/Adam_1*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Ў
*train/nn/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@nn/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
д
train/nn/dense/bias/Adam
VariableV2*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
▀
train/nn/dense/bias/Adam/AssignAssigntrain/nn/dense/bias/Adam*train/nn/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
і
train/nn/dense/bias/Adam/readIdentitytrain/nn/dense/bias/Adam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
Џ
,train/nn/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@nn/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
е
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
т
!train/nn/dense/bias/Adam_1/AssignAssigntrain/nn/dense/bias/Adam_1,train/nn/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
ј
train/nn/dense/bias/Adam_1/readIdentitytrain/nn/dense/bias/Adam_1*
_output_shapes
:@*
T0* 
_class
loc:@nn/dense/bias
х
>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   
Ъ
4train/nn/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
.train/nn/dense_1/kernel/Adam/Initializer/zerosFill>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor4train/nn/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@nn/dense_1/kernel*

index_type0*
_output_shapes

:@@
Х
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
з
#train/nn/dense_1/kernel/Adam/AssignAssigntrain/nn/dense_1/kernel/Adam.train/nn/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
џ
!train/nn/dense_1/kernel/Adam/readIdentitytrain/nn/dense_1/kernel/Adam*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
и
@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   
А
6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
0train/nn/dense_1/kernel/Adam_1/Initializer/zerosFill@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*$
_class
loc:@nn/dense_1/kernel*

index_type0*
_output_shapes

:@@
И
train/nn/dense_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@nn/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
щ
%train/nn/dense_1/kernel/Adam_1/AssignAssigntrain/nn/dense_1/kernel/Adam_10train/nn/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
ъ
#train/nn/dense_1/kernel/Adam_1/readIdentitytrain/nn/dense_1/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Ю
,train/nn/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@nn/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
ф
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
у
!train/nn/dense_1/bias/Adam/AssignAssigntrain/nn/dense_1/bias/Adam,train/nn/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
љ
train/nn/dense_1/bias/Adam/readIdentitytrain/nn/dense_1/bias/Adam*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Ъ
.train/nn/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@nn/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
г
train/nn/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container *
shape:@
ь
#train/nn/dense_1/bias/Adam_1/AssignAssigntrain/nn/dense_1/bias/Adam_1.train/nn/dense_1/bias/Adam_1/Initializer/zeros*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
ћ
!train/nn/dense_1/bias/Adam_1/readIdentitytrain/nn/dense_1/bias/Adam_1*
_output_shapes
:@*
T0*"
_class
loc:@nn/dense_1/bias
Е
.train/nn/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*$
_class
loc:@nn/dense_2/kernel*
valueB@*    
Х
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
з
#train/nn/dense_2/kernel/Adam/AssignAssigntrain/nn/dense_2/kernel/Adam.train/nn/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
џ
!train/nn/dense_2/kernel/Adam/readIdentitytrain/nn/dense_2/kernel/Adam*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ф
0train/nn/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@nn/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
И
train/nn/dense_2/kernel/Adam_1
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
щ
%train/nn/dense_2/kernel/Adam_1/AssignAssigntrain/nn/dense_2/kernel/Adam_10train/nn/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
ъ
#train/nn/dense_2/kernel/Adam_1/readIdentitytrain/nn/dense_2/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ю
,train/nn/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@nn/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
ф
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
у
!train/nn/dense_2/bias/Adam/AssignAssigntrain/nn/dense_2/bias/Adam,train/nn/dense_2/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
љ
train/nn/dense_2/bias/Adam/readIdentitytrain/nn/dense_2/bias/Adam*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
Ъ
.train/nn/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@nn/dense_2/bias*
valueB*    
г
train/nn/dense_2/bias/Adam_1
VariableV2*"
_class
loc:@nn/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ь
#train/nn/dense_2/bias/Adam_1/AssignAssigntrain/nn/dense_2/bias/Adam_1.train/nn/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
ћ
!train/nn/dense_2/bias/Adam_1/readIdentitytrain/nn/dense_2/bias/Adam_1*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *дЏ─;*
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
train/Adam/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
║
+train/Adam/update_nn/dense/kernel/ApplyAdam	ApplyAdamnn/dense/kerneltrain/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@nn/dense/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
Г
)train/Adam/update_nn/dense/bias/ApplyAdam	ApplyAdamnn/dense/biastrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
use_nesterov( *
_output_shapes
:@
к
-train/Adam/update_nn/dense_1/kernel/ApplyAdam	ApplyAdamnn/dense_1/kerneltrain/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@nn/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@
╣
+train/Adam/update_nn/dense_1/bias/ApplyAdam	ApplyAdamnn/dense_1/biastrain/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense_1/bias*
use_nesterov( *
_output_shapes
:@
к
-train/Adam/update_nn/dense_2/kernel/ApplyAdam	ApplyAdamnn/dense_2/kerneltrain/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*$
_class
loc:@nn/dense_2/kernel
╣
+train/Adam/update_nn/dense_2/bias/ApplyAdam	ApplyAdamnn/dense_2/biastrain/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense_2/bias*
use_nesterov( *
_output_shapes
:
ў
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
ф
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
џ
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
«
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@nn/dense/bias
м

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam
џ
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
л
save/SaveV2/tensor_namesConst*Ѓ
valueщBШBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
І
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
в
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
Р
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ѓ
valueщBШBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1
Ю
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
■
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
ц
save/AssignAssignnn/dense/biassave/RestoreV2*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
░
save/Assign_1Assignnn/dense/kernelsave/RestoreV2:1*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
г
save/Assign_2Assignnn/dense_1/biassave/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
┤
save/Assign_3Assignnn/dense_1/kernelsave/RestoreV2:3*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel
г
save/Assign_4Assignnn/dense_2/biassave/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
┤
save/Assign_5Assignnn/dense_2/kernelsave/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
е
save/Assign_6Assigntrain/beta1_powersave/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
е
save/Assign_7Assigntrain/beta2_powersave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
│
save/Assign_8Assigntrain/nn/dense/bias/Adamsave/RestoreV2:8*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save/Assign_9Assigntrain/nn/dense/bias/Adam_1save/RestoreV2:9*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
й
save/Assign_10Assigntrain/nn/dense/kernel/Adamsave/RestoreV2:10*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┐
save/Assign_11Assigntrain/nn/dense/kernel/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
╣
save/Assign_12Assigntrain/nn/dense_1/bias/Adamsave/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
╗
save/Assign_13Assigntrain/nn/dense_1/bias/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
┴
save/Assign_14Assigntrain/nn/dense_1/kernel/Adamsave/RestoreV2:14*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
├
save/Assign_15Assigntrain/nn/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
╣
save/Assign_16Assigntrain/nn/dense_2/bias/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
╗
save/Assign_17Assigntrain/nn/dense_2/bias/Adam_1save/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
┴
save/Assign_18Assigntrain/nn/dense_2/kernel/Adamsave/RestoreV2:18*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
├
save/Assign_19Assigntrain/nn/dense_2/kernel/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
Я
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"у|И║H╬      ┌ф╬	▄љЕчС=ОAJ╗ю
»і
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ь
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
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

2	љ
Ї
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
2	љ
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
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ё
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Њ
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
ї
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
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.1-0-g6612da8951'иТ
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_1Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
ц
/nn/dense/kernel/Initializer/random_normal/shapeConst*"
_class
loc:@nn/dense/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
Ќ
.nn/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *"
_class
loc:@nn/dense/kernel*
valueB
 *    
Ў
0nn/dense/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *"
_class
loc:@nn/dense/kernel*
valueB
 *oЃ:
Щ
>nn/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/nn/dense/kernel/Initializer/random_normal/shape*
T0*"
_class
loc:@nn/dense/kernel*
seed2 *
dtype0*
_output_shapes

:@*

seed 
з
-nn/dense/kernel/Initializer/random_normal/mulMul>nn/dense/kernel/Initializer/random_normal/RandomStandardNormal0nn/dense/kernel/Initializer/random_normal/stddev*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
▄
)nn/dense/kernel/Initializer/random_normalAdd-nn/dense/kernel/Initializer/random_normal/mul.nn/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:@*
T0*"
_class
loc:@nn/dense/kernel
Д
nn/dense/kernel
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
м
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
ј
nn/dense/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:@* 
_class
loc:@nn/dense/bias*
valueB@*═╠╠=
Џ
nn/dense/bias
VariableV2* 
_class
loc:@nn/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Й
nn/dense/bias/AssignAssignnn/dense/biasnn/dense/bias/Initializer/Const*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
nn/dense/bias/readIdentitynn/dense/bias*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
ќ
nn/dense/MatMulMatMulPlaceholder_2nn/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         @*
transpose_b( 
Ѕ
nn/dense/BiasAddBiasAddnn/dense/MatMulnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
Y
nn/dense/ReluRelunn/dense/BiasAdd*
T0*'
_output_shapes
:         @
е
1nn/dense_1/kernel/Initializer/random_normal/shapeConst*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Џ
0nn/dense_1/kernel/Initializer/random_normal/meanConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ю
2nn/dense_1/kernel/Initializer/random_normal/stddevConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
ђ
@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_1/kernel/Initializer/random_normal/shape*
T0*$
_class
loc:@nn/dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:@@*

seed 
ч
/nn/dense_1/kernel/Initializer/random_normal/mulMul@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_1/kernel/Initializer/random_normal/stddev*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel
С
+nn/dense_1/kernel/Initializer/random_normalAdd/nn/dense_1/kernel/Initializer/random_normal/mul0nn/dense_1/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Ф
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
┌
nn/dense_1/kernel/AssignAssignnn/dense_1/kernel+nn/dense_1/kernel/Initializer/random_normal*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
ё
nn/dense_1/kernel/readIdentitynn/dense_1/kernel*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
њ
!nn/dense_1/bias/Initializer/ConstConst*"
_class
loc:@nn/dense_1/bias*
valueB@*═╠╠=*
dtype0*
_output_shapes
:@
Ъ
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
к
nn/dense_1/bias/AssignAssignnn/dense_1/bias!nn/dense_1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
z
nn/dense_1/bias/readIdentitynn/dense_1/bias*
_output_shapes
:@*
T0*"
_class
loc:@nn/dense_1/bias
џ
nn/dense_1/MatMulMatMulnn/dense/Relunn/dense_1/kernel/read*
transpose_a( *'
_output_shapes
:         @*
transpose_b( *
T0
Ј
nn/dense_1/BiasAddBiasAddnn/dense_1/MatMulnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
]
nn/dense_1/ReluRelunn/dense_1/BiasAdd*
T0*'
_output_shapes
:         @
е
1nn/dense_2/kernel/Initializer/random_normal/shapeConst*$
_class
loc:@nn/dense_2/kernel*
valueB"@      *
dtype0*
_output_shapes
:
Џ
0nn/dense_2/kernel/Initializer/random_normal/meanConst*$
_class
loc:@nn/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ю
2nn/dense_2/kernel/Initializer/random_normal/stddevConst*$
_class
loc:@nn/dense_2/kernel*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
ђ
@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_2/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*$
_class
loc:@nn/dense_2/kernel
ч
/nn/dense_2/kernel/Initializer/random_normal/mulMul@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_2/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
С
+nn/dense_2/kernel/Initializer/random_normalAdd/nn/dense_2/kernel/Initializer/random_normal/mul0nn/dense_2/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ф
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
┌
nn/dense_2/kernel/AssignAssignnn/dense_2/kernel+nn/dense_2/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
ё
nn/dense_2/kernel/readIdentitynn/dense_2/kernel*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
њ
!nn/dense_2/bias/Initializer/ConstConst*"
_class
loc:@nn/dense_2/bias*
valueB*═╠╠=*
dtype0*
_output_shapes
:
Ъ
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
к
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
ю
nn/dense_2/MatMulMatMulnn/dense_1/Relunn/dense_2/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
Ј
nn/dense_2/BiasAddBiasAddnn/dense_2/MatMulnn/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
[

nn/SoftmaxSoftmaxnn/dense_2/BiasAdd*
T0*'
_output_shapes
:         
y
.loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
С
Lloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnn/dense_2/BiasAddPlaceholder*
T0*
Tlabels0*6
_output_shapes$
":         :         
џ
loss/mulMulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsPlaceholder_1*
T0*#
_output_shapes
:         
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/mul
loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
^
train/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ђ
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
е
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/mul*
_output_shapes
:*
T0*
out_type0
╣
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
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
и
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
╗
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Б
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
А
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
ћ
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
Е
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*#
_output_shapes
:         *
T0
»
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
Н
3train/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/mul_grad/Shape%train/gradients/loss/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ї
!train/gradients/loss/mul_grad/MulMul&train/gradients/loss/Mean_grad/truedivPlaceholder_1*#
_output_shapes
:         *
T0
└
!train/gradients/loss/mul_grad/SumSum!train/gradients/loss/mul_grad/Mul3train/gradients/loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┤
%train/gradients/loss/mul_grad/ReshapeReshape!train/gradients/loss/mul_grad/Sum#train/gradients/loss/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
╬
#train/gradients/loss/mul_grad/Mul_1MulLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&train/gradients/loss/Mean_grad/truediv*
T0*#
_output_shapes
:         
к
#train/gradients/loss/mul_grad/Sum_1Sum#train/gradients/loss/mul_grad/Mul_15train/gradients/loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
║
'train/gradients/loss/mul_grad/Reshape_1Reshape#train/gradients/loss/mul_grad/Sum_1%train/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
ѕ
.train/gradients/loss/mul_grad/tuple/group_depsNoOp&^train/gradients/loss/mul_grad/Reshape(^train/gradients/loss/mul_grad/Reshape_1
ѓ
6train/gradients/loss/mul_grad/tuple/control_dependencyIdentity%train/gradients/loss/mul_grad/Reshape/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/mul_grad/Reshape*#
_output_shapes
:         
ѕ
8train/gradients/loss/mul_grad/tuple/control_dependency_1Identity'train/gradients/loss/mul_grad/Reshape_1/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/mul_grad/Reshape_1*#
_output_shapes
:         
Е
train/gradients/zeros_like	ZerosLikeNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:         
й
qtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*┤
messageеЦCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:         
╗
ptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
Р
ltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims6train/gradients/loss/mul_grad/tuple/control_dependencyptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
 
etrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsqtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*'
_output_shapes
:         *
T0
т
3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:
я
8train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_depsNoOpf^train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul4^train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
џ
@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*x
_classn
ljloc:@train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*'
_output_shapes
:         
Ф
Btrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
ж
-train/gradients/nn/dense_2/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencynn/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:         @
█
/train/gradients/nn/dense_2/MatMul_grad/MatMul_1MatMulnn/dense_1/Relu@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@*
transpose_b( *
T0
А
7train/gradients/nn/dense_2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_2/MatMul_grad/MatMul0^train/gradients/nn/dense_2/MatMul_grad/MatMul_1
е
?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_2/MatMul_grad/MatMul8^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul*'
_output_shapes
:         @
Ц
Atrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_2/MatMul_grad/MatMul_18^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*B
_class8
64loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul_1
й
-train/gradients/nn/dense_1/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencynn/dense_1/Relu*
T0*'
_output_shapes
:         @
Г
3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/nn/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
д
8train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad.^train/gradients/nn/dense_1/Relu_grad/ReluGrad
ф
@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/Relu_grad/ReluGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:         @
Ф
Btrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
ж
-train/gradients/nn/dense_1/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencynn/dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         @*
transpose_b(
┘
/train/gradients/nn/dense_1/MatMul_grad/MatMul_1MatMulnn/dense/Relu@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@@*
transpose_b( *
T0
А
7train/gradients/nn/dense_1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_1/MatMul_grad/MatMul0^train/gradients/nn/dense_1/MatMul_grad/MatMul_1
е
?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/MatMul_grad/MatMul8^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul
Ц
Atrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_1/MatMul_grad/MatMul_18^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
╣
+train/gradients/nn/dense/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencynn/dense/Relu*
T0*'
_output_shapes
:         @
Е
1train/gradients/nn/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+train/gradients/nn/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
а
6train/gradients/nn/dense/BiasAdd_grad/tuple/group_depsNoOp2^train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad,^train/gradients/nn/dense/Relu_grad/ReluGrad
б
>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/Relu_grad/ReluGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/Relu_grad/ReluGrad*'
_output_shapes
:         @
Б
@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1Identity1train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
с
+train/gradients/nn/dense/MatMul_grad/MatMulMatMul>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencynn/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:         
Н
-train/gradients/nn/dense/MatMul_grad/MatMul_1MatMulPlaceholder_2>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@*
transpose_b( 
Џ
5train/gradients/nn/dense/MatMul_grad/tuple/group_depsNoOp,^train/gradients/nn/dense/MatMul_grad/MatMul.^train/gradients/nn/dense/MatMul_grad/MatMul_1
а
=train/gradients/nn/dense/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/MatMul_grad/MatMul6^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/MatMul_grad/MatMul*'
_output_shapes
:         
Ю
?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/nn/dense/MatMul_grad/MatMul_16^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense/MatMul_grad/MatMul_1*
_output_shapes

:@
є
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@nn/dense/bias*
valueB
 *fff?
Ќ
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
┬
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
є
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@nn/dense/bias*
valueB
 *wЙ?
Ќ
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@nn/dense/bias
┬
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
x
train/beta2_power/readIdentitytrain/beta2_power*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
Ц
,train/nn/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*
valueB@*    *"
_class
loc:@nn/dense/kernel
▓
train/nn/dense/kernel/Adam
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
!train/nn/dense/kernel/Adam/AssignAssigntrain/nn/dense/kernel/Adam,train/nn/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
ћ
train/nn/dense/kernel/Adam/readIdentitytrain/nn/dense/kernel/Adam*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Д
.train/nn/dense/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes

:@
┤
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
ы
#train/nn/dense/kernel/Adam_1/AssignAssigntrain/nn/dense/kernel/Adam_1.train/nn/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
ў
!train/nn/dense/kernel/Adam_1/readIdentitytrain/nn/dense/kernel/Adam_1*
_output_shapes

:@*
T0*"
_class
loc:@nn/dense/kernel
Ў
*train/nn/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    * 
_class
loc:@nn/dense/bias
д
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
▀
train/nn/dense/bias/Adam/AssignAssigntrain/nn/dense/bias/Adam*train/nn/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias
і
train/nn/dense/bias/Adam/readIdentitytrain/nn/dense/bias/Adam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
Џ
,train/nn/dense/bias/Adam_1/Initializer/zerosConst*
valueB@*    * 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
е
train/nn/dense/bias/Adam_1
VariableV2*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
т
!train/nn/dense/bias/Adam_1/AssignAssigntrain/nn/dense/bias/Adam_1,train/nn/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias
ј
train/nn/dense/bias/Adam_1/readIdentitytrain/nn/dense/bias/Adam_1*
_output_shapes
:@*
T0* 
_class
loc:@nn/dense/bias
х
>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:
Ъ
4train/nn/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@nn/dense_1/kernel
Ї
.train/nn/dense_1/kernel/Adam/Initializer/zerosFill>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor4train/nn/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Х
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
з
#train/nn/dense_1/kernel/Adam/AssignAssigntrain/nn/dense_1/kernel/Adam.train/nn/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
џ
!train/nn/dense_1/kernel/Adam/readIdentitytrain/nn/dense_1/kernel/Adam*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
и
@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:
А
6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@nn/dense_1/kernel
Њ
0train/nn/dense_1/kernel/Adam_1/Initializer/zerosFill@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
И
train/nn/dense_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@nn/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
щ
%train/nn/dense_1/kernel/Adam_1/AssignAssigntrain/nn/dense_1/kernel/Adam_10train/nn/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
ъ
#train/nn/dense_1/kernel/Adam_1/readIdentitytrain/nn/dense_1/kernel/Adam_1*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel
Ю
,train/nn/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@
ф
train/nn/dense_1/bias/Adam
VariableV2*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
у
!train/nn/dense_1/bias/Adam/AssignAssigntrain/nn/dense_1/bias/Adam,train/nn/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
љ
train/nn/dense_1/bias/Adam/readIdentitytrain/nn/dense_1/bias/Adam*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Ъ
.train/nn/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@
г
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
ь
#train/nn/dense_1/bias/Adam_1/AssignAssigntrain/nn/dense_1/bias/Adam_1.train/nn/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
ћ
!train/nn/dense_1/bias/Adam_1/readIdentitytrain/nn/dense_1/bias/Adam_1*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Е
.train/nn/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*
valueB@*    *$
_class
loc:@nn/dense_2/kernel
Х
train/nn/dense_2/kernel/Adam
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
з
#train/nn/dense_2/kernel/Adam/AssignAssigntrain/nn/dense_2/kernel/Adam.train/nn/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
џ
!train/nn/dense_2/kernel/Adam/readIdentitytrain/nn/dense_2/kernel/Adam*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Ф
0train/nn/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*
valueB@*    *$
_class
loc:@nn/dense_2/kernel
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
щ
%train/nn/dense_2/kernel/Adam_1/AssignAssigntrain/nn/dense_2/kernel/Adam_10train/nn/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
ъ
#train/nn/dense_2/kernel/Adam_1/readIdentitytrain/nn/dense_2/kernel/Adam_1*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
Ю
,train/nn/dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
ф
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
у
!train/nn/dense_2/bias/Adam/AssignAssigntrain/nn/dense_2/bias/Adam,train/nn/dense_2/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
љ
train/nn/dense_2/bias/Adam/readIdentitytrain/nn/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@nn/dense_2/bias
Ъ
.train/nn/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *"
_class
loc:@nn/dense_2/bias
г
train/nn/dense_2/bias/Adam_1
VariableV2*"
_class
loc:@nn/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ь
#train/nn/dense_2/bias/Adam_1/AssignAssigntrain/nn/dense_2/bias/Adam_1.train/nn/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
ћ
!train/nn/dense_2/bias/Adam_1/readIdentitytrain/nn/dense_2/bias/Adam_1*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *дЏ─;*
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
 *wЙ?
W
train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
║
+train/Adam/update_nn/dense/kernel/ApplyAdam	ApplyAdamnn/dense/kerneltrain/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@nn/dense/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
Г
)train/Adam/update_nn/dense/bias/ApplyAdam	ApplyAdamnn/dense/biastrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1*
T0* 
_class
loc:@nn/dense/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( 
к
-train/Adam/update_nn/dense_1/kernel/ApplyAdam	ApplyAdamnn/dense_1/kerneltrain/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@nn/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
╣
+train/Adam/update_nn/dense_1/bias/ApplyAdam	ApplyAdamnn/dense_1/biastrain/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*"
_class
loc:@nn/dense_1/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( 
к
-train/Adam/update_nn/dense_2/kernel/ApplyAdam	ApplyAdamnn/dense_2/kerneltrain/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@nn/dense_2/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
╣
+train/Adam/update_nn/dense_2/bias/ApplyAdam	ApplyAdamnn/dense_2/biastrain/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense_2/bias*
use_nesterov( *
_output_shapes
:
ў
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
ф
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
џ
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
«
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
м

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam
џ
initNoOp^nn/dense/bias/Assign^nn/dense/kernel/Assign^nn/dense_1/bias/Assign^nn/dense_1/kernel/Assign^nn/dense_2/bias/Assign^nn/dense_2/kernel/Assign^train/beta1_power/Assign^train/beta2_power/Assign ^train/nn/dense/bias/Adam/Assign"^train/nn/dense/bias/Adam_1/Assign"^train/nn/dense/kernel/Adam/Assign$^train/nn/dense/kernel/Adam_1/Assign"^train/nn/dense_1/bias/Adam/Assign$^train/nn/dense_1/bias/Adam_1/Assign$^train/nn/dense_1/kernel/Adam/Assign&^train/nn/dense_1/kernel/Adam_1/Assign"^train/nn/dense_2/bias/Adam/Assign$^train/nn/dense_2/bias/Adam_1/Assign$^train/nn/dense_2/kernel/Adam/Assign&^train/nn/dense_2/kernel/Adam_1/Assign
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
л
save/SaveV2/tensor_namesConst*Ѓ
valueщBШBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
І
save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
в
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
Р
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ѓ
valueщBШBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
Ю
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
■
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
ц
save/AssignAssignnn/dense/biassave/RestoreV2*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
░
save/Assign_1Assignnn/dense/kernelsave/RestoreV2:1*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
г
save/Assign_2Assignnn/dense_1/biassave/RestoreV2:2*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
┤
save/Assign_3Assignnn/dense_1/kernelsave/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
г
save/Assign_4Assignnn/dense_2/biassave/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
┤
save/Assign_5Assignnn/dense_2/kernelsave/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
е
save/Assign_6Assigntrain/beta1_powersave/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
е
save/Assign_7Assigntrain/beta2_powersave/RestoreV2:7*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
│
save/Assign_8Assigntrain/nn/dense/bias/Adamsave/RestoreV2:8*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save/Assign_9Assigntrain/nn/dense/bias/Adam_1save/RestoreV2:9*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias
й
save/Assign_10Assigntrain/nn/dense/kernel/Adamsave/RestoreV2:10*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┐
save/Assign_11Assigntrain/nn/dense/kernel/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
╣
save/Assign_12Assigntrain/nn/dense_1/bias/Adamsave/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
╗
save/Assign_13Assigntrain/nn/dense_1/bias/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
┴
save/Assign_14Assigntrain/nn/dense_1/kernel/Adamsave/RestoreV2:14*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
├
save/Assign_15Assigntrain/nn/dense_1/kernel/Adam_1save/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
╣
save/Assign_16Assigntrain/nn/dense_2/bias/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
╗
save/Assign_17Assigntrain/nn/dense_2/bias/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
┴
save/Assign_18Assigntrain/nn/dense_2/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
├
save/Assign_19Assigntrain/nn/dense_2/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Я
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""├
trainable_variablesФе
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


train/Adam"ж
	variables█п
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
ћ
train/nn/dense/kernel/Adam:0!train/nn/dense/kernel/Adam/Assign!train/nn/dense/kernel/Adam/read:02.train/nn/dense/kernel/Adam/Initializer/zeros:0
ю
train/nn/dense/kernel/Adam_1:0#train/nn/dense/kernel/Adam_1/Assign#train/nn/dense/kernel/Adam_1/read:020train/nn/dense/kernel/Adam_1/Initializer/zeros:0
ї
train/nn/dense/bias/Adam:0train/nn/dense/bias/Adam/Assigntrain/nn/dense/bias/Adam/read:02,train/nn/dense/bias/Adam/Initializer/zeros:0
ћ
train/nn/dense/bias/Adam_1:0!train/nn/dense/bias/Adam_1/Assign!train/nn/dense/bias/Adam_1/read:02.train/nn/dense/bias/Adam_1/Initializer/zeros:0
ю
train/nn/dense_1/kernel/Adam:0#train/nn/dense_1/kernel/Adam/Assign#train/nn/dense_1/kernel/Adam/read:020train/nn/dense_1/kernel/Adam/Initializer/zeros:0
ц
 train/nn/dense_1/kernel/Adam_1:0%train/nn/dense_1/kernel/Adam_1/Assign%train/nn/dense_1/kernel/Adam_1/read:022train/nn/dense_1/kernel/Adam_1/Initializer/zeros:0
ћ
train/nn/dense_1/bias/Adam:0!train/nn/dense_1/bias/Adam/Assign!train/nn/dense_1/bias/Adam/read:02.train/nn/dense_1/bias/Adam/Initializer/zeros:0
ю
train/nn/dense_1/bias/Adam_1:0#train/nn/dense_1/bias/Adam_1/Assign#train/nn/dense_1/bias/Adam_1/read:020train/nn/dense_1/bias/Adam_1/Initializer/zeros:0
ю
train/nn/dense_2/kernel/Adam:0#train/nn/dense_2/kernel/Adam/Assign#train/nn/dense_2/kernel/Adam/read:020train/nn/dense_2/kernel/Adam/Initializer/zeros:0
ц
 train/nn/dense_2/kernel/Adam_1:0%train/nn/dense_2/kernel/Adam_1/Assign%train/nn/dense_2/kernel/Adam_1/read:022train/nn/dense_2/kernel/Adam_1/Initializer/zeros:0
ћ
train/nn/dense_2/bias/Adam:0!train/nn/dense_2/bias/Adam/Assign!train/nn/dense_2/bias/Adam/read:02.train/nn/dense_2/bias/Adam/Initializer/zeros:0
ю
train/nn/dense_2/bias/Adam_1:0#train/nn/dense_2/bias/Adam_1/Assign#train/nn/dense_2/bias/Adam_1/read:020train/nn/dense_2/bias/Adam_1/Initializer/zeros:0ЪK