       гK"	  Ах=╫Abrain.Event:2Щ@гD│      ХбGХ	,╞жх=╫A"╖ц
f
PlaceholderPlaceholder*
shape:         *
dtype0*#
_output_shapes
:         
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
д
/nn/dense/kernel/Initializer/random_normal/shapeConst*
valueB"   @   *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
:
Ч
.nn/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
: 
Щ
0nn/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *oГ:*"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes
: 
·
>nn/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/nn/dense/kernel/Initializer/random_normal/shape*
T0*"
_class
loc:@nn/dense/kernel*
seed2 *
dtype0*
_output_shapes

:@*

seed 
є
-nn/dense/kernel/Initializer/random_normal/mulMul>nn/dense/kernel/Initializer/random_normal/RandomStandardNormal0nn/dense/kernel/Initializer/random_normal/stddev*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
▄
)nn/dense/kernel/Initializer/random_normalAdd-nn/dense/kernel/Initializer/random_normal/mul.nn/dense/kernel/Initializer/random_normal/mean*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
з
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
╥
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
О
nn/dense/bias/Initializer/ConstConst*
valueB@*═╠╠=* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
Ы
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
╛
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
Ц
nn/dense/MatMulMatMulPlaceholder_2nn/dense/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
Й
nn/dense/BiasAddBiasAddnn/dense/MatMulnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
Y
nn/dense/ReluRelunn/dense/BiasAdd*'
_output_shapes
:         @*
T0
и
1nn/dense_1/kernel/Initializer/random_normal/shapeConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:
Ы
0nn/dense_1/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@nn/dense_1/kernel
Э
2nn/dense_1/kernel/Initializer/random_normal/stddevConst*
valueB
 *oГ:*$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 
А
@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_1/kernel/Initializer/random_normal/shape*
T0*$
_class
loc:@nn/dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:@@*

seed 
√
/nn/dense_1/kernel/Initializer/random_normal/mulMul@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_1/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
ф
+nn/dense_1/kernel/Initializer/random_normalAdd/nn/dense_1/kernel/Initializer/random_normal/mul0nn/dense_1/kernel/Initializer/random_normal/mean*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel
л
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
nn/dense_1/kernel/AssignAssignnn/dense_1/kernel+nn/dense_1/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel
Д
nn/dense_1/kernel/readIdentitynn/dense_1/kernel*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Т
!nn/dense_1/bias/Initializer/ConstConst*
valueB@*═╠╠=*"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@
Я
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
╞
nn/dense_1/bias/AssignAssignnn/dense_1/bias!nn/dense_1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
z
nn/dense_1/bias/readIdentitynn/dense_1/bias*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Ъ
nn/dense_1/MatMulMatMulnn/dense/Relunn/dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         @*
transpose_a( 
П
nn/dense_1/BiasAddBiasAddnn/dense_1/MatMulnn/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:         @*
T0
]
nn/dense_1/ReluRelunn/dense_1/BiasAdd*
T0*'
_output_shapes
:         @
и
1nn/dense_2/kernel/Initializer/random_normal/shapeConst*
valueB"@      *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
:
Ы
0nn/dense_2/kernel/Initializer/random_normal/meanConst*
valueB
 *    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
: 
Э
2nn/dense_2/kernel/Initializer/random_normal/stddevConst*
valueB
 *oГ:*$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes
: 
А
@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_2/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:@*

seed *
T0*$
_class
loc:@nn/dense_2/kernel*
seed2 
√
/nn/dense_2/kernel/Initializer/random_normal/mulMul@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_2/kernel/Initializer/random_normal/stddev*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
ф
+nn/dense_2/kernel/Initializer/random_normalAdd/nn/dense_2/kernel/Initializer/random_normal/mul0nn/dense_2/kernel/Initializer/random_normal/mean*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
л
nn/dense_2/kernel
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
┌
nn/dense_2/kernel/AssignAssignnn/dense_2/kernel+nn/dense_2/kernel/Initializer/random_normal*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Д
nn/dense_2/kernel/readIdentitynn/dense_2/kernel*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Т
!nn/dense_2/bias/Initializer/ConstConst*
valueB*═╠╠=*"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
Я
nn/dense_2/bias
VariableV2*"
_class
loc:@nn/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
╞
nn/dense_2/bias/AssignAssignnn/dense_2/bias!nn/dense_2/bias/Initializer/Const*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
z
nn/dense_2/bias/readIdentitynn/dense_2/bias*
_output_shapes
:*
T0*"
_class
loc:@nn/dense_2/bias
Ь
nn/dense_2/MatMulMatMulnn/dense_1/Relunn/dense_2/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
П
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
ф
Lloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnn/dense_2/BiasAddPlaceholder*6
_output_shapes$
":         :         *
Tlabels0*
T0
Ъ
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
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Б
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
и
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/mul*
_output_shapes
:*
T0*
out_type0
╣
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*#
_output_shapes
:         *

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
╖
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
╗
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
г
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
б
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ф
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
й
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*#
_output_shapes
:         *
T0
п
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
╒
3train/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/mul_grad/Shape%train/gradients/loss/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Н
!train/gradients/loss/mul_grad/MulMul&train/gradients/loss/Mean_grad/truedivPlaceholder_1*
T0*#
_output_shapes
:         
└
!train/gradients/loss/mul_grad/SumSum!train/gradients/loss/mul_grad/Mul3train/gradients/loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
╞
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
И
.train/gradients/loss/mul_grad/tuple/group_depsNoOp&^train/gradients/loss/mul_grad/Reshape(^train/gradients/loss/mul_grad/Reshape_1
В
6train/gradients/loss/mul_grad/tuple/control_dependencyIdentity%train/gradients/loss/mul_grad/Reshape/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/mul_grad/Reshape*#
_output_shapes
:         
И
8train/gradients/loss/mul_grad/tuple/control_dependency_1Identity'train/gradients/loss/mul_grad/Reshape_1/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/mul_grad/Reshape_1*#
_output_shapes
:         
й
train/gradients/zeros_like	ZerosLikeNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         *
T0
╜
qtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:         *┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
╗
ptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
         
т
ltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims6train/gradients/loss/mul_grad/tuple/control_dependencyptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
 
etrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulltrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsqtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:         
х
3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
data_formatNHWC*
_output_shapes
:*
T0
▐
8train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_depsNoOpf^train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul4^train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
Ъ
@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*x
_classn
ljloc:@train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul
л
Btrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
щ
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
б
7train/gradients/nn/dense_2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_2/MatMul_grad/MatMul0^train/gradients/nn/dense_2/MatMul_grad/MatMul_1
и
?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_2/MatMul_grad/MatMul8^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul*'
_output_shapes
:         @
е
Atrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_2/MatMul_grad/MatMul_18^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
╜
-train/gradients/nn/dense_1/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencynn/dense_1/Relu*
T0*'
_output_shapes
:         @
н
3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/nn/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ж
8train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad.^train/gradients/nn/dense_1/Relu_grad/ReluGrad
к
@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/Relu_grad/ReluGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/Relu_grad/ReluGrad
л
Btrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
щ
-train/gradients/nn/dense_1/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencynn/dense_1/kernel/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b(*
T0
┘
/train/gradients/nn/dense_1/MatMul_grad/MatMul_1MatMulnn/dense/Relu@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
б
7train/gradients/nn/dense_1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_1/MatMul_grad/MatMul0^train/gradients/nn/dense_1/MatMul_grad/MatMul_1
и
?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/MatMul_grad/MatMul8^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul
е
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
й
1train/gradients/nn/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+train/gradients/nn/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
а
6train/gradients/nn/dense/BiasAdd_grad/tuple/group_depsNoOp2^train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad,^train/gradients/nn/dense/Relu_grad/ReluGrad
в
>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/Relu_grad/ReluGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/Relu_grad/ReluGrad*'
_output_shapes
:         @
г
@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1Identity1train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*D
_class:
86loc:@train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad
у
+train/gradients/nn/dense/MatMul_grad/MatMulMatMul>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencynn/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
╒
-train/gradients/nn/dense/MatMul_grad/MatMul_1MatMulPlaceholder_2>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Ы
5train/gradients/nn/dense/MatMul_grad/tuple/group_depsNoOp,^train/gradients/nn/dense/MatMul_grad/MatMul.^train/gradients/nn/dense/MatMul_grad/MatMul_1
а
=train/gradients/nn/dense/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/MatMul_grad/MatMul6^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/MatMul_grad/MatMul*'
_output_shapes
:         
Э
?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/nn/dense/MatMul_grad/MatMul_16^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense/MatMul_grad/MatMul_1*
_output_shapes

:@
Ж
train/beta1_power/initial_valueConst*
valueB
 *fff?* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
: 
Ч
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
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
x
train/beta1_power/readIdentitytrain/beta1_power*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
Ж
train/beta2_power/initial_valueConst*
valueB
 *w╛?* 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
: 
Ч
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
е
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
ы
!train/nn/dense/kernel/Adam/AssignAssigntrain/nn/dense/kernel/Adam,train/nn/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
Ф
train/nn/dense/kernel/Adam/readIdentitytrain/nn/dense/kernel/Adam*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
з
.train/nn/dense/kernel/Adam_1/Initializer/zerosConst*"
_class
loc:@nn/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
┤
train/nn/dense/kernel/Adam_1
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *"
_class
loc:@nn/dense/kernel
ё
#train/nn/dense/kernel/Adam_1/AssignAssigntrain/nn/dense/kernel/Adam_1.train/nn/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
Ш
!train/nn/dense/kernel/Adam_1/readIdentitytrain/nn/dense/kernel/Adam_1*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Щ
*train/nn/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@nn/dense/bias*
valueB@*    
ж
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
▀
train/nn/dense/bias/Adam/AssignAssigntrain/nn/dense/bias/Adam*train/nn/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
К
train/nn/dense/bias/Adam/readIdentitytrain/nn/dense/bias/Adam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
Ы
,train/nn/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@nn/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
и
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
О
train/nn/dense/bias/Adam_1/readIdentitytrain/nn/dense/bias/Adam_1*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
╡
>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Я
4train/nn/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
.train/nn/dense_1/kernel/Adam/Initializer/zerosFill>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor4train/nn/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@nn/dense_1/kernel*

index_type0*
_output_shapes

:@@
╢
train/nn/dense_1/kernel/Adam
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
є
#train/nn/dense_1/kernel/Adam/AssignAssigntrain/nn/dense_1/kernel/Adam.train/nn/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ъ
!train/nn/dense_1/kernel/Adam/readIdentitytrain/nn/dense_1/kernel/Adam*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
╖
@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
б
6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
У
0train/nn/dense_1/kernel/Adam_1/Initializer/zerosFill@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel*

index_type0
╕
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
∙
%train/nn/dense_1/kernel/Adam_1/AssignAssigntrain/nn/dense_1/kernel/Adam_10train/nn/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ю
#train/nn/dense_1/kernel/Adam_1/readIdentitytrain/nn/dense_1/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Э
,train/nn/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*"
_class
loc:@nn/dense_1/bias*
valueB@*    
к
train/nn/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container *
shape:@
ч
!train/nn/dense_1/bias/Adam/AssignAssigntrain/nn/dense_1/bias/Adam,train/nn/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
Р
train/nn/dense_1/bias/Adam/readIdentitytrain/nn/dense_1/bias/Adam*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Я
.train/nn/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@nn/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
м
train/nn/dense_1/bias/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@nn/dense_1/bias
э
#train/nn/dense_1/bias/Adam_1/AssignAssigntrain/nn/dense_1/bias/Adam_1.train/nn/dense_1/bias/Adam_1/Initializer/zeros*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Ф
!train/nn/dense_1/bias/Adam_1/readIdentitytrain/nn/dense_1/bias/Adam_1*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
й
.train/nn/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@nn/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
╢
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
є
#train/nn/dense_2/kernel/Adam/AssignAssigntrain/nn/dense_2/kernel/Adam.train/nn/dense_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
Ъ
!train/nn/dense_2/kernel/Adam/readIdentitytrain/nn/dense_2/kernel/Adam*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
л
0train/nn/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*$
_class
loc:@nn/dense_2/kernel*
valueB@*    
╕
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
∙
%train/nn/dense_2/kernel/Adam_1/AssignAssigntrain/nn/dense_2/kernel/Adam_10train/nn/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Ю
#train/nn/dense_2/kernel/Adam_1/readIdentitytrain/nn/dense_2/kernel/Adam_1*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
Э
,train/nn/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@nn/dense_2/bias*
valueB*    
к
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
Р
train/nn/dense_2/bias/Adam/readIdentitytrain/nn/dense_2/bias/Adam*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
Я
.train/nn/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@nn/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
м
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
#train/nn/dense_2/bias/Adam_1/AssignAssigntrain/nn/dense_2/bias/Adam_1.train/nn/dense_2/bias/Adam_1/Initializer/zeros*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ф
!train/nn/dense_2/bias/Adam_1/readIdentitytrain/nn/dense_2/bias/Adam_1*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
]
train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *жЫ─;
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
 *w╛?
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
н
)train/Adam/update_nn/dense/bias/ApplyAdam	ApplyAdamnn/dense/biastrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0* 
_class
loc:@nn/dense/bias
╞
-train/Adam/update_nn/dense_1/kernel/ApplyAdam	ApplyAdamnn/dense_1/kerneltrain/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@nn/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
╣
+train/Adam/update_nn/dense_1/bias/ApplyAdam	ApplyAdamnn/dense_1/biastrain/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*"
_class
loc:@nn/dense_1/bias
╞
-train/Adam/update_nn/dense_2/kernel/ApplyAdam	ApplyAdamnn/dense_2/kerneltrain/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@nn/dense_2/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
╣
+train/Adam/update_nn/dense_2/bias/ApplyAdam	ApplyAdamnn/dense_2/biastrain/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@nn/dense_2/bias
Ш
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
к
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
Ъ
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
о
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@nn/dense/bias
╥

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam
Ъ
initNoOp^nn/dense/bias/Assign^nn/dense/kernel/Assign^nn/dense_1/bias/Assign^nn/dense_1/kernel/Assign^nn/dense_2/bias/Assign^nn/dense_2/kernel/Assign^train/beta1_power/Assign^train/beta2_power/Assign ^train/nn/dense/bias/Adam/Assign"^train/nn/dense/bias/Adam_1/Assign"^train/nn/dense/kernel/Adam/Assign$^train/nn/dense/kernel/Adam_1/Assign"^train/nn/dense_1/bias/Adam/Assign$^train/nn/dense_1/bias/Adam_1/Assign$^train/nn/dense_1/kernel/Adam/Assign&^train/nn/dense_1/kernel/Adam_1/Assign"^train/nn/dense_2/bias/Adam/Assign$^train/nn/dense_2/bias/Adam_1/Assign$^train/nn/dense_2/kernel/Adam/Assign&^train/nn/dense_2/kernel/Adam_1/Assign
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
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
╨
save/SaveV2/tensor_namesConst*Г
value∙BЎBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
Л
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
save/RestoreV2/tensor_namesConst"/device:CPU:0*Г
value∙BЎBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
Э
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
д
save/AssignAssignnn/dense/biassave/RestoreV2*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias
░
save/Assign_1Assignnn/dense/kernelsave/RestoreV2:1*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
м
save/Assign_2Assignnn/dense_1/biassave/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
┤
save/Assign_3Assignnn/dense_1/kernelsave/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
м
save/Assign_4Assignnn/dense_2/biassave/RestoreV2:4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
┤
save/Assign_5Assignnn/dense_2/kernelsave/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
и
save/Assign_6Assigntrain/beta1_powersave/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
и
save/Assign_7Assigntrain/beta2_powersave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
│
save/Assign_8Assigntrain/nn/dense/bias/Adamsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
╡
save/Assign_9Assigntrain/nn/dense/bias/Adam_1save/RestoreV2:9*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
╜
save/Assign_10Assigntrain/nn/dense/kernel/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
┐
save/Assign_11Assigntrain/nn/dense/kernel/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
╣
save/Assign_12Assigntrain/nn/dense_1/bias/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias
╗
save/Assign_13Assigntrain/nn/dense_1/bias/Adam_1save/RestoreV2:13*
use_locking(*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@
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
save/Assign_16Assigntrain/nn/dense_2/bias/Adamsave/RestoreV2:16*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_18Assigntrain/nn/dense_2/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
├
save/Assign_19Assigntrain/nn/dense_2/kernel/Adam_1save/RestoreV2:19*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
р
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"тu─ЧH╬      ┌к╬	█азх=╫AJ╗Ь
пК
:
Add
x"T
y"T
z"T"
Ttype:
2	
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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

2	Р
Н
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
2	Р
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
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
У
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.1-0-g6612da8951'╖ц
f
PlaceholderPlaceholder*
shape:         *
dtype0*#
_output_shapes
:         
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
p
Placeholder_2Placeholder*
shape:         *
dtype0*'
_output_shapes
:         
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
д
/nn/dense/kernel/Initializer/random_normal/shapeConst*"
_class
loc:@nn/dense/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
Ч
.nn/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *"
_class
loc:@nn/dense/kernel*
valueB
 *    
Щ
0nn/dense/kernel/Initializer/random_normal/stddevConst*"
_class
loc:@nn/dense/kernel*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
·
>nn/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/nn/dense/kernel/Initializer/random_normal/shape*
T0*"
_class
loc:@nn/dense/kernel*
seed2 *
dtype0*
_output_shapes

:@*

seed 
є
-nn/dense/kernel/Initializer/random_normal/mulMul>nn/dense/kernel/Initializer/random_normal/RandomStandardNormal0nn/dense/kernel/Initializer/random_normal/stddev*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
▄
)nn/dense/kernel/Initializer/random_normalAdd-nn/dense/kernel/Initializer/random_normal/mul.nn/dense/kernel/Initializer/random_normal/mean*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
з
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
╥
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
О
nn/dense/bias/Initializer/ConstConst* 
_class
loc:@nn/dense/bias*
valueB@*═╠╠=*
dtype0*
_output_shapes
:@
Ы
nn/dense/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@nn/dense/bias*
	container 
╛
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
Ц
nn/dense/MatMulMatMulPlaceholder_2nn/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         @*
transpose_b( 
Й
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
и
1nn/dense_1/kernel/Initializer/random_normal/shapeConst*$
_class
loc:@nn/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Ы
0nn/dense_1/kernel/Initializer/random_normal/meanConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
2nn/dense_1/kernel/Initializer/random_normal/stddevConst*$
_class
loc:@nn/dense_1/kernel*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
А
@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:@@*

seed *
T0*$
_class
loc:@nn/dense_1/kernel*
seed2 
√
/nn/dense_1/kernel/Initializer/random_normal/mulMul@nn/dense_1/kernel/Initializer/random_normal/RandomStandardNormal2nn/dense_1/kernel/Initializer/random_normal/stddev*
_output_shapes

:@@*
T0*$
_class
loc:@nn/dense_1/kernel
ф
+nn/dense_1/kernel/Initializer/random_normalAdd/nn/dense_1/kernel/Initializer/random_normal/mul0nn/dense_1/kernel/Initializer/random_normal/mean*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
л
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
Д
nn/dense_1/kernel/readIdentitynn/dense_1/kernel*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Т
!nn/dense_1/bias/Initializer/ConstConst*"
_class
loc:@nn/dense_1/bias*
valueB@*═╠╠=*
dtype0*
_output_shapes
:@
Я
nn/dense_1/bias
VariableV2*
shared_name *"
_class
loc:@nn/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
╞
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
Ъ
nn/dense_1/MatMulMatMulnn/dense/Relunn/dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         @*
transpose_b( 
П
nn/dense_1/BiasAddBiasAddnn/dense_1/MatMulnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
]
nn/dense_1/ReluRelunn/dense_1/BiasAdd*'
_output_shapes
:         @*
T0
и
1nn/dense_2/kernel/Initializer/random_normal/shapeConst*$
_class
loc:@nn/dense_2/kernel*
valueB"@      *
dtype0*
_output_shapes
:
Ы
0nn/dense_2/kernel/Initializer/random_normal/meanConst*$
_class
loc:@nn/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
2nn/dense_2/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *$
_class
loc:@nn/dense_2/kernel*
valueB
 *oГ:
А
@nn/dense_2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1nn/dense_2/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*$
_class
loc:@nn/dense_2/kernel
√
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
л
nn/dense_2/kernel
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
┌
nn/dense_2/kernel/AssignAssignnn/dense_2/kernel+nn/dense_2/kernel/Initializer/random_normal*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Д
nn/dense_2/kernel/readIdentitynn/dense_2/kernel*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Т
!nn/dense_2/bias/Initializer/ConstConst*"
_class
loc:@nn/dense_2/bias*
valueB*═╠╠=*
dtype0*
_output_shapes
:
Я
nn/dense_2/bias
VariableV2*"
_class
loc:@nn/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
╞
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
Ь
nn/dense_2/MatMulMatMulnn/dense_1/Relunn/dense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
П
nn/dense_2/BiasAddBiasAddnn/dense_2/MatMulnn/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
[

nn/SoftmaxSoftmaxnn/dense_2/BiasAdd*'
_output_shapes
:         *
T0
y
.loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapePlaceholder*
_output_shapes
:*
T0*
out_type0
ф
Lloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnn/dense_2/BiasAddPlaceholder*
T0*
Tlabels0*6
_output_shapes$
":         :         
Ъ
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
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Б
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
и
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
&train/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
╖
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
: *

Tidx0*
	keep_dims( 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
г
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
б
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ф
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
й
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:         
п
#train/gradients/loss/mul_grad/ShapeShapeLloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
r
%train/gradients/loss/mul_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
╒
3train/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/mul_grad/Shape%train/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Н
!train/gradients/loss/mul_grad/MulMul&train/gradients/loss/Mean_grad/truedivPlaceholder_1*
T0*#
_output_shapes
:         
└
!train/gradients/loss/mul_grad/SumSum!train/gradients/loss/mul_grad/Mul3train/gradients/loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
╞
#train/gradients/loss/mul_grad/Sum_1Sum#train/gradients/loss/mul_grad/Mul_15train/gradients/loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
║
'train/gradients/loss/mul_grad/Reshape_1Reshape#train/gradients/loss/mul_grad/Sum_1%train/gradients/loss/mul_grad/Shape_1*#
_output_shapes
:         *
T0*
Tshape0
И
.train/gradients/loss/mul_grad/tuple/group_depsNoOp&^train/gradients/loss/mul_grad/Reshape(^train/gradients/loss/mul_grad/Reshape_1
В
6train/gradients/loss/mul_grad/tuple/control_dependencyIdentity%train/gradients/loss/mul_grad/Reshape/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/mul_grad/Reshape*#
_output_shapes
:         
И
8train/gradients/loss/mul_grad/tuple/control_dependency_1Identity'train/gradients/loss/mul_grad/Reshape_1/^train/gradients/loss/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/mul_grad/Reshape_1*#
_output_shapes
:         
й
train/gradients/zeros_like	ZerosLikeNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:         
╜
qtrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientNloss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         *┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
╗
ptrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
т
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
х
3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
data_formatNHWC*
_output_shapes
:*
T0
▐
8train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_depsNoOpf^train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul4^train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad
Ъ
@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityetrain/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*x
_classn
ljloc:@train/gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*'
_output_shapes
:         
л
Btrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
щ
-train/gradients/nn/dense_2/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependencynn/dense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         @*
transpose_b(
█
/train/gradients/nn/dense_2/MatMul_grad/MatMul_1MatMulnn/dense_1/Relu@train/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@*
transpose_b( 
б
7train/gradients/nn/dense_2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_2/MatMul_grad/MatMul0^train/gradients/nn/dense_2/MatMul_grad/MatMul_1
и
?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_2/MatMul_grad/MatMul8^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul*'
_output_shapes
:         @
е
Atrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/nn/dense_2/MatMul_grad/MatMul_18^train/gradients/nn/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
╜
-train/gradients/nn/dense_1/Relu_grad/ReluGradReluGrad?train/gradients/nn/dense_2/MatMul_grad/tuple/control_dependencynn/dense_1/Relu*
T0*'
_output_shapes
:         @
н
3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/nn/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ж
8train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad.^train/gradients/nn/dense_1/Relu_grad/ReluGrad
к
@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/Relu_grad/ReluGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:         @
л
Btrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad9^train/gradients/nn/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/nn/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
щ
-train/gradients/nn/dense_1/MatMul_grad/MatMulMatMul@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependencynn/dense_1/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:         @
┘
/train/gradients/nn/dense_1/MatMul_grad/MatMul_1MatMulnn/dense/Relu@train/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@@*
transpose_b( 
б
7train/gradients/nn/dense_1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/nn/dense_1/MatMul_grad/MatMul0^train/gradients/nn/dense_1/MatMul_grad/MatMul_1
и
?train/gradients/nn/dense_1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/nn/dense_1/MatMul_grad/MatMul8^train/gradients/nn/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/nn/dense_1/MatMul_grad/MatMul*'
_output_shapes
:         @
е
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
й
1train/gradients/nn/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+train/gradients/nn/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
а
6train/gradients/nn/dense/BiasAdd_grad/tuple/group_depsNoOp2^train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad,^train/gradients/nn/dense/Relu_grad/ReluGrad
в
>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/Relu_grad/ReluGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/Relu_grad/ReluGrad*'
_output_shapes
:         @
г
@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1Identity1train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad7^train/gradients/nn/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/nn/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
у
+train/gradients/nn/dense/MatMul_grad/MatMulMatMul>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependencynn/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
╒
-train/gradients/nn/dense/MatMul_grad/MatMul_1MatMulPlaceholder_2>train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@*
transpose_b( 
Ы
5train/gradients/nn/dense/MatMul_grad/tuple/group_depsNoOp,^train/gradients/nn/dense/MatMul_grad/MatMul.^train/gradients/nn/dense/MatMul_grad/MatMul_1
а
=train/gradients/nn/dense/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/nn/dense/MatMul_grad/MatMul6^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/dense/MatMul_grad/MatMul*'
_output_shapes
:         
Э
?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/nn/dense/MatMul_grad/MatMul_16^train/gradients/nn/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*@
_class6
42loc:@train/gradients/nn/dense/MatMul_grad/MatMul_1
Ж
train/beta1_power/initial_valueConst* 
_class
loc:@nn/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ч
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
Ж
train/beta2_power/initial_valueConst* 
_class
loc:@nn/dense/bias*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Ч
train/beta2_power
VariableV2*
shared_name * 
_class
loc:@nn/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
┬
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
е
,train/nn/dense/kernel/Adam/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes

:@
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
ы
!train/nn/dense/kernel/Adam/AssignAssigntrain/nn/dense/kernel/Adam,train/nn/dense/kernel/Adam/Initializer/zeros*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
Ф
train/nn/dense/kernel/Adam/readIdentitytrain/nn/dense/kernel/Adam*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
з
.train/nn/dense/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense/kernel*
dtype0*
_output_shapes

:@
┤
train/nn/dense/kernel/Adam_1
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *"
_class
loc:@nn/dense/kernel
ё
#train/nn/dense/kernel/Adam_1/AssignAssigntrain/nn/dense/kernel/Adam_1.train/nn/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
Ш
!train/nn/dense/kernel/Adam_1/readIdentitytrain/nn/dense/kernel/Adam_1*
T0*"
_class
loc:@nn/dense/kernel*
_output_shapes

:@
Щ
*train/nn/dense/bias/Adam/Initializer/zerosConst*
valueB@*    * 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
ж
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
train/nn/dense/bias/Adam/AssignAssigntrain/nn/dense/bias/Adam*train/nn/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
К
train/nn/dense/bias/Adam/readIdentitytrain/nn/dense/bias/Adam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
Ы
,train/nn/dense/bias/Adam_1/Initializer/zerosConst*
valueB@*    * 
_class
loc:@nn/dense/bias*
dtype0*
_output_shapes
:@
и
train/nn/dense/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@nn/dense/bias*
	container 
х
!train/nn/dense/bias/Adam_1/AssignAssigntrain/nn/dense/bias/Adam_1,train/nn/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
О
train/nn/dense/bias/Adam_1/readIdentitytrain/nn/dense/bias/Adam_1*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
:@
╡
>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:
Я
4train/nn/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
: 
Н
.train/nn/dense_1/kernel/Adam/Initializer/zerosFill>train/nn/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor4train/nn/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
╢
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
є
#train/nn/dense_1/kernel/Adam/AssignAssigntrain/nn/dense_1/kernel/Adam.train/nn/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ъ
!train/nn/dense_1/kernel/Adam/readIdentitytrain/nn/dense_1/kernel/Adam*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
╖
@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@nn/dense_1/kernel*
dtype0*
_output_shapes
:
б
6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@nn/dense_1/kernel
У
0train/nn/dense_1/kernel/Adam_1/Initializer/zerosFill@train/nn/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6train/nn/dense_1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:@@*
T0*

index_type0*$
_class
loc:@nn/dense_1/kernel
╕
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
∙
%train/nn/dense_1/kernel/Adam_1/AssignAssigntrain/nn/dense_1/kernel/Adam_10train/nn/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ю
#train/nn/dense_1/kernel/Adam_1/readIdentitytrain/nn/dense_1/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_1/kernel*
_output_shapes

:@@
Э
,train/nn/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *"
_class
loc:@nn/dense_1/bias*
dtype0*
_output_shapes
:@
к
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
!train/nn/dense_1/bias/Adam/AssignAssigntrain/nn/dense_1/bias/Adam,train/nn/dense_1/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Р
train/nn/dense_1/bias/Adam/readIdentitytrain/nn/dense_1/bias/Adam*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
Я
.train/nn/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *"
_class
loc:@nn/dense_1/bias
м
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
э
#train/nn/dense_1/bias/Adam_1/AssignAssigntrain/nn/dense_1/bias/Adam_1.train/nn/dense_1/bias/Adam_1/Initializer/zeros*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Ф
!train/nn/dense_1/bias/Adam_1/readIdentitytrain/nn/dense_1/bias/Adam_1*
T0*"
_class
loc:@nn/dense_1/bias*
_output_shapes
:@
й
.train/nn/dense_2/kernel/Adam/Initializer/zerosConst*
valueB@*    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes

:@
╢
train/nn/dense_2/kernel/Adam
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
є
#train/nn/dense_2/kernel/Adam/AssignAssigntrain/nn/dense_2/kernel/Adam.train/nn/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Ъ
!train/nn/dense_2/kernel/Adam/readIdentitytrain/nn/dense_2/kernel/Adam*
_output_shapes

:@*
T0*$
_class
loc:@nn/dense_2/kernel
л
0train/nn/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *$
_class
loc:@nn/dense_2/kernel*
dtype0*
_output_shapes

:@
╕
train/nn/dense_2/kernel/Adam_1
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
∙
%train/nn/dense_2/kernel/Adam_1/AssignAssigntrain/nn/dense_2/kernel/Adam_10train/nn/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
Ю
#train/nn/dense_2/kernel/Adam_1/readIdentitytrain/nn/dense_2/kernel/Adam_1*
T0*$
_class
loc:@nn/dense_2/kernel*
_output_shapes

:@
Э
,train/nn/dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
к
train/nn/dense_2/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@nn/dense_2/bias
ч
!train/nn/dense_2/bias/Adam/AssignAssigntrain/nn/dense_2/bias/Adam,train/nn/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias
Р
train/nn/dense_2/bias/Adam/readIdentitytrain/nn/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@nn/dense_2/bias
Я
.train/nn/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *"
_class
loc:@nn/dense_2/bias*
dtype0*
_output_shapes
:
м
train/nn/dense_2/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@nn/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
э
#train/nn/dense_2/bias/Adam_1/AssignAssigntrain/nn/dense_2/bias/Adam_1.train/nn/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:
Ф
!train/nn/dense_2/bias/Adam_1/readIdentitytrain/nn/dense_2/bias/Adam_1*
T0*"
_class
loc:@nn/dense_2/bias*
_output_shapes
:
]
train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *жЫ─;
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
 *w╛?
W
train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
║
+train/Adam/update_nn/dense/kernel/ApplyAdam	ApplyAdamnn/dense/kerneltrain/nn/dense/kernel/Adamtrain/nn/dense/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon?train/gradients/nn/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@nn/dense/kernel*
use_nesterov( *
_output_shapes

:@
н
)train/Adam/update_nn/dense/bias/ApplyAdam	ApplyAdamnn/dense/biastrain/nn/dense/bias/Adamtrain/nn/dense/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/nn/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0* 
_class
loc:@nn/dense/bias
╞
-train/Adam/update_nn/dense_1/kernel/ApplyAdam	ApplyAdamnn/dense_1/kerneltrain/nn/dense_1/kernel/Adamtrain/nn/dense_1/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@@*
use_locking( *
T0*$
_class
loc:@nn/dense_1/kernel
╣
+train/Adam/update_nn/dense_1/bias/ApplyAdam	ApplyAdamnn/dense_1/biastrain/nn/dense_1/bias/Adamtrain/nn/dense_1/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*"
_class
loc:@nn/dense_1/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( 
╞
-train/Adam/update_nn/dense_2/kernel/ApplyAdam	ApplyAdamnn/dense_2/kerneltrain/nn/dense_2/kernel/Adamtrain/nn/dense_2/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/nn/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@nn/dense_2/kernel*
use_nesterov( *
_output_shapes

:@
╣
+train/Adam/update_nn/dense_2/bias/ApplyAdam	ApplyAdamnn/dense_2/biastrain/nn/dense_2/bias/Adamtrain/nn/dense_2/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/nn/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*"
_class
loc:@nn/dense_2/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
Ш
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
к
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
Ъ
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@nn/dense/bias*
_output_shapes
: 
о
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@nn/dense/bias
╥

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1*^train/Adam/update_nn/dense/bias/ApplyAdam,^train/Adam/update_nn/dense/kernel/ApplyAdam,^train/Adam/update_nn/dense_1/bias/ApplyAdam.^train/Adam/update_nn/dense_1/kernel/ApplyAdam,^train/Adam/update_nn/dense_2/bias/ApplyAdam.^train/Adam/update_nn/dense_2/kernel/ApplyAdam
Ъ
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
╨
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*Г
value∙BЎBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1
Л
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
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
т
save/RestoreV2/tensor_namesConst"/device:CPU:0*Г
value∙BЎBnn/dense/biasBnn/dense/kernelBnn/dense_1/biasBnn/dense_1/kernelBnn/dense_2/biasBnn/dense_2/kernelBtrain/beta1_powerBtrain/beta2_powerBtrain/nn/dense/bias/AdamBtrain/nn/dense/bias/Adam_1Btrain/nn/dense/kernel/AdamBtrain/nn/dense/kernel/Adam_1Btrain/nn/dense_1/bias/AdamBtrain/nn/dense_1/bias/Adam_1Btrain/nn/dense_1/kernel/AdamBtrain/nn/dense_1/kernel/Adam_1Btrain/nn/dense_2/bias/AdamBtrain/nn/dense_2/bias/Adam_1Btrain/nn/dense_2/kernel/AdamBtrain/nn/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
Э
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
■
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
д
save/AssignAssignnn/dense/biassave/RestoreV2*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@nn/dense/bias
░
save/Assign_1Assignnn/dense/kernelsave/RestoreV2:1*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel*
validate_shape(*
_output_shapes

:@
м
save/Assign_2Assignnn/dense_1/biassave/RestoreV2:2*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
┤
save/Assign_3Assignnn/dense_1/kernelsave/RestoreV2:3*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel
м
save/Assign_4Assignnn/dense_2/biassave/RestoreV2:4*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
┤
save/Assign_5Assignnn/dense_2/kernelsave/RestoreV2:5*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@nn/dense_2/kernel
и
save/Assign_6Assigntrain/beta1_powersave/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
: 
и
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
╡
save/Assign_9Assigntrain/nn/dense/bias/Adam_1save/RestoreV2:9*
use_locking(*
T0* 
_class
loc:@nn/dense/bias*
validate_shape(*
_output_shapes
:@
╜
save/Assign_10Assigntrain/nn/dense/kernel/Adamsave/RestoreV2:10*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
┐
save/Assign_11Assigntrain/nn/dense/kernel/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@nn/dense/kernel
╣
save/Assign_12Assigntrain/nn/dense_1/bias/Adamsave/RestoreV2:12*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
╗
save/Assign_13Assigntrain/nn/dense_1/bias/Adam_1save/RestoreV2:13*
T0*"
_class
loc:@nn/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
┴
save/Assign_14Assigntrain/nn/dense_1/kernel/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@nn/dense_1/kernel
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
save/Assign_16Assigntrain/nn/dense_2/bias/Adamsave/RestoreV2:16*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╗
save/Assign_17Assigntrain/nn/dense_2/bias/Adam_1save/RestoreV2:17*
T0*"
_class
loc:@nn/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_19Assigntrain/nn/dense_2/kernel/Adam_1save/RestoreV2:19*
T0*$
_class
loc:@nn/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
р
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""├
trainable_variablesли
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
	variables█╪
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
Ф
train/nn/dense/kernel/Adam:0!train/nn/dense/kernel/Adam/Assign!train/nn/dense/kernel/Adam/read:02.train/nn/dense/kernel/Adam/Initializer/zeros:0
Ь
train/nn/dense/kernel/Adam_1:0#train/nn/dense/kernel/Adam_1/Assign#train/nn/dense/kernel/Adam_1/read:020train/nn/dense/kernel/Adam_1/Initializer/zeros:0
М
train/nn/dense/bias/Adam:0train/nn/dense/bias/Adam/Assigntrain/nn/dense/bias/Adam/read:02,train/nn/dense/bias/Adam/Initializer/zeros:0
Ф
train/nn/dense/bias/Adam_1:0!train/nn/dense/bias/Adam_1/Assign!train/nn/dense/bias/Adam_1/read:02.train/nn/dense/bias/Adam_1/Initializer/zeros:0
Ь
train/nn/dense_1/kernel/Adam:0#train/nn/dense_1/kernel/Adam/Assign#train/nn/dense_1/kernel/Adam/read:020train/nn/dense_1/kernel/Adam/Initializer/zeros:0
д
 train/nn/dense_1/kernel/Adam_1:0%train/nn/dense_1/kernel/Adam_1/Assign%train/nn/dense_1/kernel/Adam_1/read:022train/nn/dense_1/kernel/Adam_1/Initializer/zeros:0
Ф
train/nn/dense_1/bias/Adam:0!train/nn/dense_1/bias/Adam/Assign!train/nn/dense_1/bias/Adam/read:02.train/nn/dense_1/bias/Adam/Initializer/zeros:0
Ь
train/nn/dense_1/bias/Adam_1:0#train/nn/dense_1/bias/Adam_1/Assign#train/nn/dense_1/bias/Adam_1/read:020train/nn/dense_1/bias/Adam_1/Initializer/zeros:0
Ь
train/nn/dense_2/kernel/Adam:0#train/nn/dense_2/kernel/Adam/Assign#train/nn/dense_2/kernel/Adam/read:020train/nn/dense_2/kernel/Adam/Initializer/zeros:0
д
 train/nn/dense_2/kernel/Adam_1:0%train/nn/dense_2/kernel/Adam_1/Assign%train/nn/dense_2/kernel/Adam_1/read:022train/nn/dense_2/kernel/Adam_1/Initializer/zeros:0
Ф
train/nn/dense_2/bias/Adam:0!train/nn/dense_2/bias/Adam/Assign!train/nn/dense_2/bias/Adam/read:02.train/nn/dense_2/bias/Adam/Initializer/zeros:0
Ь
train/nn/dense_2/bias/Adam_1:0#train/nn/dense_2/bias/Adam_1/Assign#train/nn/dense_2/bias/Adam_1/read:020train/nn/dense_2/bias/Adam_1/Initializer/zeros:0Cy▒s