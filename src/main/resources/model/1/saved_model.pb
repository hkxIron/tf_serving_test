
Ű´
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 
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
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
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
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
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
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
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
Ttype"serve*1.9.02v1.9.0-0-g25c197e023Śv
d
xPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
n
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$w/Initializer/truncated_normal/shapeConst*
_class

loc:@w*
valueB"      *
dtype0*
_output_shapes
:
~
#w/Initializer/truncated_normal/meanConst*
_class

loc:@w*
valueB
 *    *
dtype0*
_output_shapes
: 

%w/Initializer/truncated_normal/stddevConst*
_class

loc:@w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ě
.w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal$w/Initializer/truncated_normal/shape*

seed *
T0*
_class

loc:@w*
seed2 *
dtype0*
_output_shapes

:
ż
"w/Initializer/truncated_normal/mulMul.w/Initializer/truncated_normal/TruncatedNormal%w/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*
_class

loc:@w
­
w/Initializer/truncated_normalAdd"w/Initializer/truncated_normal/mul#w/Initializer/truncated_normal/mean*
T0*
_class

loc:@w*
_output_shapes

:

w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *
_class

loc:@w

w/AssignAssignww/Initializer/truncated_normal*
T0*
_class

loc:@w*
validate_shape(*
_output_shapes

:*
use_locking(
T
w/readIdentityw*
T0*
_class

loc:@w*
_output_shapes

:
v
b/Initializer/zerosConst*
_class

loc:@b*
valueB*    *
dtype0*
_output_shapes
:

b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class

loc:@b*
	container *
shape:

b/AssignAssignbb/Initializer/zeros*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:*
use_locking(
P
b/readIdentityb*
T0*
_class

loc:@b*
_output_shapes
:
"
initNoOp	^b/Assign	^w/Assign
s
MatMulMatMulxw/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
J
yAddMatMulb/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
subSubyPlaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
H
powPowsubpow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
V
MeanMeanpowConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
\
gradients/Mean_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/Mean_grad/Shape_1Shapepow*
T0*
out_type0*
_output_shapes
:
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

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
gradients/pow_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
]
gradients/pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
´
(gradients/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pow_grad/Shapegradients/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
s
gradients/pow_grad/mulMulgradients/Mean_grad/truedivpow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
gradients/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients/pow_grad/subSubpow/ygradients/pow_grad/sub/y*
_output_shapes
: *
T0
l
gradients/pow_grad/PowPowsubgradients/pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/pow_grad/mul_1Mulgradients/pow_grad/mulgradients/pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/pow_grad/SumSumgradients/pow_grad/mul_1(gradients/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/pow_grad/ReshapeReshapegradients/pow_grad/Sumgradients/pow_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
gradients/pow_grad/GreaterGreatersubgradients/pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
gradients/pow_grad/LogLogsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/pow_grad/zeros_like	ZerosLikesub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
gradients/pow_grad/SelectSelectgradients/pow_grad/Greatergradients/pow_grad/Loggradients/pow_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
gradients/pow_grad/mul_2Mulgradients/Mean_grad/truedivpow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/pow_grad/mul_3Mulgradients/pow_grad/mul_2gradients/pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/pow_grad/Sum_1Sumgradients/pow_grad/mul_3*gradients/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/pow_grad/Reshape_1Reshapegradients/pow_grad/Sum_1gradients/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/pow_grad/tuple/group_depsNoOp^gradients/pow_grad/Reshape^gradients/pow_grad/Reshape_1
Ú
+gradients/pow_grad/tuple/control_dependencyIdentitygradients/pow_grad/Reshape$^gradients/pow_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/pow_grad/Reshape
Ď
-gradients/pow_grad/tuple/control_dependency_1Identitygradients/pow_grad/Reshape_1$^gradients/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/pow_grad/Reshape_1*
_output_shapes
: 
Y
gradients/sub_grad/ShapeShapey*
_output_shapes
:*
T0*
out_type0
e
gradients/sub_grad/Shape_1ShapePlaceholder*
_output_shapes
:*
T0*
out_type0
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients/sub_grad/SumSum+gradients/pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
¸
gradients/sub_grad/Sum_1Sum+gradients/pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ú
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
gradients/y_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
b
gradients/y_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Ž
&gradients/y_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/y_grad/Shapegradients/y_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
°
gradients/y_grad/SumSum+gradients/sub_grad/tuple/control_dependency&gradients/y_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/y_grad/ReshapeReshapegradients/y_grad/Sumgradients/y_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/y_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency(gradients/y_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/y_grad/Reshape_1Reshapegradients/y_grad/Sum_1gradients/y_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
a
!gradients/y_grad/tuple/group_depsNoOp^gradients/y_grad/Reshape^gradients/y_grad/Reshape_1
Ň
)gradients/y_grad/tuple/control_dependencyIdentitygradients/y_grad/Reshape"^gradients/y_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/y_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
+gradients/y_grad/tuple/control_dependency_1Identitygradients/y_grad/Reshape_1"^gradients/y_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/y_grad/Reshape_1*
_output_shapes
:
ą
gradients/MatMul_grad/MatMulMatMul)gradients/y_grad/tuple/control_dependencyw/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ľ
gradients/MatMul_grad/MatMul_1MatMulx)gradients/y_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ä
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
×Ł;
ë
-GradientDescent/update_w/ApplyGradientDescentApplyGradientDescentwGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*
_class

loc:@w
â
-GradientDescent/update_b/ApplyGradientDescentApplyGradientDescentbGradientDescent/learning_rate+gradients/y_grad/tuple/control_dependency_1*
T0*
_class

loc:@b*
_output_shapes
:*
use_locking( 
w
GradientDescentNoOp.^GradientDescent/update_b/ApplyGradientDescent.^GradientDescent/update_w/ApplyGradientDescent

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_b33c1298886040ef9ad4d1d959502d87/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
t
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBbBw*
dtype0*
_output_shapes
:
v
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B 

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbw"/device:CPU:0*
dtypes
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
w
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBbBw*
dtype0*
_output_shapes
:
y
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
¤
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes

::*
dtypes
2

save/AssignAssignbsave/RestoreV2*
use_locking(*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:

save/Assign_1Assignwsave/RestoreV2:1*
use_locking(*
T0*
_class

loc:@w*
validate_shape(*
_output_shapes

:
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
trainable_variablesus
=
w:0w/Assignw/read:02 w/Initializer/truncated_normal:08
2
b:0b/Assignb/read:02b/Initializer/zeros:08"
train_op

GradientDescent"
	variablesus
=
w:0w/Assignw/read:02 w/Initializer/truncated_normal:08
2
b:0b/Assignb/read:02b/Initializer/zeros:08"$
legacy_init_op

legacy_init_op*u

predictiong
#
input
x:0˙˙˙˙˙˙˙˙˙$
output
y:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict