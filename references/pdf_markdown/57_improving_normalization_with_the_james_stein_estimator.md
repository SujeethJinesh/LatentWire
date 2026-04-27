# references/57_improving_normalization_with_the_james_stein_estimator.pdf

<!-- page 1 -->

Improving Normalization with the James-Stein Estimator
Seyedalireza Khoshsirat Chandra Kambhamettu
Video/Image Modeling and Synthesis (VIMS) Lab, University of Delaware
{alireza, chandrak }@udel.edu
Abstract
Stein’s paradox holds considerable sway in high-
dimensional statistics, highlighting that the sample mean,
traditionally considered the de facto estimator, might not
be the most efficacious in higher dimensions. To address
this, the James-Stein estimator proposes an enhancement
by steering the sample means toward a more centralized
mean vector. In this paper, first, we establish that normal-
ization layers in deep learning use inadmissible estimators
for mean and variance. Next, we introduce a novel method
to employ the James-Stein estimator to improve the estima-
tion of mean and variance within normalization layers. We
evaluate our method on different computer vision tasks: im-
age classification, semantic segmentation, and 3D object
classification. Through these evaluations, it is evident that
our improved normalization layers consistently yield supe-
rior accuracy across all tasks without extra computational
burden. Moreover, recognizing that a plethora of shrink-
age estimators surpass the traditional estimator in perfor-
mance, we study two other prominent shrinkage estimators:
Ridge and LASSO. Additionally, we provide visual represen-
tations to intuitively demonstrate the impact of shrinkage on
the estimated layer statistics. Finally, we study the effect of
regularization and batch size on our modified batch normal-
ization. The studies show that our method is less sensitive
to batch size and regularization, improving accuracy under
various setups.
1. Introduction
Deep neural networks have an influential role in many
applications, especially computer vision. One milestone
improvement in these networks was the addition of normal-
ization layers [21,29,34]. Since then, a plethora of research
has tried to improve normalization layers through various
means, including better estimating the layer statistics. In
this paper, we take a statistical approach to estimate layer
statistics and introduce an improved way of estimating the
mean and variance in normalization layers. The rest of this
section is devoted to introducing prerequisite statistical con-
cepts and normalization layers.
Estimators. An estimator refers to a method used to
calculate an approximation of a specific value from the data
observed. Therefore, one should differentiate between the
estimator itself, the targeted value of interest, and the result-
ing approximation [37].
Shrinkage. In statistics, shrinkage is the reduction of
the effects of sampling noise. In regression analysis, a fitted
relationship appears to perform worse on a new dataset than
on the dataset used for fitting [11,56]. More specifically, the
value of the coefficient of determination ‘shrinks.’
Shrinkage Estimators. A shrinkage estimator is an esti-
mator that explicitly or implicitly uses the effects of shrink-
age. In loose terms, a naive or basic estimate is improved
by combining it with other information. This term refers
to the notion that the improved estimate is pushed closer to
the value provided by the ‘other information’ than the basic
estimate [24, 56]. In this sense, shrinkage is used to reg-
ularize the estimation process. In terms of mean squared
error (MSE), many standard estimators can be improved
by shrinking them towards zero or some other value. In
other words, the improvement in the estimate from the cor-
responding reduction in the width of the confidence inter-
val is likely to outweigh the worsening of the estimate in-
structed by biasing the estimate towards zero [23].
Admissibility. Assume x is distributed according to
p(x|θ), with θ being a member of the set Θ. Consider ˆθ
as an estimator for θ and let R(ˆθ, θ) represent the risk of
using ˆθ, which is calculated based on a specific loss func-
tion. The risk is defined by R(ˆθ, θ) = E[ℓ(ˆθ, θ)], where ℓ
signifies the loss function and the expectation is taken over
x drawn from p(x|θ), with ˆθ being a derived function of x.
An estimator is deemed ’inadmissible’ if there is another es-
timator, ˜θ, which performs better; that is,R(˜θ, θ) ≤ R(ˆθ, θ)
for all θ within Θ, and there’s at least one θ for which the
inequality is strict. If no such dominating estimator exists,
then ˆθ is considered ’admissible’ [6].
James-Stein. Calculating the mean of a multivariate
normal distribution stands as a key issue in the field of statis-
tics. Typically, the sample mean is used, which also hap-
pens to be the maximum-likelihood estimator. However, the
This WACV paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
2041


<!-- page 2 -->

...
  features
  samples
...
...
Mean and Variance
...
...
James-Stein Normalize
... ...
...
Scale and Shift
...
JSNorm
Figure 1. The overall structure of our proposed JSNorm for batch normalization with the key components involved at each step. JSNorm
integrates the James-Stein estimator into normalization layers to refine the originally estimated statistics. This design choice ensures that
the computational overhead associated with JSNorm is minimal and practically negligible.
James-Stein (JS) estimator, which is known to be biased, is
used for estimating the mean of c correlated Gaussian dis-
tributed random vectors whose means are not known. The
development of this estimator unfolded over two major pa-
pers; the initial version was introduced by Charles Stein in
1956 [47], leading to the surprising discovery that the stan-
dard mean estimate is admissible when c ≤ 2, but becomes
inadmissible for c ≥ 3. This work suggested an enhance-
ment that involves shrinking the sample means towards a
central vector of means, a concept often cited as Stein’s ex-
ample or paradox [57].
Normalization Layers. The primary objective of nor-
malization techniques is to enhance training stability and
facilitate the design of network architectures. While vari-
ous normalization layers are available [19, 35, 43, 51, 58],
this paper specifically focuses on batch normalization and
layer normalization. These two normalization techniques
are widely utilized in computer vision networks and serve
as the most common choices within the field.
Batch Normalization (BatchNorm or BN) [21] is a key
deep learning technique that enhances computer vision ap-
plications by normalizing feature-maps using mean and
variance computed over batches. This aids optimization
[44], promotes convergence in deep networks [3], and re-
duces the number of iterations to converge, improving per-
formance [26].
Layer Normalization (LayerNorm or LN) [2] is a dif-
ferent approach that addresses training issues in Recurrent
Neural Networks [2]. It normalizes statistics within a sin-
gle sample, making it independent of batch size, and was
specifically designed to work with the variable statistics of
recurrent neurons. It is a critical component in transformer
networks [33, 53].
Contributions. Our contributions in this study are three-
fold:
• Identification of Inadmissible Estimators: We es-
tablish that widely-used normalization layers in deep
learning networks employ inadmissible estimators for
calculating mean and variance. This insight draws at-
tention to potential limitations in the current practice.
• Introduction of Improved Normalization Methods:
We present an innovative approach wherein the James-
Stein estimator is adapted to more accurately estimate
the mean and variance in normalization layers. This
is a significant contribution, as it bridges the gap be-
tween classical statistical methods and contemporary
deep learning techniques.
• Empirical Validation Across Domains: Through ex-
tensive experimentation, we substantiate the efficacy
of our proposed method across three distinct computer
vision tasks. Additionally, we conduct a series of anal-
yses to ascertain the robustness and performance en-
hancements attributed to our approach under varying
configurations.
2. Related Work
The James-Stein estimator has motivated a rich litera-
ture on the theme of “shrinkage” in statistics. Just a small
selection of examples include LASSO [50], ridge regres-
sion [16], the Ledoit-Wolf covariance estimator [30] and
Elastic Net [67]. Excellent textbook treatments of the con-
cepts behind Stein’s paradox and James-Stein shrinkage in-
clude [12, 14].
Before batch normalization became prevalent, various
types of normalization layers were already being utilized
in deep neural networks. Local Response Normalization
(LRN) featured in AlexNet [29] and was adopted by subse-
quent models [17, 28, 45, 65]. LRN normalizes the values
around a given pixel within a specified neighborhood.
Batch Normalization (BN) [21], introduced later, applies
a more comprehensive form of normalization across the
entire batch of data and suggests applying this technique
across all layers of the network. In the wake of BN, other
normalization approaches [2,25,27,51] were developed that
do not rely on the batch dimension. For example, Layer
Normalization (LN) [2] adjusts the data across the chan-
nel dimension, while Instance Normalization (IN) [51] car-
ries out a batch normalization-like process for each indi-
vidual instance. In a different approach from normalizing
2042


<!-- page 3 -->

Batch Normalization Layer Normalization Description
µB = 1
n×h×w
P
n,h,w xi,j,k µB = 1
h×w
P
h,w xi,j Calculating the corresponding mean
σ2
B = 1
n×h×w
P
n,h,w(xi,j,k − µB)2 σ2
B = 1
h×w
P
h,w(xi,j − µB)2 Calculating the corresponding variance
µµB = 1
c
P
c µBi µµB = 1
c
P
c µBi Mean of the estimated means
σ2
µB = 1
c
P
c(µBi − µµB)2 σ2
µB = 1
c
P
c(µBi − µµB)2 Variance of the estimated means
µJ S =

1 −
(c−2)σ2
µB
∥µB∥2
2

µB µJ S =

1 −
(c−2)σ2
µB
∥µB∥2
2

µB JS estimation of the mean
µσ2
B
= 1
c
P
c σ2
Bi µσ2
B
= 1
c
P
c σ2
Bi Mean of the estimated variances
σ2
σ2
B
= 1
c
P
c(σ2
Bi − µσ2
B
)2 σ2
σ2
B
= 1
c
P
c(σ2
Bi − µσ2
B
)2 Variance of the estimated variances
σ2
J S =

1 −
(c−2)σ2
σ2
B
∥σ2
B∥2
2

σ2
B σ2
J S =

1 −
(c−2)σ2
σ2
B
∥σ2
B∥2
2

σ2
B JS estimation of the variance
ˆx = x−µJ S√
σ2
J S +ϵ ˆx = x−µJ S√
σ2
J S +ϵ Standardization using the JS estimations
y = γˆx + β y = γˆx + β Scale and shift
Table 1. Our proposed batch normalization and layer normalization which integrate the James-Stein estimator. Given an input feature-map
x ∈ Rn×c×h×w, for batch normalization, each channel ( c) is processed separately, and for layer normalization, each sample ( n). The
operations are done in order, from top to bottom.
data, Weight Normalization [43] focuses on normalizing the
weights of the filters within the network. Group Normaliza-
tion (GN) [58] generalizes LN, dividing the neurons into
groups and standardizing the layer input within the neurons
of each group for each sample independently.
A diverse set of works aiming at improving normal-
ization layers exist. Decorrelated Batch Normalization
[20] employs whitening techniques to solve the so-called
“stochastic axis swapping” problem. In [60], a simple ver-
sion of the layer normalization is introduced that removes
the bias and gain parameters to cope with over-fitting. Batch
group normalization [66] extends the grouping mechanism
of GN from being over only channels dimension to being
over both channels and batch dimensions. MoBN [43] is
a mean-only batch normalization that performs centering
only along the batch dimension. It performs well when
combined with weight normalization. A scaling-only batch
normalization is proposed in [62] that performs well in
training with a small batch size.
Although the James-Stein estimator has not directly been
used in deep learning, some methods use this estimator
alongside deep learning. In [1], the James-Stein estima-
tor is used for feature extraction before feeding data to a
deep neural network. This method combines Pinsker’s the-
orem with James-Stein to leverage the advantages of non-
parametric regression to use deep learning in limited data
setups. C-SURE [7] is a novel shrinkage estimator based on
the James-Stein estimator for the complex-valued manifold.
It is incorporated in a complex-valued classifier network for
estimating the statistical mean of the per-class wFM fea-
tures.
Our research stands at the forefront by being the first
study to investigate the admissibility of estimators within
normalization layers in deep learning. Furthermore, we
break new ground by introducing an innovative methodol-
ogy to address this issue, thereby pioneering a potentially
transformative approach in the field.
3. Method
In this section, we begin by providing a concise overview
of batch normalization [21] and layer normalization [2]
techniques. We then delve into our underlying motiva-
tion for proposing modified versions of these normalization
methods. Finally, we present our novel approaches to nor-
malization.
Batch Normalization. The batch normalization layer
performs feature standardization within each batch by nor-
malizing each feature individually, followed by learning
a shared weight and bias for the entire batch. Formally,
given the input to a batch normalization layer X ∈ Rn×c,
2043


<!-- page 4 -->

Method Original JSNorm
ResNet-18 [15] 69.7 71.0±0.3 (+1.3)
ResNet-50 [15] 76.1 77.2±0.3 (+1.1)
ResNet-152 [15] 77.9 78.9±0.2 (+1.0)
EfficientNet-B1 [48] 79.2 80.2±0.2 (+1.0)
EfficientNet-B3 [48] 81.7 82.6±0.1 (+0.9)
EfficientNet-B5 [48] 83.7 84.6±0.1 (+0.9)
GENet-light [31] 75.7 77.2±0.3 (+1.5)
SwinV2-T, win8x8 [32] 81.8 82.8±0.2 (+1.0)
SwinV2-T, win16x16 [32] 82.8 83.7±0.2 (+0.9)
SwinV2-S, win8x8 [32] 83.7 84.6±0.1 (+0.9)
Table 2. Comparison of top-1 accuracy on ImageNet with
mean±std in five random evaluations. Notably, all the networks
were trained without any fine-tuning of hyper-parameters.
where n denotes the batch size, and c is the feature dimen-
sion, batch normalization layer first normalizes each feature
i ∈ {1 . . . c}:
ˆxi = xi − E[xi]p
V ar[xi]
, (1)
where the expectation and variance are computed over the
training data set. Next, for each feature dimension, a pair of
parameters γi and βi are used to scale and shift the normal-
ized value:
yi = γiˆxi + βi. (2)
These parameters are learned during the training.
Layer Normalization. Similar to batch normaliza-
tion, layer normalization begins by standardizing the input.
However, unlike batch normalization, layer normalization
operates independently on each sample, without relying on
the batch size. As a result, the standardization process oc-
curs individually for each sample.
Therefore, for each sample i ∈ {1 . . . n}:
ˆxi = xi − µxip
V ar[xi]
. (3)
Similar to batch normalization, it is followed by learning a
pair of parameters γi and βi:
yi = γiˆxi + βi. (4)
Motivation. One notable concern pertaining to Equa-
tions 1 and 3 lies in the estimation of the mean and vari-
ance. The conventional approach suggests independently
calculating the mean and variance using “usual estimators”.
For batch normalization:
E[xi] = 1
n
nX
j=1
xi,j, (5)
V ar[xi] = 1
n
nX
j=1
(xi,j − E[xi])2, (6)
Method Original JSNorm
HRNetV2 [54] 81.6 83.0±0.4 (+1.4)
HRNetV2+OCR [64] 83.0 84.2±0.2 (+1.2)
HRNetV2+OCR† [64] 84.2 85.3±0.1 (+1.1)
EfficientPS† [36] 84.2 85.3±0.2 (+1.1)
Lawin [61] 84.4 85.4±0.1 (+1.0)
Table 3. Evaluation of mIoU scores on the Cityscapes test set.
We use † to mark methods pre-trained on Mapillary Vistas dataset
[38].
and for layer normalization:
µxi = 1
c
cX
j=1
xi,j, (7)
V ar[xi] = 1
c
cX
j=1
(xi,j − µxi)2, (8)
are the estimators.
Given that all the features contribute to a shared loss
function, according to Stein’s paradox [47], these estimators
are inadmissible when c ≥ 3. Notably, in computer vision
networks, it is consistently observed that c ≥ 3. To address
this, we propose a novel method to adopt admissible shrink-
age estimators, which effectively enhance the estimation of
the mean and variance in both normalization layers.
James-Stein. Let X = {x1, x2, ..., xc} with un-
known means θ = {θ1, θ2, ..., θc} and estimates ˆθ =
{ˆθ1, ˆθ2, ..., ˆθc}. The basic formula for the James-Stein esti-
mator is:
ˆθJS = ˆθ + s(µ ˆθ − ˆθ), (9)
where µ ˆθ − ˆθ is the difference between the total mean (aver-
age of averages) and each individual estimated mean, and s
is a shrinking factor. Among the numerous perspectives that
motivate the James-Stein estimator, the empirical Bayes
perspective [10] is exquisite. Taking a Gaussian prior on the
unknown means leads us to the following formula [22,23]:
ˆθJS =

1 − (c − 2)σ2
∥ ˆθ − v∥2
2
!
( ˆθ − v) + v, (10)
where ∥ · ∥ 2 denotes the L2 norm of the argument, σ2 is
the variance, v is an arbitrary fixed vector that shows the
shrinkage direction, and c ≥ 3. Setting v = 0 results the
following:
ˆθJS =

1 − (c − 2)σ2
∥ ˆθ∥2
2
!
ˆθ. (11)
The above estimator shrinks the estimates towards the ori-
gin 0.
2044


<!-- page 5 -->

ScanObjectNN ModelNet40
Method Original JSNorm Original JSNorm
PointNet [39] 68.2 70.0±0.4 (+1.8) 89.2 90.8±0.3 (+1.6)
PointNet++ [40] 77.9 79.5±0.3 (+1.6) 91.9 93.5±0.4 (+1.6)
DGCNN [55] 81.9 83.3±0.1 (+1.4) 93.5 94.8±0.2 (+1.3)
Point-BERT [63] 83.1 84.3±0.2 (+1.2) 93.8 94.8±0.1 (+1.0)
PointNeXt-S [41] 87.7 88.8±0.1 (+1.1) 93.2 94.2±0.1 (+1.0)
Table 4. Evaluation of classification accuracy across two 3D datasets: ScanObjectNN and ModelNet40.
Commonly in transformer networks for computer vision
[32, 33, 53], the statistics for layer normalization are cal-
culated on the patch size dimensions h, and w. We take
advantage of this design and use the James-Stein estimator
such that our layer normalization stays independent of batch
size, and each sample is processed independently.
In this paper, we employ Equation 11 to avoid the ‘mean
shift’ problem [4,5]. To integrate this equation into normal-
ization layers, we substitute the ˆθ in Equation 11 with the
estimated mean and variance obtained through the original
method. By applying the James-Stein estimator to the es-
timated statistics from the original method, the additional
processing required is minimal and can be considered neg-
ligible. Consequently, we utilize the James-Stein estimator
for both the mean and variance in the normalization lay-
ers. For batch normalization, E[x] and V ar[x] are vectors
of length c (for the whole batch). Therefore, they can di-
rectly be used in place of ˆθ. For layer normalization, µx
and V ar[x] are in the form of Rn×c, and since in layer nor-
malization each sample should be processed independently,
each vector from the second dimension c is separately used
in place of ˆθ. Table 1 shows the detailed definition of our
proposed normalization layers, while Figure 1 depicts the
overview of JSNorm for batch normalization.
3.1. Gaussian Prior
Employing a variant of the James-Stein estimator that
assumes a Gaussian prior might raise queries regarding the
general applicability of our methodology. This concern can
be addressed through two key observations:
• Alignment with Normal Distribution in Normaliza-
tion Layers: Normalization layers aim to standardize
the features-maps within a layer and make them re-
semble a standard Gaussian distribution with a mean of
zero and a standard deviation of one. Thus, adopting a
Gaussian prior aligns with the intrinsic characteristics
of the feature-maps and does not substantially deviate
their distribution within the network. Moreover, nu-
merous approaches to developing normalization-free
networks [4, 5] incorporate Gaussian weight initializa-
tion and standardization to promote a Gaussian-like
distribution for feature-maps. This underscores the
practical advantages of adopting a Gaussian prior.
• Distinction Between Feature-Maps and Input Dis-
tributions: It is crucial to acknowledge that the distri-
bution of the feature-maps within the neural network
need not mirror that of the input data. The succession
of transformations applied by the network layers often
results in an evolution of the input data distribution. As
the network trains, it learns to change data representa-
tion in a manner useful for the specific task. Therefore,
the efficacy of our method is not strictly tethered to the
distribution of inputs.
4. Experiments
We conduct a comprehensive evaluation of our proposed
method across various computer vision tasks. For each task,
we utilize well-established state-of-the-art networks with
readily available implementations. The sole modification
we introduce involves replacing the normalization layers
with our proposed batch normalization or layer normaliza-
tion counterparts. As a result, all other hyper-parameters
and training configurations remain consistent with those
outlined in the original papers.
4.1. Image Classification
Our evaluation of the proposed method for image clas-
sification involves using the ImageNet dataset [42], which
consists of 1.28 million training images and 50,000 vali-
dation images across 1,000 classes. We train the networks
from scratch and report the top-1 accuracy achieved. The
results are presented and compared in Table 2, where we
assess the performance of our modified batch normalization
on various model sizes from the ResNet [15], EfficientNet
[48], and GENet [31] families, as well as the SwinV2 [32]
model with layer normalization.
Across the ResNet models, we observe a maximum
accuracy improvement of 1.3% for ResNet-18, while the
larger models show slightly lesser improvement. This trend
holds true for other network architectures as well. For
GENet-light, we achieve a maximum accuracy improve-
ment of 1.5%, and EfficientNet-B5 demonstrates a mini-
mum accuracy improvement of 0.9%. While it is true that
2045


<!-- page 6 -->

the gains are more pronounced for ResNet-18 or GENet-
light, it is important to note that our method also yields
steady improvements on larger networks, demonstrating its
broad applicability. It is generally accepted that larger
networks, which already attain higher accuracy levels and
reach performance saturation, exhibit less incremental im-
provement when more regularizers or data augmentation
techniques are introduced. This behavior is consistent with
that of the JSNorm, which functions as a regularizer and
is inherently similar to other regularizers. The smaller im-
provements observed in larger networks can be attributed to
the presence of existing regularizers, which can limit the ad-
ditional boost provided by introducing another regularizer.
This is one of the motivations for performing the Regular-
ization Effect ablation study (Section 5.2).
The findings in this section highlight the compatibility
of our proposed JSNorm with both convolutional and trans-
former networks, indicating its effectiveness in improving
model performance.
4.2. Semantic Segmentation
The Cityscapes dataset [8] serves as our primary eval-
uation dataset for semantic segmentation, consisting of
5,000 high-quality street images with pixel-level annota-
tions. These finely annotated images are divided into sub-
sets of 2,975 for training, 500 for validation, and 1,525
for testing. Additionally, the dataset includes an additional
20,000 coarsely annotated images. It contains 30 classes,
with 19 classes used for performance assessment.
Our experiments involve the use of HRNetV2 [54] and
its augmented version HRNetV2+OCR [64], as well as Ef-
ficientPS [36] and Lawin [61]. HRNetV2 is a fully con-
volutional network that maintains high-resolution represen-
tations throughout the network. EfficientPS employs Effi-
cientNet as a backbone for semantic and panoptic segmen-
tation. Meanwhile, Lawin is a multi-scale transformer net-
work that utilizes a window attention mechanism. In our
evaluation, we replace the normalization layers in the afore-
mentioned networks with our proposed normalization layer
and compare the results.
Table 3 presents the mean intersection over union
(mIoU) measure for these models on the Cityscapes dataset.
Our improved batch normalization demonstrates notable
enhancements for HRNetV2, achieving an improvement of
1.4%. Additionally, we observe a minimum improvement
of 1.0%, which boosts Lawin’s accuracy to reach 85.4% on
this dataset. These results showcase the efficacy of our pro-
posed normalization approach in improving the segmenta-
tion performance across different models.
4.3. 3D Object Classification
A 3D point cloud, which comprises an unordered collec-
tion of 3D points, necessitates distinct network architectures
Method Original Ridge LASSO
ResNet-18 [15] 69.7 70.1 (+0.4) 70.1 (+0.4)
ResNet-152 [15] 77.9 78.0 (+0.1) 78.1 (+0.2)
EfficientNet-B1 [48] 79.2 79.4 (+0.2) 79.5 (+0.3)
EfficientNet-B5 [48] 83.7 83.7 (+0.0) 83.8 (+0.1)
GENet-light [31] 75.7 76.0 (+0.3) 76.0 (+0.3)
SwinV2-T [32] 81.8 81.9 (+0.1) 81.9 (+0.1)
SwinV2-S [32] 83.7 83.7 (+0.0) 83.7 (+0.0)
Table 5. Comparative analysis of two widely used shrinkage esti-
mators: Ridge and LASSO. The figures represent the Top-1 accu-
racy on the ImageNet dataset.
compared to those tailored for 2D images. This, in turn,
creates an alternative platform for evaluation. We have con-
ducted experiments utilizing two datasets tailored for 3D
object classification. The first dataset, ScanObjectNN [52],
is derived from real 3D scenes, and its inherent complex-
ity, augmented by occlusions and noise, poses substantial
challenges for prevailing 3D classification methodologies.
ScanObjectNN encompasses 2,309 training and 581 testing
point clouds, distributed across 15 object classes. The sec-
ond dataset, ModelNet40 [59], is widely recognized in the
realm of 3D object classification and comprises synthetic
object point clouds. The dataset contains 12,311 CAD-
generated meshes categorized into 40 classes, and is parti-
tioned into 9,843 training and 2,468 testing samples. We as-
sessed the efficacy of our proposed technique on five mod-
els, four incorporating batch normalization - PointNet [39],
PointNet++ [40], DGCNN [55], PointNeXt [41] - and one
employing layer normalization, namely Point-BERT [63].
Table 4 presents the classification accuracy for both
ScanObjectNN and ModelNet40 datasets. Remarkably, by
employing our enhanced JSNorm, PointNet attains a sub-
stantial improvement in classification accuracy, increasing
by 1.8% on the ScanObjectNN dataset. Even at the lower
end of the spectrum, Point-BERT on ModelNet40 exhibits a
noteworthy enhancement in accuracy by 1.0%. These find-
ings underscore the versatility of our proposed approach,
indicating that its application extends beyond 2D image net-
works and is adaptable to a diverse array of data formats and
network architectures.
5. Extra Studies
In addition to our primary study, we conduct an addi-
tional study that compares two alternative shrinkage esti-
mators. Furthermore, we carry out three ablation studies to
gain insights into the impact of regularization, shrinkage,
and batch size on the performance of our proposed method.
2046


<!-- page 7 -->

66
68
70
72
74
76
78
10% 30% 50% 70% 90%
Accuracy
Regularization
Original JSNorm
Figure 2. Comparative analysis of our enhanced batch normal-
ization performance across various regularization intensities. JS-
Norm not only boosts accuracy but also exhibits increased robust-
ness, particularly under lower regularization.
5.1. Ridge and LASSO Estimators
In this study, we assess two widely utilized shrinkage es-
timators, namely Ridge [16] and LASSO [50], to determine
whether alternative shrinkage estimators possess the capac-
ity to enhance accuracy. In terms of the Ridge estimator, ˆθ
can be estimated via:
ˆθRidge = argmin
ˆθ
h
ℓ( ˆθ, θ) + λ∥ ˆθ∥2
2
i
, (12)
and in terms of LASSO:
ˆθLASSO = argmin
ˆθ
h
ℓ( ˆθ, θ) + λ∥ ˆθ∥1
i
, (13)
where λ is called the regularization parameter and controls
the amount of shrinkage. Both the Ridge and LASSO es-
timators perform regularization of the estimated parame-
ters, with the LASSO estimator offering the added benefit of
variable selection, which enhances the interpretability of the
model. Both estimators also introduce shrinkage towards 0
as part of their regularization process.
We incorporate the Ridge and LASSO estimators as reg-
ularization components acting upon the estimated mean and
variance within batch normalization, a methodology that
can be seamlessly extended to layer normalization as well.
It is imperative to highlight that there is no alteration to the
original formulas of batch normalization when employing
Ridge and LASSO estimators. To integrate these regular-
ization components, we modify the training loss function in
the following manner:
ℓf inal = ℓoriginal + λ
X
f(µB) + f(σ2
B), (14)
Method 32 64 128 256 512 1024
Original 75.1 75.4 75.6 75.7 75.6 75.5
JSNorm 76.8 77.0 77.1 77.1 77.0 76.9
∆ +1.7 +1.6 +1.5 +1.4 +1.4 +1.4
Table 6. Impact of varying training batch sizes on ImageNet accu-
racy. Our JS batch normalization improves accuracy and demon-
strates superior robustness across different batch sizes.
whereP is summation over all batch normalization layers
and f is the regularization term, ∥ · ∥2
2 for Ridge and ∥ · ∥1
for LASSO. Since the value of ℓoriginal can be very much
larger or smaller thanP f(µB) + f(σ2
B) for different tasks
and networks, λ needs to be tuned accordingly. To solve
this problem, we re-scale λ proportionally to the value of
the regularization part and ℓoriginal:
λ = λoriginal( ℓoriginalP f(µB) + f(σ2
B)). (15)
This way, λoriginal is the hyper-parameter that should be
chosen. Re-scaling λ happens outside the computational
graph to prevent it from affecting gradient calculation.
We subject the regularized normalization layers to eval-
uation within the context of the image classification task.
Table 5 presents the top-1 accuracy on the ImageNet dataset
for several distinct networks. While Ridge and LASSO
contribute to enhanced performance, they do not match the
level of improvement achieved by the James-Stein estima-
tor. Given that the James-Stein estimator is predicated on
the assumption of a Gaussian distribution underlying the
data, this may account for its superior performance rela-
tive to Ridge and LASSO. In our experimental assessments,
LASSO marginally outperforms Ridge in terms of accu-
racy, though their performances are largely analogous. The
results underscore the capability of various shrinkage esti-
mators to bolster accuracy, albeit not to the same extent as
James-Stein.
5.2. Regularization Effect
In the scholarly domain, it is well-established that
shrinkage estimators inherently exhibit regularization ef-
fects [16, 50]. In this investigation, we illuminate the per-
formance characteristics of our novel JSNorm layers across
an array of regularization magnitudes. For this purpose, we
deploy three regularization techniques, namely RandAug-
ment [9], stochastic depth [18], and dropout [46]. We
initiate the experiment with a baseline configuration using
GENet-light [31] and progressively escalate the regulariza-
tion factors. Guided by the methodologies delineated in [9]
and [49], we establish the upper bounds for regularization
factors. The combined regularization is represented in per-
centage to foster clarity and streamline the visualisation.
2047


<!-- page 8 -->

/uni00000014/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000018/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000013
/uni00000015/uni00000018
/uni00000018/uni00000013
/uni0000001a/uni00000018
/uni00000014/uni00000013/uni00000013
/uni00000014/uni00000015/uni00000018
/uni00000014/uni00000018/uni00000013 /uni00000032/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni00000003/uni00000025/uni00000031
/uni00000032/uni00000058/uni00000055/uni00000003/uni0000002d/uni00000036/uni00000003/uni00000025/uni00000031
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b/uni00000014/uni00000011/uni00000013/uni00000013
/uni00000015/uni00000013/uni00000013
/uni00000017/uni00000013/uni00000013
/uni00000019/uni00000013/uni00000013
/uni0000001b/uni00000013/uni00000013
/uni00000014/uni00000013/uni00000013/uni00000013 /uni00000032/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni00000003/uni00000025/uni00000031
/uni00000032/uni00000058/uni00000055/uni00000003/uni0000002d/uni00000036/uni00000003/uni00000025/uni00000031
Figure 3. The shrinkage effect of JSNorm on ResNet-18 [15]. The figures illustrate the distribution of running means (left) and running
variances (right) in the original batch normalization compared to JSNorm, showcasing the shrinkage effect induced by our JSNorm layer.
We train the network under two scenarios – one employing
our enhanced batch normalization and the other without.
Figure 2 presents a graphical representation that com-
pares the two approaches across a range of regularization
factors. An intriguing observation is that our method ex-
hibits its most pronounced enhancement in scenarios char-
acterized by low regularization. Nonetheless, even in sce-
narios with maximal regularization, our method registers
noteworthy accuracy improvements. This exemplifies the
robustness and stability of our proposition across a spec-
trum of regularization parameters, thereby outclassing the
conventional approach.
5.3. Shrinkage Effect
Shrinkage estimators exert a shrinkage effect on the esti-
mated parameters. We are keen to analyze how this shrink-
age influences the normalization layers. To accomplish this,
we provide a visual comparison of the distribution of the
running statistics within the batch normalization layers of
a ResNet-18 model [15]. Two networks are trained, one
employing the standard batch normalization and the other
utilizing our enhanced batch normalization.
Figure 3 offers a graphical representation of the his-
tograms of all the running means (left subfigure) and vari-
ances (right subfigure), both with and without the incorpo-
ration of our JSNorm. The subfigures illuminate how our
JSNorm layer nudges the distributions toward zero. Pertain-
ing to the running means, the employment of our JSNorm
facilitates a distribution that is less skewed and more bell-
shaped. As for the running variances, directing the distribu-
tion towards zero does not hinder the network’s capability
to learn layers with elevated variances. In essence, it con-
tributes positively to the diversity of the distribution.
5.4. Batch Size Effect
Batch size plays a pivotal role in the training process
of networks that incorporate batch normalization layers.
Given that our method enhances the estimation of normal-
ization statistics, it is intriguing to investigate its perfor-
mance across varying batch sizes. This study is designed to
elucidate the behavior of our James-Stein-augmented batch
normalization in the context of different batch sizes. To
achieve this, we train a GENet-light [31] network on the
ImageNet dataset using a range of batch sizes, spanning
from 32 to 1024. We employ the linear learning rate scaling
rule [13] to adjust to the alterations in batch size.
Table 6 presents the findings of this investigation. Our
enhanced batch normalization exhibits optimal performance
with smaller batch sizes, attributable to the James-Stein es-
timator’s proficiency in estimating normalization statistics
with a limited sample pool. Notably, even with larger batch
sizes, our method surpasses the performance of the orig-
inal batch normalization by a considerable margin. Con-
sequently, our JSNorm demonstrates not only an enhance-
ment in accuracy but also exhibits robustness with respect
to batch size variations.
6. Conclusion
Through the lens of Stein’s paradox, we illustrated that
normalization layers employ inadmissible estimators, re-
sulting in suboptimal estimations of layer statistics. To ad-
dress this issue, we introduced an innovative technique uti-
lizing the James-Stein estimator, which enhances the accu-
racy of mean and variance estimations. We evaluated our
proposed method rigorously across three distinct computer
vision tasks. Our findings demonstrated that the technique
not only bolstered the accuracy of both convolutional and
transformer networks but did so without incurring addi-
tional computational overhead. We performed extra stud-
ies to unveil that our approach exhibits robustness and is
less susceptible to changes in batch size and regularization,
leading to consistent improvements in accuracy under di-
verse configurations.
2048


<!-- page 9 -->

References
[1] Marko Angjelichinoski, Mohammadreza Soltani, John Choi,
Bijan Pesaran, and Vahid Tarokh. Deep pinsker and james-
stein neural networks for decoding motor intentions from
limited data. IEEE Transactions on Neural Systems and Re-
habilitation Engineering, 29:1058–1067, 2021. 3
[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hin-
ton. Layer normalization. arXiv preprint arXiv:1607.06450,
2016. 2, 3
[3] Nils Bjorck, Carla P Gomes, Bart Selman, and Kilian Q
Weinberger. Understanding batch normalization. Advances
in neural information processing systems, 31, 2018. 2
[4] Andrew Brock, Soham De, and Samuel L Smith. Character-
izing signal propagation to close the performance gap in un-
normalized resnets. arXiv preprint arXiv:2101.08692, 2021.
5
[5] Andy Brock, Soham De, Samuel L Smith, and Karen Si-
monyan. High-performance large-scale image recognition
without normalization. In International Conference on Ma-
chine Learning, pages 1059–1071. PMLR, 2021. 5
[6] Lawrence D Brown. Admissible estimators, recurrent diffu-
sions, and insoluble boundary value problems. The Annals
of Mathematical Statistics, 42(3):855–903, 1971. 1
[7] Rudrasis Chakraborty, Yifei Xing, Minxuan Duan, and
Stella X Yu. C-sure: Shrinkage estimator and prototype clas-
sifier for complex-valued deep learning. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition Workshops, pages 80–81, 2020. 3
[8] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo
Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe
Franke, Stefan Roth, and Bernt Schiele. The cityscapes
dataset for semantic urban scene understanding. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 3213–3223, 2016. 6
[9] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V
Le. Randaugment: Practical automated data augmen-
tation with a reduced search space. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition workshops, pages 702–703, 2020. 7
[10] Bradley Efron and Carl Morris. Data analysis using stein’s
estimator and its generalizations. Journal of the American
Statistical Association, 70(350):311–319, 1975. 4
[11] Brian S Everitt and Anders Skrondal. The cambridge dictio-
nary of statistics, 2010. 1
[12] Dominique Fourdrinier, William E Strawderman, and Mar-
tin T Wells. Shrinkage estimation. Springer, 2018. 2
[13] Priya Goyal, Piotr Doll ´ar, Ross Girshick, Pieter Noord-
huis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch,
Yangqing Jia, and Kaiming He. Accurate, large mini-
batch sgd: Training imagenet in 1 hour. arXiv preprint
arXiv:1706.02677, 2017. 8
[14] Marvin HJ Gruber. Improving efficiency by shrinkage: the
James-Stein and ridge regression estimators . Routledge,
2017. 2
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 4, 5, 6, 8
[16] Arthur E Hoerl and Robert W Kennard. Ridge regression:
Biased estimation for nonorthogonal problems. Technomet-
rics, 12(1):55–67, 1970. 2, 7
[17] Parisa Hosseini, Seyedalireza Khoshsirat, Mohammad
Jalayer, Subasish Das, and Huaguo Zhou. Application of
text mining techniques to identify actual wrong-way driving
(wwd) crashes in police reports. International Journal of
Transportation Science and Technology, 2022. 2
[18] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kil-
ian Q Weinberger. Deep networks with stochastic depth. In
European conference on computer vision , pages 646–661.
Springer, 2016. 7
[19] Lei Huang, Jie Qin, Yi Zhou, Fan Zhu, Li Liu, and
Ling Shao. Normalization techniques in training dnns:
Methodology, analysis and application. arXiv preprint
arXiv:2009.12836, 2020. 2
[20] Lei Huang, Dawei Yang, Bo Lang, and Jia Deng. Decorre-
lated batch normalization. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
791–800, 2018. 3
[21] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal co-
variate shift. In International conference on machine learn-
ing, pages 448–456. PMLR, 2015. 1, 2, 3
[22] W James and C Stein. Estimation with quadratic loss. vol-
ume 1 of proc. fourth berkeley symp. on math. statist. and
prob, 1961. 4
[23] William James and Charles Stein. Estimation with quadratic
loss. In Breakthroughs in statistics, pages 443–460. Springer,
1992. 1, 4
[24] Seyedalireza Khoshsirat and Chandra Kambhamettu. Se-
mantic segmentation using neural ordinary differential equa-
tions. In Advances in Visual Computing: 17th International
Symposium, ISVC 2022, San Diego, CA, USA, October 3–5,
2022, Proceedings, Part I, pages 284–295. Springer, 2022. 1
[25] Seyedalireza Khoshsirat and Chandra Kambhamettu. Em-
bedding attention blocks for the vizwiz answer grounding
challenge. VizWiz Grand Challenge Workshop, 2023. 2
[26] Seyedalireza Khoshsirat and Chandra Kambhamettu. Em-
powering visually impaired individuals: A novel use of apple
live photos and android motion photos. In25th Irish Machine
Vision and Image Processing Conference, 2023. 2
[27] Seyedalireza Khoshsirat and Chandra Kambhamettu. Sen-
tence attention blocks for answer grounding. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 6080–6090, 2023. 2
[28] Seyedalireza Khoshsirat and Chandra Kambhamettu. A
transformer-based neural ode for dense prediction. Machine
Vision and Applications, 34(6):1–11, 2023. 2
[29] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
Imagenet classification with deep convolutional neural net-
works. Communications of the ACM, 60(6):84–90, 2017. 1,
2
[30] Olivier Ledoit and Michael Wolf. A well-conditioned esti-
mator for large-dimensional covariance matrices. Journal of
multivariate analysis, 88(2):365–411, 2004. 2
2049


<!-- page 10 -->

[31] Ming Lin, Hesen Chen, Xiuyu Sun, Qi Qian, Hao Li, and
Rong Jin. Neural architecture design for gpu-efficient net-
works. arXiv preprint arXiv:2006.14090, 2020. 4, 5, 6, 7,
8
[32] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie,
Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al.
Swin transformer v2: Scaling up capacity and resolution. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 12009–12019, 2022. 4,
5, 6
[33] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng
Zhang, Stephen Lin, and Baining Guo. Swin transformer:
Hierarchical vision transformer using shifted windows. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 10012–10022, 2021. 2, 5
[34] Elham Maserat, Reza Safdari, Hamid Asadzadeh Aghdaei,
Alireza Khoshsirat, and Mohammad Reza Zali. 43: Design-
ing evidence based risk assessment system for cancer screen-
ing as an applicable approach for the estimating of treatment
roadmap. BMJ Open, 7(Suppl 1):bmjopen–2016, 2017. 1
[35] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and
Yuichi Yoshida. Spectral normalization for generative ad-
versarial networks. arXiv preprint arXiv:1802.05957, 2018.
2
[36] Rohit Mohan and Abhinav Valada. Efficientps: Efficient
panoptic segmentation. International Journal of Computer
Vision, 129(5):1551–1579, 2021. 4, 6
[37] F Mosteller. Data analysis, including statistics, the collected
works of john w. tukey: Philosophy and principles of data
analysis 1965–1986., 1987. 1
[38] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and
Peter Kontschieder. The mapillary vistas dataset for semantic
understanding of street scenes. In Proceedings of the IEEE
international conference on computer vision , pages 4990–
4999, 2017. 4
[39] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas.
Pointnet: Deep learning on point sets for 3d classification
and segmentation. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 652–660,
2017. 5, 6
[40] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J
Guibas. Pointnet++: Deep hierarchical feature learning on
point sets in a metric space. Advances in neural information
processing systems, 30, 2017. 5, 6
[41] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai,
Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, and
Bernard Ghanem. Pointnext: Revisiting pointnet++ with
improved training and scaling strategies. arXiv preprint
arXiv:2206.04670, 2022. 5, 6
[42] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-
jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
Aditya Khosla, Michael Bernstein, et al. Imagenet large
scale visual recognition challenge. International journal of
computer vision, 115(3):211–252, 2015. 5
[43] Tim Salimans and Durk P Kingma. Weight normalization:
A simple reparameterization to accelerate training of deep
neural networks. Advances in neural information processing
systems, 29, 2016. 2, 3
[44] Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and
Aleksander Madry. How does batch normalization help op-
timization? Advances in neural information processing sys-
tems, 31, 2018. 2
[45] Pierre Sermanet, David Eigen, Xiang Zhang, Micha ¨el Math-
ieu, Rob Fergus, and Yann LeCun. Overfeat: Integrated
recognition, localization and detection using convolutional
networks. arXiv preprint arXiv:1312.6229, 2013. 2
[46] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya
Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way
to prevent neural networks from overfitting. The journal of
machine learning research, 15(1):1929–1958, 2014. 7
[47] C Stein. Inadmissibility of the usual estimator for the mean
of a multivariate distribution, vol. 1, 1956. 2, 4
[48] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model
scaling for convolutional neural networks. In International
conference on machine learning, pages 6105–6114. PMLR,
2019. 4, 5, 6
[49] Mingxing Tan and Quoc Le. Efficientnetv2: Smaller models
and faster training. In International Conference on Machine
Learning, pages 10096–10106. PMLR, 2021. 7
[50] Robert Tibshirani. Regression shrinkage and selection via
the lasso. Journal of the Royal Statistical Society: Series B
(Methodological), 58(1):267–288, 1996. 2, 7
[51] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. In-
stance normalization: The missing ingredient for fast styliza-
tion. arXiv preprint arXiv:1607.08022, 2016. 2
[52] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua,
Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud
classification: A new benchmark dataset and classification
model on real-world data. In Proceedings of the IEEE/CVF
international conference on computer vision , pages 1588–
1597, 2019. 6
[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. Advances in neural
information processing systems, 30, 2017. 2, 5
[54] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang,
Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui
Tan, Xinggang Wang, et al. Deep high-resolution repre-
sentation learning for visual recognition. arXiv preprint
arXiv:1908.07919, 2019. 4, 6
[55] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma,
Michael M Bronstein, and Justin M Solomon. Dynamic
graph cnn for learning on point clouds. Acm Transactions
On Graphics (tog), 38(5):1–12, 2019. 5, 6
[56] Wikipedia. Shrinkage (statistics). Wikipedia, Sep 2021. 1
[57] Wikipedia. James–stein estimator. Wikipedia, 2023. 2
[58] Yuxin Wu and Kaiming He. Group normalization. In Pro-
ceedings of the European conference on computer vision
(ECCV), pages 3–19, 2018. 2, 3
[59] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Lin-
guang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d
shapenets: A deep representation for volumetric shapes. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 1912–1920, 2015. 6
2050


<!-- page 11 -->

[60] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and
Junyang Lin. Understanding and improving layer normaliza-
tion. Advances in Neural Information Processing Systems ,
32, 2019. 3
[61] Haotian Yan, Chuang Zhang, and Ming Wu. Lawin trans-
former: Improving semantic segmentation transformer with
multi-scale representations via large window attention.arXiv
preprint arXiv:2201.01615, 2022. 4, 6
[62] Junjie Yan, Ruosi Wan, Xiangyu Zhang, Wei Zhang, Yichen
Wei, and Jian Sun. Towards stabilizing batch statistics in
backward propagation of batch normalization.arXiv preprint
arXiv:2001.06838, 2020. 3
[63] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie
Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud
transformers with masked point modeling. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 19313–19322, 2022. 5, 6
[64] Yuhui Yuan, Xilin Chen, and Jingdong Wang. Object-
contextual representations for semantic segmentation. In
European conference on computer vision , pages 173–190.
Springer, 2020. 4, 6
[65] Matthew D Zeiler and Rob Fergus. Visualizing and under-
standing convolutional networks. InEuropean conference on
computer vision, pages 818–833. Springer, 2014. 2
[66] Xiao-Yun Zhou, Jiacheng Sun, Nanyang Ye, Xu Lan, Qi-
jun Luo, Bo-Lin Lai, Pedro Esperanca, Guang-Zhong Yang,
and Zhenguo Li. Batch group normalization. arXiv preprint
arXiv:2012.02782, 2020. 3
[67] Hui Zou and Trevor Hastie. Regularization and variable se-
lection via the elastic net. Journal of the royal statistical
society: series B (statistical methodology) , 67(2):301–320,
2005. 2
2051
