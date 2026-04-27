# references/67_distributed_indirect_source_coding_with_decoder_side_information.pdf

<!-- page 1 -->

Distributed Indirect Source Coding with Decoder
Side Information
Jiancheng Tang∗, Qianqian Yang ∗, Deniz G ¨und¨uz†
∗College of information Science and Electronic Engineering, Zhejiang University, Hangzhou 310007, China
∗Email: {jianchengtang, qianqianyang20}@zju.edu.cn
†Department of Electrical and Electronic Engineering Imperial College London London, UK
†Email:d.gunduz@imperial.ac.uk
Abstract—This paper studies a variant of the rate-distortion
problem motivated by task-oriented semantic communication and
distributed learning problems, where M correlated sources are
independently encoded for a central decoder . The decoder has
access to a correlated side information in addition to the messages
received from the encoders, and aims to recover a latent random
variable correlated with the sources observed by the encoders
within a given distortion constraint rather than recovering the
sources themselves. We provide bounds on the rate-distortion
region for this scenario in general, and characterize the rate-
distortion function exactly when the sources are conditionally
independent given the side information.
Index Terms —Semantic communication, distributed source
coding, rate-distortion theory, side information.
I. I NTRODUCTION
Consider the multiterminal source coding setup as shown
in Fig. 1. Let (T, X1, ..., XM , Y ) ∼ p(t, x1, ..., xM , y) be a
discrete memoryless source (DMS) taking values in the finite
alphabets T × X 1 × · · · × X M × Y according to a fixed and
known probability distribution p(t, x1, ..., xM , y). In this setup,
the encoder m, m ∈ M := {1, ..., M} has local observations
X n
m := (X n
1 , . . . , Xn
M ). The agents independently encode their
observations into binary sequences at rates {R1, . . . , RM } bits
per input symbol, respectively. The decoder with side infor-
mation Y n = (Y1, . . . , Yn) aims to recover some task-oriented
latent information T n := (T1, . . . , Tn) which is correlated with
(X n
1 , . . . , Xn
M ), but it is not observed directly by any of the
encoders. We are interested in the lossy reconstruction of T n
with the average distortion measured by E
h
1
n
Pn
i=1 d(Ti, ˆTi)
i
,
for some prescribed single-letter distortion measure d(·, ·). A
formal (2nR1 , ..., 2nRM , n) rate-distortion code for this setup
consists of
• M independent encoders, where encoder m ∈ M assigns
an index sm(xn
m) ∈

1, . . . ,2nRm

to each sequence
xn
m ∈ X n
m;
• a decoder that produces the estimate ˆtn(s1, ..., sM , yn) ∈
T n to each index tuple (s1, ..., sM ) and side information
yn ∈ Y n.
*This work is partly supported by NSFC under grant No. 62293481, No.
62201505, partly by the SUTD-ZJU IDEA Grant (SUTD-ZJU (VP) 202102).
Fig. 1. Distributed remote compression of a latent variable by M correlated
sources with side information at the receiver.
A rate tuple (R1, ..., RM ) is said to be achievable with the
distortion measure d(·, ·) and the distortion value D if there
exists a sequence of (2nR1 , ..., 2nRM , n) codes that satisfy
lim sup
n→∞
E
"
1
n
nX
i=1
d(Ti, ˆTi)
#
≤ D. (1)
The rate-distortion region R∗
X1,...,Xm|Y (D) for this dis-
tributed source coding problem is the closure of the set
of all achievable rate tuples (R1, . . . , RM ) that permit the
reconstruction of the latent variable T n within the average
distortion constraint D.
The problem as illustrated in Fig. 1 is motivated by se-
mantic/ task-oriented communication and distributed learning
problems. In semantic/task-oriented communication, the de-
coder only needs to reconstruct some task-oriented information
implied by the sources. For instance, it might extract hidden
features from a scene captured by multiple cameras positioned
at various angles. Here, Ti may also be a deterministic function
of the source samples (X1,i, . . . , XM,i), which then reduces to
the problem of lossy distributed function computation [1]–[3].
A similar problem also arises in distributed training. Consider
Y n as the global model available at the server at an iteration
of a federated learning process, and (X n
1 , . . . , Xn
M ) as the
independent correlated versions of this model after downlink
transmission and local training. The server aims to recover
the updated global model, T n, based on the messages received
from all M clients. It is often assumed that the global model is
arXiv:2405.13483v1  [cs.IT]  22 May 2024

<!-- page 2 -->

transmitted to the clients intact, but in practical scenarios where
downlink communication is limited, the clients may receive
noisy or compressed versions of the global model [4]–[6].
For the case of M = 1 , the considerd problem reduces to
the remote compression in a point-to-point scenario with side
information available at the decoder. In [7], [8], the authors
studied this problem without the correlated side information
at the receiver, motivated in the context of semantic com-
munication. This problem is known in the literature as the
remote rate-distortion problem [9], [10], and the rate-distortion
trade-off is fully characterized in the general case. The authors
studied this trade-off in detail for specific source distributions
in [7]. Similarly, the authors of [11] characterized the remote
rate-distortion trade-off when correlated side information is
available both at the encoder and decoder. Our problem for
M = 1 can be solved by combining the remote rate-distortion
problem with the classical Wyner-Ziv rate-distortion function
[12], [13].
The rate-distortion region for the multi-terminal version of
the remote rate-distortion problem considered here remains
open. Sung et al. proposed an achievable rate region for the
distributed lossy computation problem, but no conclusive rate-
distortion function can be given [14]. Gwanmoet al. considered
a special case in which the sources are independent and derived
a single-letter expression for the rate-distortion region [15].
Gastpar [16] considered the lossy compression of the source
sequences in the presence of side information at the receiver.
He characterized the rate-distortion region for the special case,
in which Xi’s are conditionally independent given the side
information.
In this paper, we are interested in the rate-distortion region
R∗
X1,...,Xm|Y (D) for the general problem. We will pay par-
ticular attention to the special case in which the sources are
conditionally independent given the side information, moti-
vated by the aforementioned examples. For the sake of brevity
of the presentation, we set M = 3 in this paper, with the
understanding that the results can be readily extended to an
arbitrary number of sources.
In Section II, we derive an achievable region Ra (D) ⊆
R∗
X1,X2,X3|Y (D). In Section III, we determine a general outer
bound Ro (D) ⊇ R∗
X1,X2,X3|Y (D). In Section IV , we show
that the two regions coincide and the region is optimal when
the sources (X1, X2, X3) are conditionally independent given
the side information Y .
II. A N ACHIEVABLE RATE REGION
In this section, we introduce an achievable rate region
Ra (D), which is contained within the goal rate-distortion
region Ra (D) ⊆ R∗
X1,X2,X3|Y (D).
Theorem 1: Ra (D) ⊆ R∗
X1,X2,X3|Y (D), where
Ra (D) is the set of all rate tuples (R1, R2, R3) such
that there exists a tuples (W1, W2, W3) of discrete
random variables with p (w1, w2, w3, x1, x2, x3, y) =
p (w1|x1) p (w2|x2) p (w3|x3) p (x1, x2, x3, y), for which the
following conditions are satisfied
R1 ⩾ I (X1; W1) − I (W1; W2, W3, Y ) (2a)
R2 ⩾ I (X2; W2) − I (W2; W1, W3, Y ) (2b)
R3 ⩾ I (X3; W3) − I (W3; W1, W2, Y ) (2c)
R1 + R2 ⩾ I (X1; W1) + I (X2; W2) − I (W1; W2, W3, Y )
− I (W2; W1, W3, Y ) + I (W1; W2|W3, Y ) (2d)
R1 + R3 ⩾ I (X1; W1) + I (X3; W3) − I (W1; W2, W3, Y )
− I (W3; W1, W2, Y ) + I (W1; W3|W2, Y ) (2e)
R2 + R3 ⩾ I (X2; W2) + I (X3; W3) − I (W2; W1, W3, Y )
− I (W3; W1, W2, Y ) + I (W2; W3|W1, Y ) , (2f)
R1 + R2 + R3 ⩾ I (X1; W1) + I (X2; W2) + I (X3; W3)
− I (W1; W2, W3, Y ) − I (W2; W1, W3, Y )
− I (W3; W1, W2, Y ) + I (W1; W2|W3, Y )
+ I (W1, W2; W3|Y ) , (2g)
and there exist a decoder g (·) such that
Ed(T, g(W1, W2, W3, Y )) ⩽ D. (3)
The rigorous proof of Theorem 1 is provided in Appendix
A.
Corollary 2: The conditions (19) of Theorem 1 can be
expressed equivalently as
R1 ⩾ I (X1, X2, X3; W1|W2, W3, Y ) (4a)
R2 ⩾ I (X1, X2, X3; W2|W1, W3, Y ) (4b)
R3 ⩾ I (X1, X2, X3; W3|W1, W2, Y ) (4c)
R1 + R2 ⩾ I (X1, X2, X3; W1, W2|W3, Y ) (4d)
R1 + R3 ⩾ I (X1, X2, X3; W1, W3|W2, Y ) (4e)
R2 + R3 ⩾ I (X1, X2, X3; W2, W3|W1, Y ) (4f)
R1 + R2 + R3 ⩾ I(X1, X2, X3; W1, W2, W3|Y ) (4g)
Proof: First we prove R1 ⩾ I (X1; W1) −
I (W1; W2, W3, Y ) = I (X1, X2, X3; W1|W2, W3, Y ).
The bound of (4a) can be written as
I (X1, X2, X3; W1|W2, W3, Y )
= I (X1; W1|W2, W3, Y ) + I (X2, X3; W1|X1, W2, W3, Y )| {z }
=0
,
(5)
where I (X2, X3; W1|X1, W2, W3, Y ) = 0 because
(X2, X3, Y ) is conditionally independent of W1 for given X1.
For the first term of the right in (5), we have
I (X1; W1|W2, W3, Y ) + I (W1; W2, W3, Y )
= I (W1; X1, W2, W3, Y )
= I (W1; X1) + I (W2, W3, Y ; W1|X1)| {z }
=0
,
(6)
where I (W2, W3, Y ; W1|X1) = 0 because (W2, W3, Y ) is
conditionally independent of W1 given X1. Then we have
I (X1, X2, X3; W1|W2, W3, Y )
= I (X1; W1|W2, W3, Y )
= I (W1; X1) − I (W1; W2, W3, Y )
≤ R1.
(7)

<!-- page 3 -->

This completes the proof of (4a). (4b) and (4c) can be proved in
the same way. The rigorous proof of the rest sum rate bounds
will be provided in a longer version.
III. A N GENERAL OUTER BOUND
In this section, we derive a region Ro (D) which contains
the goal rate-distortion region Ro (D) ⊇ R∗
X1,X2,X3|Y (D).
Theorem 3: Ro (D) ⊇ R∗
X1,X2,X3|Y (D), where Ro (D)
is the set of all rate triples (R1, R2, R3) such that there
exists a triple (W1, W2, W3) of discrete random variables
with p (w1|x1, x2, x3, y) = p (w1|x1), p (w2|x1, x2, x3, y) =
p (w2|x2) and p (w3|x1, x2, x3, y) = p (w3|x3), for which the
following conditions are satisfied
R1 ⩾ I (X1, X2, X3; W1|W2, W3, Y )
R2 ⩾ I (X1, X2, X3; W2|W1, W3, Y )
R3 ⩾ I (X1, X2, X3; W3|W1, W2, Y )
R1 + R2 ⩾ I (X1, X2, X3; W1, W2|W3, Y )
R1 + R3 ⩾ I (X1, X2, X3; W1, W3|W2, Y )
R2 + R3 ⩾ I (X1, X2, X3; W2, W3|W1, Y )
R1 + R2 + R3 ⩾ I(X1, X2, X3; W1, W2, W3|Y )
(8)
and there exist a decoding function g (·) such that
Ed(T, g1(W1, W2, W3, Y )) ⩽ D, (9)
The rigorous proof of Theorem 3 is provided in Appendix
B.
While the expressions of the inner bound (4) and the outer
bound (8) are the same, these two regions do not coincide
because the marginal constrains p (w1, w2, w3, x1, x2, x3, y) =
p (w1|x1) p (w2|x2) p (w3|x3) p (x1, x2, x3, y) in Theorem 1
limit the degree of freedom for choosing the auxiliary random
variables (W1, W2, W3) compared with the marginal constrains
in Theorem 3. In the next section, we will demonstrate that the
additional degree of freedom in choosing the auxiliary random
variables (W1, W2, W3) in Theorem 3 cannot lower the value
of the rate-distortion functions.
IV. CONCLUSIVE RATE -DISTORTION RESULTS
Corollary 4: If X1, X2, X3 are conditionally independent
given the side information Y , Ra (D) ⊆ R∗
X1,X2,|Y (D)
where Ra (D) is the set of all rate triples (R1, R2, R3)
such that there exists a triple (W1, W2, W3) of
random variables with p (w1, w2, w3, x1, x2, x3, y) =
p (w1|x1) p (w2|x2) p (w3|x3) p (x1|y) p (x2|y) p (x3|y) p (y),
for which the following conditions are satisfied
R1 ⩾ I (X1; W1) − I (W1; Y )
R2 ⩾ I (X2; W2) − I (W2; Y )
R3 ⩾ I (X3; W3) − I (W3; Y )
(10)
and there exist decoding functions g (·) such that
Ed(T, g(W1, W2, W3, Y )) ⩽ D. (11)
Proof: Since the joint distribution can be written as
p (w1, w2, w3, x1, x2, x3, y)
= p (w1, w2, w3|x1, x2, x3, y) p (x1, x2, x3|y) p (y)
= p (w1|x1) p (w2|x2) p (w3|x3) p (x1|y)
× p (x2|y) p (x3|y) p (y) ,
(12)
the terms I (W1; W2|W3, Y ) in the sum rate bound (2d)
is 0 because W2 is conditionally independent of W1. Sim-
ilarly, the terms I (W1; W3|W2, Y ) , I (W2; W3|W1, Y ) and
I (W1; W2|W3, Y ) +I (W1, W2; W3|Y ) in the sum rate bound
(2e)-(2g) are all 0. Therefore, the sum rate bound can be
expressed as the combination of the side bounds, and hence
can be omitted. Meanwhile, The term I (W1; W2, W3, Y ) in
the side bound in (2a) can be written as
I (W1; W2, W3, Y ) = I (W1; Y ) + I (W2, W3; W1|Y )| {z }
=0
. (13)
Similarly, we have
I (W2; W1, W3, Y ) = I (W2; Y ) + I (W1, W3; W2|Y )| {z }
=0
I (W3; W1, W2, Y ) = I (W3; Y ) + I (W1, W2; W3|Y )| {z }
=0
. (14)
This completes the proof Corollary 4.
Corollary 5: If X1, X2, X3 are conditionally independent
given the side information Y , R
′
o (D) ⊇ Ro (D), and
hence R
′
o (D) ⊇ R∗
X1,X2,W3|Y (D) where R
′
o (D) is the
set of all rate triples (R1, R2, R3) such that there exists
a triple (W1, W2, W3) of discrete random variables with
p (w1|x1, x2, w3, y) = p (w1|x1), p (w2|x1, x2, w3, y) =
p (w2|x2) and p (w3|x1, x2, x3, y) = p (w3|x3), for which the
following conditions are satisfied
R1 ⩾ I (X1; W1) − I (W1; Y )
R2 ⩾ I (X2; W2) − I (W2; Y )
R3 ⩾ I (X3; W3) − I (W3; Y ) ,
(15)
and there exists decoding functions g (·) such that
Ed(T, g1(W1, W2, W3, Y )) ⩽ D, (16)
Proof: First, we can enlarge the region Ro (D) by omitting the
sum rate bound in (8). Then, the side rate bounds in (8) can
be relaxed as
R1 ⩾ I (X1, X2, X3; W1|W2, W3, Y )
⩾ I (X1; W1|W2, W3, Y ) + I (X2, X3; W1|W2, W3, Y, X1)
⩾ I (X1; W1|W2, W3, Y )
= I (X1; W1, W2, W3|Y ) − I (X1; W2, W3|Y )| {z }
=0
.
(17)
According to the conditional independence relations, we
have I (X1; W2, W3|Y ) = 0 , and then we have
R1 ⩾ I (X1; W1, W2, W3|Y )
= I (X1; W1|Y ) + I (X1; W2, W3|Y, W1)| {z }
=0
(18a)

<!-- page 4 -->

= I (X1, Y ; W1) − I (W1; Y ) (18b)
= I (X1; W1) − I (W1; Y ) (18c)
where (18a) is obtained by the condition that X1, X2, X3 are
conditionally independent given the side information Y , (18b)
follows from the chain rule of mutual information and (18c) is
derived by the Markov chain Y −X1−W1. The same derivation
can be applied to R2 and R3, this proves the corollary 5.
Theorem 6: If X1, X2, X3 are conditionally independent
given the side information Y ,
Ra (D) = Ro (D) = R∗
X1,X2,X3|Y (D) . (19)
Proof: We note that the only difference between Ra (D) and
Ro (D) is the degrees of freedom when choosing the auxiliary
random variables (W1, W2, W3), and all of the mutual infor-
mation functions in (10) and (15) only depend on the marginal
distribution (X1, W1, Y ), (X2, W2, Y ) and (X3, W3, Y ). Ran-
domly choose a certain rate triple (R1, R2, R3) with a auxiliary
random variable triple (W1, W2, W3) meeting the conditions
of Corollary 5, the corresponding joint distribution is given
in (12) Then we construct the auxiliary random variables
(W
′
1, W
′
2, W
′
3) such that
pW ′
1 |X1
(w1|s1) =
X
w2,s2,w3,s3
p (w1, w2, w3|s1, s2, s3)p (s2, s3|s1)
pW ′
2 |X2
(w2|s2) =
X
w1,s1,w3,s3
p (w1, w2, w3|s1, s2, s3)p (s1, s3|s2)
pW ′
3 |X3
(w3|s3) =
X
w1,s1,w2,s2
p (w1, w2, w3|s1, s2, s3)p (s1, s2|s3) .
(20)
The joint distribution
p

w
′
1, w
′
2, w
′
3, x1, x2, x3, Y

= p

w
′
1|x1

p

w
′
2|x2

p

w
′
3|x3

p (x1|y) p (x2|y) p (x3|y) p (y)
(21)
has the same marginal distributions on (X1, W1, Y ),
(X2, W2, Y ) and (X3, W3, Y ). Therefore, the additional de-
gree of freedom for choosing the auxiliary random variables
(W1, W2, W3) in Corollary 5 can not lower the value of rate-
distortion functions. This proves the Theorem 6. The arguments
leading to Theorem 6 indicates that the result extends to the
M sources scenario.
V. C ONCLUSION
A variant rate-distortion problem was studied in this paper.
We first derived an achievable rate-distortion region of this
problem, and subsequently, we derived a general outer bound
of the rate-distortion region. We show that the two regions
coincide and characterize the rate-distortion function exactly
under the scenario that the sources are conditionally indepen-
dent given the side information.
APPENDIX A
Here we provide the rigorous proof of Theorem 1.
Lemma 7 (Extended Markov Lemma) : Let
p (w1, w2, w3, x1, x2, x3, y)
= p (w1|x1) p (w2|x2) p (w3|x3) p (x1, x2, x3, y) . (22)
For a fixed (xn
1 , xn
2 , xn
3 yn) ∈ A∗(n)
ϵ , wn
1 , wn
2 and wn
3 are drown
from p(w1|x1) , p(w2|x2) and p(w3|x3), respectively. Then
lim
n→∞
P r{(wn
1 , wn
2 , wn
3 , xn
1 , xn
2 , xn
3 , yn) ∈ A∗(n)
ϵ } = 1. (23)
Proof: We can use the Markov Lemma of joint typicality
multiple times to prove this lemma, the details are omitted
here.
Proof of Theorem 1. For m = 1 , 2, 3, fix p (wm|xm)
and g(W1, W2, W3, Y ) such that the distortion constraint
Ed(T, ˆT ) ⩽ D is satisfied. Calculate p (wm) =P
xm p (xm)p (wm|xm).
Generation of codebooks: Generate 2nR
′
m i.i.d codewords
wmn (sm) ∼ Qn
i=1 p (wm,i), and index them by sm ∈n
1, 2, ..., 2nR
′
m
o
. Provide 2nRm random bins with indices
tm ∈

1, 2, ..., 2nRm

. Randomly assign the codewords
wmn (sm) to one of 2nRm bins using a uniform distribution.
Let Bm(tm) denote the set of codeword indices sm assigned
to bin index tm.
Encoding: Given a source sequence X n
m, the encoder looks
for a codeword W n
m(sm) such that (X n
m, W n
m (sm)) ∈ A∗(n)
ϵ .
The encoder sends the index of the bintm in which sm belongs.
Decoding: The decoder looks for a pair
(W n
1 (s1), W n
2 (s2), W n
3 (s3)) such that sm ∈ Bm(tm)
and (W n
1 (s1), W n
2 (s2), W n
3 (s3), Y n) ∈ A∗(n)
ϵ . If the decoder
finds a unique triple (s1, s2, s3), he then calculates ˆT n, where
ˆTi = g(W1,i, W2,i, W3,i, Yi).
Analysis of the probability of error:
1. The encoders cannot find the codewords W n
m(sm) such
that (X n
m, W n
m(sm)) ∈ A∗(n)
ϵ . The probability of this event is
small if
R
′
m > I (Xm, Wm) (24)
2. The pair of sequences (X n
1 , W n
1 (s1)) ∈ A∗(n)
ϵ ,
(X n
2 , W n
2 (s2)) ∈ A∗(n)
ϵ and (X n
3 , W n
3 (s3)) ∈ A∗(n)
ϵ
but the codewords {W n
1 (s1), W n
2 (s2), W n
3 (s3)} are not
jointly typical with the side information sequences Y n, i.e.,
(W n
1 (s1), W n
2 (s2), W n
3 (s3), Y n) /∈ A∗(n)
ϵ . We have assume
that
p (w1, w2, w3, x1, x2, x3, y)
= p (w1|x1) p (w2|x2) p (w3|x3) p (x1, x2, x3, y) . (25)
Hence, by the Markov lemma, the probability of this event
goes to zero if n is large enough.
3. There exists another s
′
with the same bin index that
is jointly typical with the side information sequences. The
correct codeword indices are denoted by s1 , s2 and s3. We
first consider the situation where the codeword index s1 is in
error. The probability that a randomly chosen W n
1 (s
′
1) is jointly
typical with (W n
2 (s2), W n
3 (s3), Y n) can be bounded as
Pr
n
W n
1

s
′
1

, W n
2 (s2) , W n
3 (s3) , Y n

∈ A∗(n)
ϵ
o
⩽ 2−n(I(W1;W2,W3,Y )−3ϵ).
(26)

<!-- page 5 -->

The probability of this error event is bounded by the number of
codewords in the bin t1 times the probability of joint typicality
Pr
n
∃s
′
1 ∈ B1 (t1) , s
′
1 ̸= s1 :

W n
1

s
′
1

, W n
2 (s2) , W n
3 (s3) , Y n

∈ A∗(n)
ϵ
o
⩽
X
s′
1 ̸=s1 ,
s′
1 ∈B1 (t1 )
Pr
n
W n
1

s
′
1

, W n
2 (s2) , W n
3 (s3) , Y n

∈ A∗(n)
ϵ
o
⩽ 2
n

R1−R
′
1

2−n(I(W1;W2,W3,Y )−3ϵ).
(27)
Similarly, the probability that the codeword index s2 or s3 is
in error can be bounded by
Pr
n
∃s
′
2 ∈ B2 (t2) , s
′
2 ̸= s2 :

W n
1 (s1) , W n
2

s
′
2

, W n
3 (s3) , Y n

∈ A∗(n)
ϵ
o
⩽ 2
n

R2−R
′
2

2−n(I(W2;W1,W3,Y )−3ϵ),
Pr
n
∃s
′
3 ∈ B3 (t3) , s
′
3 ̸= s3 :

W n
1 (s1) , W n
2 (s2) , W n
3

s
′
3

, Y n

∈ A∗(n)
ϵ
o
⩽ 2
n

R3−R
′
3

2−n(I(W2;W1,W3,Y )−3ϵ).
(28)
We then consider the case that two of the three codeword
indices are in error. The probability that the randomly chosen
W n
1 (s
′
1) and W n
2 (s
′
2) are jointly typical with (W n
3 (s3), Y n)
can be bounded as
Pr
n
W n
1

s
′
1

, W n
2

s
′
2

, W n
2 (s3) , Y n

∈ A∗(n)
ϵ
o
=
X
(W1,W2,W3,Y )∈A∗(n)
ϵ
p (wn
1 ) p (wn
2 ) p (wn
3 , yn)
⩽ 2−n(I(W1;W2,W3,Y )+I(W2;W1,W3,Y )−I(W1;W2|W3,Y )−4ϵ).
(29)
Hence, the error probability can be bounded as
Pr
n
∃s
′
1 ∈ B1 (t1) , s
′
1 ̸= s1, ∃s
′
2 ∈ B2 (t2) , s
′
2 ̸= s2 :

W n
1

s
′
1

, W n
2

s
′
2

, W n
2 (s3) , yn

∈ A∗(n)
ϵ
o
⩽ 2n(R1−R
′
1+R2−R
′
2)
× 2−n(I(W1;W2,W3,Y )+I(W2;W1,W3,Y )−I(W1;W2|W3,Y )−4ϵ).
(30)
Similarly, we can obtain the probability that the codeword
indices (s1, s3) or (s2, s3) are in error, which we omit here.
For the case where all the codeword indices s1, s2 and
s3 are in error. The probability that the randomly chosen
W n
1 (s
′
1), W n
2 (s
′
2) and W n
3 (s
′
3) are jointly typical with Y n can
be bounded as
Pr
n
W n
1

s
′
1

, W n
2

s
′
2

, W n
2

s
′
3

, Y n

∈ A∗(n)
ϵ
o
=
X
(W1,W2,W3,Y )∈A∗(n)
ϵ
p (wn
1 ) p (wn
2 ) p (wn
3 ) p (Y n)
⩽ 2−n(I(W1;W2,W3,Y )+I(W2;W1,W3,Y ))
× 2−n(I(W3;W1,W2,Y )−I(W1;W2|W3,Y )−I(W1,W2;W3|Y )−5ϵ).
(31)
The probability of the above error events goes to 0 when
R
′
1 − R1 ⩽ I (W1; W2, W3, Y )
...
R
′
1 − R1 + R
′
2 − R2 + R
′
3 − R3 ⩽ I (W1; W2, W3, Y )
+ I (W2; W1, W3, Y )
+ I (W3; W1, W2, Y )
− I (W1; W2|W3, Y )
− I (W1, W2; W3|Y ) .
(32)
Therefore, (2) can be obtained by combining (24) and (32).
If (s1, s2, s3) are correctly decoded, we have
(X n
1 , Xn
2 , Xn
3 , W n
1 (s1), W n
2 (s2), W n
3 (s3), Y n) ∈ A∗(n)
ϵ .
Therefore, the empirical joint distribution is close to the
distribution p (w1|x1) p (w2|x2) p (w3|x3) p (x1, x2, x3, y) that
achieves distortion D.
APPENDIX B
Here we provide the rigorous proof of Theorem 3. Define a
series of encoders fm : X n
m →

1, . . . ,2nRm

and decoders
g : Q
m∈M

1, . . . ,2nRm

× Y n → ˆT n that achieving the
given distortion D. We can derive the following inequalities
nR1
≥ H(f1(X n
1 ))
≥ H(f1(X n
1 )|f2(X n
2 ), f3(X n
3 ), Y n) (33a)
≥ H(f1(X n
1 )|f2(X n
2 ), f3(X n
3 ), Y n)
− H(f1(X n
1 )|f2(X n
2 ), f3(X n
3 ), Y n, Xn
1 , Xn
2 , Xn
3 )
= I(X n
1 , Xn
2 , Xn
3 ; f1(X n
1 )|f2(X n
2 ), f3(X n
3 ), Y n) (33b)
=
nX
i=1
I(X1,i, X2,i, X3,i; f1(X n
1 )|f2(X n
2 ), f3(X n
3 ),
Y n, Xi−1
1,1 , Xi−1
2,1 , Xi−1
3,1 ) (33c)
=
nX
i=1
H(X1,i, X2,i|f2(X n
2 ), Y n, Xi−1
1,1 , Xi−1
2,1 )
− H(X1,i, X2,i|f1(X n
1 ), f2(X n
2 ), Y n, Xi−1
1,1 , Xi−1
2,1 ) (33d)
=
nX
i=1
H(X1,i, X2,i|W2,i, Yi) − H(X1,i, X2,i|W1,i, W2,i, Yi)
(33e)
=
nX
i=1
I(X1,i, X2,i; W1,i|W2,i, Yi),
where (33a) follows from the fact that conditioning re-
duces entropy, (33b) and (33d) are obtained by the def-
inition of conditional mutual information and (33c) is
the chain rule of mutual information. In (33e), we let
W1,i =
 
f1(X n
1 ), Xi−1
1,1 , Xi−1
2,1 , Y i−1
1 ,Y n
i+1

, and W2,i = 
f2(X n
2 ), Xi−1
1,1 , Xi−1
2,1 , Y i−1
1 ,Y n
i+1

. Similarly we have
nR2 ≥
nX
i=1
I(X1,i, X2,i, X3,i; W2,i|W1,i, W3,i, Yi),
nR3 ≥
nX
i=1
I(X1,i, X2,i, X3,i; W3,i|W2,i, W3,i, Yi).
(34)
The rigorous proof of the sum rate part will be provided in
a longer version.

<!-- page 6 -->

REFERENCES
[1] T. Adikari and S. Draper, “Two-terminal source coding with common
sum reconstruction,” in 2022 IEEE International Symposium on Infor-
mation Theory (ISIT) . IEEE, 2022, pp. 1420–1424.
[2] J. Korner and K. Marton, “How to encode the modulo-two sum of binary
sources (corresp.),” IEEE Transactions on Information Theory , vol. 25,
no. 2, pp. 219–221, 1979.
[3] A. Pastore, S. H. Lim, C. Feng, B. Nazer, and M. Gastpar, “Distributed
lossy computation with structured codes: From discrete to continuous
sources,” in 2023 IEEE International Symposium on Information Theory
(ISIT). IEEE, 2023, pp. 1681–1686.
[4] M. M. Amiri, D. Gunduz, S. R. Kulkarni, and H. V . Poor, “Federated
learning with quantized global model updates,” arXiv:2006.10672, 2020.
[5] K. Gruntkowska, A. Tyurin, and P. Richt ´arik, “Improving the worst-
case bidirectional communication complexity for nonconvex distributed
optimization under function similarity,” arXiv:2402.06412, 2024.
[6] M. M. Amiri, D. G ¨und¨uz, S. R. Kulkarni, and H. V . Poor, “Convergence
of federated learning over a noisy downlink,” IEEE Transactions on
Wireless Communications, vol. 21, no. 3, pp. 1422–1437, 2022.
[7] P. A. Stavrou and M. Kountouris, “The role of fidelity in goal-oriented
semantic communication: A rate distortion approach,” IEEE Transactions
on Communications , 2023.
[8] J. Liu, W. Zhang, and H. V . Poor, “A rate-distortion framework for char-
acterizing semantic information,” in 2021 IEEE International Symposium
on Information Theory (ISIT) . IEEE, 2021, pp. 2894–2899.
[9] R. Dobrushin and B. Tsybakov, “Information transmission with addi-
tional noise,” IRE Transactions on Information Theory , vol. 8, no. 5, pp.
293–304, 1962.
[10] J. Wolf and J. Ziv, “Transmission of noisy information to a noisy receiver
with minimum distortion,” IEEE Transactions on Information Theory ,
vol. 16, no. 4, pp. 406–411, 1970.
[11] T. Guo, Y . Wang, J. Han, H. Wu, B. Bai, and W. Han, “Semantic
compression with side information: A rate-distortion perspective,” arXiv
preprint arXiv:2208.06094, 2022.
[12] A. Wyner and J. Ziv, “The rate-distortion function for source coding
with side information at the decoder,” IEEE Transactions on information
Theory, vol. 22, no. 1, pp. 1–10, 1976.
[13] D. Slepian and J. Wolf, “Noiseless coding of correlated information
sources,” IEEE Transactions on information Theory , vol. 19, no. 4, pp.
471–480, 1973.
[14] S. H. Lim, C. Feng, A. Pastore, B. Nazer, and M. Gastpar, “Towards an
algebraic network information theory: Distributed lossy computation of
linear functions,” in 2019 IEEE International Symposium on Information
Theory (ISIT) . IEEE, 2019, pp. 1827–1831.
[15] G. Ku, J. Ren, and J. M. Walsh, “Computing the rate distortion region
for the ceo problem with independent sources,” IEEE Transactions on
Signal Processing, vol. 63, no. 3, pp. 567–575, 2014.
[16] M. Gastpar, “The Wyner-Ziv problem with multiple sources,” IEEE
Transactions on Information Theory , vol. 50, no. 11, pp. 2762–2768,
2004.
