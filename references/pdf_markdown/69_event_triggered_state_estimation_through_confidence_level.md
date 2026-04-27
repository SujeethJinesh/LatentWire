# references/69_event_triggered_state_estimation_through_confidence_level.pdf

<!-- page 1 -->

arXiv:2403.15289v1  [eess.SY]  22 Mar 2024
1
Event-Triggered State Estimation Through
Conﬁdence Level
Wei Liu, Senior Member , IEEE
Abstract—This paper considers the state estimation problem
for discrete-time linear systems under event-triggered sc heme. In
order to improve performance, a novel event-triggered sche me
based on conﬁdence level is proposed using the chi-square
distribution and mild regularity assumption. In terms of th e novel
event-triggered scheme, a minimum mean squared error (MMSE )
state estimator is proposed using some results presented in this
paper . Two algorithms for communication rate estimation of the
proposed MMSE state estimator are developed where the ﬁrst
algorithm is based on information with one-step delay, and t he
second algorithm is based on information with two-step dela y.
The performance and effectiveness of the proposed MMSE stat e
estimator and the two communication rate estimation algori thms
are illustrated using a target tracking scenario.
Index Terms —Event-triggered state estimation, Conﬁdence
level, Communication rate estimation, Discrete-time line ar sys-
tems, Sensor networks
I. I NTRODUCTION
With the development of wireless sensor network tech-
nology, wireless networked control systems (WNCS) have
attracted increasing attention, and have been successfull y
applied in various ﬁelds such as control, signal processing ,
robotics, power electronics, etc [1] − [6]. In WNCS, sensors,
controllers, estimators and actuators are spatially distr ibuted
where sensor and estimator are usually far away from each
other. In this case, the communication from sensor to remote
estimator is costly because the communication requires con -
suming power of energy limited battery in the sensor where th e
battery is probably hard to replace due to its physical posit ion.
Event-triggered scheme is an effective means to reduce sens or-
to-estimator communication cost since communication is no t
permitted unless a pre-deﬁned triggered condition is satis ﬁed.
Previous studies have shown that event-triggered scheme ca n
strike a proper balance of trade-offs between communicatio n
cost and estimation performance [7] − [10].
As a fundamental issue, event-triggered state estimation
has been extensively studied [10] − [24]. In [11], for a ﬁrst
order discrete-time linear system, the pre-processor and e s-
timator were sought to minimize a cost with two terms.
In [12], a centralized sensor network with multiple nodes
were considered where each node yields measurement of the
original system. Local event-triggered transmission stra tegies
were developed, and the strategies’s stability and perform ance
were studied. For the balance between communication rate an d
state estimation performance, an event-triggered sensor d ata
This work was partially supported by the National Nature Sci ence Foun-
dation of China (6207312).
Wei Liu is with the School of Information and Electronic Engi neering,
Zhejiang Gongshang University, Hangzhou 310018, China (e- mail: inter-
valm@163.com).
scheduler was presented in [10] where, for a speciﬁc thresho ld,
this scheduler is determined by the H¨ older inﬁnity-norm of
the innovation’s linear operation. Using an approximation
technique in nonlinear ﬁltering, an approximate MMSE state
estimator was proposed. The results of [10] were extended in
[14] and [15] where the results presented in [14] considered
separate transmission for each element of the measurement,
and measurements from multiple sensors with separate event -
triggering conditions for each measurement were studied in
[15]. The measurement prediction variance was used in [16]
to determine whether the measurement is transmitted. Based
on this kind of measurement transmission, the state estimat or
was designed, and the corresponding Riccati equation with p e-
riodic behavior was developed. In [17], the set-valued Kalm an
ﬁltering problem for additional information with stochast ic
uncertainty was studied, and it was applied to event-trigge red
estimation. Two stochastic event-triggered sensor schedu les
were proposed in [18] where one schedule depends on the
current measurement, and the other one depends on the inno-
vation. Based on the two schedules, the MMSE state estimator s
were proposed, and the communication rates were analyzed.
The results of [18] were generalized and extended in [19]
and [20], respectively, where single-sensor was generaliz ed
to the case of multi-sensor in [19], and a stochastic event-
triggered mechanism based on information-state contribut ion
was proposed in [20]. More results about event-triggered st ate
estimation were provided in [21] − [24] and references therein.
Because additional information was introduced to the remot e
estimator when the measurement is not transmitted to the
estimator, the state estimator developed in [10] can yield b etter
performance. However, the results developed in [10] does
not establish the connection between the innovation and the
trigger threshold, which means the performance can be furth er
improved through establishing a proper connection between
them. So, it is necessary to propose an event-triggered sche me
which can establish a proper connection of the innovation, t he
trigger threshold and other related parameters, and to desi gn
state estimator based on this scheme, which motivates our
research.
In this paper, using the chi-square distribution, regular G aus-
sian assumption and the method of conﬁdence level, we ﬁrst
propose an event-triggered scheme which establishes a prop er
connection of the tolerable upper bound of the innovation
covariance, the innovation and the trigger threshold. Howe ver,
the results proposed in [10] do not obtain any connection of
these parameters. Also, to the best of the author’s knowledg e,
the event-triggered scheme proposed in this paper is novel a nd
different from the existing results. The novel event-trigg ered
scheme paves the way for the design of state estimator with

<!-- page 2 -->

2
better performance.
Then, based on the novel event-triggered scheme, a MMSE
state estimator is proposed in a recursive form. It is worth
mentioning that, due to the use of the novel event-triggered
scheme, the strategy in computing the error covariance of th e
proposed MMSE state estimator is different in contrast to th e
existing results. Two algorithms for estimating the commu-
nication rate of the proposed state estimator are developed
where the ﬁrst algorithm uses information with one-step del ay,
and the second algorithm utilizes information with two-ste p
delay. As far as the author knows, the strategy used in the
second algorithm, namely, using information with two-step
delay, cannot be found in the existing results for estimatin g
the communication rate of event-triggered state estimator . In
addition, the simulation results show that the second algo-
rithm yields a better communication rate estimation of the
proposed state estimator, which means that the strategy of
using information with two-step delay is effective. Due to
using information with two-step delay, the proof of the seco nd
algorithm becomes very challenging. In order to prove the
second algorithm, we ﬁrst prove Lemma 4 and Theorem 2
in Appendix C, and then we prove the second algorithm in
Appendix D.
The remainder of this paper is organized as follows. The
system and problem under consideration are provided in Sec-
tion II. We also present an event-triggered scheme in Sectio n
II. A MMSE state estimator based on the presented event-
triggered scheme is proposed in Section III. Two algorithms
for estimating the communication rate of the proposed MMSE
state estimator are developed in Section IV. In Section V, th e
performance and effectiveness of the proposed results, inc lud-
ing the MMSE state estimator and the communication rate
estimation algorithms, are demonstrated via a target track ing
scenario. The conclusion is drawn in Section VI.
Notation: The n-dimensional real Euclidean space is de-
noted by Rn, and N > 0 is used to denote the positive deﬁnite
matrix N. For a matrix A, its transpose, determinant and
inverse are represented by AT, |A| and A− 1, respectively. The
probability density function is denoted by f , and the n × n
identity matrix is denoted by In.
∫
Φ(η)dη is used to stand
for
∫
Φ(η)dη1dη2 · · ·dηn where η = (η1,η2,· · ·,ηn)T ∈ Rn,
and Φ(η) is a function of η. We use E [·] and V ar(·) to stand
for the expectation operation and the covariance operation ,
respectively.
II. P ROBLEM FORMULATION
A. System Description
Consider the following system
xk+1 =Axk + ωk, (1)
yk =Cxk + υk,k = 0,1,· · · (2)
where xk ∈ Rn is the unknown state; ωk ∈ Rn is the process
noise; yk ∈ Rp is the measurement; υk ∈ Rp is the measurement
noise; A and C are matrices of appropriate dimensions; and the
initial state x0 is a random vector with mean ¯x0 and covariance
matrix ¯P0.




 !"#$%%&

'$(%)!$*$+,&

& -$,."!/& 0%,1*(,"!
2!133$!&
4#5$*$
&
&
Fig. 1. Structure of event-triggered state estimation.
Throughout the paper, we introduce the following two
assumptions.
1) ωk and υk are zero-mean white Gaussian noise se-
quences with covariance matrices Q and R, respectively.
2) ωk is independent of υk, and x0 is independent of ωk
and υk.
Considering the event-triggered state estimation problem
whose structure is given in Fig. 1. γk has two possible values
0 and 1, and the value of γk is determined by the trigger
scheme. When γk = 1, the measurement yk is transmitted
to the estimator via network, and the information I k avail-
able for the estimator is I k = (Ik− 1,yk). When γk = 0, there
is no data tramission, and the information I k available for
the estimator is I k = ( Ik− 1,γk = 0). Hence, we have I k ={
(Ik− 1,γk = 0), γk = 0;
(Ik− 1,yk), γk = 1 with I 0 =
{
(γ0 = 0), γ0 = 0;
(y0), γ0 = 1 .
The trigger scheme is based on conﬁdence level, and it will
be proposed in the next part.
Remark 1: For a joint probability density function f (x,y) in
probability theory, f (x = a,y = b) denotes the joint probability
density of x and y given (x = a,y = b) where (x = a,y = b)
stands for {x = a} ∩ { y = b} instead of {x = a} ∪ { y = b}.
As an extension, the conditional probability density of x = a
given y = b is deﬁned as f (x = a|y = b) = f (x=a,y=b)
f (y=b) . It has to
be said that the conditional probability f (x = a|y = b) is not
equal to f (x=a∪ y=b)
f (y=b) , that is, f (x = a|y = b) ̸= f (x=a∪ y=b)
f (y=b) , where
(x = a ∪ y = b) denotes {x = a} ∪{ y = b} which is the union of
two random variables. The conditional probability was used in
the wrong way in [10], which expressed the probability densi ty
with the union of two random variables. For example, in [10],
the term f
εk (ε|Ik− 1) is equal to f (εk=ε∪ Ik− 1)
f (Ik− 1) , and fεk (ε|Ik− 1)
is obviously not equal to the conditional probability densi ty
of εk = ε given Ik− 1. However, the term fεk (ε|Ik− 1) was used
as the conditional probability density, that is, the result s were
developed based on the condition f (εk=ε∪ Ik− 1)
f (Ik− 1) = f (εk=ε,Ik− 1)
f (Ik− 1) .
Hence, the correctness of the corresponding results presen ted
in [10] is doubtful unless the information Ik− 1 ∩ { γk = 0} is
used to replace Ik− 1 ∪ { γk = 0}.
Remark 2: In [18], Ik is deﬁned by Ik ≜
{γ0,γ1,· · ·,γk,· · ·,γ0y0,γ1y1,γkyk} = {Ik− 1,γk,γkyk}. Because
the relation between Ik− 1, γk and γkyk in Ik is not clearly
stated, we ﬁrst assume that Ik = Ik− 1 ∩ (γk ∪ γkyk). Then, we
have γk ∪ γkyk =
{
γk = 0, γk = 0;
(γk = 1) ∪ yk, γk = 1 , which means

<!-- page 3 -->

3
that Ik =
{ Ik− 1 ∩ (γk = 0), γk = 0;
Ik− 1 ∩
(
(γk = 1) ∪ yk
)
, γk = 1 . We easily
see that, under the above assumption, Ik is equal to the
description presented in [18] for γk = 0, but is not equal to
that for γk = 1. Second, we assume that Ik = Ik− 1 ∩ (γk ∩ γkyk).
Then, we get γk ∩ γkyk = 0 in γk = 0. As a result, we derive
Ik = Ik− 1 in γk = 0. It is obvious that, under the assumption
of Ik = Ik− 1 ∩ (γk ∩ γkyk), Ik is not equal to the description
presented in [18] for γk = 0. Hence, no matter how we choose
the relation between Ik− 1, γk and γkyk, Ik cannot completely
embody the description Ik =
{ (Ik− 1,γk = 0), γk = 0;
(Ik− 1,yk), γk = 1
presented in [18]. This means that the deﬁnition of
Ik ≜ {
γ0,γ1,· · ·,γk,· · ·,γ0y0,γ1y1,γkyk} given in [18] is not
rigorous.
B. Event-Triggered Scheme based on Conﬁdence Level
In order to propose the event-triggered scheme based on
conﬁdence level, we ﬁrst present the following two remarks
and one lemma.
Remark 3: Let w ∈ Rp be a Gaussian random vector with
mean ¯w and covariance matrix S. Then, it was presented in
Result 4.7 of [28] that (w − ¯w)TS− 1(w − ¯w) obeys the distri-
bution of
χ 2
p where χ 2
p stands for the chi-square distribution
with p degrees of freedom.
Remark 4: Let µ 1, µ 2, · · ·, µ m be Gaussian and mutually
independent. Then, it is well known that the linear combinat ion
of µ 1, µ 2, · · ·, µ m is still Gaussian.
Lemma 1: Under the assumption that f (xk− 1|Ik− 1) is Gaus-
sian, it holds that:
1). f (xk|Ik− 1) is Gaussian.
2). f (yk|Ik− 1) is Gaussian.
Proof: See Appendix A.
For notational simplicity, deﬁne
ˆyk,k− 1 ≜E[yk|Ik− 1], (3)
˜yk ≜yk − ˆyk,k− 1, (4)
Nk ≜V ar(yk|Ik− 1). (5)
Under the assumption that f (xk− 1|Ik− 1) is Gaussian, we see
from 2) of Lemma 1 that f (yk|Ik− 1) is Gaussian. Then, using
Remark 3, we ﬁnd that ˜ yT
k N− 1
k ˜yk is distributed as
χ 2
p where ˜yk
and Nk are deﬁned in (4) and (5), respectively. Let N > 0 be
a tolerable upper bound of Nk. When Nk exceeds the tolerable
upper bound, namely, Nk > N, we need to take γk = 1 so that
Nk+1 does not exceed the tolerable upper bound. When Nk > N,
we have
ϕk > ˜yT
k N− 1
k ˜yk (6)
where
ϕk = ˜yT
k
Σ ˜yk,Σ = N− 1. (7)
Let χ 2
α (p) denote the upper (l00 α)th percentile of the χ 2
p
distribution, that is, P
(
˜yT
k N− 1
k ˜yk ≤
χ 2
α (p)
)
= 1 − α. In fre-
quentist statistics, the 95% conﬁdence level is the most oft en.
Hence, we can take 1 − α = 0.95 in conﬁdence level for
˜yT
k N− 1
k ˜yk. Then, we conclude from the theory of conﬁdence
level that ˜ yT
k N− 1
k ˜yk does not obey the
χ 2
p distribution if
˜yT
k N− 1
k ˜yk >
χ 2
α (p). Then, using (6), we get ϕk > ˜yT
k N− 1
k ˜yk >
χ 2
α (p) =⇒ ϕk > χ 2
α (p) when Nk > N. Based on the above
discussion, we present the following event-triggered sche me:
γk =
{
γk = 0, ϕk ≤ χ 2
α (p);
γk = 1, ϕk > χ 2
α (p). (8)
where 1 − α = 0.95 is suggested to be taken considering the
theory of conﬁdence level. Since Σ is positive deﬁnite, Σ can
be expressed as Σ = Φ TΦ such that Φ is invertible. Then,
using (7) and noticing yk ∈ Rp, we have
ϕk = ˜yT
k
Σ ˜yk = ˜yT
k
Φ TΦ ˜yk = zT
k zk =
p
∑
i=1
z2
k,i (9)
where
zk ≜
Φ ˜yk (10)
and zk,i denotes the ith element of zk.
Remark 5: An advantage of the event-triggered scheme
proposed in this paper is that a proper connection of the
tolerable upper bound
N, the innovation ˜ yk and the trigger
threshold χ 2
α (p) is established via conﬁdence level and chi-
square distribution, which leads to the performance improv e-
ment of the state estimator proposed in Section III in contra st
to the state-of-the-art state estimator proposed in [10]. H ow-
ever, the event-triggered schemes provided in [10] and [17]
do not establish any connection of these parameters. Also, t he
proposed event-triggered scheme is novel and different tho se
presented in [11] − [16] and [18] − [27].
Let ˆxk ≜ E[xk|Ik] which is the optimal MMSE estimate of
xk given I k. The rest of the study has two main objectives.
The ﬁrst objective is to design a MMSE state estimator using
the above event-triggered scheme based on conﬁdence level,
which can recursively compute ˆxk under a regular assumption.
The second objective is to develop two algorithms for esti-
mating the communication rate of the proposed MMSE state
estimator.
III. MMSE S TATE ESTIMATION
In this section, we study the MMSE state estimation prob-
lem based on the event-triggered scheme. More precisely, th e
computational strategy for the MMSE estimate ˆ xk is studied
in the section. For notational simplicity, let
Pk ≜V ar(xk|Ik), (11)
Ωk ≜{zk ∈ Rp|
ϕk ≤ χ 2
α (p)}, (12)
ˆx[z]
k ≜E[xk|Ik− 1,zk], (13)
P[z]
k ≜V ar(xk|Ik− 1,zk), (14)
ˆxk,k− 1 ≜E[xk|Ik− 1], (15)
˜xk ≜xk − ˆxk,k− 1, (16)
Mk ≜V ar(xk|Ik− 1), (17)
N[z]
k ≜V ar(zk|Ik− 1), (18)
Kk ≜E
[
˜xk(zk − E[zk|Ik− 1])T
](
N[z]
k
) − 1. (19)

<!-- page 4 -->

4
Lemma 2: Under the assumption that f (xk− 1|Ik− 1) is Gaus-
sian, it holds that:
1). f (zk|Ik− 1) is Gaussian.
2). f (zk|Ik− 1) = g(zk)
(2π)0.5p
⏐
⏐N[z]
k
⏐
⏐0.5 with
g(zk) ≜ exp
{
− 0.5zT
k (N[z]
k )− 1zk
}
.
Proof: See Appendix A.
Lemma 3: Under the assumption that f (xk− 1|Ik− 1) is Gaus-
sian, it holds that:
1).
∫
Ωk f (zk|Ik− 1)dzk = hk
(2π)0.5p
⏐
⏐N[z]
k
⏐
⏐0.5 where
hk =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2
ˇzk,2∫
− ˇzk,2
dzk,3 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
g(zk)dzk,p
with
ˇb ≜
√
χ 2α (p),ˇzk,1 ≜
√
χ 2α (p) − z2
k,1,
ˇzk,2 ≜
√
χ 2α (p) −
2
∑
i=1
z2
k,i,· · ·,ˇzk,p− 1 ≜



√
χ 2α (p) −
p− 1
∑
i=1
z2
k,i.
2).
∫
Ωk f (zk|Ik− 1)zkzT
k dzk =
Ψk
(2π)0.5p
⏐
⏐N[z]
k
⏐
⏐0.5 with
Ψk = (ψk,i j)p× p
where ψk,i j =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
g(zk)zk,izk,jdzk,p.
Proof: See Appendix A.
Considering that γk has two possible values 0 and 1, we
deal with the problem under the following two cases:
1) γk = 1. Noticing the deﬁnition of I k, we have I k =
(Ik− 1,yk). Then, using Kalman ﬁlter, we derive
ˆxk = ˆxk,k− 1 + MkCT(CMkCT + R)− 1 ˜yk, (20)
Pk =Mk − MkCT(CMkCT + R)− 1CMk (21)
where ˜yk = yk − C ˆxk,k− 1, ˆxk,k− 1 = A ˆxk− 1 and Mk = APk− 1AT +
Q.
2) γk = 0. We have I k = (Ik− 1,γk = 0) when γk = 0. Then,
we present the following theorem to compute ˆ xk in γk = 0.
Theorem 1: When f (xk− 1|Ik− 1) is Gaussian, the MMSE
state estimation ˆxk and the corresponding error covariance Pk in
γk = 0 can be computed according to the following equalities:
ˆxk = ˆxk,k− 1, (22)
Kk =MkCT(CMkCT + R)− 1Φ − 1, (23)
P[z]
k =Mk − MkCT(CMkCT + R)− 1CMk, (24)
Pk =P[z]
k + 1
hk
KkΨkKT
k . (25)
Proof: See Appendix B.
Remark 6: In fact, ˆxk in (22) of Theorem 1 should be com-
puted via ˆxk = ˆxk,k− 1 + ek with ek ≜
Kk
∫
Ωk f (zk|Ik− 1)zkdzk∫
Ωk f (zk|Ik− 1)dzk
in which
∫
Ωk f (zk|Ik− 1)zkdzk can be obtained via
∫
Ωk f (zk|Ik− 1)zkdzk =
ψk
(2π)0.5p
⏐
⏐N[z]
k
⏐
⏐0.5 where
ψk = (ψk,1,ψk,2,· · ·,ψk,p)T with ψk,i =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
g(zk)zk,idzk,p, i = 1,2,· · ·,p. Since ψk is
almost equal to zero vector, we conclude that ek is almost
equal to zero vector. Hence, ˆ xk in Theorem 1 is calculated
using (22).
Now, we can present the MMSE state estimator.
Starting with ˆ xk− 1 and Pk− 1, the MMSE state estimator in-
cludes the following two steps.
Step 1: Compute ˆxk,k− 1, ˜yk, Mk, hk and
Ψk according to
ˆxk,k− 1 =A ˆxk− 1, (26)
˜yk =yk − C ˆxk,k− 1, (27)
Mk =APk− 1AT + Q, (28)
N[z]
k =Φ(CMkCT + R)Φ T, (29)
hk =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
g(zk)dzk,p, (30)
ψk,i j =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
g(zk)zk,izk,jdzk,p, (31)
Ψk =(ψk,i j)p× p (32)
where g(zk) is deﬁned in 2) of Lemma 2, as well as ˇb and ˇzk,i
with i = 1,2,· · ·,p − 1 are deﬁned in 1) of Lemma 3.
Step 2: Compute ˆxk and Pk in terms of
P[z]
k =Mk − MkCT(CMkCT + R)− 1CMk, (33)
ˆxk = ˆxk,k− 1 + γkMkCT(CMkCT + R)− 1 ˜yk, (34)
Kk =MkCT(CMkCT + R)− 1Φ − 1, (35)
Pk =P[z]
k + (1 − γk)
hk
KkΨkKT
k (36)
where γk is determined by (8) and (7). For the proposed MMSE
state estimator, we easily see that we only need to prove (34)
and (36) where we easily obtain (34) by using (20) and (22).
Using (21) and (24), we see that
Pk = P[z]
k when
γk = 1. (37)
Putting (37) and (25) together, we prove (36).
Remark 7: The results for MMSE state estimation problem
based on different event-triggered schemes were presented in
[10], [14], [15], [16], [18], [19], [21], [22], [23] and [27] .
However, compared with the results, the MMSE state estimato r
presented in this paper has a different strategy in computin g
the error covariance Pk because a novel conﬁdence level based
event-triggered scheme is applied to the design of the MMSE
state estimator.
Remark 8: For the MMSE state estimator presented in
(26)− (36), we need to know the initial conditions ˆ x0 and P0.
Hence, we present the following scheme to obtain ˆ x0 and P0.
ˆx0 and P0 can be computed according to
ˆx0 = ¯x0 +
γ0 ¯P0CT(C ¯P0CT + R)− 1(y0 − C ¯x0), (38)
¯N[z]
0 =Φ(C ¯P0CT + R)Φ T, (39)

<!-- page 5 -->

5
¯ψ0,i j =
ˇb∫
− ˇb
dz0,1
ˇz0,1∫
− ˇz0,1
dz0,2 · · ·
ˇz0,p− 1∫
− ˇz0,p− 1
ρ(z0)z0,iz0,jdz0,p, (40)
¯Ψ0 =( ¯ψ0,i j)p× p, (41)
α0 =
ˇb∫
− ˇb
dz0,1
ˇz0,1∫
− ˇz0,1
dz0,2 · · ·
ˇz0,p− 1∫
− ˇz0,p− 1
ρ(z0)dz0,p, (42)
¯K0 = ¯P0CT(C ¯P0CT + R)− 1Φ − 1, (43)
P[z]
0 = ¯P0 − ¯P0CT(C ¯P0CT + R)− 1C ¯P0, (44)
P0 =P[z]
0 + (1 − γ0)
α0
¯K0 ¯Ψ0 ¯KT
0 (45)
where γ0 is determined by (8) with ϕ0 = (y0 − C ¯x0)TΣ (y0 −
C ¯x0), as well as α0 ≜
∫
Ω0 ρ(z0)dz0, ρ(z0) ≜ exp
{
−
0.5zT
0 ( ¯N[z]
0 )− 1z0
}
, ¯N[z]
0 ≜ V ar(z0) and ¯K0 ≜ Cov(x0,z0)
(¯N[z]
0
) − 1.
Making reference to the derivation of the MMSE state esti-
mator, we can easily obtain (38) − (45).
IV. C OMMUNICATION RATE ESTIMATION
In this section, we study the communication rate estima-
tion problem for the proposed MMSE state estimator. More
precisely, we will present two strategies for approximatel y
computing E [
γk].
E[γk] can be expressed as
E[γk] =0 × P(γk = 0) +1 × P(γk = 1)
=P(γk = 1) = 1 − P(γk = 0) (46)
where
P(γk = 0) =
∫
f (γk = 0,Ik− 1)dIk− 1
=
∫
f (Ik− 1)P(γk = 0|Ik− 1)dIk− 1. (47)
Remark 9: From (47), we see that the computation of
P(γk = 0) is intractable because the computational complex
of
∫
f (Ik− 1)P(γk = 0|Ik− 1)dIk− 1 increases with k. Then, it
follows from (46) that the computation of E [γk] is intractable.
In order to approximately compute E [γk], we will use two
types of approximations where one type of approximation is
E[
γk] ≈ E[γk|Ik− 1], and the other one is E [γk] ≈ E[γk|Ik− 2]. We
will present a strategy for computing E [γk|Ik− 1] and E[γk|Ik− 2],
and we will test the two different approximations for E [γk] in
Section V -A.
For notational simplicity, let
ˆγk,k− i ≜E[γk|Ik− i], (48)
Pk,k− i(0) ≜P(γk = 0|Ik− i), (49)
⃗P[z]
k (0) ≜P(γk = 0|Ik− 2,zk− 1), (50)
˘Pk(0) ≜P(γk = 0|Ik− 2,γk− 1 = 0), (51)
ˆz⊲
k,k− 1 ≜E[zk|Ik− 2,zk− 1], (52)
˜z⊲
k ≜zk − ˆz⊲
k,k− 1, (53)
⃗N[z]
k ≜V ar(zk|Ik− 2,zk− 1), (54)
ˆzk,k− 1(0) ≜E[zk|Ik− 2,
γk− 1 = 0], (55)
Nk(0) ≜V ar(zk|Ik− 2,γk− 1 = 0), (56)
˘Ωk ≜{zk ∈ Rp|ϕk > χ 2
α (p)} (57)
with i = 1,2.
Starting with ˆ xk− 1 and Pk− 1, ˆγk,k− 1 can be recursively
computed according to Algorithm 1. For Algorithm 1, we only
Algorithm 1 : Communication rate based on information
up to k − 1
Step 1: Compute ˆγk,k− 1 according to
Pk,k− 1(0) = hk
(2π)0.5p|N[z]
k |0.5
, (58)
ˆγk,k− 1 =1 − Pk,k− 1(0) (59)
where N[z]
k and hk are computed using (28) − (30) in sequence.
Step 2: Compute and store ˆ xk and Pk for the derivation
of ˆγk+1,k where ˆxk and Pk are computed via (26) − (36) in
sequence.
need to prove (58) and (59). P(γk = 0|Ik− 1) can be rewritten
as
P(γk = 0|Ik− 1) =P(zk ∈ Ωk|Ik− 1)
=
∫
Ωk
f (zk|Ik− 1)dzk = hk
(2π)0.5p⏐
⏐N[z]
k
⏐
⏐0.5 (60)
where Ωk is deﬁned in (12), and the last equality is due to 1)
of Lemma 3. Then, replacing P(
γk = 0|Ik− 1) by Pk,k− 1(0), we
prove (58). Making reference to (46), we easily obtain (59).
In order to compute ˆγk,k− 2, we propose the following
theorem.
Theorem 2: When f (xk− 1|Ik− 1) is Gaussian, ˆγk,k− 2 can be
computed in terms of the following equalities:
⃗N[z]
k =Φ
(
C(AP[z]
k− 1AT + Q)CT + R)Φ T, (61)
Nk(0) = Φ
(
C
(
A(P[z]
k− 1 + 1
hk− 1
Kk− 1Ψk− 1KT
k− 1)AT + Q
)
CT + R
)
× Φ T, (62)
f (zk|Ik− 2,zk− 1) = ⃗g(zk)
(2π)0.5p⏐
⏐⃗N[z]
k
⏐
⏐0.5 , (63)
f (zk|Ik− 2,zk− 1) = ⃗g(zk)
(2π)0.5p⏐
⏐⃗N[z]
k
⏐
⏐0.5 , (64)
f (zk|Ik− 2,
γk− 1 = 0) = ˘g(zk)
(2π)0.5p⏐
⏐Nk(0)
⏐
⏐0.5 , (65)
⃗P[z]
k (0) =
∫
Ωk
f (zk|Ik− 2,zk− 1)dzk, (66)
˘Pk(0) =
∫
Ωk
f (zk|Ik− 2,
γk− 1 = 0)dzk, (67)
Pk,k− 2(0) =Pk− 1,k− 2(0) ˘Pk(0) +
∫
˘Ωk− 1
⃗P[z]
k (0) f (zk− 1|Ik− 2)dzk− 1,
(68)
ˆγk,k− 2 =1 − Pk,k− 2(0) (69)
where

<!-- page 6 -->

6
⃗g(zk) ≜ exp
{
− 0.5zT
k (⃗N[z]
k )− 1zk
}
,
˘g(zk) ≜ exp
{
− 0.5zT
k Nk(0)− 1zk
}
. (70)
Proof: See Appendix C.
Based on the above discussion, we present an algorithm to
compute ˆ
γk,k− 2 in a recursive structure. Starting with P[z]
k− 1,
hk− 1, Kk− 1, Ψk− 1, ˆxk− 1, Pk− 1 and Pk− 1,k− 2(0), ˆγk,k− 2 can be
recursively computed according to Algorithm 2 where the
proof of Algorithm 2 is presented in Appendix D.
Algorithm 2 : Communication rate based on information
up to k − 2
Step 1: Compute ⃗N[z]
k and Nk(0) using (61) and (62), respec-
tively.
Step 2: Compute ⃗P[z]
k (0) and ˘Pk(0) according to
⃗P[z]
k (0) =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
⃗g(zk)dzk,p
(2π)0.5p⏐
⏐⃗N[z]
k
⏐
⏐0.5 , (71)
˘Pk(0) =
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
˘g(zk)dzk,p
(2π)0.5p|Nk(0)|0.5 (72)
where ⃗g(zk) and ˘g(zk) are deﬁned in (70).
Step 3: Compute Pk,k− 2(0) using
Pk,k− 2(0) =⃗P[z]
k (0) +Pk− 1,k− 2(0)
(˘Pk(0) − ⃗P[z]
k (0)
)
. (73)
Step 4: Compute ˆγk,k− 2 in terms of (69).
Step 5: For computing the communication rate at time step
k + 1, update P[z]
k , hk, Kk, Ψk, ˆxk, Pk and Pk,k− 1(0) using the
MMSE state estimator presented in (26) − (36) and using (58).
Remark 10: Under the assumption that f (xk− 1|Ik− 1) is
Gaussian, we obtain ˆγk,k− 2 using Algorithm 2 in a recursive
form where the proof of the algorithm is very challenging.
In order to prove Algorithm 2, Lemma 4 and Theorem 2 are
ﬁrst proved in Appendix C, and then Algorithm 2 is proved
in Appendix D.
Remark 11: If we take the approximation E [
γk] ≈ ˆγk,k− 1
where ˆγk,k− 1 can be obtained from Algorithm 1, we need to
additionally obtain E [γ0] since Algorithm 1 starts with k = 1.
In the same way, we need to know E [γ0] and E [γ1] if we take
the approximation E [γk] ≈ ˆγk,k− 2 using Algorithm 2. Hence,
we need to obtain E [γ0] and E [γ1].
We present a strategy for computing E [γ0] and E [γ1] with the
following content:
P(γ0 = 0) = α0
(2π)0.5p|¯N[z]
0 |0.5
, (74)
E[γ0] =1 − P(γ0 = 0), (75)
¯N[z]
1 =Φ
(
C(AP[z]
0 AT + Q)CT + R)Φ T, (76)
¯N1(0) =Φ
(
C
(
A(P[z]
0 + 1
α0
¯K0 ¯Ψ0 ¯KT
0 )AT + Q
)
CT + R
)
Φ T,
(77)
¯P1(0) =
ˇb∫
− ˇb
dz1,1
ˇz1,1∫
− ˇz1,1
dz1,2 · · ·
ˇz1,p− 1∫
− ˇz1,p− 1
˘ρ(z1)dz1,p
(2π)0.5p⏐
⏐ ¯N1(0)
⏐
⏐0.5 , (78)
¯P[z]
1 (0) =
ˇb∫
− ˇb
dz1,1
ˇz1,1∫
− ˇz1,1
dz1,2 · · ·
ˇz1,p− 1∫
− ˇz1,p− 1
⃗
ρ(z1)dz1,p
(2π)0.5p⏐
⏐ ¯N[z]
1
⏐
⏐0.5 , (79)
P(
γ1 = 0) = ¯P[z]
1 (0) +P(γ0 = 0)( ¯P1(0) − ¯P[z]
1 (0)), (80)
E[γ1] =1 − P(γ1 = 0) (81)
where ¯N[z]
0 , α0 and P[z]
0 are computed using (39), (42) and (44);
and ¯N1(0) ≜ V ar(z1|γ0 = 0), ¯N[z]
1 ≜ V ar(z1|z0), ˘ρ(z1) ≜ exp
{
−
0.5zT
1
(¯N1(0)
) − 1z1
}
, ⃗
ρ(z1) ≜ exp
{
− 0.5zT
1
(¯N[z]
1
) − 1z1
}
, ¯P1(0) ≜
P(
γ1 = 0|γ0 = 0) and ¯P[z]
1 (0) ≜ P(γ1 = 0|z0). Making reference
to the derivation of Algorithm 1, we easily derive (74). Usin g
(46), we directly derive (75) and (81). Similar to the deriva tion
of Algorithm 2, we easily obtain (76) − (80).
V. S IMULATION EXAMPLE
In this section, we illustrate the performance of the MMSE
state estimator and the communication rate estimation algo -
rithms proposed in this paper via a target tracking scenario
including two parts. More precisely, we test the performanc e
of the proposed results in Section V -A, and we compare the
MMSE state estimator proposed in this paper with the state
estimator proposed in [10] in Section V -B. The state estimat or
proposed in [10] is referred as SEHI considering that its eve nt-
triggered scheduler is based on H¨ older inﬁnity-norm.
A. Performance Evaluation of the Proposed Results
Consider a target tracking problem [29] where the state-
space formulation of the target can be written as (1) with
xk =


pk
vk
ak

 , A =


1 T T 2
0 1 T
0 0 1

 ,
Q = 2a
σ 2
m


T 5/ 20 T 4/ 8 T 3/ 6
T 4/ 8 T 3/ 3 T 2/ 2
T 3/ 6 T 2/ 2 T

 .
pk, vk and ak stand for the position, velocity and acceleration,
respectively, of the target. T , a and
σ 2
m denote the sampling
period, the maneuver time constant’s reciprocal and the tar get
acceleration’s variance, respectively. The measurement o f the
target can be modeled as (2) where C =
( 1 0 0
0 0 1
)
and
R =
( 60 0
0 10
)
. The initial position, velocity and accelera-
tion of this target are 3410m, 30m/s and 0m/s 2, respectively.
We select
¯x0 =


3500
40
0

 , ¯P0 =


602 602/ T 0
602/ T 2 × 602/ T 2 0
0 0 0

 .
Also, we take T = 1, a = 2 and
σ 2
m = 0.5. We select 1 − α =
0.95 in conﬁdence level. Then, noticing p = 2 and using the

<!-- page 7 -->

7
0 10 20 30 40 50 60 70 80 90 100
Time instant k
4.5
5
5.5
6
6.5
7
7.5
8
8.5Position errors (m)
Case 1
Case 2
Case 3
Fig. 2. RMS Position errors of SECL for three different cases .
0 10 20 30 40 50 60 70 80 90 100
Time instant k
2.2
2.4
2.6
2.8
3
3.2
3.4
3.6
3.8
4
Velocity errors (m/s)
Case 1
Case 2
Case 3
Fig. 3. RMS velocity errors of SECL for three different cases .
chi-square distribution table, we have χ 2
α (p) = 5.991. For the
tolerable upper bound N, we take three different parameter
values given by the following three cases:
Case 1:
N =
( 50 4
4 8
)
; Case 2: N = 0.5×
( 50 4
4 8
)
; Case
3: N =
(
60 10
10 20
)
.
We test the performance of the presented results using
a Monte Carlo simulation with N = 5000 trials, and we
take k = 0,1,· · ·,100 for each trial. We use the root-mean-
square (RMS) error, the communication rate and the average
communication rate as the performance evaluation criteria . At
time step k, the RMS error is deﬁned as
√
1
N ∑N
i=1(
ςk,i − ˆςk,i)2
for N trials where ςk,i stands for the state of ςk at the ith
trial, and ˆςk,i stands for an estimate of ςk,i. Let γk,i denote
the state of γk at the ith trial, and the communication rate
for the proposed MMSE state estimator at time step k can be
computed using the approximation E [γk] ≈ 1
N ∑N
i=1
γk,i where
E[γk] = 1
N ∑N
i=1
γk,i when N approaches inﬁnity. The average
communication rate is deﬁned as γ = lim
k→ ∞
1
k ∑k− 1
j=0 E[γ j], and
the average communication rate is approximately computed
via
γ ≈ 1
101 ∑100
j=0 E[γ j] in the Monte Carlo simulation. The
0 10 20 30 40 50 60 70 80 90 100
Time instant k
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Communication rates
SECL
Algorithm 1
Algorithm 2
Fig. 4. Communication rates of SECL, Algorithm 1 and Algorit hm 2 at
Case 1.
TABLE I
AVERAGE COMMUNICATION RATES OF SECL, A LGORITHM 1 AND
ALGORITHM 2
Case SECL Algorithm 1 Algorithm 2
1 0.3812 0.3730 0.3761
2 0.5684 0.5696 0.5678
3 0.2798 0.2750 0.2712
MMSE state estimator based on conﬁdence level proposed
in this paper is referred to as SECL. The RMS position and
velocity errors of SECL for three different cases are given
in Figs. 2 and 3, respectively. The communication rates of
SECL, Algorithm 1 and Algorithm 2 at Cases 1, 2 and 3 are
provided in Figs. 4, 5 and 6, respectively, where, without lo ss
of generality, we take the information at the 40th trial for
running Algorithms 1 and 2. From Figs. 2 − 6, we ﬁnd that,
for both position and velocity estimate at the three differe nt
cases, SECL has different performances, and the performanc e
is connected with the communication rate. More precisely,
the performance of SECL becomes better and better with the
increase of communication rate. This indicates the effecti ve-
ness of the MMSE state estimator based on conﬁdence level
proposed in this paper. Figs. 4 − 6 also shows that Algorithm
2 provides a good estimate for the communication rate of
SECL because the communication rate yielded by Algorithm
2 is close to the communication rate of SECL. The average
communication rates of SECL, Algorithm 1 and Algorithm 2
for the three cases are provided in Table I. We see from Table
I that the average communication rates of SECL, Algorithm 1
and Algorithm 2 are very close at all the three cases, which
means that both Algorithm 1 and Algorithm 2 yield good
performance for estimating the average communication rate s
of SECL.
Hence, the simulation results indicate that SECL is effecti ve
in solving state estimation with the trade-off between com-
munication rate and state estimation performance, and that
Algorithm 2 yields a good performance in estimating the
communication rate and the average communication rate of
SECL.

<!-- page 8 -->

8
0 10 20 30 40 50 60 70 80 90 100
Time instant k
0.5
0.55
0.6
0.65
0.7
0.75
0.8
0.85
0.9
0.95
1
Communication rates
SECL
Algorithm 1
Algorithm 2
Fig. 5. Communication rates of SECL, Algorithm 1 and Algorit hm 2 at
Case 2.
0 10 20 30 40 50 60 70 80 90 100
Time instant k
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9Communication rates
SECL
Algorithm 1
Algorithm 2
Fig. 6. Communication rates of SECL, Algorithm 1 and Algorit hm 2 at
Case 3.
B. Comparison with SEHI
We compare SECL with SEHI still using the above target
tracking example where we take the same parameter values
except for the tolerable upper bound
N. In the comparison
with SEHI, we take N = Na ≜ 0.695 ×
( 60 10
10 20
)
which is
a slight change of
N at Case 3 in Section V -A through multi-
plying by 0.695. After a Monte Carlo simulation, we get the
average communication rate of SECL is 0 .35 when
N = Na.
Considering fairness, we compare SECL with SEHI under the
same average communication rate. Using Monte Carlo simula-
tion, we get
δ = 1.5565 when the average communication rate
of SEHI is 0 .35. The target’s RMS position errors of SECL
and SEHI under the same average communication rate 0.35
are provided in Fig. 7, and the corresponding velocity error s
are provided in Fig. 8. From observing Figs. 7 and 8, we
ﬁnd that SECL performs better than SEHI in target tracking
accuracy for both position and velocity. Hence, the simulat ion
results show that SECL yields better tracking performance i n
contrast to SEHI.
0 10 20 30 40 50 60 70 80 90 100
Time instant k
6
6.5
7
7.5
8
8.5
9
Position errors (m)
SEHI
SECL
Fig. 7. RMS Position errors of SEHI and SECL under the same ave rage
communication rate 0.35.
0 10 20 30 40 50 60 70 80 90 100
Time instant k
2.8
3
3.2
3.4
3.6
3.8
4
4.2Velocity errors (m/s)
SEHI
SECL
Fig. 8. RMS velocity errors of SEHI and SECL under the same ave rage
communication rate 0.35.
VI. C ONCLUSION
Based on conﬁdence level, a novel event-triggered scheme
has been proposed using the chi-square distribution and reg ular
Gaussian assumption. The novel scheme was applied to the
state estimation problem for discrete-time linear systems in
the environment of wireless sensor network so that a MMSE
state estimator was proposed. Two algorithms for estimatin g
the communication rate of the proposed state estimator have
been developed where, at time step k, the ﬁrst algorithm is
based on information up to k − 1, and the second algorithm is
based on information up to k− 2. A target tracking scenario has
been given to examine the performance of the proposed result s,
and the simulation results have shown that the proposed
state estimator performs better than SEHI under the same
average communication rate. The simulation results have al so
shown that Algorithm 2 provides a good estimate for the
communication rate of the proposed state estimator.
APPENDIX A
PROOF OF LEMMAS 1− 3
Proof of Lemma 1: Using (1), we have

<!-- page 9 -->

9
f (xk|Ik− 1) = f (Axk− 1 + ωk− 1|Ik− 1). (82)
From Assumptions 1 and 2, we see that ωk− 1 is indepen-
dent of xk− 1 and I k− 1. Hence, f (xk|Ik− 1) is a linear com-
bination of two Gaussian and mutually independent random
vectors f (xk− 1|Ik− 1) and
ωk− 1, that is A f(xk− 1|Ik− 1) + ωk− 1.
Then, using Remark 4, we prove 1). Applying (2), we have
f (yk|Ik− 1) = f (Cxk + υk|Ik− 1). Then, using 1) of Lemma 1,
and referring to the proof of 1) of Lemma 1, we prove 2).
Proof of Lemma 2: 1). Using (10), (4) and (2), we get
zk = ΦCxk + Φυ k − Φ ˆyk,k− 1. (83)
Then, using 1) of Lemma 1, and making reference to the proof
of 1) of Lemma 1, we prove 1).
2). Applying (10), (4) and (3), we get
E[zk|Ik− 1] =
ΦE[ ˜yk|Ik− 1] = Φ( ˆyk,k− 1 − ˆyk,k− 1) = 0. (84)
Then, applying 1) of Lemma 2, we prove 2).
Proof of Lemma 3: 1). From (8) and (9), it follows that
ϕk ≤ χ 2
α (p) is equivalent to ∑p
i=1 z2
k,i ≤
χ 2
α (p). Then, using the
deﬁnition of Ωk presented in (12), we obtain
Ωk =
{
zk,1,zk,2,· · ·,zk,p ∈ R
⏐
⏐
⏐
⏐
p
∑
i=1
z2
k,i ≤
χ 2
α (p)
}
. (85)
Then, using (85) and 2) of Lemma 2, we prove 1).
2). Noticing zk ∈ Rp, we have zkzT
k = (
ζk,i j)p× p with ζk,i j =
zk,izk,j. Then, using 2) of Lemma 2, we have
∫
Ωk
f (zk|Ik− 1)zkzT
k dzk
=
∫
Ωk
g(zk)
(2π)0.5p⏐
⏐N[z]
k
⏐
⏐0.5 (
ζk,i j)p× pdzk,1dzk,2 · · ·dzk,p
= 1
(2π)0.5p⏐
⏐N[z]
k
⏐
⏐0.5
( ∫
Ωk
g(zk)
ζk,i jdzk,1dzk,2 · · ·dzk,p
)
p× p
.
(86)
Using (85) and noticing ζk,i j = zk,izk,j, we have
∫
Ωk
g(zk)ζk,i jdzk,1dzk,2 · · ·dzk,p
=
ˇb∫
− ˇb
dzk,1
ˇzk,1∫
− ˇzk,1
dzk,2 · · ·
ˇzk,p− 1∫
− ˇzk,p− 1
g(zk)zk,izk,jdzk,p (87)
where ˇb and ˇzk,i with i = 1,2,· · ·,p − 1 are deﬁned in 1) of
Lemma 3. Substituting (87) into (86), as well as using the
deﬁnition of
ψk,i j and Ψk given in 2) of Lemma 3, we have
∫
Ωk
f (zk|Ik− 1)zkzT
k dzk = (
ψk,i j)p× p
(2π)0.5p⏐
⏐N[z]
k
⏐
⏐0.5 =
Ψk
(2π)0.5p⏐
⏐N[z]
k
⏐
⏐0.5 .
(88)
This complete the proof of the statement.
APPENDIX B
PROOF OF THEOREM 1
Derivation of (22). When
γk = 0, we have
ˆxk =
∫
Rn
xk f (xk|Ik)dxk =
∫
Rn
xk f (xk|Ik− 1,γk = 0)dxk. (89)
By (8), f (xk|Ik− 1,γk = 0) can be expressed as
f (xk|Ik− 1,γk = 0) = f
(
xk|Ik− 1,ϕk ≤ χ 2
α (p)
)
= f (xk|Ik− 1,zk ∈ Ωk)
= f (xk,Ik− 1,zk ∈ Ωk)
f (Ik− 1,zk ∈ Ωk)
=
∫
Ωk f (xk,Ik− 1,zk)dzk
∫
Ωk f (Ik− 1,zk)dzk
=
∫
Ωk f (xk|Ik− 1,zk) f (zk|Ik− 1)dzk
∫
Ωk f (zk|Ik− 1)dzk
(90)
where the second and last equalities are due to (12) and Bayes ’
rule, respectively. Substituting (90) into (89) yields tha t
ˆxk =
∫
Rn
xk
∫
Ωk f (xk|Ik− 1,zk) f (zk|Ik− 1)dzk
∫
Ωk f (zk|Ik− 1)dzk
dxk
=
∫
Rn xk
∫
Ωk f (xk|Ik− 1,zk) f (zk|Ik− 1)dzkdxk
∫
Ωk f (zk|Ik− 1)dzk
=
∫
Ωk f (zk|Ik− 1)
∫
Rn xk f (xk|Ik− 1,zk)dxkdzk
∫
Ωk f (zk|Ik− 1)dzk
=
∫
Ωk f (zk|Ik− 1) ˆx[z]
k dzk
∫
Ωk f (zk|Ik− 1)dzk
. (91)
Making reference to the proof of Theorem 3.2 in [10], we can
obtain
∫
Ωk f (zk|Ik− 1) ˆx[z]
k dzk
∫
Ωk f (zk|Ik− 1)dzk
= ˆxk,k− 1. (92)
where ˆxk,k− 1 is deﬁned in (15). Substituting (92) into (91), we
derive (22).
Derivation of (23). Using (18) and the deﬁnition of conditio nal
covariance matrix, we have
N[z]
k = E
[
(zk − E[zk|Ik− 1])(zk − E[zk|Ik− 1])T]
. (93)
Substituting (84) into (93), and using (10), we derive
N[z]
k = E
[
zkzT
k
]
= E
[
Φ ˜yk ˜yT
k
Φ T]
= Φ(CMkCT + R)Φ T (94)
where Mk is deﬁned in (17). Similarly, we have
E
[
˜xk(zk − [zk|Ik− 1])T
]
= E
[
˜xkzT
k
]
= MkCT
Φ T. (95)
Substituting (94) and (95) into (19) yields (23).
Derivation of (24). Utilizing Kalman ﬁlter, we get
ˆx[z]
k = ˆxk,k− 1 + Kk(zk − E[zk|Ik− 1]) (96)
where ˆx[z]
k and Kk are deﬁned in (13) and (19), respectively,
and N[z]
k in Kk is deﬁned in (18). Substituting (84) into (96),
we obtain
ˆx[z]
k = ˆxk,k− 1 + Kkzk. (97)
From (14), (13) and the deﬁnition of conditional covariance
matrix, it follows that
P[z]
k =E
[(
xk − ˆx[z]
k
)(
xk − ˆx[z]
k
) T]
. (98)
Substituting (97) into (98) and using (16), we obtain

<!-- page 10 -->

10
P[z]
k =E
[
( ˜xk − Kkzk)( ˜xk − Kkzk)T]
=Mk − MkCTΦ TKT
k − KkΦCMk + KkN[z]
k KT
k (99)
where Mk is deﬁned in (17), and the last equality is due to
(94) and (95). Applying (23) and (94), we easily obtain
MkCTΦ TKT
k =KkΦCMk
=KkN[z]
k KT
k = MkCT(CMkCT + R)− 1CMk. (100)
Substituting (100) into (99), we obtain (24).
Derivation of (25). Using (11) and (22), we have
Pk =E
[
(xk − ˆxk)(xk − ˆxk)T]
=
∫
Rn
(xk − ˆxk)(xk − ˆxk)T f (xk|Ik− 1,
γk = 0)dxk
=
∫
Rn
˜xk ˜xT
k f (xk|Ik− 1,
γk = 0)dxk (101)
where ˜xk is deﬁned in (16). Substituting (90) into (101), and
referring to (91), we obtain
Pk =
∫
Ωk f (zk|Ik− 1)M[z]
k dzk
∫
Ωk f (zk|Ik− 1)dzk
(102)
where
M[z]
k ≜
∫
Rn
˜xk ˜xT
k f (xk|Ik− 1,zk)dxk. (103)
Using (103) and making reference to the proof of Theorem
3.2 in [10], we obtain
M[z]
k
=
∫
Rn
( ˜xk − Kkzk + Kkzk)( ˜xk − Kkzk + Kkzk)T f (xk|Ik− 1,zk)dxk
=
( ∫
Rn
( ˜xk − Kkzk)( ˜xk − Kkzk)T f (xk|Ik− 1,zk)dxk
)
+ KkzkzT
k KT
k
=P[z]
k + KkzkzT
k KT
k . (104)
Substituting (104) into (102), we get
Pk =
∫
Ωk f (zk|Ik− 1)(P[z]
k + KkzkzT
k KT
k )dzk
∫
Ωk f (zk|Ik− 1)dzk
=P[z]
k +
Kk
∫
Ωk f (zk|Ik− 1)zkzT
k dzkKT
k
∫
Ωk f (zk|Ik− 1)dzk
=P[z]
k + 1
hk
KkΨkKT
k (105)
where the last equality is because of Lemma 3. This completes
the derivation of (25).
APPENDIX C
PROOF OF THEOREM 2
The following lemma is required to obtain Theorem 2.
Lemma 4: Under the assumption that f (xk− 2|Ik− 2) is Gaus-
sian, it holds that E [yk|Ik− 2,yk− 1] = E[yk|Ik− 2,zk− 1].
Proof: f (yk|Ik− 2,zk− 1) can be given by
f (yk|Ik− 2,zk− 1) = f (Ik− 2,zk− 1,yk)
f (Ik− 2,zk− 1)
= f (zk− 1,yk|Ik− 2)
f (zk− 1|Ik− 2) = f (ηk|Ik− 2)
f (zk− 1|Ik− 2) (106)
where ηk ≜ (zT
k− 1,yT
k )T. Under the assumption that
f (xk− 2|Ik− 2) is Gaussian, we easily see that f (
ηk|Ik− 2)
is Gaussian by making reference to the proof of Lemmas 1
and 2. Hence, we have
f (
ηk|Ik− 2) = g1(ηk)
(2π)0.5× 2p⏐
⏐N[
η]
k
⏐
⏐0.5 = g1(
ηk)
(2π)p⏐
⏐N[
η]
k
⏐
⏐0.5 (107)
where
g1(
ηk) ≜exp
{
− 0.5(ηk − ˆηk,k− 2)T(N[η]
k )− 1(ηk − ˆηk,k− 2)
}
,
(108)
ˆηk,k− 2 ≜E[ηk|Ik− 2],N[η]
k ≜ V ar(ηk|Ik− 2). (109)
Utilizing Lemma 2, we have
f (zk− 1|Ik− 2) = g(zk− 1)
(2π)0.5p⏐
⏐N[z]
k− 1
⏐
⏐0.5 . (110)
Substituting (107) and (110) into (106) yields that
f (yk|Ik− 2,zk− 1) = g1(
ηk)
⏐
⏐N[z]
k− 1
⏐
⏐0.5
(2π)0.5pg(zk− 1)
⏐
⏐N[
η]
k
⏐
⏐0.5 . (111)
In the same way, we can derive
f (yk|Ik− 2,yk− 1) = g1(
ζk)
⏐
⏐Nk− 1
⏐
⏐0.5
(2π)0.5pg2(yk− 1)
⏐
⏐N[
ζ ]
k
⏐
⏐0.5 (112)
where
g1(
ζk) ≜ exp
{
− 0.5(ζk − ˆζk,k− 2)T(N[ζ ]
k )− 1(ζk − ˆζk,k− 2)
}
,
(113)
ζk ≜ (yT
k− 1,yT
k )T,ˆ
ζk,k− 2 ≜ E[ζk|Ik− 2],N[ζ ]
k ≜ V ar(ζk|Ik− 2),
(114)
g2(yk− 1) ≜ exp
{
− 0.5 ˜yT
k− 1N− 1
k− 1 ˜yk− 1
}
. (115)
Using (10) and (84), we have
ηk =
( Φ ˜yk− 1
yk
)
,ˆηk,k− 2 =
( 0
E[yk|Ik− 2]
)
(116)
which means
ηk − ˆηk,k− 2 =
( Φ ˜yk− 1
yk − E[yk|Ik− 2]
)
= Φa(ζk − ˆζk,k− 2) (117)
where
Φa ≜
( Φ 0
0 Ip
)
. (118)
Using the deﬁnition of conditional covariance matrix and
utilizing (117), we easily obtain
N[
η]
k = ΦaN[ζ ]
k Φ T
a (119)
where N[η]
k and N[ζ ]
k are deﬁned in (109) and (114), respec-
tively. Substituting (117) and (119) into (108) yields that
g1(ηk) =exp
{
− 0.5(ζk − ˆζk,k− 2)TΦ T
a Φ − T
a (N[ζ ]
k )− 1Φ − 1
a Φa(ζk
− ˆζk,k− 2)
}
=exp
{
− 0.5(ζk − ˆζk,k− 2)T(N[ζ ]
k )− 1(ζk − ˆζk,k− 2)
}
.
(120)

<!-- page 11 -->

11
Putting (120) and (113) together, we have
g1(ηk) = g1(ζk). (121)
In the same way, we can obtain
g(zk− 1) = g2(yk− 1). (122)
From (119), it follows that
⏐
⏐N[
η]
k
⏐
⏐ =
⏐
⏐
Φa
⏐
⏐⏐
⏐N[
ζ ]
k
⏐
⏐⏐
⏐
Φ T
a
⏐
⏐ =
⏐
⏐
Φa
⏐
⏐2⏐
⏐N[
ζ ]
k
⏐
⏐ (123)
where the last equality is because the determinant of a matri x
is equal to the determinant of its transpose. Noting that
Φa
deﬁned in (118) is a block-diagonal matrix, we have
|Φa|= |Φ||Ip|= |Φ|. (124)
Substituting (124) into (123) yields that
⏐
⏐N[
η]
k
⏐
⏐ =
⏐
⏐
Φ
⏐
⏐2⏐
⏐N[
ζ ]
k
⏐
⏐. (125)
Similarly, we can get
⏐
⏐N[z]
k− 1
⏐
⏐ = |
Φ|2|Nk− 1|. (126)
Substituting (121), (122), (125) and (126) into (111), we de rive
f (yk|Ik− 2,zk− 1) = g1(ζk)(|Φ|2|Nk− 1|)0.5
(2π)0.5pg2(yk− 1)
(⏐
⏐
Φ
⏐
⏐2⏐
⏐N[
ζ ]
k
⏐
⏐) 0.5
= g1(
ζk)|Nk− 1|0.5
(2π)0.5pg2(yk− 1)
⏐
⏐N[
ζ ]
k
⏐
⏐0.5 . (127)
Combining (112) and (127), we derive f (yk|Ik− 2,yk− 1) =
f (yk|Ik− 2,zk− 1) which means Lemma 4 holds.
Now, we provide a proof of Theorem 2.
Derivation of (61). Substituting (1) at time step k into (2), we
obtain
yk = CAxk− 1 + C
ωk− 1 + υk. (128)
When γk− 1 = 1, we have I k− 1 = (Ik− 2,yk− 1). Hence, we have
ˆyk,k− 1 = E[yk|Ik− 2,yk− 1] = E[yk|Ik− 2,zk− 1] (129)
where ˆyk,k− 1 is deﬁned in (3), and the last equality is due to
Lemma 4. Substituting (10) into (52), we get
ˆz⊲
k,k− 1 =
ΦE[ ˜yk|Ik− 2,zk− 1] = 0 (130)
where ˜yk is deﬁned in (4), and the last equality is due to (129).
Substituting (130) into (53), we derive
˜z⊲
k =zk. (131)
Substituting (128) into (3) at I k− 1 = (Ik− 2,yk− 1), as well as
using Assumptions 1 and 2, we get
ˆyk,k− 1 =CAE[xk− 1|Ik− 2,yk− 1]
=CA( ˆxk− 1,k− 2 + Kk− 1
Φ ˜yk− 1) = CA ˆx[z]
k− 1 (132)
where the second equality is due to the Kalman ﬁlter and (35),
and the last equality is because of (10) and (97). Substituti ng
(1) and (132) into (83), we obtain
zk =
ΦCAxk− 1 + ΦCωk− 1 + Φυ k − ΦCA ˆx[z]
k− 1
=Φ
(
CA(xk− 1 − ˆx[z]
k− 1) +Cωk− 1 + υk
)
(133)
given (Ik− 2,zk− 1). Utilizing (52) − (54) and the deﬁnition of
conditional covariance matrix, we have
⃗N[z]
k = E
[
˜z⊲
k (˜z⊲
k)T]
. (134)
Substituting (131) into (134) and using (133), we get
⃗N[z]
k =E
[(
Φ
(
CA(xk− 1 − ˆx[z]
k− 1) +Cωk− 1 + υk
))(
Φ
(
CA(xk− 1
− ˆx[z]
k− 1) +Cωk− 1 + υk
)) T]
=Φ
(
C(AP[z]
k− 1AT + Q)CT + R)Φ T (135)
where P[z]
k is deﬁned in (14). This completes the derivation of
(61).
Derivation of (62). Making reference to (129) and (130), we
get
ˆzk,k− 1(0) =
ΦE[ ˜yk|Ik− 2,γk− 1 = 0] = 0 (136)
where ˆzk,k− 1(0) is deﬁned in (55). Substituting (128) into (3),
as well as using Assumptions 1 and 2, we have
ˆyk,k− 1 =CA ˆxk− 1. (137)
Substituting (1) and (137) into (83) yields that
zk =ΦCAxk− 1 + ΦCωk− 1 + Φυ k − ΦCA ˆxk− 1
=Φ
(
CA(xk− 1 − ˆxk− 1) +Cωk− 1 + υk
)
. (138)
Using (56), (55) and the deﬁnition of conditional covarianc e
matrix, we obtain
Nk(0) =E
[(
zk − ˆzk,k− 1(0)
)(
zk − ˆzk,k− 1(0)
) T]
. (139)
Substituting (138) and (136) into (139), we get
Nk(0) =E
[(
Φ(CA(xk− 1 − ˆxk− 1) +Cωk− 1 + υk)
)(
Φ(CA(xk− 1
− ˆxk− 1) +Cωk− 1 + υk)
) T]
=Φ
(
C
(
A(P[z]
k− 1 + 1
hk− 1
Kk− 1Ψk− 1KT
k− 1)AT + Q
)
CT + R
)
× Φ T (140)
where the last equality is due to (11) and (25) of Theorem
1 for given (Ik− 2,
γk− 1 = 0). This completes the derivation of
(62).
Derivation of (64) and (65). When f (xk− 1|Ik− 1) is Gaussian,
we see that f (xk− 1|Ik− 2) is Gaussian. Then, using Remark
4 and (133), we conclude that f (zk|Ik− 2,zk− 1) is Gaussian.
Hence, we have
f (zk|Ik− 2,zk− 1) = exp
{
− 0.5(˜z⊲
k)T(⃗N[z]
k )− 1 ˜z⊲
k
}
(2π)0.5p⏐
⏐⃗N[z]
k
⏐
⏐0.5 (141)
where ˜z⊲
k and ⃗N[z]
k are deﬁned in (53) and (54), respec-
tively. Substituting (131) into (141) and replacing exp
{
−
0.5zT
k (⃗N[z]
k )− 1zk
}
by ⃗g(zk) yield (64). Similarly to the deriva-
tion of (64), we can obtain (65) by using (136) and (56).
Derivation of (66) and (67). P(
γk = 0|Ik− 2,zk− 1) can be given
by

<!-- page 12 -->

12
P(γk = 0|Ik− 2,zk− 1) =P(zk ∈ Ωk|Ik− 2,zk− 1)
=
∫
Ωk
f (zk|Ik− 2,zk− 1)dzk. (142)
Then, replacing P(γk = 0|Ik− 2,zk− 1) by ⃗P[z]
k (0), we derive (66).
In the same way, we can obtain (67) where ˘Pk(0) is deﬁned
in (51).
Derivation of (68). P(
γk = 0|Ik− 2) can be given by
P(γk = 0|Ik− 2)
=P(γk− 1 = 0|Ik− 2)P(γk = 0|Ik− 2,γk− 1 = 0)
+ P(γk− 1 = 1|Ik− 2)P(γk = 0|Ik− 2,γk− 1 = 1). (143)
P(γk = 0|Ik− 2,γk− 1 = 1) can be given by
P(γk = 0|Ik− 2,γk− 1 = 1)
= P(Ik− 2,γk− 1 = 1,γk = 0)
P(Ik− 2,γk− 1 = 1)
= P(Ik− 2,zk− 1 ∈ ˘Ωk− 1,γk = 0)
P(Ik− 2,zk− 1 ∈ ˘Ωk− 1)
=
∫
˘Ωk− 1
f (Ik− 2,zk− 1,γk = 0)dzk− 1
∫
˘Ωk− 1
f (Ik− 2,zk− 1)dzk− 1
=
∫
˘Ωk− 1
⃗P[z]
k (0) f (Ik− 2,zk− 1)dzk− 1
∫
˘Ωk− 1
f (Ik− 2,zk− 1)dzk− 1
=
∫
˘Ωk− 1
⃗P[z]
k (0) f (zk− 1|Ik− 2)dzk− 1
∫
˘Ωk− 1
f (zk− 1|Ik− 2)dzk− 1
=
∫
˘Ωk− 1
⃗P[z]
k (0) f (zk− 1|Ik− 2)dzk− 1
P(γk− 1 = 1|Ik− 2) (144)
where ˘Ωk and ⃗P[z]
k (0) are deﬁned in (57) and (50), respectively.
Substituting (144) into (143), as well as replacing P(γk− 1 =
0|Ik− 2) by Pk− 1,k− 2(0), we obtain
P(γk = 0|Ik− 2) =Pk− 1,k− 2(0)P(γk = 0|Ik− 2,γk− 1 = 0)
+
∫
˘Ωk− 1
⃗P[z]
k (0) f (zk− 1|Ik− 2)dzk− 1. (145)
Then, replacing P(γk = 0|Ik− 2) and P(γk = 0|Ik− 2,γk− 1 = 0)
by Pk,k− 2(0) and ˘Pk(0), respectively, we derive (68). Making
reference to (46), we easily obtain (69).
APPENDIX D
PROOF OF ALGORITHM 2
Making reference to the proof of 1) of Lemma 3, we can
obtain (71) by using (64) and (66). Similarly, we obtain (72)
by using (65) and (67). From (66), (64) and (70), we see that
⃗P[z]
k (0) does not contain the random vector zk− 1. Hence, we
have∫
˘Ωk− 1
⃗P[z]
k (0) f (zk− 1|Ik− 2)dzk− 1
=⃗P[z]
k (0)
∫
˘Ωk− 1
f (zk− 1|Ik− 2)dzk− 1
=⃗P[z]
k (0)
(∫
Rp
f (zk− 1|Ik− 2)dzk− 1 −
∫
Ωk− 1
f (zk− 1|Ik− 2)dzk− 1
)
=⃗P[z]
k (0)
(
1 −
∫
Ωk− 1
f (zk− 1|Ik− 2)dzk− 1
)
=⃗P[z]
k (0)
(
1 − hk− 1
(2π)0.5p⏐
⏐N[z]
k− 1
⏐
⏐0.5
)
=⃗P[z]
k (0)
(
1 − Pk− 1,k− 2(0)
)
(146)
where the fourth equality is due to 1) of Lemma 3, and the
last equality is because of (58). Substituting (146) into (6 8),
we get
Pk,k− 2(0) =Pk− 1,k− 2(0) ˘Pk(0) +⃗P[z]
k (0)
(
1 − Pk− 1,k− 2(0)
)
=⃗P[z]
k (0) +Pk− 1,k− 2(0)
(˘Pk(0) − ⃗P[z]
k (0)
)
, (147)
which means that (73) holds.
REFERENCES
[1] J. P . Hespanha, P . Naghshtabrizi, and Y . Xu, “A survey of r ecent results
in networked control systems”, Proc. IEEE , vol. 95, no. 1, pp. 138-162,
Jan. 2007.
[2] S. Wildhagen, J. Berberich, M. Hertneck, and F. Allg¨ owe r, “Data-driven
analysis and controller design for discrete-time systems u nder aperiodic
sampling”, IEEE Trans. Autom. Control , vol. 68, no. 6, pp. 3210-3225,
Jun. 2023.
[3] J. Shang, H. Y u, and T. Chen, “Worst-case stealthy attack s on stochastic
event-based state estimation”, IEEE Trans. Autom. Control , vol. 67, no.
4, pp. 2052-2059, Apr. 2022.
[4] W. Liu, P . Shi, and S. Wang, “Distributed Kalman ﬁltering through trace
proximity”, IEEE Trans. Autom. Control , vol. 67, no. 9, pp. 4908-4915,
Sep. 2022.
[5] J. Hu, B. Lennox, and F. Arvin, “Robust formation control for networked
robotic systems using negative imaginary dynamics”, Automatica, vol.
140, 2022, Art. no. 110235.
[6] I. Z. Petric, P . Mattavelli, and S. Buso, “Multi-sampled grid-connected
VSCs: A path toward inherent admittance passivity”, IEEE Trans. Power
Electron., vol. 37, no. 7, pp. 7675-7687, Jul. 2022.
[7] J. J. Xiao, A. Ribeiro, Z. Q. Luo, and G. B. Giannakis, “Dis tributed
compression-estimation using wireless sensor networks”, IEEE Trans.
Signal Process. Mag. , vol. 23, no. 4, pp. 27-41, Jul. 2006.
[8] A. Ribeiro, G. B. Giannakis, and S. I. Roumeliotis, “SOI- KF: Distributed
Kalman ﬁltering with low-cost communications using the sig n of inno-
vations”, IEEE Trans. Signal Process. , vol. 54, no. 12, pp. 4782-4795,
Dec. 2006.
[9] E. J. Msechu, S. I. Roumeliotis, A. Ribeiro, and G. B. Gian nakis,
“Decentralized quantized Kalman ﬁltering with scalable co mmunication
cost”, IEEE Trans. Signal Process. , vol. 56, no. 8, pp. 3727-3741, Aug.
2008.
[10] J. Wu, Q. S. Jia, K. H. Johansson, and L. Shi, “Event-base d sensor
data scheduling: Trade-off between communication rate and estimation
quality”, IEEE Trans. Autom. Control , vol. 58, no. 4, pp. 1041-1046,
Apr. 2013.
[11] G. M. Lipsa and N. C. Martins, “Remote state estimation w ith commu-
nication costs for ﬁrst-order LTI systems”, IEEE Trans. Autom. Control ,
vol. 56, no. 9, pp. 2013-2025, Sep. 2011.
[12] G. Battistelli, A. Benavoli, and L. Chisci, “Data-driv en communication
for state estimation with sensor networks”, Automatica, vol. 48, no. 5,
pp. 926-935, May 2012.
[13] J. Sijs and M. Lazar, “Event based state estimation with time syn-
chronous updates”, IEEE Trans. Autom. Control , vol. 57, no. 10, pp.
2650-2655, Oct. 2012.
[14] K. Y ou and L. Xie, “Kalman ﬁltering with scheduled measu rements”,
IEEE Trans. Signal Process. , vol. 61, no. 6, pp. 1520-1530, Mar. 2013.
[15] D. Shi, T. Chen, and L. Shi, “An event-triggered approac h to state esti-
mation with multiple point- and set-valued measurements”, Automatica,
vol. 50, no. 6, pp. 1641-1648, Jun. 2014.
[16] S. Trimpe and R. D’Andrea, “Event-based state estimati on with
variance-based triggering”, IEEE Trans. Autom. Control , vol. 59, no. 12,
pp. 3266-3281, Dec. 2014.
[17] D. Shi, T. Chen, and L. Shi, “On set-valued Kalman ﬁlteri ng and its
application to event-based state estimation”, IEEE Trans. Autom. Control,
vol. 60, no. 5, pp. 1275-1290, May 2015.

<!-- page 13 -->

13
[18] D. Han, Y . Mo, J. Wu, S. Weerakkody, B. Sinopoli, and L. Sh i,
“Stochastic event-triggered sensor schedule for remote st ate estimation”,
IEEE Trans. Autom. Control , vol. 60, no. 10, pp. 2661-2675, Oct. 2015.
[19] S. Weerakkody, Y . Mo, B. Sinopoli, D. Han, and L. Shi, “Mu lti-sensor
scheduling for state estimationwith event-based, stochas tic triggers”,
IEEE Trans. Autom. Control , vol. 61, no. 9, pp. 2695-2701, Sep. 2016.
[20] A. Mohammadi and K. N. Plataniotis, “Event-based estim ation with
information-based triggering and adaptive update”, IEEE Trans. Signal
Process., vol. 65, no. 18, pp. 4924-4939, Sep. 2017.
[21] L. He, J. Chen, and Y . Qi, “Event-based state estimation : Optimal
algorithm with generalized closed skew normal distributio n”, IEEE Trans.
Autom. Control, vol. 64, no. 1, pp. 321-328, Jan. 2019.
[22] Z. Hu, B. Chen, R. Wang, and L. Y u, “Remote state estimati on with
posterior-based stochastic event-triggered schedule”, IEEE Trans. Autom.
Control, vol. 69, no. 2, pp. 1194-1201, Feb. 2024.
[23] H. Y u, J. Shang, and T. Chen, “On stochastic and determin istic event-
based state estimation”, Automatica, vol. 123, 2021, Art. no. 109314.
[24] G. Battistelli, L. Chisci, and D. Selvi, “A distributed Kalman ﬁlter with
event-triggered communication and guaranteed stability” , Automatica,
vol. 93, pp. 75-82, Jul. 2018.
[25] M. Miskowicz, “Send-on-delta concept: An event-based data reporting
strategy”, Sensors, vol. 6, no. 1, pp. 49-63, Jan. 2006.
[26] R. Cogill, S. Lall, and J. P . Hespanha, “A constant facto r approximation
algorithm for event-based sampling”, in Proc. 2007 Amer . Control Conf. ,
New Y ork, USA, Jul. 2007, pp. 305-311.
[27] L. Li, M. Lemmon, and X. Wang, “Event-triggered state es timation in
vector linear processes”, in Proc. 2010 Amer . Control Conf. , Baltimore,
USA, Jun. 2010, pp. 2138-2143.
[28] R. A. Johnson and D. W. Wichern, Applied multivariate statistical
analysis, 6th ed., Upper Saddle River: Pearson prentice hall, 2007.
[29] R. A. Singer, “Estimating optimal tracking ﬁlter perfo rmance for manned
maneuvering targets”, IEEE Trans. Aerosp. Electron. Syst. , vol. AES-6,
no. 4, pp. 473-483, Jul. 1970.
