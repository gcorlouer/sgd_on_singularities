
In singular learning theory (SLT) we take a bayesian perspective and understand model selection as minimisation of the free energy for some model. One key insight of SLT is that free energy minisation is achieved by achieving some optimal tradeoff between accuracy and another term which governs the geometry of the model (the real log canonical threshold, RLCT). More specifically, the RLCT counts the effective dimension of the model near the selected region which depends on the singularity at the critical point of the relative information.

Intuitively, after observing data, the posterior narrows down around a singularity of the relative information, chosing the singularity with lowest RLCT.

Model selection in deep learning is done via stochastic gradient descent - not Bayes updating - so to apply the insight of SLT to neural networks, we really want to understand how the SLT story translates to SGD. At a high level, we want to understand how SGD behaves around the singularities, i.e. we want to understand the relationships between the geometry of the relative information and where SGD ends up in the course of training and how phase transitions that induce a shift in the posterior and/or free energy affect SGD.

## Statistical learning 

Let  $D_n := \{X_1,...,X_n\}$ be a set of random variable where each random variable $X_i$ is distributed according to a distribution $q(X_i)$ - also known as the true distribution. In SLT, we make the i.i.d assumption, i.e., each $X_i$ has is distributed by $q$ and the random variables $X_i$ are statistically independant.

The goal of statistical learning is about finding a parametrised distribution $p(X|w)$ for $w\in W$ that approximates well the true distribution $q$. We can operationalise "approximation" here by stating that, more specifically, statistical learning is about finding the parameters that minimise the relative information between the true distribution $q$ and $p(X|w)$: i.e. we want to find $W^*\subset W$ such that:

$$W^* := \arg\min_{w}\left(K(w)\right)$$
With 
$$K(w) := K\left[q(X)||p(X|w)\right] = \mathbb{E_q}\left[\ln\left(\frac{q(X)}{p(X|w)}\right)\right] $$

Let $W_0 := \{w\in W| K(w) = 0\}$ be the set of *true* parameters. Let $\mathcal{M}:=\{p(x|w), w\in W\}$ be a model. A model is *realisable* if $W^* = W_0$. A model is *identifiable* if the map $W \to \mathcal{P}$; $w \mapsto p(x|w)$ is injective. In singular learning theory we will often work with models that are not identifiable and sometimes also not realisable - which are hypotheses that often assumed by statisticians.

Note the following identity, with $S(q)$ the entropy of $q$ and $S(q,p)$ the relative entropy:

$$K(w) = -S[q] + S[q,p(X|w)] \geq 0$$

Hence minising the KL divergence with respect to $w$ is equivalant to minisming the relative entropy $S[q,p(X|w)]$, the entropy $S[q]$ being the intrinsic uncertainty of $q$.

Define the empirical KL divergence as:

$$K_n(w) = \frac{1}{n} \sum_{i=1}^{n}\ln\left(\frac{q(X_i)}{p(X_i|w)}\right) = \frac{1}{n} \sum_{X\in D_n}\ln\left(\frac{q(X)}{p(X|w)}\right) $$

We have $K(w) = \mathbb{E}_q[K_n(w)]$ and, thanks to the law of large numbers, $K_n(w) \to_{a.s.} K(w)$ 

Observe that:

$$K_n(w) = - S_n + \frac{1}{n}L_n(w)  $$

where $L_n(w):=-\sum_{i=0}^{n}\log p(X_i|w)$ is the negative log-likelihood. It is interesting to note finding the parameters that best represents the probability $q$ via minimising the KL divergence is equivalent to minimising the cross-entropy. Empirically, we can learn $q$ via minising the empirical KL divergence which is equivalent to minising the negative log-likelihodd $L_n$. 

>Note: Define Fischer information.

## The deep learning setup

A deep neural network is a function 

$$
\begin{align}
f:\ & \mathcal{X}\times W \to \mathcal{Y} \\
    & (x,w)            \mapsto f(x,w)
\end{align}
$$

Typically - in deep-learning - we have $f = W^{L}_b\prod_{l=L-1}^{1}\sigma\circ  W^{l}_b$; a composition of non-linear (activation) functions $\sigma$ and affine functions $W^{l}_b$ along a number of layers $L$. Let $(x_i,y_i)\in \mathcal{X}\times\mathcal{Y}$, a deep neural network can learn any continuous function by selecting the parameters $W^*\subset W$ that minise the loss function:

$$L^{f}_n(w) := \sum_{i=1}^{n} ||y_i - f(x_i,w)||$$

A deep neural network learns by updating its weights $w\in W$ via stochastic gradient descent - an iterative algorithm. For a random sample of indices - with fixed size - at each time t, $b(t)\subset\{1,...,n\}$ - a batch - we have randomly initialise $w_0\in W$ and for all $t$, $w_t\in W$ we have the weight update:

$$
w_{t+1} = w_t - \nabla_{w_t} L^{f}_{b(t)}(w_t)
$$

In expectation, for all $t$, we have $L^{f}_n(w) = \mathbb{E}[L^{f}_{b(t)}(w)]$ (see, for example, this [paper](https://arxiv.org/pdf/1704.04289.pdf))


The deep learning setup is a particular case of the statistical learning setup covered in the previous section. Indeed, for $(x_i,y_i)\in\mathcal{X}\times\mathcal{Y}$ we want to represents the true distribution $q$ of the data $D_n=\{(x_1,y_1),...,(x_n,y_n)\}$ with the parametrised distribution $p((x,y)|w):=C\exp\left(-L^{f}_n(w)\right)$ where $C$ is a normalisation constant by minising the KL divergence $K(w)$. By computing the negative log-likelihood, one can check that for such $p((x,y)|w)$:

$$ L_n(w) = L^{f}_n(w) - C$$

Furthermore, since $K_n(w)$ only depends on $w$ via $L_n(w)$ we have

$$
\nabla_w K_n(w) = \nabla_w L_n(w)
$$

The two later facts allow us to re-write SGD as the following update equation, for all time $t$ and batch $b(t)$:

$$
w_{t+1} = w_t - \nabla_{w_t} K_{b(t)}(w_t)
$$

## Geometry of learning

Need to cover:
* Fisher information
* regular and singular models
* resolution of singularities

### Fisher information




## Free energy minisation



## SGD and relative information (deprecated)

A deep neural network learns a function $f(x,\omega)$ by minimising a loss function 

$$L_n(w) = \frac{1}{2n}\sum_i ||y_i - f(x_i,w)|^2 = \frac{1}{2n}\sum_i l_i(w)$$ 

Let $i_t\subset \{1,...,n\}$ such that the weights are updated upon sampling a batch $(y_{ik},x_{i_k})$ via:

$$w_{t+1} = w_t -\eta \nabla_{w_t} L_{i_t}(w_t)$$ 

In SLT we minimise the empirical KL divergence $K_n$ as a sum of :

$$ K_n(w) =  \frac{1}{n} \sum_i k_i = \frac{1}{n} \sum_i \log\frac{q(x_i)}{p(x_i|w)}$$
with $k_i := \log\frac{q(x_i)}{p(x_i|w)}$.
> Shouldn't the KL divergence involve $y$ somehow? Also can you briefly define $q$ and $p$? (I keep forgetting definitions, so it'd be useful for me to have them in one place).

Chosing for the model $p(x_i|w):= C\exp(-\mathcal{L}_n(w))$ for some normalisation constant $C$ we readily compute that:

$$\nabla L_{i_t}(w) = \nabla K_{i_t}(w) $$

> This is not obvious to me; can you perhaps point to some resource (or show it if it's quick)? I guess it's non-obvious to me partly because I don't recall the relationship between $p$, $q$, $f$ and $y$.

and re-write SGD in terms of the gradient of the in-sample relative information:

$$w_{t+1} = w_t - \eta \nabla_{w_t} K_{i_t}(w_t)$$

The SGD scheme updates the weights $w\in W$ by descending along the gradient of $K_{i_t}$ upon seeing the batch $x_{i_t}$

## First Watanabe fundamental's theorem

Recall Watanabe main theorem 6.1 that under some (mild?) assumptions (see fundamental assumptions I): 

Let $W_{\epsilon}:=\{w\in W| K(w)\leq \epsilon \}$
There exist a constant $\epsilon > 0$, a real analytic manifold $\mathcal{M}^{R}$ and a real analytic function $g:\mathcal{M}^{R}\to W_{\epsilon}^{R}$ such that in every local coordinate U of $\mathcal{M}^{R}$:

$$K(g(u)) = u^{2k}$$ 

and the empirical relative information is represented in $U$ by:

$$K_n(g(u)) = u^{2k} - \frac{1}{\sqrt{n}}u^k\xi_n(u)$$

where $\xi_n(u)$ is an empirical process that converges in distribution to a gaussian process $\xi(u)$ defined by:

$$\xi_n(u):=\frac{1}{\sqrt{n}}\sum_{i=1}^{n}\left[u^{k} - a(X_i,u)\right ]$$

with $a(X_i,u)$ such that:

$$k_i(u) = k(X_i,u) = a(X_i,u)u^k $$

(see Watanabe introduction main formula 1 and section 6.1 for more details)

Note that by $k_i(u)$ we mean the representant of $k_i(w)$ in a given chart of the Manifold $\mathcal{M}$ resolving the singularities.
> Do you have any intuition for this result?

## SGD and stochastic fluctuations

> I'm still confused why one can apply the Watanabe theorem in the case of SGD. You've claimed above that the loss gradient is equal to the empirical KLD gradient. That seems like a step in the direction of applying Watanabe. But you still need some assumptions about the dynamics, I assume. One question there: I assume in practice the model you select depends on the batch size, is that right? Whereas in the Bayesian learning picture, I presume it doesn't (since Bayesian posteriors do not depend on the order in which you update on several pieces of evidence)? What's going on here? Maybe the hope is that Bayesian conditionalization correctly approximates SGD when hyperparameters like batch size are chosen "optimally", for some notion of optimality?

The weight update under SGD is governed by $\nabla K_{i_t}$. Because the stochastic fluctuations of $K_{i_t}$ are ill define on the singularities we need to pull back SGD in the resolution where we will study the field $\xi_n(u)$ representing the fluctuations of $K_{i_t}$ in the resolution.   Let's for now assume that the main formula 1 in Watanabe holds for batches - which seems true - i.e. in the resolution of a singularity, locally:

$$K_{i_t}(u) = u^{2k} - \frac{1}{b}u^k\xi_{i_t}(u)$$

with $\xi_{i_t}= \frac{1}{\sqrt{b}}\sum_{i=i_t(0)}^{b} \left[ u^k - a(X_i,u) \right]$

Let $m=i_t$, we derive:

$$\nabla K_m(u) = 2k u^{2k'} - \frac{k}{b} u^{k'} \xi_{m}(u) -\frac{1}{b} u^k \nabla\xi_m(u)$$

where $k'$ is such that $u^{k'} = vec(u_1^{k_1-1}u_2^{k_2}...u_d^{k_d}, ..., u_1^{k_1}u_2^{k_2}...u_d^{k_d-1})$

Intuition: To understand the influence of the geometry around the singularity on SGD we want to understand the influence of the geometry of $\mathcal{M}^R$ on $\xi_m(u)$ and $\nabla\xi_m(u)$

Questions: 
- Can we further simplify $\nabla K_m(u)$?
- Can we neglect some terms?
- What can we say about $\nabla\xi_m(u)$?
- How to study SGD in a way that does not depends on the choice of local chart of resolution?
- Take a simple example!
- What does it mean for a neural network to be singular, in-terms of the loss? Are neural netwoks generally singular?

# SLT for SGD

## Toy model of SGD on singular models

### Linear regression with polynomial coefficients.

Consider the model $p(y|x,w) = \mathcal{N}(Q(w)x, \sigma)$ and the truth $q(y|x) = \mathcal{N}(Q(w_0)x, \sigma)$. 
We know that for a linear regression the KL divergence $K(w)$ writes: 
$$K(w)  = \frac{(Q(w_0) - Q (w))^2 x^2}{\sigma^2}$$

If $w_0$ is a zero of $K$, integrating over x (?) then we also have $K(w) = Q(w)^2$. 

In particular, we chose $Q$ such that: 
$$ Q(w) = (w - w_0)^2(w+w_0)$$
While this choice of model seems contrived - in the sense of not corresponding to a realistic situation - it should enable us to gain better intuition about the behaviour of SGD on singular models. Such choise of singular model and truth leads to a KL divergence:
$$K(w) = (w-w_0)^4(w+w_0)^2$$
Which is probably one of the simplest workable model containing two minima, one of which corresponds to a singular model and the other a regular model.

The corresponding empirical KL divergence writes:

$$K_n(w) = - \frac{1}{n}\sum_{i=1}^n y_i^2 - (y_i - Q(w)x_i)^2$$

When expanding it:

$$K_n(w) = - \frac{Q(w)}{2n}∑_{i=1}^n [2x_iy_i - Q(w)x_i^2].$$

When the x,y are independant and in large sample we recover $K(w)$ using the law of large number.

Watanabe formula I:

$$K_b(u) = u^{2k} - \frac{1}{\sqrt{b}}u^k\xi_{b_t}(u)$$

We consider the limit of large sample i.e. $\xi_b \to \xi$ where $\xi$ is a gaussian field over $W$.

We want to understand how the noise varies around the singularities to improve our understanding of SGD for singular models. Formally, we want to compute 

$$C(w_1,w_2) = Cov(\nabla \left(\sqrt{K}\xi\right)(w_1), \nabla\left(\sqrt{K}\xi\right)(w_2))$$

Note that the field $\sqrt{K}\xi$ is Gaussian.
We use the fact that the covariance of the derivative of a gaussian field with 0 mean is equal to the second derivative of the covariance of that same Gaussian field. Hence:

$$C(w_1, w_2) = \nabla^2 Cov(\sqrt{K}\xi(w_1), \sqrt{K}\xi(w_2)) = \nabla^2 \sqrt{K}(w_1) \sqrt{K}(w_2) \Gamma(w_1,w_2)$$

Note that we are heavily abusing notation and that to be rigorous we should look at the latter equality within local charts.

It is particularly informative to look at the point: $w_1 = w_2 = w$ in which case:

$C(w_1,w_2)|_{w_1=w_2} = \nabla^2\left(K(w)\Gamma(w)\right)$

Which leads to the following expression for the covariance:

$$\nabla^2K(w). \Gamma(w) + 2 \nabla K(w)\nabla \Gamma(w) + K(w)\nabla^2\Gamma(w)$$

Let's calculate each derivative at $w_0 + \epsilon$ and at $- w_0 + \epsilon$. First let's observe that the latter equation will be dominated by $\nabla^2K(w). \Gamma(w)$ around the singularity.

$$K(w_0 + \epsilon) = 4\epsilon^4w_0 + o(\epsilon^5)$$

$$\nabla^2 K(w) = 12(w - w_0)^2(w + w_0)^2 + 16(w-w_0)^3(w+w_0) + 2(w-w_0)^4$$

Close to the singularity $w = w_0 + \epsilon$ we have:

$$\nabla^2 K(w_0 + \epsilon) =  12(ε^2)(4w_0^2 + 4w_0ε + ε^2) + 16(ε^3)(2w_0 + ε) + 2(ε^4)$$

We find that:

$$\nabla^2 K(w_0 + \epsilon)(w_0 + ε) ≈ 12(ε^2)(4w_0^2) = 48w_0^2ε^2 + o (\epsilon^3) $$

Let's now look at the singularity $-w_0 + \epsilon$:

We find that 

$$\nabla^2 K(-w_0 + ε) = 32w_0 + o(\epsilon) $$

We see that around the regular point the curvature dominate the curvature around the singular point: the consequence is that the covarience of the noise will dominate and gradient descent will "push" more strongly outside of the regular point. 


Questions:
* Numerical experiments compare Bayes et SGD on simple model
* Look in the resolution
* Look into 2 singularity with differents RLCT
* Chat with Edmund and discord

