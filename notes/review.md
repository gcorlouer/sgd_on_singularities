# Singular learning theory

Some core ideas:


## A learning machine 

A learning machine approximates a sampling distribution $q(D_n)$ of some data $D_n$, with a model $p(x|w)$. The approximation is performed by finding the model $p$ minising the KL divergence:
$$
d(q,p) := K(q||p) := K_n(w) := \mathbb{E}_q[\log(q(D_n))] - \mathbb{E}_q[\log(p(X|w))] = H(q,p) - H(q) 
$$

Question: Maybe there is a better notion of distance?

The entropy term does not depends on $p$, it measures the amount of average uncertainty about the data that any model can hope to resolve. Minimising the KL divergence amounts to maximising the cross-entropy term.

A learning machine computes a function $f(x,w)$ minimising a cost function $\mathcal{L}_f(x,w)$ via some optimisation procedure (for example) gradient or stochastic gradient descent. In the context of bayesian learning, a cost function is given by the negative cross entropy-term and the function learned is the MLE $p(x|w^*)$ if we fix the prior. Importantly there might be multiple parameters compatible with the learned function. 

## Fischer information

The Fischer information matrix is defined by:

$$
I(w) = \mathbb{E}[\nabla_w \log p(x|w).\nabla_w \log p(x|w)^{\intercal}]
$$

It is related to curvature: the flatter the model, the more difficult it is to distinguish hypotheses, the lower the information. Under some assumptions to specify:

$$
I(w) = - \mathbb{E}[\nabla_w^2 \log p(x|w)]
$$

## The partition function

The partition function is a central object in SLT:

Define the negative log-likelihood:

$$
L_n(w) = \frac{1}{n} \sum_{x\in D_n} \log p(x|w)
$$

The parition function writes:

$$
Z = \int_{W}\exp(-nL_n(w))\phi(w)dw
$$

The integrand is simply the numerator in the Bayes equation.

We can normalise Z:
$$
\bar{Z} = \int_{W}\exp(-nK_n(w))\phi(w)dw
$$

with $K_n(w) = L_n(w) - L_n(\hat{w})$ (note that the second term is just the entropy when the MLE coincides with the true parameter)

To ignore stochasticity we can defined the anneal quantity $K(w) := \mathbb{E}[K_n(w)]$

## Examples

### Regular models

A model is regular if the hessian of $K(w)$ is non degenerate. Typical examples of regular models include linear regressions with gaussian noise. When $K(w)$ is a quadratic function, given the regularity assumptions, one can use the Bernstein-von Mises theorem and use the Laplace approximation to compute the partition function in large n. One then finds that the free energy writes:

$$ F = -\ln Z = nL_n(w_0) +\frac{d}{2} \ln(n)  + O(1) $$

In this case F depends only on the truth and the dimension of the model, but not on the geometry of $K$. 

### Singular models

A model is singular if the critical points of $K$ are degenerate i.e. the hessian of $K$ on these points is degenerate. We can use the Laplace and Mellin transforms to compute the partition function.

