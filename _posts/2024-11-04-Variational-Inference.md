---
layout: post
title: Variational Inference with Normalizing Flow review
date: 2024-11-04 20:15:00-0400
description: Variational Inference.
tags:
  - Variational
  - Inference
categories: Variational Inference
related\_posts: false
pseudocode: true
---

# Central Limit Theorem for BNNs trained with Variational Inference

## Variational Inference: A Review for Statisticians

### Bayesian Statistics Recap

## 1. Train

Instead of tuning $\mathbf{w}$, tune $p(\mathbf{w}\mid \mathcal{D}\_{\text{train}})$. The posterior distribution is:

$$
p(\mathbf{w}\mid \mathcal{D}\_{\text{train}}) = \frac{p(\mathcal{D}\_{\text{train}}\mid \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D}\_{\text{train}})}
$$

Calculating the posterior is often difficult since calculating the evidence $p(\mathcal{D}\_{\text{train}})$ is nearly impossible.

## 2. Inference

To approximate:

$$
p(y\mid \mathcal{D}) = \int p(y\mid \mathbf{w}) p(\mathbf{w}\mid \mathcal{D}) d\mathbf{w} = E\_{p(\mathbf{w}\mid \mathcal{D})}[p(\hat{y}\mid \mathbf{w})]
$$

This can be approximated by:

$$
LHS \approx \frac{1}{n} \frac{\sum\_{s=1}^n w\_i\cdot\hat{y}(x)}{\sum\_{s=1}^n w\_i}
$$

where $\mathbf{w}\_i \sim p(\mathbf{w}\mid \mathcal{D})$ and $\hat{y}\_{\mathbf{w}}(x)=f(x,\mathbf{w})$.
That is, replacing
\mid Neural Network (NN) \mid Bayesian Neural Network (BNN) \mid
\mid ----------------------\mid ----------------------------------------------------\mid
\mid $\mathbf{w}$ \mid $p(\mathbf{w}\mid \mathcal{D})$ \mid
\mid $f\_{\mathbf{w}}(x)$ \mid $\int f\_{\mathbf{w}}(x) p(\mathbf{w}\mid \mathcal{D}) d\mathbf{w}$ \mid

1. **Sampling Methods**: Sample $\mathbf{w}$ from $p(\mathbf{w}\mid \mathcal{D})$ and compute $p(y\mid \mathbf{w})$ for each sample.
2. **Variational Methods**: Model $p(\mathbf{w}\mid \mathcal{D})$ using a parametrized distribution $q(\mathbf{w})$ and optimize to approximate the true posterior.

# Variational Inferences: A Review for statistics

Consider a joint density of latent variables $\mathbf{z}$ and observations $\mathbf{x}$:

$$
p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x}\mid \mathbf{z}) p(\mathbf{z})
$$

## A short review of Bayesian statistics

### Weighted average of means

One of the main premises of Bayesian statistics is that we set the population parameters as random variables.
Note the expectation of bayes estimator is weighted average of the expectation of likelihood distribution and of prior distribution.

$$
\mathbb{E}[\hat{\theta}\_{\text{Bayes}}] = r\_1 \mathbb{E}[\theta \mid  \mathcal{D}] + r\_2 \mathbb{E}[\theta_{prior}]
$$

$$
\text{where } \mathbb{E}[\theta \mid  \mathcal{D}]= \mathbb{E}[\hat{\theta}_{MLE}] \text{ (of frequentist)}
$$

### How to calculate predictive posterior distribution

$$
p(\tilde{y}\mid y) = \int p(\tilde{y}, \theta\mid y)d\theta = \int p(\tilde{y}\mid \theta, y)p(\theta\mid y)d\theta \text{ }(\because \tilde{y}\mid \theta \perp \!\!\! \perp y\mid \theta)
$$

where $p(\tilde{y}\mid \theta)$ is likelihood and $p(\theta\mid y)$ is posterior distribution.

### Numerical Integration

#### Monte Carlo

To calculate expectations of functions of the posterior distribution, $E[h(\theta)\mid y]$, use Monte Carlo methods or importance sampling. This is available when we know the posterior distribution of parameters,$\theta$, which is easily sampled.

$$
E[h(\theta)\mid y] = \int h(\theta)\cdotp(\theta\mid y)d\theta
$$

1. Sample $\theta\_1, \cdots, \theta\_N$ from posterior distribution $\theta\mid y$.
2. calculate $h(\theta\_1), \cdots, h(\theta\_N)$.
3. $\frac{1}{n}\sum_{i=1}^{N}{h(\theta\_i)} = \bar{h}_{MC} \xrightarrow{n\rightarrow\infty}\int h(\theta)p(\theta \mid y)d\theta = E[h(\theta\mid y)]$.

#### Rejection Sampling

We try rejection sampling if we don't know the prior distribution of $\theta$.
We want to find the expectation of (the function of) the posterior distribution$h(\theta)$or$h(\theta\mid \mathcal{D}) $. WLOG,

$$
\mathbb{E}[h(\theta)] = \int h(\theta) p(\theta\mid \mathcal{D}) \, d\theta
$$

Since we cannot directly sample from$p(\theta\mid \mathcal{D}) $:

1. Sample $\theta\_{1}, \theta\_{2}, \dots, \theta\_{S}$, where $\theta\_{i} \sim g(\theta)$, and $g(\theta)$ is the proposal density.
2. Choose $M$ such that $p(\theta\mid \mathcal{D}) \leq M g(\theta)$ for all $\theta$.
3. Sample auxiliary variable $u*1, \dots, u\_S \sim \text{Uniform}(0, 1)$.
4. a

- If $u \leq \frac{p(\theta^\* \mid \mathcal{D})}{M g(\theta^\*)}$, accept $\theta^\* $.
- If $u > \frac{p(\theta^\* \mid \mathcal{D})}{M g(\theta^\*)}$, reject $\theta^\* $.

5. Repeat steps 3 and 4.
6. Calculate:
   $$
   \frac{1}{n} \sum_{i=1}^{n} h(\theta_{i}) \approx \mathbb{E}_{R.S.}[h(\theta)] = \bar{h}_{R.S.} \xrightarrow{n\rightarrow\infty} \int h(\theta) p(\theta\mid \mathcal{D}) \, d\theta = \mathbb{E}[h(\theta)].
   $$

Sampling auxiliary variable and accepting/declining $\theta^\*$ is equivalent to accepting $\theta*$ with possibility $\frac{p(\theta^\* \mid \mathcal{D})}{M g(\theta^\*)}$.

### Variational Inference

#### Intractability of Posterior

The posterior density:

$$
p(\mathbf{z}\mid \mathbf{x}) = \frac{p(\mathbf{x}\mid \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}
$$

is often intractable.

#### Evidence Lower Bound (ELBO)

To approximate the intractable posterior, maximize the ELBO:

$$
\text{ELBO}(q) = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}\mid \mathbf{z})] - \text{KL}(q(\mathbf{z}) \mid \mid  p(\mathbf{z}))
$$

Maximizing ELBO is equivalent to minimizing the KL divergence between the approximate and true posterior.

## Advanced Inference Techniques

### Sampling Methods

- Monte Carlo integration: Sample $\mathbf{z}$ from $p(\mathbf{z}\mid \mathbf{x})$.
- Importance Sampling: Use a proposal distribution $g(\mathbf{z})$ to sample and weight accordingly.

### Markov Chain Monte Carlo (MCMC)

- Use MCMC methods like Gibbs Sampling or Metropolis-Hastings when direct sampling is not feasible.

## Conclusion

Variational Inference provides a powerful framework for approximating intractable posteriors, balancing computational efficiency and flexibility.
