---
layout: post
title: Variational Inference with Normalizing Flow review
date: 2024-11-04 20:15:00-0400
description: Variational Inference.
tags:
  - Variational
  - Inference
categories: Variational Inference
related_posts: false
---

\documentclass{article}
\usepackage{amsmath}
\begin{document}

# Central Limit Theorem for BNNs trained with Variational Inference

## Variational Inference: A Review for Statisticians

### Bayesian Statistics Recap

## 1. Train

Instead of tuning $\mathbf{w}$, tune $p(\mathbf{w}|\mathcal{D}\_{\text{train}})$. The posterior distribution is:

$$
p(\mathbf{w}|\mathcal{D}_{\text{train}}) = \frac{p(\mathcal{D}_{\text{train}}|\mathbf{w}) p(\mathbf{w})}{p(\mathcal{D}_{\text{train}})}
$$

Calculating the posterior is often difficult since calculating the evidence $p(\mathcal{D}\_{\text{train}})$ is nearly impossible.

## 2. Inference

To approximate:

$$
p(y|\mathcal{D}) = \int p(y|\mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w} = E_{p(\mathbf{w}|\mathcal{D})}[p(\hat{y}|\mathbf{w})]
$$

This can be approximated by:

$$
LHS \approx \frac{1}{n} \frac{\sum_{s=1}^n w_i\cdot\hat{y}(x)}{\sum_{s=1}^n w_i}
$$

where $\mathbf{w}_{i} \sim p(\mathbf{w}|\mathcal{D})$ and $\hat{y}_{\mathbf{w}}(x)=f(x,\mathbf{w})$.
That is, replacing
| Neural Network (NN) | Bayesian Neural Network (BNN)                    |
|---------------------|--------------------------------------------------|
|$\mathbf{w}$  |$p(\mathbf{w}|\mathcal{D}) $                  |
|$f\_{\mathbf{w}}(x)$|$\int f\_{\mathbf{w}}(x) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}$ |

1. **Sampling Methods**: Sample $\mathbf{w}$ from $p(\mathbf{w}|\mathcal{D})$ and compute $p(y|\mathbf{w})$ for each sample.
2. **Variational Methods**: Model $p(\mathbf{w}|\mathcal{D})$ using a parametrized distribution $q(\mathbf{w})$ and optimize to approximate the true posterior.

# Variational Inferences: A Review for statistics

Consider a joint density of latent variables $\mathbf{z}$ and observations $\mathbf{x}$:

$$
p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x}|\mathbf{z}) p(\mathbf{z})
$$

## A short review of Bayesian statistics

### Weighted average of means

One of the main premises of Bayesian statistics is that we set the population parameters as random variables.
Note the expectation of bayes estimator is weighted average of the expectation of likelihood distribution and of prior distribution.

$$
\mathbb{E}[\hat{\theta}_{\text{Bayes}}] = r_1 \mathbb{E}[\theta | \mathcal{D}] + r_2 \mathbb{E}[\theta_{\text{prior}}]
$$

$$
\text{where } \mathbb{E}[\theta | \mathcal{D}]= \mathbb{E}[\hat{\theta}_{\text{MLE}}] \text{ (of frequentist)}
$$

### How to calculate predictive posterior distribution

$$
p(\tilde{y}|y) = \int p(\tilde{y}, \theta|y)d\theta = \int p(\tilde{y}|\theta, y)p(\theta|y)d\theta \text{ }(\because \tilde{y}|\theta \perp \!\!\! \perp y|\theta)
$$

where $p(\tilde{y}|\theta)$ is likelihood and $p(\theta|y)$ is posterior distribution.

### Numerical Integration

#### Monte Carlo

To calculate expectations of functions of the posterior distribution, $E[h(\theta)|y]$, use Monte Carlo methods or importance sampling. This is available when we know the posterior distribution of parameters,$\theta$, which is easily sampled.

$$
E[h(\theta)|y] = \int h(\theta)\cdotp(\theta|y)d\theta
$$

1. Sample $\theta_1, \cdots, \theta_N$ from posterior distribution $\theta|y$.
2. calculate $h(\theta_1), \cdots, h(\theta_N)$. 3.$\frac{1}{n}\sum*{i=1}^{N}{h(\theta_i)} = \bar{h*{MC}} \rightarrow^{n\rightarrow\infty}\int h(\theta)p(\theta|y)d\theta = E[h(\theta|y)]$.

#### Rejection Sampling

We try rejection sampling if we don't know the prior distribution of $\theta$.
We want to find the expectation of (the function of) the posterior distribution$h(\theta)$or$h(\theta|\mathcal{D}) $. WLOG,

$$
\mathbb{E}[h(\theta)] = \int h(\theta) p(\theta|\mathcal{D}) \, d\theta
$$

Since we cannot directly sample from$p(\theta|\mathcal{D}) $:

1. Sample $\theta^_, \theta_1, \dots, \theta_S$, where $\theta^_ \sim g(\theta)$, and $g(\theta)$ is the proposal density.
2. Choose$M$ such that $p(\theta|\mathcal{D}) \leq M g(\theta)$ for all$\theta $.
3. Sample auxiliary variable $u*1, \dots, u_S \sim \text{Uniform}(0, 1)$.
4. a

- If $u \leq \frac{p(\theta^_|\mathcal{D})}{M g(\theta^_)}$, accept $\theta^\* $.
- If $u > \frac{p(\theta^_|\mathcal{D})}{M g(\theta^_)}$, reject $\theta^\_ $.

5. Repeat steps 3 and 4.
6. Calculate:
   $$
   \frac{1}{n} \sum*{i=1}^{n} h(\theta_i) \approx \mathbb{E}*{R.S.}[h(\theta)] = \bar{h}\_{R.S.} \rightarrow^{n\rightarrow\infty} \int h(\theta) p(\theta|\mathcal{D}) \, d\theta = \mathbb{E}[h(\theta)].
   $$

Sampling auxiliary variable and accepting/declining $\theta^*$ is equivalent to accepting $\theta*$ with possibility $\frac{p(\theta^_|\mathcal{D})}{M g(\theta^_)}$.

### Variational Inference

#### Intractability of Posterior

The posterior density:

$$
p(\mathbf{z}|\mathbf{x}) = \frac{p(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}
$$

is often intractable.

#### Evidence Lower Bound (ELBO)

To approximate the intractable posterior, maximize the ELBO:

$$
\text{ELBO}(q) = \mathbb{E}\_{q(\mathbf{z})}[\log p(\mathbf{x}|\mathbf{z})] - \text{KL}(q(\mathbf{z}) || p(\mathbf{z}))
$$

Maximizing ELBO is equivalent to minimizing the KL divergence between the approximate and true posterior.

## Advanced Inference Techniques

### Sampling Methods

- Monte Carlo integration: Sample $\mathbf{z}$ from $p(\mathbf{z}|\mathbf{x})$.
- Importance Sampling: Use a proposal distribution $g(\mathbf{z})$ to sample and weight accordingly.

### Markov Chain Monte Carlo (MCMC)

- Use MCMC methods like Gibbs Sampling or Metropolis-Hastings when direct sampling is not feasible.

## Conclusion

Variational Inference provides a powerful framework for approximating intractable posteriors, balancing computational efficiency and flexibility.
