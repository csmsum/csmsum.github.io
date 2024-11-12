---
layout: post
title: SVM and Kernel Method
date: 2024-10-05 11:12:00-0400
description: Kernel Method explained with SVM.
tags: Kernel Method
categories: kernel method
related_posts: false
---

# Kernel Based Learning

This post deals with kernel based learning that are largely used nowadays in many neural network models, starting from maximal margin classifier through support vector classifier/machine to how they are employed in feature maps in neural networks.

## Support Vector Machine Review

Support vector machines (SVMs) was developed in 1990s. It is originally from 'maximal margin classifier', which was further developed into 'support vector classifier', leading to SVM.

### Separating Hyperplane (Maximal Margin Classifier)

Suppose a $p$-dimensional space, a hyperplane is a flat affine subspace of dimension $p-1$.

For example, in two dimensions, a hyperplane is defined by the equation

$$
\beta_0 + \beta_1X_1 + \beta_2X_2 = 0
$$

Without loss of generality(WLOG), we can extend to the $p$-dimensional setting:

$$
\beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p = 0
$$

Suppose there exists a point $ x_i \in \mathbb{R}^p $ , i.e. $ x_i = (x_1, x_2, \cdots, x_n) $ then the hyperplane divides the $p$-dimensional space into two halves s.t.

$$
\begin{aligned}
\text{(1) : } \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p > 0 \\\
\text{(2) : } \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p < 0
\end{aligned}
$$

Based on the location of subspace, if we classify (1) as $y_i =1$ and (2) as $y_i=-1$, we may say that:

$$
y_i(\beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p) > 0, \forall \mathbf{x}\in\mathbb{R}^p
$$

### How to decide classifier with Lagrange Multiplier

Alright, now let's say that we have an observation $x^\ast$ Also, let $f(x^\ast)=\beta_0 + \beta_1x^\ast_1 + \beta_2x^\ast_2 + \cdots + \beta_px^\ast_p$. Then we can assign $x^\ast$ into class 1 or to class -1. That is,

$$
\hat{y}^\ast = \text{sign}f(x^\ast)
$$

You may easily think that if the magnitude of $f(x^\ast)$ is far from 0, then $x^\ast$ lies far from the hyperplane, and otherwise $x^\ast$ lies right beside of the (separating) hyperplane.

The smallest distance from such $x^\ast$ from the hyperplane is called 'margin', so the choice of $f(x^\ast)$ maximizing the margin is called 'maximal margin classifier'.

Maximal margin classifier is at last decided by only a handful of data that are close to the hyperplane but not others, and we say that the data 'support' the maximal margin hyperplane.

So what we basically want here is to maximize $f(x^\ast)$, but we need to be cautious of the choices of $\beta$s because $\lbrace\beta\rbrace_p=\lbrace1,1, \cdots ,1\rbrace$ is equivalent to $\lbrace\beta\rbrace_p = \lbrace2,2,\cdots,2\rbrace$, as they do not affect the sign of $f(x^\ast)$ but only the magnitude. So we constrain $\beta \text{ s.t.} \sum_{j=1}^p\beta_j^{2}=1$. That is,

$$
\begin{aligned}
&\max_{\beta_0, \cdots, \beta_p}M \\
&\text{subject to } \sum_{j=1}^p\beta_j^{2}=1 \\
&\text{and } y_i(\beta_0 + \beta_1x^\ast_{i1} + \beta_2x^\ast_{i2} + \cdots + \beta_px^\ast_{ip}) \geq M \\
\end{aligned}
$$

Let $\textbf{w} = \frac{1}{M}(\beta_1, \cdots \beta_p)^T$ and $b=\frac{\beta_0}{M}$. Then since $\bf{w^Tw}=\frac{1}{M^2}$ , it suffices to

$$
\begin{aligned}
&\min_{\mathbf{w}, b}\frac{1}{2}\mathbf{w^T w} \\
&\text{subject to } 1-y_i(\mathbf{w^T x} + b) \leq 0, \forall i = 1,\cdots,n
\end{aligned}
$$

for all data point $\lbrace x_i \rbrace _{i=1}^n$. To minimize the Lagrangian Function $\mathcal{L}$ :

$$
\begin{aligned}
\mathcal{L} = \frac{1}{2}\mathbf{w^T w} + \sum_{i=1}^{n}\alpha_i(1-y_i(\mathbf{w^T x_i} + b)) \\
\alpha_i \geq 0, \forall i = 1,\cdots,n
\end{aligned}
$$

The partial derivatives:

$$
\begin{aligned}
&\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = \mathbf{w} +\sum_{i=1}^{n}\alpha_i(-y_i)(\mathbf{x_i})  =0 \implies \mathbf{w} = \sum_{i=1}^{n}\alpha_i y_i \mathbf{x}_i\\
&\frac{\partial}{\partial b} = \sum_{i=1}^{n} \alpha_iy_i = 0
\end{aligned}
$$

From the above equation, you may see that $\mathbf{w} = \sum_{i=1}^{n}\alpha_i y_i \mathbf{x}_i$. Let's substitute this $\mathbf{w}$ back to the $\mathcal{L}$. Then the Lagrangian

$$
\begin{aligned}
\mathcal{L} = -\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_j\mathbf{x}^T\mathbf{x} + \sum_{i=1}^{n} \alpha_i \\
\text{subject to } \sum_{i=1}^n\alpha_i y_i = 0, \forall i = 1, \cdots, n
\end{aligned}
$$

However, the maximal margin classifier does not exists if the data overlaps. If an extraordinary point exists in the other side of the data, we cannot find a classifier since it does not exists. Instead, we can give some margin to the maximal margin classifier, and we say them 'support vector classifier'.

### Support Vector Classifier

The core idea for support vector classifier is to give margin to each data point so that the equation makes sense even with some outliers (**soft margin**). To address the problem, the SVC suggest slack variable $\xi_i$. The modified equation is as following:

$$
\begin{aligned}
&\max_{\beta_0, \cdots, \beta_p}M \\
\text{subject to } &\sum_{j=1}^p\beta_j^2=1 \\
&y_i(\beta_0 + \beta_1x^\ast_{i1} + \beta_2x^\ast_{i2} + \cdots + \beta_px^\ast_{ip}) \geq M(1 - \xi_i) \\
&\xi_i \geq 0, \\
&\sum_{i=1}^n\xi_n \leq const.
\end{aligned}
$$

Intuitively, We may see that if a datapoint $x_i$ lies near the borderline, the $\xi_i$ for that datapoint will be larger, leading to smaller $M(1-\xi_i)$. Also, if $x_i$ lies at the other side than where it was suppoossed to be, the $\xi_i$ will be larger than 1 so that the $M(1-\xi_i) < 0$.

Now, to maximize the equation, Let $\mathbf{w} = \frac{1}{M}(\beta_1, \cdots, \beta_p)^T$. This optimization problem is equvalent to minimizing:

$$
\mathcal{L} = \frac{1}{2}\mathbf{w^T w} + C\left[ \sum_{i=1}^n\xi_i\right] + \sum_{i=1}^n\alpha_i\left[ 1 - \xi_i - y_i(\mathbf{w^T x_i} + b)\right] - \sum_{i=1}^n\eta_i\xi_i
$$

where the regularization parameters for the lagrange are $\alpha_i, \eta_i$ for $i=1,\cdots,n$, and $C$ is the regularization parameter or the penalty term for the equation. $C$ is hyperparameter WE choose how much to penalty the classification errors.

Again, the partial derivatives are:

$$
\begin{equation}
\begin{aligned}
&\frac{\partial}{\partial\mathbf{w}}\mathcal{L} = \mathbf{w} +\sum_{i=1}^{n}\alpha_i(-y_i)(\mathbf{x_i})  =0 \implies \mathbf{w} = \sum_{i=1}^{n}\alpha_i y_i \mathbf{x}_i\\
&\frac{\partial}{\partial b} = \sum_{i=1}^{n} \alpha_iy_i = 0 \\
&\frac{\partial}{\partial\xi_i}\mathcal{L} = C - \alpha_i - \eta_i = 0 \iff \alpha_i = C-\eta_i
\end{aligned}
\end{equation}
$$

Using the above equation, let's go back to $\mathcal{L}$. Then it is the dual optimization problem (Proofs to be updated):

$$
\begin{aligned}
&\text{Maximize } \mathcal{L} = \sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_j\mathbf{x_i^T x_j} \\
&\text{subject to } \sum_{i=1}^n\alpha_iy_i=0, C\geq\alpha_i\geq0,(\eta_i = C-\alpha_i), \forall i = 1,\cdots,n
\end{aligned}
$$

The following conditions are called Karush-Kuhn-Tucker (KKT) conditions.

$$
\begin{aligned}
\alpha_i\left[1-\xi_i-y_i(\mathbf{w^T x_i} + b)\right] = 0 \\
\eta_i\xi_i = 0, \forall i=1,\dots,n, \\
yi(\mathbf{w^T x_i + b}) \geq 1 - \xi_i, \forall i = 1,\cdots,n.

\end{aligned}
$$

Since from the above equation of $\frac{\partial}{\partial \mathbf{w}}\mathcal{L}$, the solution is:

$$
\begin{aligned}
\mathbf{\hat{w}} = \sum_{i=1}^n \hat{\alpha_i}y_ix_i
\end{aligned}
$$

Therefore,

$$
\hat{Y}(\mathbf{x}) = sign\left[\hat{\mathcal{f}}(\mathbf{x})\right] = sign\left[\mathbf{\hat{w}^T x} + \hat{b}\right]
$$

To be updated more. \ref{eq1} asdf

### Support Vector Machine

However, data does not always are classifiable with a line. In much more case, they are nonlinearly distributed. We may try some quadtratic terms in the support vector classifier, but this makes computation extremely difficult.

The Kernel Method appears in that regards.

- We want to enlarge the feature space with higher-order (polynomial) terms
- We want the equation be simple nevertheless.

### Brief Conclusion

We can show that the support vector classifier can be written as:

$$
\begin{aligned}
f(x) &= \sum_{i=1}^n\alpha_iy_i<\mathbf{x,x_i}> + b \\
&= \sum_{i\in S}^n\alpha_iy_i<\mathbf{x,x_i}> + b
\end{aligned}
$$

Let $\phi$ a $\mathbb{R}^p\rightarrow \mathbb{R}^M$ ($\mathbb{R}^p\rightarrow \mathcal{H}$ to be precise) function such that $\phi(\mathbf{x})=(\phi_1(\mathbf{x}),\cdots,\phi_M(\mathbf{x}))$ that moves each datapoint into (possibly high-dimensional) feature space. The inner product of such space will be

$$
\begin{aligned}
f(x) &= \sum_{i=1}^n\alpha_iy_iK<\mathbf{x,x_i}>_\mathcal{H} + b \\
&= \sum_{i\in S}^n\alpha_iy_iK<\mathbf{x,x_i}> + b
\end{aligned}
$$

So if we find adequate $\phi$ where the feature maps' inner product is can be expressed as the original data's inner product, it is equivalent to calculating on higher ($M$ in this case) dimension, but maintaining the calculation simple.

**EXAMPLES and MORE EXPLANATIONS FOR KERNELS TO BE UPDATED**

### Acknowledges

I would like to express my sincere gratitude to Professor [Wonkuk Kim](https://sites.google.com/view/wonkukkim/home) at Chung-Ang University, as this post is largely based on insights gained from his Data Mining class.
