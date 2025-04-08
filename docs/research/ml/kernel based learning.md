---
title: Kernel Based Learning
description: An exploration into the beauty of kernels and Hilbert space magic
---

# Kernel Based Learning
This post deals with kernel based learning that are largely used nowadays in many neural network models, starting from maximal margin classifier through support vector classifier/machine to how they are employed in feature maps in neural networks.

## Support Vector Machine Review

Support vector machine (SVM) was developed in 1990s. It is originally from 'maximal margin classifier', which was further developed into 'support vector classifier', leading to SVM.

### Separating Hyperplane (Maximal Margin Classifier)

Suppose a $p$-dimensional space, a hyperplane is a flat affine subspace of dimension $p-1$.

For example, in two dimensions, a hyperplane is defined by the equation
$$ 
\beta_0 + \beta_1X_1 + \beta_2X_2 = 0
$$ 
Without loss of generality(WLOG), we can extend to the $p$-dimensional setting:
$
\beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_nX_n = 0
$

Suppose there exists a point $x_i \in \mathbb{R}^p$ , i.e. $x_i = (x_1, x_2, \cdots, x_n)$  then the hyperplane divides the $p$-dimensional space into two halves s.t.
$$
\begin{aligned}
\text{(1) : } \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n > 0 \\
\text{(2) : } \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n < 0
\end{aligned}
$$

Based on the location of subspace, if we classify (1) as $y_i =1$ and (2) as $y_i=-1$, we may say that:


$$
y_i(\beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n) > 0, \forall x\in\mathbb{R}^p
$$


### How to decide classifier with Lagrange Multiplier



Alright, now let's say that we have an observation $x^\ast$ Also, let $f(x^\ast)=\beta_0 + \beta_1x^\ast_1 + \beta_2x^\ast_2 + \cdots + \beta_nx^\ast_n$. Then we can assign $x^\ast$ into class 1 or to class -1. That is,
$$
\hat{y}^\ast = \text{sign}f(x^\ast)
$$

You may easily think that if the magnitude of $f(x^\ast)$ is far from 0, then $x^\ast$ lies far from the hyperplane, and otherwise $x^\ast$ lies right beside of the (separating) hyperplane.

The smallest distance from such $x^\ast$ from the hyperplane is called 'margin', so the choice of $f(x^\ast)$ maximizing the margin is called 'maximal margin classifier'.

Maximal margin classifier is at last decided by only a handful of data that are close to the hyperplane but not others, and we say that the data 'support' the maximal margin hyperplane. 

So what we basically want here is to maximize $f(x^\ast)$, but we need to be cautious of the choices of $\beta$s because $\lbrace\beta\rbrace_n=\lbrace1,1, \cdots ,1\rbrace$ is equivalent to  $\lbrace\beta\rbrace_n = \lbrace2,2,\cdots,2\rbrace$, as they do not affect the sign of $f(x^\ast)$ but only the magnitude. So we constrain $\beta \text{ s.t.} \sum_{j=1}^n\beta_j^{2}=1$. That is,

$$
\begin{aligned}
&\max_{\beta_0, \cdots, \beta_p}M \\ 
&\text{subject to } \sum_{j=1}^n\beta_j^{2}=1 \\
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


However, the maximal margin classifier does not exists if the data overlaps. If an extraordinary point exists in the other side of the data, we cannot find a classifer since it does not exists. Instead, we can give some margin to the maximal margin classifier, and we say them 'support vector classifier'.

### Support Vector Classifier
To be updated
### Support Vector Machine
To be updated
## What is kernel based leraning?
To be updated

### Acknowledges
I would like to express my sincere gratitude to Professor [Wonkuk Kim](https://sites.google.com/view/wonkukkim/home) at Chung-Ang University, as this post is largely based on insights gained from his Data Mining class.