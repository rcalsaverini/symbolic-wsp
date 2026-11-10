---
title: Could Neural Nets be used for Symbolic Optimization? Maybe.
date: 2021-07-02T00:00:00.000Z
categories:
  - machine-learning
tags:
  - Machine Learning
  - Neural Networks
  - Deep Learning
  - Symbolic Optimization
author: Rafael S. Calsaverini
draft: false
---

### What's Symbolic Optimization

A while ago I was entertaining problems in the intersection of symbolic manipulation of expressions and Deep Learning. In particular I was interested in finding "optimal" expressions in some way. So, imagine you have some grammar $G$ that describe a set of expressions, let's call it $\mathrm{Exp}_G$, and suppose we have a real-valued function that takes an expression and maps into a number $f: \mathrm{Exp}_G \to \mathbb{R}$. The
problem I want to discuss is finding expressions that minimize that function:

$$
e^{\star} = \arg \min_{e \in \mathrm{Exp}\_G} f(e)
$$

### A toy version of the problem

Let's discuss one particular toy version of this problem. It might look like a silly problem, but it's simple enough to illustrate the concept and non-trivial enough to avoid simple solutions.

Suppose we have a training set that consist of a sample of strings. We hold the hypothesis that there's an underlying logic or grammar that generated those strings and we want to describe commmon patterns in these strings by finding a regular expression that would match as many of those strings as possible. We'd like to use this regular expression to check if future out-of-sample strings we come across display the same patterns and belong to this group.

As an example, suppose our training set consists of the names "Rafael", "Gabriel" and "Manoel". Of course we have infinite regular expressions that match all those names. In particular we have the really trivial option `/Rafael|Gabriel|Manoel/` that would certainly match all strings but would probably fail to match out-of-sample items that do match interesting patterns in the training set – for example "Emanuel". In a way, this looks a lot like overfitting. Another failure mode would be underfitting and choosing the regex `/.\*/`, that certainly fits all in-sample items but would also accept all out-of-sample items, even those that don't match any interesting patterns in the training set – for example "Pedro". Probably what we want is something like `/.+el/` (or perhaps something with a bit more structure if this is not sufficiently constrained and we want to avoid matching the string "chapel").

So, we have to impose additional requirements and constraints or, in ML parlance, additional regularizations that would make our regular expression regularize better. It could be interesting, for example, to find the regular expression that fits as many items as possible but also fail to match a set of randomly chosen strings (a negative sample). Or to find the regular expression that generates a reasonably sized match set but still match as many items as possible. That can be perhaps stated as finding the regular expression $e$ that minimizes a function:

$$
f(e) = − \sum_{s \in \mathrm{D}} \mathtt{Match}_e(s) + \lambda \Lambda(e)
$$

Where:

- $\mathtt{Match}_e(s)$ is 1 if the regular expression e matches the string s and 0 otherwise; and
- $\Lambda(e)$ is some regularization that penalizes regular expressions that are either too "broad" or too "restricted". For example we could penalizing matching random strings and penalize expressions that have a finite (or
  too small) match set.

### What other things we could solve if we learn how to do this?

This particular problem of symbolic optimization have been bugging me for a while. There seems to be a lot of things we could do if we could solve this problem. Here's a non-exhaustive list.

- **Symbolic Regression**: we could be interested in finding the best analytical function to represent a set of data points. In this case the grammar $G$ would be a simple grammar of mathematical expressions containing things nodes (numerical values, input variables), unary operations (like $\sin$, $\cos$, $\exp$, etc), binary operators (like $+$, $−$, $×$, etc) and so on. An expression in this grammar would be a mathematical expression like $y = \sin(\pi × x_1) × exp(−4.6 × x_2)$. The function we want to minimize would be, for example, the least-squares error of fitting this expression over a set of points:
  $$f(e) = \sum_{k=1}^{N} (y_k − \mathtt{Eval}_e(\mathbf{x}_k))^2$$
  Where $\mathtt{Eval}_e(\mathbf{x})$ is a procedure that evaluates the expression $e$ at the input $\mathbf{x}$.
- **Learning to Parse**: another intriguing problem is learning how to parse strings in a certain domain. For example, imagine we have a bunch of strings representing addresses in a culture or systems we're not familiar with and we want to learn how to extract parts. How would we go about learning what structures are interesting and what parts deserve a name? We could optimize through grammars that parse those strings into parts that are reusable and build an objective function that rewards reusability of parts. That's a very handwavy example, but it can be made more concrete.
- **Enhancing code-completion by leveraging test cases**: there are many ML-based code completion products today (like [Tabnine](https://www.tabnine.com/), [kite](https://www.kite.com/), the new [Visual Studio Intellisense](https://devblogs.microsoft.com/visualstudio/the-making-of-intellicodes-first-deep-learning-model-a-research-journey/) and [Github's Copilot](https://copilot.github.com/)) but all of them look only into the immediate vicinity of the code and offer suggestions based on the environment of the code. Perhaps one could rank order the code suggestions by how well it passes a set of test cases? That would probably help the programmer narrow down the correct code completing even better. Such a procedure could be framed as symbolic optimization[^idris].

### Why is it hard?

Now, this looks interesting but it actually seems very intractable. First of all, the function
$f$ might be expensive to calculate. Even if it isn't, the space of possible expressions is combinatorially large. Probably factorially large in the size of the grammar G and depth of the expressions being considered. Pure enumeration is unfeasible and techniques like genetic programming don't deal well with high dimensional inputs. Something like Simulated Annealing can deal better with big combinatorial spaces, but convergence is slow.

Wouldn't it be nice if we could somehow turn this into gradient descent over a continuous space of analytical functions? That's the easiest and most tractable type of optimization we know. If we can map somehow, even approximately, a combinatorial optimization into a continous, $\mathbb{R}^n$ valued optimization, that would the best deal we could get, wouldn't it?

#### How to make it easier?

So, here's an idea that ocurred to me a while ago, when I was reading the Neural Architecture Optimization paper[^naopaper]. Suppose we had a pair of function, let's call it an Encoder/Decoder pair, that map our expressions into and onto real valued vectors. That is, we a pair of functions:

$$E: \mathrm{Exp}_G \to \mathbb{R}^k$$

and

$$D: \mathbb{R}^k \to \mathrm{Exp}_G$$

such that $D(E(e)) = e$ for all $e \in \mathrm{Exp}_G$.

If we had such a pair of functions, we could, instead of finding the expression $e$ that optimizes $f(e)$, we could try to find the vector $\mathbf{x} \in \mathbb{R}^k$ that optimizes $f(D(\mathbf{x}))$. Now we're optimizing over a continuous space!!! That's a bit of progress but it's still not perfect: we don't know how to calculate gradients of $f(D(\mathbf{x}))$ with respect to the vector $\mathbf{x}$, since there's the intervening combinatorial space of the expressions.

#### Neural Networks to the rescue.

Let's first focus on the Encoder/Decoder pair. What if we could train functions to learn this bijection? Even if it's not an exact bijection, it could be a start, right?

So, imagine we have a parametric family of functions we could use to find a suitable encoder:

$$E: \Theta_E \times \mathrm{Exp}_G \to \mathbb{R}^k$$

where $\Theta_E$ is some space of parameters. We could try to find a function that evaluates how well the vector $E(\theta_E, e)$ encodes the expression $e \in \mathrm{Exp}_G$ and find the parameter $\theta_E \in \Theta_E$ that optimizes that "well-encodability" function. If the parameter space is nice and the "well-encodability" function is continuous this could be done by gradient descent.

We could do the same with the decoder:

$$D: \Theta_D \times \mathbb{R}^k \to \mathrm{Exp}_G $$

If we put the encoder and decoder together, one good candidate for a measure of "well-encodability/decodability" is the reconstruction loss[^granted]:

$$
\ell(\theta_E, \theta_D) = \sum_{k=1}^N d(e_k, D(\theta_D, E(\theta_E, e_k)))
$$

where $d(e, e')$ is a measure of how distant two expressions are. In summary, we would:

- sample a random expression $e$
- use the encoder to generate it's latent encoding vector $\mathbf{x} = E(\theta_E, e)$;
- use the decoder to recover the a new expression $e' = D(\theta_D, \mathbf{x})$;
- adjust the parameters $\theta_E$ and $\theta_D$ to minimize the distance between the expressions $e$ with $e'$.

Now, after we have learned the encoder and decoder, we could now go back to our original optimization problem. We still want to find the best expression that minimizes $f(e)$ and we want to do that by minimizing $f(D(x, \theta_D))$. Perhaps we can apply the same trick above? We could try to learn a function:

$$V: \Theta_V \times \mathbb{R}^k \to \mathbb{R} $$

that given the vector embedding $\mathbf{x} = E(\theta_E, e)$ associated to an expression $e$, learns to estimate the value of $f(e)$. One possible way would be to:

- sample a random expression $e$
- use the encoder to recover its associated embedding $\mathbf{x} = E(\theta_E, e)$;
- compare the value of $f(e)$ with the value of $V(\theta_V, \mathbf{x})$ and adjust the parameter $\theta_V$ to minimize the error.

That could be achieved by gradient descending the loss function:

$$
\ell(\theta_V) = \sum_{k=1}^{N} (V(\theta_V, E(\theta_E, e)) - f(e))^2
$$

So, in the end we could simulaneously descend into the three parameters $\theta_E$, $\theta_D$ and $\theta_V$ to minimize the following loss:

$$
\ell(\theta_E, \theta_D, \theta_V) = \sum_{k=1}^N\left[d(e_k, D(\theta_D, E(\theta_E, e_k))) + (V(\theta_V, E(\theta_E, e)) - f(e))^2\right]
$$

Once we learn the best parameters $\theta_E^\star$, $\theta_D^\star$, $\theta_V^\star$, finding the best expression $e$ that maximizes $f(e)$ can be done by:

- find the vector $\mathbf{x}^\star$ that maximizes $V(\theta_V^\star, \mathbf{x})$;
- decode it to find the optimal expression associated with it: $e^\star = D(\theta_D^\star, \mathbf{x}^\star)$.

This will probably not be an exact solution, but hopefully one that is good enough.

### Problems and conclusions

That's a nice story, but is it actually possible to do? That's a great question. I have no idea. I tried attacking the regex problem with this idea before and I hit several brick walls (that mainly stem from my lack of knowledge in symbolic computation since my focus is 100% in ML and not in Computer Science in general).

The first difficulty is how to define the parametric family of encoders $E(\theta, e)$. Those functions must take values in the set of possible expressions $\mathrm{Exp}_G$ (the set of all regular expressions in our toy example). We could work with a string representation of the expression and use whatever fancy architecture kids are using for processing strings nowadays. But it would be nice to be able to feed the Abstract Syntax Tree (AST) of the expression and process this. I wonder if that would add significant amounts of inductive bias about the problem and help the model to learn. I tried using TreeLSTMs[^treelstm] for this problem the past and the result was interesting. So I think this part is really feasible.

The parametric family of decoders $D(\theta_D, \mathbf{x})$ is a much worse problem though. Those functions have to produce syntatically correct expressions in order for the optimization to make sense. Using simple string generation neural nets to produce a string representation of the expression will probably not garantee syntatically valid expressions for every latent vector we input. One possibility is to build it as a generative process that produces valid ASTs recursively. That can work but on my past tries I had a lot of trouble with performance.

Also randomly generating trees is not easy and naive processes tend to produce trees with a very heavily tailed depth distribution. In my experiments frequently my decode would generate mostly very shallow ASTs but every once in a while it would generate a monstrously deep AST. Manually clipping the maximum depth didn't work very well for me, but I might have done something dumb.

So, I didn't reach the point where I could test if this idea would work at all, but that looks like a promising idea for someone with actual time to test it.

### References and Footnotes

[^naopaper]: Luo, R., Tian, F., Qin, T., Chen, E., & Liu, T. Y. (2018). Neural architecture optimization. [_arXiv preprint arXiv:1808.07233_](https://arxiv.org/abs/1808.07233).
[^treelstm]: Tai, K. S., Socher, R., & Manning, C. D. (2015). Improved semantic representations from tree-structured long short-term memory networks. [_arXiv preprint arXiv:1503.00075_](https://arxiv.org/abs/1503.00075).
[^granted]: Granted that we are able to find a distance function between expressions that can provide continuous gradients with respect to $\theta_D$ and $\theta_E$.
[^idris]: Alternatively, some other strategies like [type driven development](https://www.idris-lang.org/courses/MGS2018/idris-mgs4.pdf) in [Idris](https://www.idris-lang.org/) use information about typing and dependently typed proofs to suggest code by case analysis. Given a strong enough type constraint Idris is able to generate the desired code for a function. Perhaps there's a way to marry this with symbolic manipulation of ASTs as well.
