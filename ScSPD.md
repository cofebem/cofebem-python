# $S_c$ SPD $\Longrightarrow$ existence & uniqueness of $\operatorname{LCP}(S_c,g_0)$

$$
S_c \in \mathbb{R}^{N_c \times N_c}
$$

SPD

---

## 1. SPD $\Longrightarrow$ P-matrix

A matrix $S_c$ is SPD if:

$$
S_c = S_c^T
$$

and

$$
x^T S_c x > 0
\qquad
\forall x \neq 0.
$$

A matrix $M \in \mathbb{R}^{n \times n}$ is a **P-matrix** if

$$
\det(M_{II}) > 0
\qquad
\forall I \subset \{1,\dots,n\}, \ I \neq \emptyset,
$$

where $M_{II}$ is the principal submatrix of $M$ obtained by keeping the rows and columns indexed by $I$.

Let $I \subset \{1,\dots,N_c\}$ be a nonempty index set.  
The principal submatrix of $S_c$ associated with $I$ is denoted by

$$
(S_c)_{II}.
$$

Since $S_c$ is SPD, every principal submatrix $(S_c)_{II}$ is also SPD.

Therefore, all eigenvalues of $(S_c)_{II}$ are strictly positive:

$$
\lambda_i > 0.
$$

Hence,

$$
\det((S_c)_{II})
=
\prod_i \lambda_i
> 0.
$$

Thus, all principal minors of $S_c$ are strictly positive.

Therefore,

$$
\boxed{
S_c \text{ SPD}
\quad \Longrightarrow \quad
S_c \text{ is a P-matrix}.
}
$$

---

## 2. Cottle-Pang-Stone theorem

$$
M \text{ is a P-matrix}
\quad \Longleftrightarrow \quad
\operatorname{LCP}(M,q)
\text{ has a unique solution for every } q.
$$


Since $S_c$ is SPD, it is a P-matrix.

Therefore,

$$
\boxed{
\forall g_0 \in \mathbb{R}^{N_c},
\qquad
\operatorname{LCP}(S_c,g_0)
\text{ admits a unique solution}.
}
$$

Equivalently, there exists a unique contact pressure vector $p$ such that

$$
0 \leq p \perp g = S_c p + g_0 \geq 0.
$$


---

## 3. $S_c$  $\mathcal{H}$-matrix approximation


$$
S_c \approx \widetilde{S}_c.
$$

Where $\widetilde{S}_c$ is a $\mathcal{H}$-matrix.
Therefore, 

$$
\widetilde{S}_c = S_c + E,
$$

with

$$
E = \widetilde{S}_c - S_c.
$$

Since we use the same low rank approx for symmetric blocks:

$$
\widetilde{S}_c = \widetilde{S}_c^T,\quad E^T = E
$$


The remaining question is whether $\widetilde{S}_c$ is still positive definite.



For any $x \neq 0$, we have

$$
x^T \widetilde{S}_c x
=
x^T S_c x + x^T E x.
$$

Since $S_c$ is SPD, its smallest eigenvalue is strictly positive:

$$
\lambda_{\min}(S_c) > 0.
$$

Using the Rayleigh quotient characterization of the smallest eigenvalue,

$$
x^T S_c x
\geq
\lambda_{\min}(S_c)\|x\|^2.
$$

Moreover, by the definition of the spectral norm,

$$
|x^T E x|
\leq
\|x\| \, \|Ex\|
\leq
\|E\|_2 \|x\|^2.
$$

Hence,

$$
x^T E x
\geq
-\|E\|_2 \|x\|^2.
$$

Therefore,

$$
x^T \widetilde{S}_c x
\geq
\lambda_{\min}(S_c)\|x\|^2
-
\|E\|_2 \|x\|^2.
$$

So,

$$
x^T \widetilde{S}_c x
\geq
\left(
\lambda_{\min}(S_c) - \|E\|_2
\right)
\|x\|^2.
$$

Thus, a sufficient condition to have

$$
x^T \widetilde{S}_c x > 0
\qquad
\forall x \neq 0
$$

is

$$
\lambda_{\min}(S_c) - \|E\|_2 > 0.
$$

Equivalently,

$$
\|E\|_2 < \lambda_{\min}(S_c).
$$

Since

$$
E = \widetilde{S}_c - S_c,
$$

we obtain the sufficient condition

$$
\boxed{
\|S_c - \widetilde{S}_c\|_2
<
\lambda_{\min}(S_c)
\quad \Longrightarrow \quad
\widetilde{S}_c \text{ is SPD}.
}
$$


Moreover,

$$
\|E\|_2 \leq \|E\|_F,
$$

i.e:


$$
\boxed{
\left(
\sum_{i=1}^{N_c} \sigma_i(E)^2
\right)^{1/2}
<
\lambda_{\min}(S_c)
}
$$



## 4. Bebendorf--Hackbusch stabilization

Bebendorf and Hackbusch stabilization to preserves positive definiteness.

For an admissible symmetric block pair, let the discarded residual be

$$
R_{ts}
=
(S_c)_{ts} - (\widetilde{S}_c)_{ts}
=
P_{ts}Q_{ts}^T.
$$

The off-diagonal approximation is therefore

$$
(\widetilde{S}_c)_{ts}
=
(S_c)_{ts} - P_{ts}Q_{ts}^T,
$$

with the symmetric block

$$
(\widetilde{S}_c)_{st}
=
(S_c)_{st} - Q_{ts}P_{ts}^T.
$$

The stabilization adds the following corrections to the corresponding diagonal blocks:

$$
(\widetilde{S}_c)_{tt}
\leftarrow
(\widetilde{S}_c)_{tt} + P_{ts}P_{ts}^T,
\qquad
(\widetilde{S}_c)_{ss}
\leftarrow
(\widetilde{S}_c)_{ss} + Q_{ts}Q_{ts}^T.
$$

The matrices $P_{ts}P_{ts}^T$ and $Q_{ts}Q_{ts}^T$ are positive semidefinite, since for every vector $x$,

$$
x^T P_{ts}P_{ts}^T x
=
\|P_{ts}^T x\|^2
\geq 0,
$$

and similarly,

$$
x^T Q_{ts}Q_{ts}^T x
=
\|Q_{ts}^T x\|^2
\geq 0.
$$

The complete correction on the index set $t \cup s$ is

$$
C_{ts}
=
\begin{bmatrix}
P_{ts}P_{ts}^T & -P_{ts}Q_{ts}^T \\
-Q_{ts}P_{ts}^T & Q_{ts}Q_{ts}^T
\end{bmatrix}.
$$

It can be factorized as

$$
C_{ts}
=
\begin{bmatrix}
P_{ts} \\
-Q_{ts}
\end{bmatrix}
\begin{bmatrix}
P_{ts} \\
-Q_{ts}
\end{bmatrix}^T
\succeq 0.
$$

Indeed, for any vector

$$
z
=
\begin{bmatrix}
x \\
y
\end{bmatrix},
$$

we have

$$
z^T C_{ts} z
=
\|P_{ts}^T x - Q_{ts}^T y\|^2
\geq 0.
$$

Therefore, the stabilized approximation satisfies

$$
\widetilde{S}_c^{\mathrm{stab}}
=
S_c + C_{ts}.
$$

Since $S_c$ is SPD and $C_{ts}$ is positive semidefinite,

$$
z^T \widetilde{S}_c^{\mathrm{stab}} z
=
z^T S_c z + z^T C_{ts} z
> 0
\qquad
\forall z \neq 0.
$$

Hence,

$$
\boxed{
S_c \succ 0
\quad\Longrightarrow\quad
\widetilde{S}_c^{\mathrm{stab}} \succ 0.
}
$$

The stabilized approximation therefore remains SPD and consequently remains a P-matrix.

To apply this stabilization, an explicit factorization of the residual must be available. With a truncated SVD,

$$
R_{ts}
=
U_{\perp}\Sigma_{\perp}V_{\perp}^T
=
\left(
U_{\perp}\Sigma_{\perp}^{1/2}
\right)
\left(
V_{\perp}\Sigma_{\perp}^{1/2}
\right)^T.
$$

Therefore, one may define

$$
P_{ts}
=
U_{\perp}\Sigma_{\perp}^{1/2},
\qquad
Q_{ts}
=
V_{\perp}\Sigma_{\perp}^{1/2}.
$$

Standard ACA provides only the approximation $UV^T$ and does not provide a factorization of the exact residual

$$
R_{ts}
=
(S_c)_{ts} - UV^T
$$

without assembling the dense block. Therefore, in our implementation, the Bebendorf--Hackbusch stabilization requires truncated SVD rather than standard ACA.

**Reference:** M. Bebendorf and W. Hackbusch, *Stabilized rounded addition of hierarchical matrices*, Numerical Linear Algebra with Applications, 14(5), 407--423, 2007.

## Jacobi method o solve Ax = b

$$ A = L + D + U $$
$$ (L + D + U)x = b \Rightarrow Dx = b - (L + U)x$$

$$ x^{k+1} = D^{-1} [b - (L + U)x^{k}]$$

$$ L = U^{T}$$

$$ x^{k+1} =max(0, D^{-1} [b - (U + U^{\top})x^{k}])$$

