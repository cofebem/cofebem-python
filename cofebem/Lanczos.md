# Lanczos Method for Approximating Entries of $S_c$ and $K^{-1}$

## 1. Objective

We consider the linear elasticity problem

$$
K u = f,
$$

where:

- $K\in\mathbb{R}^{N\times N}$ is the stiffness matrix,
- $u\in\mathbb{R}^{N}$ is the global displacement vector,
- $f\in\mathbb{R}^{N}$ is the global force vector.

After applying sufficient Dirichlet boundary conditions, we assume that $K$ is **symmetric positive definite**:

$$
K=K^T,
$$

and

$$
x^T K x>0
\qquad
\forall x\neq 0.
$$

Therefore, $K$ is invertible and

$$
u=K^{-1}f.
$$

The contact compliance matrix is

$$
\boxed{S_c = B K^{-1} B^T,}
$$

where $B$ maps global finite-element displacements to contact quantities, such as the normal displacement at the contact degrees of freedom.

The goal is to understand how the Lanczos method can approximate selected entries of $S_c$, and consequently selected entries of $K^{-1}$, without explicitly computing the full inverse matrix.

---

## 2. Expression of one entry of $S_c$

Let $e_i^c$ denote the $i$-th canonical basis vector in the contact space:

$$
e_i^c=
\begin{bmatrix}
0\\
\vdots\\
1\\
\vdots\\
0
\end{bmatrix}.
$$

The entry $(S_c)_{ij}$ is

$$
(S_c)_{ij}
=
(e_i^c)^T S_c e_j^c.
$$

Using

$$
S_c=BK^{-1}B^T,
$$

we obtain

$$
(S_c)_{ij}
=
(e_i^c)^T B K^{-1} B^T e_j^c.
$$

Define the global vectors

$$
b_i=B^T e_i^c,
\qquad
b_j=B^T e_j^c.
$$

Then

$$
\boxed{(S_c)_{ij}=b_i^T K^{-1} b_j.}
$$

This is the fundamental relation.

Approximating entries of $S_c$ therefore reduces to approximating bilinear inverse forms of the type

$$
\boxed{b_i^T K^{-1}b_j.}
$$

---

## 3. Physical interpretation of $b_i$

The vector

$$
b_i=B^T e_i^c
$$

is the global finite-element force vector associated with a unit contact force applied at contact degree of freedom $i$.

Consider the elasticity problem

$$
K u_i=b_i.
$$

Its exact solution is

$$
u_i=K^{-1}b_i.
$$

Then

$$
(S_c)_{ij}
=
b_j^T u_i.
$$

Therefore, $(S_c)_{ij}$ can be interpreted as follows:

1. apply a unit contact force at contact degree of freedom $i$,
2. solve the global elasticity problem,
3. measure the resulting contact displacement at contact degree of freedom $j$.

Thus, $(S_c)_{ij}$ represents the displacement response at contact degree of freedom $j$ due to a unit force applied at contact degree of freedom $i$.

For $i=j$,

$$
(S_c)_{ii}=b_i^T K^{-1}b_i
$$

is the local self-compliance at contact degree of freedom $i$.

---

## 4. Direct computation of an entry

To compute $(S_c)_{ij}$ directly, solve

$$
K u_i=b_i.
$$

Then evaluate

$$
(S_c)_{ij}=b_j^T u_i.
$$

To compute the complete $i$-th column of $S_c$, observe that

$$
S_c e_i^c
=
BK^{-1}B^T e_i^c.
$$

Since

$$
b_i=B^Te_i^c,
$$

we have

$$
S_c e_i^c
=
BK^{-1}b_i.
$$

Thus:

1. solve

   $$
   K u_i=b_i,
   $$

2. compute

   $$
   B u_i.
   $$

The vector $Bu_i$ is the complete $i$-th column of $S_c$.

If the contact space has $N_c$ degrees of freedom, forming the full matrix exactly generally requires approximately $N_c$ elasticity solves.

---

## 5. Main idea of the Lanczos method

We first focus on a diagonal inverse form

$$
q=b^T K^{-1}b.
$$

This includes:

$$
(K^{-1})_{ii}=e_i^T K^{-1}e_i,
$$

when $b=e_i$, and

$$
(S_c)_{ii}=b_i^TK^{-1}b_i,
$$

when $b=b_i$.

The Lanczos method does not explicitly compute $K^{-1}$. Instead, it constructs a small tridiagonal matrix $T_m$ that represents the action of $K$ in a Krylov subspace.

The final approximation will be

$$
\boxed{
b^TK^{-1}b
\approx
\|b\|_2^2 e_1^T T_m^{-1}e_1.
}
$$

The large matrix $K$ has size $N\times N$, whereas $T_m$ has size $m\times m$, with typically

$$
m\ll N.
$$

---

## 6. Krylov subspace

Starting from the vector $b$, consider

$$
b,\quad Kb,\quad K^2b,\quad \ldots,\quad K^{m-1}b.
$$

These vectors span the Krylov subspace

$$
\boxed{
\mathcal{K}_m(K,b)
=
\operatorname{span}
\left\{
b,Kb,K^2b,\ldots,K^{m-1}b
\right\}.
}
$$

For example,

$$
\mathcal{K}_1(K,b)=\operatorname{span}\{b\},
$$

$$
\mathcal{K}_2(K,b)=\operatorname{span}\{b,Kb\},
$$

and

$$
\mathcal{K}_3(K,b)=\operatorname{span}\{b,Kb,K^2b\}.
$$

The exact solution of

$$
Ku=b
$$

is

$$
u=K^{-1}b.
$$

Lanczos seeks an approximation

$$
u_m\in\mathcal{K}_m(K,b).
$$

As $m$ increases, the subspace becomes richer and the approximation can improve.

---

## 7. Why an orthonormal basis is needed

The vectors

$$
b,Kb,K^2b,\ldots
$$

are not generally orthogonal. They may also become nearly linearly dependent, which is numerically undesirable.

Lanczos constructs an orthonormal basis

$$
v_1,v_2,\ldots,v_m
$$

of the same Krylov subspace.

Define

$$
V_m=
\begin{bmatrix}
v_1&v_2&\cdots&v_m
\end{bmatrix}.
$$

The orthonormality condition is

$$
V_m^TV_m=I_m.
$$

The first Lanczos vector is

$$
\boxed{v_1=\frac{b}{\|b\|_2}.}
$$

Hence,

$$
b=\|b\|_2v_1.
$$

---

## 8. First Lanczos iteration

Start from

$$
v_1=\frac{b}{\|b\|_2}.
$$

Apply $K$:

$$
w=Kv_1.
$$

The component of $w$ in the direction of $v_1$ is

$$
\alpha_1=v_1^T Kv_1.
$$

Remove it:

$$
\widetilde w
=
Kv_1-\alpha_1v_1.
$$

This new vector is orthogonal to $v_1$, because

$$
v_1^T\widetilde w
=
v_1^TKv_1-\alpha_1v_1^Tv_1
=
\alpha_1-\alpha_1
=
0.
$$

Now define

$$
\beta_1=\|\widetilde w\|_2.
$$

Then normalize:

$$
v_2=\frac{\widetilde w}{\beta_1}.
$$

Therefore,

$$
\boxed{Kv_1=\alpha_1v_1+\beta_1v_2.}
$$

---

## 9. Second Lanczos iteration

Apply $K$ to $v_2$:

$$
w=Kv_2.
$$

Because $K$ is symmetric, the component of $Kv_2$ along $v_1$ is $\beta_1$.

Subtract it:

$$
w\leftarrow Kv_2-\beta_1v_1.
$$

Compute

$$
\alpha_2=v_2^Tw.
$$

Subtract the component along $v_2$:

$$
\widetilde w
=
Kv_2-\beta_1v_1-\alpha_2v_2.
$$

Define

$$
\beta_2=\|\widetilde w\|_2,
$$

and normalize:

$$
v_3=\frac{\widetilde w}{\beta_2}.
$$

Then

$$
\boxed{
Kv_2
=
\beta_1v_1+\alpha_2v_2+\beta_2v_3.
}
$$

---

## 10. General three-term recurrence

At iteration $k$, Lanczos satisfies

$$
\boxed{
Kv_k
=
\beta_{k-1}v_{k-1}
+
\alpha_kv_k
+
\beta_kv_{k+1}.
}
$$

The practical recurrence is

$$
w=Kv_k-\beta_{k-1}v_{k-1},
$$

$$
\alpha_k=v_k^Tw,
$$

$$
w\leftarrow w-\alpha_kv_k,
$$

$$
\beta_k=\|w\|_2,
$$

$$
v_{k+1}=\frac{w}{\beta_k}.
$$

Only the two previous Lanczos vectors are needed in exact arithmetic:

$$
v_{k-1},\qquad v_k.
$$

This is why Lanczos is called a three-term recurrence.

---

## 11. Construction of the tridiagonal matrix $T_m$

The coefficients $\alpha_k$ and $\beta_k$ define

$$
\boxed{
T_m=
\begin{bmatrix}
\alpha_1 & \beta_1 & & &\\
\beta_1 & \alpha_2 & \beta_2 & &\\
&\beta_2&\alpha_3&\ddots&\\
&&\ddots&\ddots&\beta_{m-1}\\
&&&\beta_{m-1}&\alpha_m
\end{bmatrix}.
}
$$

This matrix is:

- symmetric,
- tridiagonal,
- of size $m\times m$.

It satisfies

$$
\boxed{T_m=V_m^TKV_m.}
$$

Thus, $T_m$ is the representation of $K$ restricted to the Krylov subspace.

For example, $K$ may have dimension $N=10^6$, whereas $T_m$ may have dimension $m=50$.

---

## 12. Approximate solution in the Krylov subspace

We want to solve

$$
Ku=b.
$$

We search for an approximation of the form

$$
u_m=V_my_m,
$$

where

$$
y_m\in\mathbb{R}^m.
$$

Thus,

$$
u_m
=
y_1v_1+y_2v_2+\cdots+y_mv_m.
$$

The residual is

$$
r_m=b-Ku_m.
$$

The Galerkin condition imposes

$$
V_m^Tr_m=0.
$$

Therefore,

$$
V_m^T(b-KV_my_m)=0.
$$

Expanding,

$$
V_m^Tb-V_m^TKV_my_m=0.
$$

Since

$$
V_m^TKV_m=T_m,
$$

we obtain

$$
T_my_m=V_m^Tb.
$$

Now

$$
b=\|b\|_2v_1.
$$

Because $v_1$ is the first column of $V_m$,

$$
V_m^Tb=\|b\|_2e_1,
$$

where

$$
e_1=
\begin{bmatrix}
1\\
0\\
\vdots\\
0
\end{bmatrix}
\in\mathbb{R}^m.
$$

Hence the reduced system is

$$
\boxed{T_my_m=\|b\|_2e_1.}
$$

Therefore,

$$
y_m=\|b\|_2T_m^{-1}e_1,
$$

and the approximate global solution is

$$
\boxed{
u_m
=
\|b\|_2V_mT_m^{-1}e_1.
}
$$

---

## 13. Approximation of $b^TK^{-1}b$

The exact quantity is

$$
q=b^TK^{-1}b.
$$

Since

$$
u=K^{-1}b,
$$

we have

$$
q=b^Tu.
$$

Use the approximation $u_m$:

$$
q_m=b^Tu_m.
$$

Substitute

$$
u_m=\|b\|_2V_mT_m^{-1}e_1.
$$

Then

$$
q_m
=
b^T\left(\|b\|_2V_mT_m^{-1}e_1\right).
$$

Since

$$
b=\|b\|_2v_1,
$$

we have

$$
b^TV_m
=
\|b\|_2v_1^TV_m.
$$

Because $v_1$ is the first column of $V_m$,

$$
v_1^TV_m=e_1^T.
$$

Therefore,

$$
q_m
=
\|b\|_2^2e_1^TT_m^{-1}e_1.
$$

Hence

$$
\boxed{
b^TK^{-1}b
\approx
\|b\|_2^2e_1^TT_m^{-1}e_1.
}
$$

This is the principal Lanczos formula for diagonal inverse forms.

---

## 14. Approximation of a diagonal entry of $K^{-1}$

The $i$-th diagonal entry of $K^{-1}$ is

$$
(K^{-1})_{ii}=e_i^TK^{-1}e_i.
$$

Choose

$$
b=e_i.
$$

Since

$$
\|e_i\|_2=1,
$$

we obtain

$$
\boxed{
(K^{-1})_{ii}
\approx
e_1^TT_m^{-1}e_1.
}
$$

The Lanczos process starts from

$$
v_1=e_i.
$$

The procedure is:

1. choose the target global degree of freedom $i$,
2. set $b=e_i$,
3. run $m$ Lanczos iterations with $K$,
4. build $T_m$,
5. solve

   $$
   T_my=e_1,
   $$

6. take the first component $y_1$.

Then

$$
(K^{-1})_{ii}\approx y_1.
$$

There is no need to form $T_m^{-1}$ explicitly.

---

## 15. Approximation of a diagonal entry of $S_c$

For the contact compliance matrix,

$$
(S_c)_{ii}=b_i^TK^{-1}b_i,
$$

with

$$
b_i=B^Te_i^c.
$$

Run Lanczos with

$$
v_1=\frac{b_i}{\|b_i\|_2}.
$$

After $m$ iterations,

$$
\boxed{
(S_c)_{ii}
\approx
\|b_i\|_2^2e_1^TT_m^{-1}e_1.
}
$$

The complete procedure is:

1. construct

   $$
   b_i=B^Te_i^c,
   $$

2. compute

   $$
   \gamma_i=\|b_i\|_2,
   $$

3. initialize

   $$
   v_1=\frac{b_i}{\gamma_i},
   $$

4. run Lanczos and construct $T_m$,
5. solve

   $$
   T_my=e_1,
   $$

6. approximate

   $$
   (S_c)_{ii}\approx \gamma_i^2y_1.
   $$

---

## 16. Small two-dimensional example

Consider

$$
K=
\begin{bmatrix}
2&-1\\
-1&2
\end{bmatrix}.
$$

Its exact inverse is

$$
K^{-1}
=
\frac13
\begin{bmatrix}
2&1\\
1&2
\end{bmatrix}.
$$

Therefore,

$$
(K^{-1})_{11}=\frac23.
$$

We now recover this entry using Lanczos.

Choose

$$
b=e_1=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

Since $\|b\|_2=1$,

$$
v_1=b.
$$

### 16.1 First iteration

Compute

$$
Kv_1
=
\begin{bmatrix}
2\\
-1
\end{bmatrix}.
$$

Then

$$
\alpha_1
=
v_1^TKv_1
=
2.
$$

Remove the component along $v_1$:

$$
w
=
Kv_1-\alpha_1v_1.
$$

Thus,

$$
w
=
\begin{bmatrix}
2\\
-1
\end{bmatrix}
-
2
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
\begin{bmatrix}
0\\
-1
\end{bmatrix}.
$$

Therefore,

$$
\beta_1=\|w\|_2=1,
$$

and

$$
v_2=
\begin{bmatrix}
0\\
-1
\end{bmatrix}.
$$

After one iteration,

$$
T_1=[2].
$$

Hence

$$
q_1=e_1^TT_1^{-1}e_1=\frac12.
$$

The first approximation is therefore

$$
(K^{-1})_{11}\approx 0.5.
$$

The exact value is

$$
\frac23\approx0.6667.
$$

### 16.2 Second iteration

Compute

$$
Kv_2
=
\begin{bmatrix}
1\\
-2
\end{bmatrix}.
$$

Subtract the previous component:

$$
Kv_2-\beta_1v_1
=
\begin{bmatrix}
1\\
-2
\end{bmatrix}
-
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
\begin{bmatrix}
0\\
-2
\end{bmatrix}.
$$

Then

$$
\alpha_2
=
v_2^T
\begin{bmatrix}
0\\
-2
\end{bmatrix}
=
2.
$$

Thus,

$$
T_2=
\begin{bmatrix}
2&1\\
1&2
\end{bmatrix}.
$$

Its inverse is

$$
T_2^{-1}
=
\frac13
\begin{bmatrix}
2&-1\\
-1&2
\end{bmatrix}.
$$

Therefore,

$$
e_1^TT_2^{-1}e_1
=
\frac23.
$$

Hence

$$
\boxed{(K^{-1})_{11}=\frac23.}
$$

The result is exact after two iterations because the original matrix has dimension two.

---

## 17. What Lanczos has achieved

For a quantity

$$
q=b^TK^{-1}b,
$$

Lanczos replaces the large matrix $K$ by the small tridiagonal matrix $T_m$:

$$
\boxed{
q
\approx
\|b\|_2^2e_1^TT_m^{-1}e_1.
}
$$

The method requires repeated matrix-vector products

$$
Kv_k,
$$

but it does not require:

- forming $K^{-1}$,
- factorizing $K$,
- storing a dense inverse,
- solving the full problem to machine precision when only one scalar quantity is required.

For a selected diagonal entry of $K^{-1}$, use $b=e_i$.

For a selected diagonal entry of $S_c$, use

$$
b=b_i=B^Te_i^c.
$$

---

## 18. Important limitation

The standard scalar Lanczos method directly approximates quadratic forms

$$
b^TK^{-1}b.
$$

Therefore, it directly gives diagonal quantities such as

$$
(K^{-1})_{ii}
$$

or

$$
(S_c)_{ii}.
$$

An off-diagonal entry has the form

$$
(S_c)_{ij}=b_i^TK^{-1}b_j,
\qquad i\neq j,
$$

which is a bilinear form rather than a quadratic form.

Approximating off-diagonal entries requires an additional construction, such as the polarization identity, or the computation of an approximate solution $K^{-1}b_i$ followed by projection onto $b_j$.

That is the natural next step after understanding the diagonal case.

---

## 19. Basic Lanczos pseudocode

```text
Input:
    SPD matrix or operator K
    starting vector b
    number of iterations m

beta_0 = norm(b)
v_prev = 0
v = b / beta_0
beta_prev = 0

for k = 1, ..., m:

    w = K(v) - beta_prev * v_prev

    alpha_k = dot(v, w)

    w = w - alpha_k * v

    beta_k = norm(w)

    store alpha_k

    if k < m:
        store beta_k
        v_prev = v
        v = w / beta_k
        beta_prev = beta_k

Construct the tridiagonal matrix T_m

Solve:
    T_m y = e_1

Return:
    q_m = norm(b)^2 * y_1
```

The result satisfies

$$
q_m
=
\|b\|_2^2e_1^TT_m^{-1}e_1.
$$

---

## 20. Summary

The essential sequence is:

1. express the desired diagonal entry as

   $$
   q=b^TK^{-1}b,
   $$

2. normalize the starting vector

   $$
   v_1=\frac{b}{\|b\|_2},
   $$

3. build the Krylov basis through the Lanczos recurrence,
4. construct the tridiagonal matrix

   $$
   T_m=V_m^TKV_m,
   $$

5. solve the reduced system

   $$
   T_my=e_1,
   $$

6. compute

   $$
   \boxed{q_m=\|b\|_2^2y_1.}
   $$

For $K^{-1}$, choose $b=e_i$.

For $S_c$, choose

$$
b=B^Te_i^c.
$$

This gives an approximation of the corresponding diagonal entry without forming the inverse matrix.