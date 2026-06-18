# Brainstorming on $S_c$ interpolation

+ $f_c$ for $c = N//2$ we get $u^c(x_i)$
+ $f_e$ for $e=0$ correspond to $x=0$, we get $u^e(x_i)$
+ Apply $f$ everywhere, and get $\tilde u(x_i)$

For the half-space, 
$$\tilde u(x_i) = \sum_{j=1,N} u^j(x_i),$$
where $j$ is the point of unit force application.


+ situation 1: push on the center
```
            |
            v
o---o---o---o---o---o---o
e           c
```

+ situation 2: push on the edge
```
|
v
o---o---o---o---o---o---o
e           c
```

+ situation 3: push everywhere
```
|   |   |   |   |   |   |
v   v   v   v   v   v   v
o---o---o---o---o---x---o
e           c
```

If the object is half-plane, then situation 3 will produce the displacement equivalent to

$$\tilde u(x_i) = \sum_{j=1,N} u^j(x_i) = \sum_{j=1,N} u^c(x_i-x_j)$$

Our geometry is finite and thus $\tilde u(x_i) \ne \tilde u(x_j)$ if $i\ne j$.

$${u}^\circ(x_i) = \tilde u(x_i) - w(x_i)\sum_{j=1,N,j\ne i} u^c(x_i-x_j) - (1-w(x_i))\sum_{j=1,N,j\ne i} u^e(x_i-x_j) 
$$



