# Methods

This document lists the inference methods implemented in spininfer and where they come from in the literature. Derivations written for this library are in the PDFs next to this file:

- [`Maximum_Likelihood.pdf`](Maximum_Likelihood.pdf): likelihood and moment-matching gradient
- [`Pseudo_Likelihood.pdf`](Pseudo_Likelihood.pdf): pseudo-likelihood and its gradient
- [`Naive_mean_field_approximations.pdf`](Naive_mean_field_approximations.pdf): naive mean field for the Ising and Potts models

The other methods are not re-derived here; see the references for each section.

## The inverse problem

Given samples, or only their first and second moments, find the fields `h` and couplings `J` of the maximum-entropy model that reproduces those moments [Schneidman 2006]. The library uses these conventions throughout:

| | Ising | Potts |
|---|---|---|
| States | σᵢ ∈ {−1, +1} | σᵢ ∈ {0, …, q−1}, used as one-hot indicators σᵢ(a) |
| Energy | H = −Σᵢ hᵢσᵢ − ½ Σ_{i≠j} Jᵢⱼσᵢσⱼ | H = −Σᵢ hᵢ(σᵢ) − ½ Σ_{i≠j} Jᵢⱼ(σᵢ, σⱼ) |
| Shapes | `h`: (N,), `J`: (N, N) | `h`: (N, q), `J`: (N, N, q, q) |
| Moments | `mean_s` = ⟨σᵢ⟩, `mean_ss` = ⟨σᵢσⱼ⟩ | `mean_s` = Pᵢ(a), `mean_ss` = Pᵢⱼ(a, b) |

J is symmetric with zero diagonal (blocks). Papers often write the pair sum as Σ_{i<j}, which is the same as ½ Σ_{i≠j} for symmetric J.

**Gauge (Potts).** The Potts parameters are not unique: per-site constants can be moved between `h` and `J` without changing the distribution. The library returns all Potts parameters in the zero-sum ("Ising") gauge, where `h` and every row and column of `J` sum to zero over states [Ekeberg 2013]. Internally, some methods first work in the lattice-gas gauge (the last state is a reference with zero field and couplings) and convert at the end; see `PottsModel.apply_gauge`.

## Overview

| Method | Code | Speed | Exact? | Derivation | References |
|---|---|---|---|---|---|
| Maximum likelihood (moment matching) | `src/spininfer/objectives/moment_matching.py` with `src/spininfer/fitters/` | slow | yes (with exact statistics) | `Maximum_Likelihood.pdf` | Schneidman 2006; Nguyen 2017 |
| Pseudo-likelihood | `src/spininfer/objectives/PLE_ising.py`, `src/spininfer/objectives/PLE_potts.py` | fast | consistent for many samples | `Pseudo_Likelihood.pdf` | Besag 1974; Besag 1975; Ekeberg 2013 |
| Naive mean field | `src/spininfer/mean_field/ising.py`, `src/spininfer/mean_field/potts.py` | closed form | weak coupling only | `Naive_mean_field_approximations.pdf` | Roudi 2009 |
| TAP | `src/spininfer/mean_field/ising.py` (Ising only) | closed form | weak coupling only | — | Thouless 1977; Roudi 2009 |
| Independent pair | `src/spininfer/mean_field/ising.py`, `src/spininfer/mean_field/potts.py` | closed form | isolated pairs | — | Roudi 2009 |
| Sessak–Monasson | `src/spininfer/mean_field/ising.py`, `src/spininfer/mean_field/potts.py` | closed form | small correlations | — | Sessak & Monasson 2009 |
| Bethe | `src/spininfer/mean_field/ising.py` (Ising only) | closed form | tree-like | — | Nguyen & Berg 2012; Ricci-Tersenghi 2012 |
| Adaptive cluster expansion | `src/spininfer/cluster_expansion/`, `src/spininfer/fitters/ACE_fitter.py` | medium | yes as threshold → 0 | — | Cocco & Monasson 2011; Barton 2016 |

## Short description
### Maximum likelihood
In this approach you are maximizing the log likelihood term given by 

```math
\mathcal{L} = \frac{1}{M}\sum_{i=1}^M \ln{P(\boldsymbol{\sigma}^i | \boldsymbol{h}, \boldsymbol{J})}.
```

For a detailed derivation see ```Maximum_Likelihood.pdf```.

### Pseudo likelihood
In the Pseudo likelihood approach, you are maximizing the log pseudo likelihood term given by:

```math
\mathcal{L} = \frac{1}{M}\sum_{j=1}^M\sum_{i=1}^N \ln{P(\sigma_i^j | \boldsymbol{\sigma}^j \backslash \sigma_i^j)}.
```

For a detailed derivation see ```Pseudo_Likelihood.pdf```.

### Naive mean field
In the naive mean field approximation, the self-consistent equations for the Ising and Potts model, respectively, are given by:

```math
\langle \sigma_i \rangle = \tanh{\left(h_i + \sum_{j \neq i} J_{ij} \langle \sigma_j \rangle \right)}
```

```math
\langle \sigma_i(a) \rangle = \frac{e^{h_i(a) + \sum_{j \neq i}\sum_{b=1}^q J_{ij}(a,b) \langle \sigma_j(b) \rangle}}{\sum_{c=1}^q e^{h_i(c) + \sum_{j \neq i}\sum_{b=1}^q J_{ij}(c,b) \langle \sigma_j(b) \rangle}},
```

where the coupling parameter $J$ can be recovered through the inversion of the covariance matrix

```math
J = - C^{-1},
```

where for the Potts model the last state is used as a reference state. After this the field parameters can be found through the self consistent equations. For a detailed derivation, see ```Naive_mean_field_approximations.pdf```.

### TAP
For the Ising model the TAP equation is given by

```math
\langle \sigma_i \rangle = \tanh\left(h_i + \sum_{j}J_{ij}\langle \sigma_j \rangle - \sum_{j}\langle \sigma_i \rangle(1-\langle \sigma_j \rangle^2)J_{ij}^2\right),
```

where first we solve for J by differentiating with respect to the average magnetization, giving:

```math
(C^{-1})_{ij} = -J_{ij} - 2 \langle \sigma_i \rangle \langle \sigma_j \rangle J_{ij}^2.
```

This yields the closed form

```math
J_{ij} = \frac{-1 + \sqrt{1 - 8 \langle \sigma_i \rangle \langle \sigma_j \rangle C^{-1}_{ij}}}{4 \langle \sigma_i \rangle \langle \sigma_j \rangle},
```

where h can be derived trivially from the TAP equation. For a more detailed derivation I refer to Thouless et al. (1977) and Roudi et al. (2009).


### Independent pair
For the Ising model in the independent pair approximation, we assume that pair $(i, j)$ is independent from all other sites, which yields the following form for the couplings:

```math
J_{ij} = \frac{1}{4}\ln{\frac{P(1, 1)P(-1, -1)}{P(1, -1)P(-1, 1)}}
```

and the following for the magnetizations

```math
h_i = \tanh^{-1}{\langle \sigma_i \rangle} + \sum_{j \neq i} \left( \frac{1}{4}\ln{\frac{P_{ij}(1, 1)P_{ij}(1, -1)}{P_{ij}(-1, 1)P_{ij}(-1, -1)}} - \tanh^{-1}{\langle \sigma_i \rangle} \right),
```

which is identical to the form by Roudi et al. (2009), written in a different format.

For the Potts model the approach is similar, but we take the last state $q$ to be the reference state, which gives:

```math
J_{ij}(a,b) = \ln{\frac{P_{ij}(a,b)P_{ij}(q,q)}{P_{ij}(a,q)P_{ij}(q,b)}},
```

where $P_{ij}(a,b)$ is approximated by the empirical pair frequency

```math
f_{ij}(a,b) = \frac{1}{M} \sum_{\mu=1}^{M} \delta_{\sigma_i^{\mu},a}\,\delta_{\sigma_j^{\mu},b}.
```

The field parameter for a pair is given by

```math
h^{(ij)}_{i}(a) = \ln{\frac{P_{ij}(a,q)}{P_{ij}(q,q)}},
```

then as in the Ising model, the total field parameter is built up from the independent site approximation with pair corrections:

```math
h_{i}(a) = \ln{\frac{P_{i}(a)}{P_{i}(q)}} + \sum_{j \neq i} \left( h^{(ij)}_i(a) - \ln{\frac{P_{i}(a)}{P_{i}(q)}} \right).
```

Finally, the parameters are put into the correct gauge as outlined in the top of the document.

### Sessak Monasson
The Sessak Monasson approximation to the Ising model has the form,

```math
J_{ij} = -C^{-1}_{ij} + J_{ij}^{IP} - \frac{C_{ij}}{(1-\langle \sigma_i \rangle^2)(1-\langle \sigma_j \rangle^2) - C_{ij}^2}, 
```

where the field parameters are determined through the TAP equation. For a detailed derivation I refer to Sessak & Monasson (2009) and Roudi et al. (2009).

For the Potts model, we take the same approach where the coupling parameter is determined as follows, where the reference state is left out:

```math
J = J^{nMF} + J^{IPA} - J^{nMF pair}.
```

Now, the naive mean field inversion per pair can be derived as follows:

$$
\begin{pmatrix}
\mathbf{C}_{ii} & \mathbf{C}_{ij} \\
\mathbf{C}_{ji} & \mathbf{C}_{jj}
\end{pmatrix}
\begin{pmatrix}
\mathbf{M}_{11} & \mathbf{M}_{12} \\
\mathbf{M}_{21} & \mathbf{M}_{22}
\end{pmatrix}
=
\begin{pmatrix}
I & 0 \\
0 & I
\end{pmatrix},
$$

where $M_{12}$ can be isolated yielding:

```math
J_{ij} = -M_{12} = C_{ii}^{-1} C_{ij}(-C_{ji}C_{ii}^{-1}C_{ij} + C_{jj})^{-1}.
```

Now the field parameters are recovered using the naive mean field self consistent equations, followed by the same form:

```math
h^{sm} = h^{nMF} + h^{IPA} - h^{nMF pair}.
```

### Bethe
In the Bethe approximation, the interactions are assumed to be tree-like. In this case I follow the method by Nguyen & Berg (2012), where the corrected correlation matrix is described by:

```math
C^{-1}_{ij} = \frac{\tilde{C}_{ij}}{(\tilde{C}_{ij})^2 - (1- \langle \sigma_i \rangle^2)(1- \langle \sigma_j \rangle^2)}.
```

Here $\tilde{C}$ can be isolated and used in the independent pair approximation described above. For a more detailed description I refer to Nguyen & Berg (2012) and Ricci-Tersenghi (2012).

### Adaptive cluster expansion
In adaptive cluster expansion, the parameters are inferred based on the extra information clusters provide. First, we start out by getting the entropy and inferred parameters for all sites as if they are independent. This is followed by getting the entropy and parameters of all pairs. The pairs that are kept are the ones where the entropy difference between the independent sites and the pairs is bigger than the hyperparameter $\theta$ in absolute value. Then, larger clusters are built by combining the smaller ones we just kept, and the same entropy difference is determined. This is continued until either there are no more clusters or a max size is reached. In this way, the entropy of the system is determined by summing the base independent entropy with all the extra differences in entropy, and the parameters by Möbius summation of the inferred parameters. The entropy is determined through exact enumeration. For a more detailed description I defer to Cocco & Monasson (2011) and Barton et al. (2016).

## Regularization

`src/spininfer/objectives/regularization.py` provides L2 (ridge) and L1 (lasso) penalties on h and/or J, which can be combined, and `src/spininfer/objectives/regularized.py` wraps any objective with them. L2 is the usual choice for plmDCA [Ekeberg 2013] and the cluster expansion [Cocco & Monasson 2011].

## References

- Thouless, D. J., Anderson, P. W., & Palmer, R. G. (1977). 'Solution of solvable model of a spin glass'. Philosophical Magazine, 35(3), 593-601.
- Roudi, Y., Aurell, E., & Hertz, J. A. (2009). Statistical physics of pairwise probability models. Frontiers in computational neuroscience, 3, 652.
- Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E. (2013). Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 87(1), 012707.
- Cocco, S., & Monasson, R. (2011). Adaptive cluster expansion for inferring Boltzmann machines with noisy data. Physical review letters, 106(9), 090601.
- Schneidman, E., Berry, M. J., Segev, R., & Bialek, W. (2006). Weak pairwise correlations imply strongly correlated network states in a neural population. Nature, 440(7087), 1007-1012.
- Sessak, V., & Monasson, R. (2009). Small-correlation expansions for the inverse Ising problem. Journal of Physics A: Mathematical and Theoretical, 42(5), 055001.
- Nguyen, H. C., & Berg, J. (2012). Bethe–Peierls approximation and the inverse Ising problem. Journal of Statistical Mechanics: Theory and Experiment, 2012(03), P03004.
- Ricci-Tersenghi, F. (2012). The Bethe approximation for solving the inverse Ising problem: a comparison with other inference methods. Journal of Statistical Mechanics: Theory and Experiment, 2012(08), P08015.
- Barton, J. P., De Leonardis, E., Coucke, A., & Cocco, S. (2016). ACE: adaptive cluster expansion for maximum entropy graphical model inference. Bioinformatics, 32(20), 3089-3097.
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. Journal of the Royal Statistical Society: Series B (Methodological), 36(2), 192-225.
- Nguyen, H. C., Zecchina, R., & Berg, J. (2017). Inverse statistical problems: from the inverse Ising problem to data science. Advances in physics, 66(3), 197-261.
- Besag, J. (1975). Statistical analysis of non‐lattice data. Journal of the Royal Statistical Society: Series D (The Statistician), 24(3), 179-195.