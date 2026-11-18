# Token uniformity in residual self-attention at random initialization comes from the mean token

Yitzchak Shmalo

We study residual self-attention networks X_{l+1} = X_l + α_l A_l X_l W_l at initialization, with residual scales
α_l ≥ 0. The value matrices W_l are Gaussian. The attention matrices A_l are row-stochastic and may depend on the
tokens and on any randomness independent of the value matrices, which covers softmax attention with query and key
matrices independent of the value matrices. Let S_l be the squared Frobenius norm of X_l after the mean token is
subtracted from every token. If the tokens have dimension d ≥ 2 and the input X_0 is deterministic with S_0 > 0, then
log S_l is a submartingale for every choice of attention and of residual scales; this can fail for d = 1.
Consequently, for a network of depth L with α_l = c/√L, the ratio S_L/‖X_L‖_F² stays bounded below in probability,
uniformly in L and in the attention. At a fixed scale α it decays at most at the exponential rate log(1 + α²n), where
n is the number of tokens.

- `paper/main.tex`, `paper/notation.tex`, `paper/main.pdf`: the paper.
- `code/c2.py`: the constant C_* = sup_{s>0} s^(-2) E log²(1+sξ)² ≈ 9.39 (attained near s = 0.51) by quadrature.
- `code/consts.py`: the function h(s) = E log(1+sξ)² by quadrature, including h(1) = -0.41699.
- `code/sim.py`: the networks of Table 1 (n = 8, d = 16, α_l = 1/√L, L = 100, 1000, 4000); the printed means and
  minima of log(S_L/S_0) are the entries of the table.
- `code/submart.py`: a Monte Carlo check of the sign of the one-step drift E[log(S_{l+1}/S_l) | G_l] for d = 1, 2, 3,
  16 and several attention matrices. For each d it prints the most negative z-score found; `(9, None)` means that no
  case fell below z = 9.

Run with `python3 code/<script>.py`. The scripts need NumPy, and `c2.py` and `consts.py` also need SciPy. On some
NumPy builds `sim.py` prints floating-point warnings from the softmax logits; the printed values are unaffected. The
proofs do not depend on the scripts.
