# Check E log(S'/S) >= 0 for d>=2 with adversarial row-stochastic A and large alpha; d=1 may fail.
import numpy as np
rng = np.random.default_rng(7)
n = 8; P = np.eye(n) - np.ones((n, n))/n; N = 40000; CH = 5
def stoch(kind):
    if kind == "twoclust":
        A = np.zeros((n, n)); A[:n//2, 0] = 1; A[n//2:, 1] = 1; return A
    if kind == "perm": return np.eye(n)[rng.permutation(n)]
    if kind == "rand": A = rng.random((n, n))**8; return A/A.sum(1, keepdims=True)
    if kind == "self": return np.eye(n)
for d in [1, 2, 3, 16]:
    worst = (9, None)
    for trial in range(16):
        Y = P@rng.standard_normal((n, d))
        if trial % 2: Y[:, 0] = P@np.r_[np.ones(n//2), -np.ones(n//2)]  # align with the two clusters
        for kind in ["twoclust", "perm", "rand", "self"]:
            A = stoch(kind); M = P@A
            for al in [0.3, 1.0, 3.0]:
                MY = M@Y; parts = []
                for _ in range(CH):
                    W = rng.standard_normal((N, d, d))/np.sqrt(d)
                    Yp = Y[None] + al*np.einsum('ij,njk->nik', MY, W)
                    parts.append(np.log((Yp**2).sum((1, 2))/(Y**2).sum()))
                lr = np.concatenate(parts)
                m, se = lr.mean(), lr.std()/np.sqrt(N*CH)
                if m/max(se, 1e-12) < worst[0]: worst = (m/max(se, 1e-12), (kind, al, round(m, 5), round(se, 5)))
    print("d", d, "most negative z of E log S'/S:", worst, flush=True)
