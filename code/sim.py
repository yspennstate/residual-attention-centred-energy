# X_{l+1} = X_l + alpha A_l X_l W_l, W_l iid N(0,1/d), A_l row-stochastic and independent of W_l.
# Reports log(S_L/S_0), S = |P X|_F^2, and the centred fraction r_L = S_L/T_L, T = |X|_F^2.
import numpy as np
rng = np.random.default_rng(1)
n, d, c, reps = 8, 16, 1.0, 200
P = np.eye(n) - np.ones((n, n))/n
def att(X, kind, WQ, WK):
    if kind == "uniform": return np.ones((n, n))/n
    if kind == "twoclust":
        A = np.zeros((n, n)); A[:n//2, 0] = 1; A[n//2:, 1] = 1; return A
    beta = float(kind[2:])
    Lg = beta*(X@WQ)@(X@WK).T/np.sqrt(d)/max(np.linalg.norm(X)/np.sqrt(n*d), 1e-12)**2
    Lg -= Lg.max(1, keepdims=True); E = np.exp(Lg); return E/E.sum(1, keepdims=True)
for kind in ["uniform", "sm1", "sm4", "sm16", "sm64", "twoclust"]:
    for L in [100, 1000, 4000]:
        al = c/np.sqrt(L); lS = []; r = []
        for _ in range(reps if L < 4000 else 60):
            X = rng.standard_normal((n, d)) + 3.0
            S0 = np.linalg.norm(P@X)**2
            for l in range(L):
                WQ = rng.standard_normal((d, d))/np.sqrt(d); WK = rng.standard_normal((d, d))/np.sqrt(d)
                A = att(X, kind, WQ, WK)
                X = X + al*A@X@(rng.standard_normal((d, d))/np.sqrt(d))
            S = np.linalg.norm(P@X)**2; lS.append(np.log(S/S0)); r.append(S/np.linalg.norm(X)**2)
        lS = np.array(lS); r = np.array(r)
        print(f"{kind:9s} L={L:5d} logS mean {lS.mean():+.3f} p5 {np.percentile(lS,5):+.3f} min {lS.min():+.3f} | r median {np.median(r):.4f} p5 {np.percentile(r,5):.4f}", flush=True)
