import numpy as np, warnings
from scipy import integrate
warnings.filterwarnings("ignore")
phi = lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
def m2(s):
    f = lambda x: np.log((1+s*x)**2)**2*phi(x)
    return integrate.quad(f, -40, 40, points=[-1/s], limit=400)[0]
best = max((m2(s)/s**2, s) for s in np.concatenate([np.linspace(0.01, 1, 100), np.linspace(1, 30, 150)]))
print("C_* = sup E log^2(1+s xi)^2 / s^2 =", best)
