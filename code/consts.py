# h(s) = E log(1+s xi)^2 = E log|1-s^2 xi^2| with xi ~ N(0,1), by quadrature; h(1) = -0.41699.
import numpy as np, warnings
from scipy import integrate
warnings.filterwarnings("ignore")
def phi(x): return np.exp(-x*x/2)/np.sqrt(2*np.pi)
def h(s):
    f = lambda x: np.log((1+s*x)**2)*phi(x)
    return integrate.quad(f, -40, 40, points=[-1.0/s], limit=400)[0]
for s in [0.05, 0.1, 0.25]:   # the bound h(s) < -0.46 s^2 is proved for s <= 1/4
    print("s =", s, " h(s) =", round(h(s), 5), " -0.46 s^2 =", round(-0.46*s*s, 5))
for s in [0.5, 1, 2]:
    print("s =", s, " h(s) =", round(h(s), 5))
