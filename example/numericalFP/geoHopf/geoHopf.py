import sympy as sp

# Coordinates
r, theta, eta, phi = sp.symbols('r theta eta phi')

# Parameters
mu0, beta, delta = sp.symbols('mu0 beta delta')
gam = sp.symbols('gam', nonzero=True)

# Paremeter mu
mu = mu0 * (1. + delta * sp.sin(theta))

beta = 0

# Vector field in (r, theta)
Fr = mu * r
Ftheta = gam - beta * r**2

# Vector field in (eta, phi)
Feta = mu0 * eta

eq = sp.Eq(sp.Derivative(eta(theta), theta), (mu0 * eta(theta) - mu * r) / Ftheta)
#eq = sp.Eq(sp.Derivative(eta(theta), theta), mu0 * eta(theta))
res = sp.dsolve(eq)
