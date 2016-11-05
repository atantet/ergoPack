import numpy as np
import sympy as sp

# [mu] = [T]**(-1) = [x]**2
# [beta] = 1
# i
mu = sp.symbols('mu', positive=True)
omega, beta, tau, T = sp.symbols('omega beta tau T')

J = sp.Matrix([[-2*mu, 0],[-2*beta*sp.sqrt(mu), 0]])

specJ = J.eigenvects()
D = sp.Matrix([[specJ[0][0], 0], [0, specJ[1][0]]])
E = sp.Matrix(np.concatenate((specJ[0][2][0], specJ[1][2][0]), 1))
F = (E**(-1)).T
e1 = E[:, 0]
f1 = F[:, 0]

#Ztau = sp.Matrix([[1, 0], [0, sp.exp(omega * tau)]])
Mtau = sp.zeros(2, 2)
Mtau[:, :] = (E * sp.exp(D*tau) * F.T) # * Ztau # Periodic part Z(t) * stability part (Guckenheimer & Holmes 1983)
invMtau = Mtau**(-1)
#ZT = sp.Matrix([[1, 0], [0, sp.exp(omega * T)]])
MT = sp.zeros(2, 2)
MT[:, :] = (E * sp.exp(D*T) * F.T) # * ZT
# omega = 2*sp.pi / T
# Mtau[1, 1] = omega * tau
# invMtau[1, 1] = -omega * tau
# MT[1, 1] = omega * T

# The diffusion matrix (we do not divide by 2 compared to Gaspard, 2003)
Q =  sp.Matrix([[1., 0], [0, 1 / mu]])
# The time T correlation matrix
CT = sp.integrate(Mtau * Q * Mtau.T, (tau, 0, T))
Cinf = sp.integrate(Mtau * Q * Mtau.T / T, (tau, 0, sp.oo))
#iCT = sp.integrate(invMtau * Q * invMtau.T, (tau, 0, T))
iCT = -sp.integrate(Mtau * Q * Mtau.T, (tau, 0, -T))
CT = sp.simplify(sp.trigsimp(CT))
#Cinf = sp.simplify(sp.trigsimp(Cinf))
iCT = sp.simplify(sp.trigsimp(iCT))
NT = MT * iCT
NT = sp.simplify(NT)

dET = - (f1.T * NT * f1)[0, 0] / (f1.T * e1)[0, 0]
print sp.simplify(dET)    
