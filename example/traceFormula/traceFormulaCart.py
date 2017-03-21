import numpy as np
import sympy as sp

# mu = sp.symbols('mu', positive=True)
# beta, tau, T, xtau = sp.symbols('beta tau T xtau')

# J = sp.Matrix([[-2*mu, 0],[-2*beta*sp.sqrt(mu), 0]])

# specJ = J.eigenvects()
# D = sp.Matrix([[specJ[0][0], 0], [0, specJ[1][0]]])
# E = sp.Matrix(np.concatenate((specJ[0][2][0], specJ[1][2][0]), 1))
# F = (E**(-1)).T
# e1 = E[:, 0]
# f1 = F[:, 0]
# x0 = sp.Matrix([[0.], [0]])

# #Mtau = E * sp.exp(D*tau) * F.T
# #MT = E * sp.exp(D*T) * F.T
# Mtau = sp.exp(J*tau)
# invMtau = Mtau**(-1)
# MT = sp.exp(J*tau)

# xtau = Mtau * x0
# #Q = sp.Rational(1, 2) * sp.Matrix([[1, 0], [0, 1 / xtau[0, 0]**2]])
# Q = sp.Rational(1, 2) * sp.Matrix([[1, 0], [0, 1 / mu]])
# NT = 2 * MT * sp.integrate(invMtau * Q * invMtau.T, (tau, 0, T))

# dET = - (f1.T * NT * f1)[0, 0] / (f1.T * e1)[0, 0]
# print sp.simplify(dET)    


mu = sp.symbols('mu', positive=True)
beta, gamma, tau, T, xtau, omega = sp.symbols('beta gamma tau T xtau omega')
x, y = sp.symbols('x y')

# Define vector field
# field = sp.Matrix([[(mu - (x**2 + y**2))*x - (gamma - beta*(x**2 + y**2))*y],
#                    [(gamma - beta*(x**2 + y**2))*x + (mu - (x**2 + y**2))*y]])
field = sp.Matrix([[(mu - (x**2 + y**2))*x - gamma * y],
                   [gamma * x + (mu - (x**2 + y**2))*y]])

# # Get Jacobian
Jxy = field.jacobian((x, y))

# Evaluate Jacobian on periodic orbit
omega = gamma
#omega = gamma - beta * mu
Jtau = sp.simplify(Jxy.subs(x, sp.sqrt(mu)*sp.cos(omega*tau)) \
                   .subs(y, sp.sqrt(mu)*sp.sin(omega*tau)))
print 'Jacobian on limit cycle Jtau = '
print Jtau

print 'Getting spectrum of Jacobian'
specJtau = Jtau.eigenvects()
D = sp.Matrix([[specJtau[0][0], 0], [0, specJtau[1][0]]])
E = sp.Matrix(np.concatenate((specJtau[0][2][0], specJtau[1][2][0]), 1))
F = (E**(-1)).T

# print 'Simplifying spectrum'
# sp.simplify(sp.trigsimp(D))
# sp.simplify(sp.trigsimp(sp.trigsimp(E)))
# sp.simplify(sp.trigsimp(sp.trigsimp(F)))

# print 'Jtau = '
# print E
# print ' * '
# print D
# print ' * '
# print F

# e1 = E[:, 0]
# f1 = F[:, 0]
# x0 = sp.Matrix([[0.], [0]])

# print 'Getting fundamental matrix'
# Mtau = E * sp.exp(D*tau) * F.T
# invMtau = E * sp.exp(-D*tau) * F.T
# MT = E * sp.exp(D*T) * F.T
# #Mtau = sp.exp(J*tau)
# #invMtau = Mtau**(-1)
# #MT = sp.exp(J*tau)

# print 'Simplifying fundamental matrix'
# sp.simplify(sp.trigsimp(Mtau))
# sp.simplify(sp.trigsimp(invMtau))
# sp.simplify(sp.trigsimp(MT))

# print 'Getting NT'
# xtau = Mtau * x0
# Q = sp.Rational(1, 2) * sp.Matrix([[1, 0], [0, 1 ]])
# NT = 2 * MT * sp.integrate(invMtau * Q * invMtau.T, (tau, 0, T))

# print 'Getting dET'
# dET = - (f1.T * NT * f1)[0, 0] / (f1.T * e1)[0, 0]
# print sp.simplify(dET)    
