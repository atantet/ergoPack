import sys, os
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from ergoNumAna import ChangCooper

# def hopf(x, mu, gamma, beta, delta):
#     f = np.empty((2,))
#     theta = np.angle(x[0] + 1j * x[1])
#     mu = mu * (1 + delta * np.sin(theta))
#     f[0] = x[0] * (mu - (x[0]**2 + x[1]**2)) \
#            - x[1] * (gamma - beta*(x[0]**2 + x[1]**2))
#     f[1] = x[0] * (gamma - beta*(x[0]**2 + x[1]**2)) \
#            + x[1] * (mu - (x[0]**2 + x[1]**2))
#     return f

def hopf(x, mu, gamma, beta, delta):
    f = np.empty((2,))
    theta = np.angle(x[0] + 1j * x[1])
    mu = mu * (1 + delta * np.sin(theta))
    f[0] = x[0] * mu \
           - x[1] * gamma
    f[1] = x[0] * gamma \
           + x[1] * mu 
    return f

# Model parameters
model = 'Hopf'
mu = float(sys.argv[1])
beta = float(sys.argv[2])
gamma = float(sys.argv[3])
delta = float(sys.argv[4])
eps = float(sys.argv[5])

# Numerical analysis parameters
nx0 = int(sys.argv[6])
nSTD = int(sys.argv[7])
nev = int(sys.argv[8])
tol = float(sys.argv[9])
saveEigVecForward = bool(int(sys.argv[10]))
saveEigVecBackward = bool(int(sys.argv[11]))
dim = 2

#mu0 = -10.
muf = 15.
#dmu = 0.1

print '\n'
print 'mu = ', mu
print 'beta = ', beta
print 'gamma = ', gamma
print 'delta = ', delta
print 'eps = ', eps
print 'nx0 = ', nx0
print 'nSTD = ', nSTD
print 'nev = ', nev
print 'tol = ', tol
print 'Save forward eigenvectors = ', saveEigVecForward
print 'Save backward eigenvectors = ', saveEigVecBackward

# Get standard deviations
B = np.eye(dim) * eps
Q = np.dot(B, B.T)

mu += 1.e-8
if mu < 0:
    signMu = 'm'
else:
    signMu = 'p'
beta += 1.e-8
if beta < 0:
    signBeta = 'm'
else:
    signBeta = 'p'
postfix = '_%s_mu%s%02d_beta%s%03d_delta%03d_eps%03d_nx%d_nSTD%d_nev%d' \
          % (model, signMu, int(round(np.abs(mu) * 10)),
             signBeta, int(round(np.abs(beta) * 100)),
             int(round(delta * 100)), int(round(eps * 100)),
             nx0, nSTD, nev)
rootDir = '../../'
resDir = '%s/results/numericalFP/%s' % (rootDir, model)
os.system('mkdir %s 2> /dev/null' % resDir)
print 'Postfix = ', postfix

# if mu < -1.e-6:
#     xlim = np.ones((dim,)) * eps / np.sqrt(-2 * mu) * nSTD
# elif mu > 1.e-6:
#     xlim = np.ones((dim,)) * (np.sqrt(mu) + eps / np.sqrt(4 * mu) * nSTD)
# else:
#     xlim = np.ones((dim,)) * eps / np.sqrt(-2 * (mu-0.1)) * nSTD
r = np.linspace(0., np.sqrt(muf)*2., 10000)
theta = np.linspace(-np.pi, np.pi, 1000)
(R, THETA) = np.meshgrid(r, theta)
Ur = (-mu*R**2/2 + R**4/4)
Ur[-2*Ur/eps**2 > 100] = 100
rho = R * (np.exp((-2*Ur / eps**2)))
rho /= rho.sum()
xrt = R * np.cos(THETA)
sigma = np.sqrt((xrt**2 * rho).sum() - (xrt * rho).sum()**2)
xlim = np.ones((dim,)) * sigma * nSTD
print 'xlim = ', xlim

# Get grid points and steps
x = []
dx = np.empty((dim,))
nx = np.ones((dim,), dtype=int) * nx0
for d in np.arange(dim):
    x.append(np.linspace(-xlim[d], xlim[d], nx[d]))
    dx[d] = x[d][1] - x[d][0]
N = np.prod(nx)
idx = np.indices(nx).reshape(dim, -1)
X = np.meshgrid(*x, indexing='ij')
points = np.empty((dim, N))
for d in np.arange(dim):
    points[d] = X[d].flatten()

# Define drift
def drift(x):
    return hopf(x, mu, gamma, beta, delta)


# Get discretized Fokker-Planck operator
print 'Discretizing Fokker-Planck operator'
FPO = ChangCooper(points, nx, dx, drift, Q)


# Get spectrum
# Forward
print 'Solving eigenvalue problem for Forward Kolmogorov Operator'
(eigValForward, eigVecForward) = linalg.eigs(FPO, k=nev,
                                             which='LR', tol=tol)

# Backward
print 'Solving eigenvalue problem for Backward Kolmogorov Operator'
(eigValBackward, eigVecBackward) = linalg.eigs(FPO.T, k=nev,
                                               which='LR', tol=tol)


# Save eigenvalues and eigenvectors
# Forward eigenvalues
print 'Saving forward eigenvalues in %s' % resDir
np.savetxt('%s/eigValForward%s.txt' % (resDir, postfix), eigValForward)

# Forward eigenvectors
if saveEigVecForward:
    print 'Saving forward eigenvectors in %s' % resDir
    np.savetxt('%s/eigVecForward%s.txt' % (resDir, postfix), eigVecForward)
    
# Backward eigenvalues
print 'Saving backward eigenvalues in %s' % resDir
np.savetxt('%s/eigValBackward%s.txt' % (resDir, postfix), eigValBackward)

# Backward eigenvectors
if saveEigVecBackward:
    print 'Saving backward eigenvectors in %s' % resDir
    np.savetxt('%s/eigVecBackward%s.txt' % (resDir, postfix), eigVecBackward)
