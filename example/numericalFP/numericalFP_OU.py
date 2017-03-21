import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from ergoNumAna import ChangCooper
#import pylibconfig2

# # Get configuration
# configFile = '../cfg/OU2d.cfg'
# cfg = pylibconfig2.Config()
# cfg.read_file(configFile)

# Grid definition
#nx0 = 400
nx0 = 50
nSTD0 = 5

# Number of eigenvalues
nev = 50
tol = 0.

# Get model
A = np.array([[0.196, 0.513], [-0.513, -0.396]])
B = np.array([[1., 0.], [0., 1.]])
dim = A.shape[0]
# dim = cfg.model.dim
#A = np.array(cfg.model.drift).reshape(dim, dim)
#B = np.array(cfg.model.diffusion).reshape(dim, dim)

# Define drift
def drift(x):
    return np.dot(A, x)

# Get standard deviations
DetmA = np.linalg.det(-A)
TrmA = np.diag(-A).sum()
Q = np.dot(B, B.T)
std = np.sqrt(np.diag((DetmA * Q + (-A - TrmA) * Q * (-A - TrmA).T) \
                      / (2 * TrmA * DetmA)))

# Get grid points and steps
x = []
dx = np.empty((dim,))
nx = np.ones((dim,), dtype=int) * nx0
nSTD = [nSTD0] * dim
for d in np.arange(dim):
    x.append(np.linspace(-nSTD[d]*std[d], nSTD[d]*std[d], nx[d]))
    dx[d] = x[d][1] - x[d][0]
N = np.prod(nx)
idx = np.indices(nx).reshape(dim, -1)
X = np.meshgrid(*x, indexing='ij')
points = np.empty((dim, N))
for d in np.arange(dim):
    points[d] = X[d].flatten()

# Get discretized Fokker-Planck operator
print 'Discretizing Fokker-Planck operator'
FPO = ChangCooper(points, nx, dx, drift, Q)

print 'Solving eigenvalue problem'
(w, v) = linalg.eigs(FPO, k=nev, which='LR', tol=tol)
isort = np.argsort(-w.real)
w = w[isort]
v = v[:, isort]

print 'Plotting'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(w.real, w.imag)

# Plot eigenvalues from natural linear combinations
# of the eigenvalues of the drift
eigDrift = np.linalg.eig(A)[0]
for k in np.arange(10):
    for l in np.arange(10):
        eig = k * eigDrift[0] + l * eigDrift[1]
        ax.plot(eig.real, eig.imag, '+k')
        
