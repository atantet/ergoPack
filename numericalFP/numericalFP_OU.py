import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg

def ChangCooper(points, idx, nx, dx, drift, Q):
    """For a constant diagonal diffusion"""
    (dim, N) = points.shape
    rows = []
    cols = []
    data = []

    for k in np.arange(N):
        j = idx[:, k]
        pj = points[:, k]

        for d in np.arange(dim):
            # Get step for this dimension
            h = dx[d]
            # Get indices +1 and -1
            jp1 = j.copy()
            jp1[d] += 1
            jm1 = j.copy()
            jm1[d] -= 1
                
            # Get points +1/2 and -1/2
            pjp = pj.copy()
            pjp[d] += h / 2
            pjm = pj.copy()
            pjm[d] -= h / 2

            # Get fields
            Bjp = - drift(pjp)[d]
            Bjm = - drift(pjm)[d]
            Cjp = Q[d, d] / 2
            Cjm = Q[d, d] / 2
            
            # Get convex combination weights
            wj = h * Bjp / Cjp
            wjm1 = h * Bjm / Cjm
            if (wj != 0.):
                deltaj = 1. / wj - 1. / (np.exp(wj) - 1)
            if (wjm1 != 0.):
                deltajm1 = 1. / wjm1 - 1. / (np.exp(wjm1) - 1)
            
            # Do not devide by step since we directly do the matrix product
            if j[d] == 0:
                kp1 = np.ravel_multi_index(jp1, nx)
                rows.append(k)
                cols.append(k)
                data.append(-(Cjp / h - deltaj * Bjp) / h)
                rows.append(k)
                cols.append(kp1)
                data.append(((1. - deltaj) * Bjp + Cjp / h) / h)
            elif j[d] + 1 == nx[d]:
                km1 = np.ravel_multi_index(jm1, nx)
                rows.append(k)
                cols.append(km1)
                data.append((Cjm / h - deltajm1 * Bjm) / h)
                rows.append(k)
                cols.append(k)
                data.append(-(Cjm / h + (1 - deltajm1) * Bjm))
            else:
                km1 = np.ravel_multi_index(jm1, nx)
                kp1 = np.ravel_multi_index(jp1, nx)
                rows.append(k)
                cols.append(km1)
                data.append((Cjm / h - deltajm1 * Bjm) / h)
                rows.append(k)
                cols.append(k)
                data.append(-((Cjp + Cjm) / h + (1 - deltajm1) * Bjm - deltaj * Bjp) / h)
                rows.append(k)
                cols.append(kp1)
                data.append(((1. - deltaj) * Bjp + Cjp / h) / h)

    # Get CRS matrix
    FPO = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return FPO

# Get configuration
configFile = '../cfg/OU2d.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

# Grid definition
nx0 = 400
nSTD0 = 20

# Number of eigenvalues
nev = 50
tol = 0.

# Get model
dim = cfg.model.dim
A = np.array(cfg.model.drift).reshape(dim, dim)
B = np.array(cfg.model.diffusion).reshape(dim, dim)

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
FPO = ChangCooper(points, idx, nx, dx, drift, Q)

print 'Solving eigenvalue problem'
(w, v) = linalg.eigs(FPO, k=nev, which='LR', tol=tol)

print 'Plotting'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(w.real, w.imag)



