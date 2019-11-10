import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from ergoNumAna import ChangCooper

# Hopf vector field
def hopf(x, p):
    f = np.empty((2,))
    f[0] = x[0] * (p['mu'] - (x[0]**2 + x[1]**2)) \
           - x[1] * (p['gamma'] - p['beta']*(x[0]**2 + x[1]**2))
    f[1] = x[0] * (p['gamma'] - p['beta']*(x[0]**2 + x[1]**2)) \
           + x[1] * (p['mu'] - (x[0]**2 + x[1]**2))
    return f

# Hopf normal form parameters
p = {}
p['gamma'] = 1.
p['beta'] = 0.4
p['mu'] = 1.
# Intensity of the noise
p['q'] = 1.

# Grid definition
# Dimensions
dim = 2
# Number of grid boxes
nx0 = 100
# Size of the domain in terms of number
nSTD = 5.

# Number of eigenvalues to look for
nev = 20
nevPlot = 4
# Tolerance of the Arpack algorithm
tol = 1.e-6

# Noise matrix
B = np.eye(dim) * p['q']

# Get diffusion matrix
Q = np.dot(B, B.T)

# Just some formatting of the file name
print 'gamma = ', p['gamma']
print 'beta = ', p['beta']
print 'mu = ', p['mu']
print 'q = ', p['q']
if p['mu'] < 0:
    signMu = 'm'
else:
    signMu = 'p'
if p['beta'] < 0:
    signBeta = 'm'
else:
    signBeta = 'p'
postfix = '_nx%d_mu%s%02d_beta%s%03d_q%03d' \
          % (nx0, signMu, int(round(np.abs(p['mu']) * 10)),
             signBeta, int(round(np.abs(p['beta']) * 100)),
             int(round(p['q'] * 100)))

# Calculate standard deviation to adapt size of the domain
# (can be removed to simplify by choosing arbitrary size)
muf = 15.
r = np.linspace(0., np.sqrt(muf)*2., 10000)
theta = np.linspace(-np.pi, np.pi, 1000)
(R, THETA) = np.meshgrid(r, theta)
# Potential
Ur = (-p['mu']*R**2/2 + R**4/4)
# Stationary density
rho = R * (np.exp((-2*Ur / p['q']**2)))
rho /= rho.sum()
xrt = R * np.cos(THETA)
# Standar deviation
sigma = np.sqrt((xrt**2 * rho).sum() - (xrt * rho).sum()**2)
# Grid limits
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

# Get small-noise expansion second eigenvalue for comparison
# with numerical results
# Supercritical case
if p['mu'] > 1.e-6:
    omega = p['gamma'] - p['beta'] * p['mu']
    Tp = 2*np.pi/omega
    dTp = - Tp * (1 + p['beta']**2) / p['mu']
    wana = -p['q']**2 * (1 + p['beta']**2) / (2*p['mu']) + 1j * omega
    print 'wAna = ', wana
# Subcritical case
else:
    omega = p['gamma']
    wana = p['mu'] + 1j * omega
    
# Define drift
def drift(x):
    return hopf(x, p)
        
# Get discretized Fokker-Planck operator
print 'Discretizing Fokker-Planck operator'
FPO = ChangCooper(points, nx, dx, drift, Q)
print 'Transpose to get Backward Kolmogorov operator'
BKO = FPO.T

# Solve eigenvalue problem for backward Kolmogorov operator
print 'Solving eigenvalue problem for backward Kolmogorov operator'
(w, v) = linalg.eigs(BKO, k=nev, which='LR', tol=tol)
# Sort eigenvalues
isort = np.argsort(-w.real)
w = w[isort]
print 'w[1] = ', w[1]
# Sort eigenvectors
v = v[:, isort]
# Save
print 'Saving eigenvalues and eigenvectors'
np.savetxt('w_hopf%s.txt' % postfix, w)
np.savetxt('ev_hopf%s.txt' % postfix, v)

# Plot
# Plot eigenvalues
print 'Plotting'
fig = plt.figure()
#fig.set_visible(False)
ax = fig.add_subplot(111)
ax.scatter(w.real, w.imag, c='b', edgecolors='face')
ax.set_xlim(-30, 0.1)
ax.set_ylim(-10, 10)
ax.set_xlabel(r'$\Re(\lambda_1)$', fontsize='xx-large')
ax.set_ylabel(r'$\Im(\lambda_1)$', fontsize='xx-large')
plt.setp(ax.get_xticklabels(), fontsize='xx-large')
plt.setp(ax.get_yticklabels(), fontsize='xx-large')
fig.savefig('w_hopf%s.eps' % postfix, bbox_inches='tight', dpi=300)

# Plot eigenvectors
# (Changes of sign in the modulus may appear due to oposite phases)
for k in np.arange(nevPlot):
    amp = np.abs(v[:, k]).reshape(nx)
    phase = np.angle(v[:, k]).reshape(nx)

    # Plot the modulus of the eigenvector
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.contourf(X[0].T, X[1].T, amp, 20, cmap=cm.RdBu_r)
    ax.set_title(r'$|\psi_{%d}|$' % k, fontsize='xx-large')
    ax.set_xlim(X[0][:, 0].min(), X[0][:, 0].max())
    ax.set_ylim(X[1][0].min(), X[1][0].max())
    cbar = plt.colorbar(h)
    ax.set_xlabel(r'$x$', fontsize='xx-large')
    ax.set_ylabel(r'$y$', fontsize='xx-large')
    plt.setp(cbar.ax.get_yticklabels(), fontsize='xx-large')
    plt.setp(ax.get_xticklabels(), fontsize='xx-large')
    plt.setp(ax.get_yticklabels(), fontsize='xx-large')
    pxlim = ax.get_xlim()
    pylim = ax.get_ylim()
    ax.text(pxlim[0]*0.9, pylim[0]*0.9, r'$\mu = %.1f$' % p['mu'],
            fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/v%d_amp_hopf%s.eps' \
                % (k, postfix), bbox_inches='tight', dpi=300)

    # Plot the phase of the eigenvectors
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.contourf(X[0].T, X[1].T, phase, 20, cmap=cm.RdBu_r)
    ax.set_title(r'$\mathrm{arg}\psi_{%d}$' % k, fontsize='xx-large')
    ax.set_xlim(X[0][:, 0].min(), X[0][:, 0].max())
    ax.set_ylim(X[1][0].min(), X[1][0].max())
    cbar = plt.colorbar(h)
    ax.set_xlabel(r'$x$', fontsize='xx-large')
    ax.set_ylabel(r'$y$', fontsize='xx-large')
    plt.setp(cbar.ax.get_yticklabels(), fontsize='xx-large')
    plt.setp(ax.get_xticklabels(), fontsize='xx-large')
    plt.setp(ax.get_yticklabels(), fontsize='xx-large')
    ax.text(pxlim[0]*0.9, pylim[0]*0.9, r'$\mu = %.1f$' % p['mu'],
            fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/v%d_phase_hopf%s.eps' \
                % (k, postfix), bbox_inches='tight', dpi=300)
