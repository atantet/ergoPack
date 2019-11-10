import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from ergoNumAna import ChangCooper

# Grid definition
# Number of grid boxes
nx0 = 100
# Size of the domain in terms of number
nSTD0 = 5.

# Number of eigenvalues to look for
nev = 20
nevPlot = 4
# Tolerance of the Arpack algorithm
tol = 1.e-6

# Get OU definition
# Drift Matrix
A = np.array([[0.196, 0.513], [-0.513, -0.396]])
# Noise Matrix
B = np.array([[1., 0.], [0., 1.]])
# Diffusion Matrix
Q = np.dot(B, B.T)
print 'A = ', A
print 'Q = ', Q
print 'wAna = ', np.sort(np.linalg.eig(A)[0])[-1]

# Define drift
def drift(x):
    return np.dot(A, x)

# Get standard deviations to define domain size
DetmA = np.linalg.det(-A)
TrmA = np.diag(-A).sum()
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
np.savetxt('w_OU.txt', w)
np.savetxt('ev_OU.txt', v)

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
fig.savefig('w_OU.eps', bbox_inches='tight', dpi=300)

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
    fig.savefig('../results/plot/numericalFP/v%d_amp_OU.eps' % k,
                bbox_inches='tight', dpi=300)

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
    fig.savefig('../results/plot/numericalFP/v%d_phase_OU.eps' % k,
                bbox_inches='tight', dpi=300)
