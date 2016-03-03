import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import cm

#readEigVal = False
readEigVal = True

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
            try:
                deltaj = 1. / wj - 1. / (np.exp(wj) - 1)
            except:
                deltaj = 0.
            try:
                deltajm1 = 1. / wjm1 - 1. / (np.exp(wjm1) - 1)
            except:
                deltajm1 = 0.
            
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

def hopf(x, mu, omega):
    f = np.empty((2,))
    f[0] = x[0] * (mu - (x[0]**2 + x[1]**2)) - omega*x[1]
    f[1] = x[1] * (mu - (x[0]**2 + x[1]**2)) + omega*x[0]
    return f

# Get model
omega = 1.
q = 0.5
#q = 1.
#q = 2.
#muRng = np.arange(-10, 15., 0.1)
#k0 = 0
#muRng = np.arange(6.6, 15., 0.1)
#k0 = 166
#muRng = np.arange(-4, 2, 0.1)
#k0 = 60
#muRng = np.arange(2, 8, 0.1)
#k0 = 120
#muRng = np.arange(8, 15, 0.1)
#k0 = 180
#muRng = np.arange(5., 10., 0.1)
#k0 = 150
muRng = np.array([8.])
k0 = 180

# Grid definition
dim = 2
nx0 = 100
#nx0 = 200
# give limits for the size of the periodic orbit
# at maximum value of control parameter (when noise
# effects transversally are small)
xlim = np.ones((dim,)) * np.sqrt(15) * 2

# Number of eigenvalues
nev = 200
tol = 1.e-6

B = np.eye(dim) * q

# Get standard deviations
Q = np.dot(B, B.T)

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

alpha = 0.0
levels = 20
fs_default = 'x-large'
fs_latex = 'xx-large'
fs_xlabel = fs_default
fs_ylabel = fs_default
fs_xticklabels = fs_default
fs_yticklabels = fs_default
fs_legend_title = fs_default
fs_legend_labels = fs_default
fs_cbar_label = fs_default
#figFormat = 'png'
figFormat = 'eps'
dpi = 300
msize = 32
bbox_inches = 'tight'
plt.rc('font',**{'family':'serif'})

print 'For q = ', q
for k in np.arange(muRng.shape[0]):
    mu = muRng[k]
    print 'For mu = ', mu
    if mu < 0:
        signMu = 'm'
    else:
        signMu = 'p'
    postfix = '_nx%d_k%03d_mu%s%02d_q%02d' \
              % (nx0, k0 + k, signMu, int(round(np.abs(mu) * 10)), int(round(q * 10)))

    if not readEigVal:
        # Define drift
        def drift(x):
            return hopf(x, mu, omega)
        
        # Get discretized Fokker-Planck operator
        print 'Discretizing Fokker-Planck operator'
        FPO = ChangCooper(points, idx, nx, dx, drift, Q)

        print 'Solving eigenvalue problem'
        (w, v) = linalg.eigs(FPO, k=nev, which='LR', tol=tol)
        isort = np.argsort(-w.real)
        w = w[isort]
        v = v[:, isort]
        rho0 = v[:, 0].real
        rho0 /= rho0.sum()
        rho0_tile = np.tile(rho0, (dim, 1))
        meanPoints = (points * rho0_tile).sum(1)
        stdPoints = np.sqrt(((points - np.tile(meanPoints, (N, 1)).T)**2 * rho0_tile).sum(1))
        print 'Mean points = ', meanPoints
        print 'Std points = ', stdPoints

        print 'Saving eigenvalues'
        np.savetxt('../results/numericalFP/w_hopf%s.txt' % postfix, w)
        np.savetxt('../results/numericalFP/statDist_hopf%s.txt' % postfix, rho0)
                   

    else:
        print 'Reading eigenvalues'
        srcFile = '../results/numericalFP/w_hopf%s.txt' % postfix
        fp = open(srcFile, 'r')
        w = np.empty((nev,), dtype=complex)
        for ev in np.arange(nev):
            line = fp.readline()
            line = line.replace('+-', '-')
            w[ev] = complex(line)
        rho0 = np.loadtxt('../results/numericalFP/statDist_hopf%s.txt' % postfix)
                
    print 'Plotting'
    fig = plt.figure()
    #fig.set_visible(False)
    ax = fig.add_subplot(111)
    ax.scatter(w.real, w.imag, edgecolors='face')
    ax.set_xlim(-30, 0.1)
    ax.set_ylim(-10, 10)
    ax.text(-29, -9, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/numFP_hopf%s.%s' \
                % (postfix, figFormat), bbox_inches='tight', dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vect = rho0.copy()
    vecAlpha = vect[vect != 0]
    if alpha > 0:
        vmax = np.sort(vecAlpha)[int((1. - alpha) \
                                     * vecAlpha.shape[0])]
        vect[vect > vmax] = vmax
    else:
        vmax = np.max(vect)
    h = ax.contourf(X[0].T, X[1].T, vect.reshape(nx), levels,
                    cmap=cm.hot_r, vmin=0., vmax=vmax)
    ax.set_xlim(X[0][:, 0].min(), X[0][:, 0].max())
    ax.set_ylim(X[1][0].min(), X[1][0].max())
    #cbar = plt.colorbar(h)
    ax.set_xlabel(r'$x$', fontsize=fs_latex)
    ax.set_ylabel(r'$y$', fontsize=fs_latex)
#    plt.setp(cbar.ax.get_yticklabels(), fontsize=fs_yticklabels)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
    ax.text(-7, -7, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/statDist_hopf%s.%s' \
                % (postfix, figFormat), bbox_inches='tight', dpi=300)
    
    plt.close()

    
