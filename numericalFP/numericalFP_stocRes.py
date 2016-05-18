import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import cm

readEigVal = False
#readEigVal = True

# With periodic boundary conditions for the second coordinate
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

            # Make the fluxes periodic
            if (d == 1) & (jp1[d] == nx[d]):
                jp1[d] = 0
                pjp[d] = points[d][0] - h / 2
            if (d == 1) & (jm1[d] == -1):
                jm1[d] = nx[d] - 1
                pjm[d] = points[d][-1] + h / 2
                
            # Get fields
            Bjp = - drift(pjp)[d]
            Bjm = - drift(pjm)[d]
            Cjp = Q[d, d] / 2
            Cjm = Q[d, d] / 2
            
            # Get convex combination weights
            wj = h * Bjp / Cjp
            deltaj = 1. / wj - 1. / (np.exp(wj) - 1)
            wjm1 = h * Bjm / Cjm
            deltajm1 = 1. / wjm1 - 1. / (np.exp(wjm1) - 1)
            
            # Do not devide by step since we directly do the matrix product
            if (d == 0) & (j[d] == 0):
                kp1 = np.ravel_multi_index(jp1, nx)
                rows.append(k)
                cols.append(k)
                data.append(-(Cjp / h - deltaj * Bjp) / h)
                rows.append(k)
                cols.append(kp1)
                data.append(((1. - deltaj) * Bjp + Cjp / h) / h)
            elif (d == 0) & (j[d] + 1 == nx[d]):
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

# def stocRes(x, t, mu, F, A, taue)
#     f = np.empty((2,))
#     f[0] = F * (1 + x[1]) - x[0] * (1 + mu * (1 - x[0])**2)
#     f[1] = A * 2*np.pi/taue * np.cos(t * 2*np.pi/taue)
#     return f

def boxStocRes(x, mu, F, A, taue):
    f = np.empty((2,))
    z = A * np.sin(x[1] * 2*np.pi/taue)
    f[0] = F * (1 + z) - x[0] * (1 + mu*(1 - x[0])**2)
    f[1] = 1 
    return f

# Get model
mu = 6.2
F = 1.1
#A = 0.1
A = 0.
taue = 200
#etaRng = np.array([0.0125, 0.0175])
#etaRng = np.array([0.015, 0.02])
etaRng = np.array([0.3])

# Grid definition
dim = 2
BI = np.array([[1, 0],[0, 0]])
#dim = 1
#BI = np.array([1])
nx = [100, 20]
# give limits for the size of the periodic orbit
# at maximum value of control parameter (when noise
# effects transversally are small)
xlim = [[-1., 2.5], [-taue/2, taue/2]]

# Number of eigenvalues
nev = 200
#nev = 400
tol = 1.e-4

# Get grid points and steps
x = []
dx = np.empty((dim,))
for d in np.arange(dim):
    x.append(np.linspace(xlim[d][0], xlim[d][1], nx[d]))
    dx[d] = x[d][1] - x[d][0]
N = np.prod(nx)
idx = np.indices(nx).reshape(dim, -1)
X = np.meshgrid(*x, indexing='ij')
points = np.empty((dim, N))
for d in np.arange(dim):
    points[d] = X[d].flatten()

alpha = 0.025
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

for eta in etaRng:
    print 'For eta = ', eta
    postfix = '_nx%dx%d_mu%d_F%d_A%d_taue%d_eta%d' \
              % (nx[0], nx[1], int(np.round(mu * 100)), int(np.round(F * 100)),
                 int(np.round(A * 100)), int(np.round(taue * 100)),
                 int(np.round(eta*10000)))

    B = BI * eta
    Q = np.matrix(np.dot(B, B.T))

    if not readEigVal:
        # Define drift
        def drift(x):
            return boxStocRes(x, mu, F, A, taue)
        
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
        rho1 = v[:, 1].real
        rho1 /= rho1.sum()
        if np.min(rho0) < np.min(rho1):
            tmp = rho0.copy()
            rho0 = rho1.copy()
            rho1 = tmp.copy()

        print 'Saving eigenvalues'
        np.savetxt('../results/numericalFP/w_stocRes%s.txt' % postfix, w)
        np.savetxt('../results/numericalFP/statDist_stocRes%s.txt' % postfix, rho0)
                   

    else:
        print 'Reading eigenvalues'
        srcFile = '../results/numericalFP/w_stocRes%s.txt' % postfix
        fp = open(srcFile, 'r')
        w = np.empty((nev,), dtype=complex)
        for ev in np.arange(nev):
            line = fp.readline()
            line = line.replace('+-', '-')
            w[ev] = complex(line)
        rho0 = np.loadtxt('../results/numericalFP/statDist_stocRes%s.txt' % postfix)
                
    print 'Plotting'
    fig = plt.figure()
    #fig.set_visible(False)
    ax = fig.add_subplot(111)
    ax.scatter(w.real, w.imag, edgecolors='face')
    ax.set_xlim(-5, 0.1)
    ax.set_ylim(-0.1, 0.1)
#    ax.text(-29, -9, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/numFP_stocRes%s.%s' \
                % (postfix, figFormat), bbox_inches='tight', dpi=300)

    # Convert to transport
    Qtr = 1 + mu*(1 - X[0])**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vect = rho0.copy()
    vecAlpha = vect[vect != 0]
    if alpha > 0:
        vmax = np.sort(vecAlpha[vecAlpha != 0])[int((1. - alpha) \
                                                    * vecAlpha.shape[0])]
        vect[vect > vmax] = vmax
    else:
        vmax = np.max(vect)
    h = ax.contourf(X[1], Qtr, vect.reshape(nx), levels,
                    cmap=cm.hot_r, vmin=0., vmax=vmax)
    ax.set_xlim(X[1][0].min(), X[1][0].max())
    ax.set_ylim(0., 8.)
    #cbar = plt.colorbar(h)
    ax.set_xlabel(r'$t$', fontsize=fs_latex)
    ax.set_ylabel(r'$q$', fontsize=fs_latex)
#    plt.setp(cbar.ax.get_yticklabels(), fontsize=fs_yticklabels)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
    ax.text(-7, -7, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/statDist_stocRes%s.%s' \
                % (postfix, figFormat), bbox_inches='tight', dpi=300)

