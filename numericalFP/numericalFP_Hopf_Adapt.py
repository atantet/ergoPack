import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from ergoNumAna import ChangCooper
import ergoPlot

readEigVal = False
#readEigVal = True

def hopf(x, mu, gamma, beta):
    f = np.empty((2,))
    f[0] = x[0] * (mu - (x[0]**2 + x[1]**2)) \
           - x[1] * (gamma - beta*(x[0]**2 + x[1]**2))
    f[1] = x[0] * (gamma - beta*(x[0]**2 + x[1]**2)) \
           + x[1] * (mu - (x[0]**2 + x[1]**2))
    return f

# Get model
gamma = 1.
#gamma = 1.5
#gamma = 2.
#beta = 0.
#beta = .05
#beta = 0.1
#beta = .15
#beta = 0.2
#beta = .25
#beta = 0.3
#beta = 0.35
beta = 0.4
#beta = 0.45
#beta = 0.5
#beta = 0.6
#beta = 0.7
#beta = 0.8
#beta = 0.9
#beta = 1.
#q = 0.1
#q = 0.25
#q = 0.5
#q = 0.75
#q = 1.
#q = 1.25
#q = 1.5
#q = 1.75
#q = 2.
#q = 2.25
#q = 2.5
#q = 2.75
#q = 3.
#q = 3.25
#q = 3.5
#q = 3.75
q = 4.
#muRng = np.arange(-10, 15., 0.1)
#muRng = np.arange(1.5, 10.1, 0.1)
#muRng = np.arange(-5, 5.05, 0.1)
muRng = np.array([0.])
#muRng = np.arange(-5., 0., 1.)
#muRng = np.arange(0., 5., 1.)
#muRng = np.arange(5., 11., 1.)
#muRng = np.array([0.])
#muRng = np.array([3.])
#muRng = np.array([7.])
#muRng = np.array([10.])
mu0 = -10.
muf = 15.
dmu = 0.1
k0 = int(np.round((-mu0 + muRng[0]) / dmu))

# Grid definition
dim = 2
#nx0 = 50
#nx0 = 100
nx0 = 200
#nSTD = 2.
nSTD = 5.
#nSTD = 10.
# give limits for the size of the periodic orbit
# at maximum value of control parameter (when noise
# effects transversally are small)

# Number of eigenvalues
nev = 20
#nev = 200
tol = 1.e-6

B = np.eye(dim) * q

# Get standard deviations
Q = np.dot(B, B.T)

print 'gamma = ', gamma
print 'q = ', q
print 'beta = ', beta
for k in np.arange(muRng.shape[0]):
    mu = muRng[k]
    print 'mu = ', mu
    if mu < 0:
        signMu = 'm'
    else:
        signMu = 'p'
    if beta < 0:
        signBeta = 'm'
    else:
        signBeta = 'p'
    postfix = '_adapt_nx%d_k%03d_mu%s%02d_beta%s%03d_q%03d' \
              % (nx0, k0 + k, signMu, int(round(np.abs(mu) * 10)),
                 signBeta, int(round(np.abs(beta) * 100)), int(round(q * 100)))

    # if mu < -1.e-6:
    #     xlim = np.ones((dim,)) * q / np.sqrt(-2 * mu) * nSTD
    # elif mu > 1.e-6:
    #     xlim = np.ones((dim,)) * (np.sqrt(mu) + q / np.sqrt(4 * mu) * nSTD)
    # else:
    #     xlim = np.ones((dim,)) * q / np.sqrt(-2 * (mu-0.1)) * nSTD
    r = np.linspace(0., np.sqrt(muf)*2., 10000)
    theta = np.linspace(-np.pi, np.pi, 1000)
    (R, THETA) = np.meshgrid(r, theta)
    Ur = (-mu*R**2/2 + R**4/4)
    rho = R * (np.exp((-2*Ur / q**2)))
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

    omega = gamma - beta*mu
    Tp = 2*np.pi/omega
    dTp = - Tp * (1 + beta**2) / mu
    wana = -q**2 * (1 + beta**2) / (2*mu) + 1j*omega
    wGaspard = -q**2/2 * np.abs(dTp)/Tp * omega**2 + 1j*omega
    print 'wAna = ', wana
    print 'wGaspard = ', wGaspard
    
    if not readEigVal:
        # Define drift
        def drift(x):
            return hopf(x, mu, gamma, beta)
        
        # Get discretized Fokker-Planck operator
        print 'Discretizing Fokker-Planck operator'
        FPO = ChangCooper(points, nx, dx, drift, Q)

        print 'Solving eigenvalue problem for backward Kolmogorov'
        (w, v) = linalg.eigs(FPO.T, k=nev, which='LR', tol=tol)
        isort = np.argsort(-w.real)
        w = w[isort]
        print 'w[1] = ', w[1]
        v = v[:, isort]
        rho0 = v[:, 0].real
        rho0 /= rho0.sum()
        rho0_tile = np.tile(rho0, (dim, 1))
        meanPoints = (points * rho0_tile).sum(1)
        stdPoints = np.sqrt(((points - np.tile(meanPoints, (N, 1)).T)**2 * rho0_tile).sum(1))
        #print 'Mean points = ', meanPoints
        #print 'Std points = ', stdPoints

        print 'Saving eigenvalues'
        np.savetxt('../results/numericalFP/w_hopf%s.txt' % postfix, w)
        np.savetxt('../results/numericalFP/ev_hopf%s.txt' % postfix, v)
        np.savetxt('../results/numericalFP/statDist_hopf%s.txt' % postfix, rho0)
                   

    else:
        print 'Reading eigenvalues'
        srcFileEigVal = '../results/numericalFP/w_hopf%s.txt' % postfix
        w = np.empty((nev,), dtype=complex)
        ergoPlot.loadtxt_complex(srcFileEigVal, w)
        srcFileEigVec = '../results/numericalFP/ev_hopf%s.txt' % postfix
        print 'Reading eigenvectors'
        v = np.empty((N, nev), dtype=complex)
        ergoPlot.loadtxt_complex(srcFileEigVec, v)
        rho0 = np.loadtxt('../results/numericalFP/statDist_hopf%s.txt' % postfix)
                
    print 'Plotting'
    fig = plt.figure()
    #fig.set_visible(False)
    ax = fig.add_subplot(111)
    ax.scatter(w.real, w.imag, c='b', edgecolors='face')
    ax.set_xlim(-30, 0.1)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$\Im(\lambda_1)$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    fig.savefig('../results/plot/numericalFP/numFP_hopf%s.%s' \
                % (postfix, ergoPlot.figFormat), bbox_inches=ergoPlot.bbox_inches,
                dpi=ergoPlot.dpi)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vect = rho0.copy()
    h = ax.contourf(X[0].T, X[1].T, vect.reshape(nx), 20,
                    cmap=cm.hot_r)
    ax.set_xlim(X[0][:, 0].min(), X[0][:, 0].max())
    ax.set_ylim(X[1][0].min(), X[1][0].max())
    #cbar = plt.colorbar(h)
    ax.set_xlabel(r'$x$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$y$', fontsize=ergoPlot.fs_latex)
#    plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    pxlim = ax.get_xlim()
    pylim = ax.get_ylim()
    ax.text(pxlim[0]*0.9, pylim[0]*0.9, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/statDist_hopf%s.%s' \
                % (postfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

for k in np.arange(0):
    amp = np.abs(v[:, k]).reshape(nx)
    phase = np.angle(v[:, k]).reshape(nx)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.contourf(X[0].T, X[1].T, amp, 20, cmap=cm.RdBu_r)
    ax.set_title(r'$|ev[%d]|$' % k)
    ax.set_xlim(X[0][:, 0].min(), X[0][:, 0].max())
    ax.set_ylim(X[1][0].min(), X[1][0].max())
    cbar = plt.colorbar(h)
    ax.set_xlabel(r'$x$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$y$', fontsize=ergoPlot.fs_latex)
    plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    ax.text(pxlim[0]*0.9, pylim[0]*0.9, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/v%d_amp_hopf%s.%s' \
                % (k, postfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.contourf(X[0].T, X[1].T, phase, 20, cmap=cm.RdBu_r)
    ax.set_title(r'$\mathrm{arg}ev[%d]$' % k)
    ax.set_xlim(X[0][:, 0].min(), X[0][:, 0].max())
    ax.set_ylim(X[1][0].min(), X[1][0].max())
    cbar = plt.colorbar(h)
    ax.set_xlabel(r'$x$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$y$', fontsize=ergoPlot.fs_latex)
    plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    ax.text(pxlim[0]*0.9, pylim[0]*0.9, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/v%d_phase_hopf%s.%s' \
                % (k, postfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
