import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg
import ergoPlot

# Get model
omega = 1.
betaRng = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, .35, .4, .45, .5, .6, .7, .8, .9, 1.])
betaRngPlot = betaRng.copy()
mu = 0.
q = 1.
mu0 = -10.
muf = 15.
dmu = 0.1
k0 = int(np.round((-mu0 + mu) / dmu))
nevPlot = 20

r = np.linspace(0., 20., 20000)
Ur = -mu*r**2/2 + r**4/4
rho = r * (np.exp((-2*Ur / q**2)))
rho /= rho.sum()
r2 = 1. / (r**2 * rho).sum()

# Grid definition
dim = 2
nx0 = 100
#nx0 = 200

# Plot config
#figFormat = 'png'
figFormat = 'eps'
xmin = -30.
xmax = 0.1
ymin = -10.
ymax = -ymin

eigVal = np.empty((betaRng.shape[0], nevPlot), dtype=complex)
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < betaRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
for ibeta in np.arange(betaRng.shape[0]):
    beta = betaRng[ibeta]
    print 'For q = ', q
    print 'For mu = ', mu
    if mu < 0.001:
        signMu = 'm'
    else:
        signMu = 'p'
    signMu = 'p'
    if beta < 0:
        signBeta = 'm'
    else:
        signBeta = 'p'
    signBeta = 'p'
    postfix = '_adapt_nx%d_k%03d_mu%s%02d_beta%s%03d_q%03d' \
              % (nx0, k0, signMu, int(round(np.abs(mu) * 10)),
                 signBeta, int(round(np.abs(beta) * 100)), int(round(q * 100)))

    print 'Reading eigenvalues'
    srcFile = '../results/numericalFP/w_hopf%s.txt' % postfix
    fp = open(srcFile, 'r')
    for ev in np.arange(nevPlot):
        line = fp.readline()
        line = line.replace('+-', '-')
        eigVal[ibeta, ev] = complex(line)

for k in np.arange(1, 12):
    X = np.matrix(np.ones((betaRng.shape[0], 2)))
    X[:, 1] = np.matrix(betaRng**2).T
    A = (X.T * X)**(-1) * (X.T * np.matrix(eigVal[:, k]).real.T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(betaRng, eigVal[:, k].real, '+k', markersize=16,
            markeredgewidth=2,
            label=r'$\mu = 0 : \quad \Re(\lambda_%d)$' % k)
    ax.plot(betaRng, np.array(X*A)[:, 0], '--k')

    if np.abs(eigVal[0, k].imag) > 1.e-5:
        Xim = np.matrix(np.ones((betaRng.shape[0], 2)))
        Xim[:, 1] = np.matrix(betaRng**2).T
        Aim = (Xim.T * Xim)**(-1) \
              * (Xim.T * np.matrix(eigVal[:, k].imag**2).T)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(betaRng, np.abs(eigVal[:, k].imag), '+k',
                markersize=16, markeredgewidth=2,
                label=r'$\mu = 0 : \quad \Im(\lambda_%d)$' % k)
        ax.plot(betaRng, np.sqrt(np.array(Xim*Aim)[:, 0]), '--k')
