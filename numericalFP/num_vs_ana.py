import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg
import ergoPlot

# Get model
omega = 1.
#q = 0.5
q = 1.
#q = 2.
#muRng = np.array([-5.])
#muRng = np.array([0.])
#muRng = np.array([3.])
muRng = np.array([7.])
mu0 = -10.
muf = 15.
dmu = 0.1
k0 = int(np.round((-mu0 + muRng[0]) / dmu))
#plotPoint = False
plotPoint = True
#plotOrbit = False
plotOrbit = True

# Grid definition
dim = 2
nx0 = 200

# Number of eigenvalues
nev = 200

# Indices for the analytical eigenvalues
ni = 200
i = np.arange(-ni/2, ni/2)
j = np.arange(ni)
(I, J) = np.meshgrid(i, j)

# Plot config
#figFormat = 'png'
figFormat = 'eps'
xlabel = r'$\Re(\lambda_k)$'
ylabel = r'$\Im(\lambda_k)$'
xmin = -30.
xmax = 0.1
ymin = -10.
ymax = -ymin

print 'For q = ', q
for k in np.arange(muRng.shape[0]):
    mu = muRng[k]
    print 'For mu = ', mu
    if mu < 0:
        signMu = 'm'
    else:
        signMu = 'p'
    postfix = '_adapt_nx%d_k%03d_mu%s%02d_q%03d' \
              % (nx0, k0 + k, signMu, int(round(np.abs(mu) * 10)), round(int(q * 100)))

    print 'Reading eigenvalues'
    srcFile = '../results/numericalFP/w_hopf%s.txt' % postfix
    fp = open(srcFile, 'r')
    eigVal = np.empty((nev,), dtype=complex)
    for ev in np.arange(nev):
        line = fp.readline()
        line = line.replace('+-', '-')
        eigVal[ev] = complex(line)

    # Calculate analytical eigenvalues
    if mu <= 0: 
        eigValAnaPoint = (J + J.T) * mu \
                         + 1j * (J - J.T) * omega
    if mu > 0:
        eigValAnaPoint = -(J + J.T + 2) * mu \
                         - 1j * (J - J.T) * omega
    eigValAnaOrbit = (-(I * q)**2 / (2 * mu) + 1j * I * omega \
                      - 2 * J * np.sqrt(mu**2 + 2 * q**2)).flatten()

    # Filter spectrum outside
    eigValAnaOrbit = eigValAnaOrbit[(eigValAnaOrbit.real >= xmin) & (eigValAnaOrbit.real <= xmax) \
                                    & (eigValAnaOrbit.imag >= ymin) & (eigValAnaOrbit.imag <= ymax)]
    eigValAnaPoint = eigValAnaPoint[(eigValAnaPoint.real >= xmin) & (eigValAnaPoint.real <= xmax) \
                                    & (eigValAnaPoint.imag >= ymin) & (eigValAnaPoint.imag <= ymax)]
        
    print 'Plotting'
    fig = plt.figure()
    #fig.set_visible(False)
    ax = fig.add_subplot(111)
    ax.scatter(eigVal.real, eigVal.imag, edgecolors='face')
    if plotOrbit:
        ax.scatter(eigValAnaOrbit.real, eigValAnaOrbit.imag,
                   marker='+', color='k', s=40)
    if plotPoint:
        ax.scatter(eigValAnaPoint.real, eigValAnaPoint.imag,
                   marker='x', color='k', s=40)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.text(-29, -9, r'$\mu = %.1f$' % mu, fontsize=ergoPlot.fs_latex)
    ax.set_xlabel(xlabel, fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(ylabel, fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_latex)
    fig.savefig('../results/plot/numericalFP/anaVSnumFP%s.%s' \
                % (postfix, ergoPlot.figFormat), bbox_inches=ergoPlot.bbox_inches,
                dpi=ergoPlot.dpi)
