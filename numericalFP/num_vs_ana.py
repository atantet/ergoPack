import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg

# Get model
omega = 1.
q = 0.5
#q = 1.
#q = 2.
#muRng = np.arange(-10, 10.1, 0.1)
#k0 = 0
muRng = np.array([5.])
k0 = 150
#muRng = np.array([8.])
#k0 = 180
#muRng = np.array([10.])
#k0 = 200
#muRng = np.array([15.])
#k0 = 250

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
xlabel = r'$\Re(\bar{\lambda}_k)$'
ylabel = r'$\Im(\bar{\lambda}_k)$'
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
    postfix = '_nx%d_k%03d_mu%s%02d_q%02d' \
              % (nx0, k0 + k, signMu, int(round(np.abs(mu) * 10)), round(int(q * 10)))

    print 'Reading eigenvalues'
    srcFile = '../results/numericalFP/w_hopf%s.txt' % postfix
    fp = open(srcFile, 'r')
    eigVal = np.empty((nev,), dtype=complex)
    for ev in np.arange(nev):
        line = fp.readline()
        line = line.replace('+-', '-')
        eigVal[ev] = complex(line)

    # Calculate analytical eigenvalues
    eigValAnaOrbit = (-(I * q)**2 / (2 * mu) + 1j * I * omega \
                      - 2 * J * np.sqrt(mu**2 + 2 * q**2)).flatten()
    if mu <= 0: 
        eigValAnaPoint = (J + J.T) * mu \
                         + 1j * (J - J.T) * omega
    if mu > 0:
        eigValAnaPoint = -(J + J.T + 2) * mu \
                         - 1j * (J - J.T) * omega

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
    ax.scatter(eigValAnaOrbit.real, eigValAnaOrbit.imag,
               marker='+', color='k', s=40)
    ax.scatter(eigValAnaPoint.real, eigValAnaPoint.imag,
               marker='x', color='k', s=40)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.text(-29, -9, r'$\mu = %.1f$' % mu, fontsize='xx-large')
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(ylabel, fontsize='xx-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large')
    plt.setp(ax.get_yticklabels(), fontsize='xx-large')
    fig.savefig('../results/plot/numericalFP/anaVSnumFP%s.%s' \
                % (postfix, figFormat), bbox_inches='tight', dpi=300)

    plt.show()
