import os
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

# Get model
model = 'Hopf'
gam = 1.
#betaRng = np.arange(0., 2.05, 0.05)
betaRng = np.arange(0., 1.85, 0.05)
betaRngPlot = np.array([0., 0.25, 0.5])
mu = 0.
eps = 1.


# Grid definition
dim = 2
nx0 = 200
nSTD = 5

# Number of eigenvalues
nev = 21
#nev = 201
nevPlot = 7

# Directories
print 'For eps = ', eps
print 'For mu = ', mu
resDir = '../results/numericalFP/%s' % model
plotDir = '../results/plot/numericalFP/%s' % model
os.system('mkdir %s 2> /dev/null' % plotDir)
mu += 1.e-8
if mu < 0:
    signMu = 'm'
else:
    signMu = 'p'
plotPostfix = '_%s_mu%s%02d_eps%03d_nx%d_nSTD%d_nev%d' \
              % (model, signMu, int(round(np.abs(mu) * 10)),
                 int(round(eps * 100)), nx0, nSTD, nev)
print 'Plot postfix = ', plotPostfix

eigValBeta = np.empty((betaRng.shape[0], nev), dtype=complex)
eigVecBeta = np.empty((betaRng.shape[0], nx0**dim, nev), dtype=complex)
for ibeta in np.arange(betaRng.shape[0]):
    beta = betaRng[ibeta]
    beta += 1.e-8
    if beta < 0:
        signBeta = 'm'
    else:
        signBeta = 'p'
    postfix = '_%s_mu%s%02d_beta%s%03d_eps%03d_nx%d_nSTD%d_nev%d' \
              % (model, signMu, int(round(np.abs(mu) * 10)),
                 signBeta, int(round(np.abs(beta) * 100)), int(round(eps * 100)),
                 nx0, nSTD, nev)
    print 'For beta = ', beta

    # Read eigenvalues
    eigVal = np.empty((nev,), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigValBackward%s.txt' \
                             % (resDir, postfix), eigVal)
    isort = np.argsort(-eigVal.real)
    eigVal = eigVal[isort]
    eigValBeta[ibeta] = eigVal

markersize = 6
markeredgewidth = 1
lw = 2
colors1 = rcParams['axes.prop_cycle'].by_key()['color']
colors = np.empty((len(colors1)*2,), dtype='|S1')
colors[::2] = colors1
colors[1::2] = colors1
xplot = np.linspace(0., betaRng[-1], 1000)
XP = np.ones((xplot.shape[0], 2))
XP[:, 1] = xplot
XP = np.matrix(XP)


# Imaginary part of leading eigenvalues vs. beta
fig = plt.figure()
ax = fig.add_subplot(111)
ev = 1
evImagMax = []
ordinate = []
steepness = []
evRank = []
flag = False
colorCount = 0
for iev in np.arange(nevPlot):
    if np.abs(eigValBeta[0, ev].imag) < 1.e-6:
        ev += 1
    else:
        #        if eigValBeta[:, ev].imag > 0:
        if not flag:
            ab = np.abs(eigValBeta[:, ev].imag)
            diff = ab[1:] - ab[:-1]
            eigVal = ab.copy()
            eigVal[np.concatenate(([False], diff > 0))] *= -1
            flag = True
        else:
            ab = -np.abs(eigValBeta[:, ev].imag)
            diff = ab[1:] - ab[:-1]
            eigVal = ab.copy()
            eigVal[np.concatenate(([False], diff < 0))] *= -1
            flag = False
        evImagMax.append(np.max(np.abs(eigVal)))
        # Linear regression
        nreg = betaRng.shape[0]
        X = np.ones((nreg, 2))
        X[:, 1] = betaRng[:nreg]
        X = np.matrix(X)
        B = np.matrix(eigVal[:nreg]).T
        A = (X.T * X)**(-1) * (X.T * B)
        Y = X * A
        Stot = np.var(B)
        Sres = np.sum((np.array(Y)[:, 0] - B)**2) / nreg
        ordinate.append(A[0, 0])
        steepness.append(A[1, 0])
        evRank.append(ev)
        ax.plot(betaRng, eigVal, 'x%s' % colors[colorCount],
                markersize=markersize, markeredgewidth=markeredgewidth,
                label=r'$\Im(\lambda_{%d})$' % (ev,))
        ax.plot(xplot, np.array(XP*A)[:, 0], '-%s' % colors[colorCount], linewidth=lw,
                label=r'$\Im(\widehat{\lambda}_%d)} = %.2f + %.2f \beta$' \
                % (ev, A[0, 0], A[1, 0]))
        ev += 1
        colorCount += 1
#plt.legend(fontsize=ergoPlot.fs_default, loc='lower left')
xlim = [betaRng[0], betaRng[-1]]
ymax = np.max(np.abs(evImagMax))
ylim = [-ymax, ymax]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# Add text labels
aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) \
         * fig.get_figheight() / fig.get_figwidth()
for iev in np.arange(len(evRank)):
    rotation = np.arctan(steepness[iev] * aspect) * 180./np.pi
    corr = 0.4
    xcorr = -corr *  np.sin(rotation*np.pi/180.)
    ycorr = corr / np.cos(rotation*np.pi/180.)
    xFitTxt = betaRng[int(betaRng.shape[0] * 0.82)] + xcorr
    yFitTxt = ordinate[iev] + xFitTxt * steepness[iev] + ycorr
    if steepness[iev] >= 0:
        sgn = '+'
    else:
        sgn = '-'
    fitTxt = r'$\Im(\widehat{ \lambda}_{%d}) = %.2f %s %.2f \beta$' \
             % (evRank[iev], ordinate[iev], sgn, np.abs(steepness[iev]))
    ax.text(xFitTxt, yFitTxt, fitTxt, fontsize='x-large',
            color=colors[iev], rotation=rotation,
            ha='center', va='center')

# Parameter labels
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.03,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\mu = %.1f$' % mu, fontsize='xx-large')
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.82,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\epsilon = %.1f$' % eps, fontsize='xx-large')
# Axes
ax.set_xlabel(r'$\beta$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Im(\lambda)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/eigValImagVSbeta%s.%s' \
            % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

