import os
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

# Get model
model = 'Hopf'
gam = 1.
betaRng = np.arange(0., 2.05, 0.05)
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
nevPlot = 6

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

    # Read eigenvalues
    eigVal = np.empty((nev,), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigValForward%s.txt' \
                             % (resDir, postfix), eigVal)
    isort = np.argsort(-eigVal.real)
    eigVal = eigVal[isort]
    eigValBeta[ibeta, :] = eigVal[:]


markersize = 6
markeredgewidth = 1
lw = 2
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < nevPlot:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
xplot = np.linspace(0., betaRng[-1], 1000)
XP = np.ones((xplot.shape[0], 2))
XP[:, 1] = xplot**2
XP = np.matrix(XP)
fig = plt.figure()
ax = fig.add_subplot(111)
ev = 0
evRealMin = []
ordinate = []
steepness = []
evRank = []
for iev in np.arange(nevPlot):
    eigVal = eigValBeta[:, ev].real
    evRealMin.append(np.min(eigVal))
    # Linear regression
    nreg = int(betaRng.shape[0] / 2)
    X = np.ones((nreg, 2))
    X[:, 1] = betaRng[:nreg]**2
    X = np.matrix(X)
    B = np.matrix(eigVal[:nreg]).T
    A = (X.T * X)**(-1) * (X.T * B)
    Y = X * A
    Stot = np.var(B)
    Sres = np.sum((np.array(Y)[:, 0] - B)**2) / nreg
    ordinate.append(A[0, 0])
    steepness.append(A[1, 0])
    evRank.append(ev)
    ax.plot(betaRng, eigVal, 'x%s' % colors[iev],
            markersize=markersize, markeredgewidth=markeredgewidth,
            label=r'$\Re(\lambda_{%d})$' % (ev,))
    ax.plot(xplot, np.array(XP*A)[:, 0], '-%s' % colors[iev], linewidth=lw,
            label=r'$\Re(\widehat{\lambda}_%d)} = %.2f + %.2f \beta^2$' % (ev, A[0, 0], A[1, 0]))
    if np.abs(eigValBeta[0, ev].imag) > 1.e-6:
        ev += 1
    ev += 1
#plt.legend(fontsize=ergoPlot.fs_default, loc='lower left')
xlim = [betaRng[0], betaRng[-1]]
ymin = np.min(evRealMin)
ylim = [ymin, -ymin / 100]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# Add text labels
for iev in np.arange(1, nevPlot):
    ycorr = (ylim[1] - ylim[0]) * 0.03
    xcorr = (xlim[1] - xlim[0]) * 0.02
    if steepness[iev] >= 0:
        sgn = '+'
    else:
        sgn = '-'
    ax.text(betaRng[0] + xcorr, ordinate[iev] + steepness[iev] * betaRng[0]**2 + ycorr,
            r'$\Re(\widehat{\lambda}_%d) = %.2f %s %.2f \beta^2$' \
            % (evRank[iev], ordinate[iev], sgn, np.abs(steepness[iev])),
            fontsize='x-large', color=colors[iev])
# Parameter labels
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.03,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\mu = %.1f$' % mu, fontsize='xx-large')
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.82,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\epsilon = %.1f$' % eps, fontsize='xx-large')
# Axes
ax.set_xlabel(r'$\beta$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/eigValRealVSbeta%s.%s' \
            % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# Scatter plot of eigenvalues
msize = 32
fig = plt.figure()
ax = fig.add_subplot(111)
evRealMin = []
evImagMin = []
markers = ['o', '+', 'x']
markersizes = [32, 60, 40]
markercolors = rcParams['axes.prop_cycle'].by_key()['color']
xlabel = r'$\Re(\lambda_k)$'
ylabel = r'$\Im(\lambda_k)$'
for ibeta in np.arange(betaRngPlot.shape[0]):
    ib = np.argmin((betaRng - betaRngPlot[ibeta])**2)
    beta = betaRng[ib]
    eigVal = eigValBeta[ib]
    ax.scatter(eigVal.real, eigVal.imag, c=markercolors[ibeta],
               s=markersizes[ibeta], marker=markers[ibeta], edgecolors='face',
               label=r'$\beta = %.2f$' % beta)
    evRealMin.append(np.min(eigVal.real))
    evImagMin.append(np.min(eigVal.imag))
xmin = np.min(evRealMin)
xmax = -xmin/100
ymin = np.min(evImagMin)
ymax = -ymin
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Parameter labels
ax.text(xmin*0.96, ymax*1.03, r'$\mu = %.1f$' % mu,
        fontsize='xx-large')
ax.text(xmin*0.18, ymax*1.03, r'$\epsilon = %.1f$' % eps,
        fontsize='xx-large')
plt.legend(fontsize=ergoPlot.fs_default, loc='lower right', frameon=True)
ax.set_xlabel(xlabel, fontsize=ergoPlot.fs_latex)
ax.set_ylabel(ylabel, fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_latex)
fig.savefig('%s/eigValVSbeta%s.%s' % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)



# for k in np.arange(1, 12):
#     X = np.matrix(np.ones((betaRng.shape[0], 2)))
#     X[:, 1] = np.matrix(betaRng**2).T
#     A = (X.T * X)**(-1) * (X.T * np.matrix(eigVal[:, k]).real.T)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(betaRng, eigVal[:, k].real, '+k', markersize=16,
#             markeredgewidth=2,
#             label=r'$\mu = 0 : \quad \Re(\lambda_{%d})$' % k)
#     ax.plot(betaRng, np.array(X*A)[:, 0], '--k')

#     if np.abs(eigVal[0, k].imag) > 1.e-5:
#         Xim = np.matrix(np.ones((betaRng.shape[0], 2)))
#         Xim[:, 1] = np.matrix(betaRng**2).T
#         Aim = (Xim.T * Xim)**(-1) \
#               * (Xim.T * np.matrix(eigVal[:, k].imag**2).T)
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.plot(betaRng, np.abs(eigVal[:, k].imag), '+k',
#                 markersize=16, markeredgewidth=2,
#                 label=r'$\mu = 0 : \quad \Im(\lambda_{%d})$' % k)
#         ax.plot(betaRng, np.sqrt(np.array(Xim*Aim)[:, 0]), '--k')
