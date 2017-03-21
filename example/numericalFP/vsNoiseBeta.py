import os
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

# Get model
model = 'Hopf'
gam = 1.
betaRng = np.arange(0., 2.05, 0.05)
#epsRng = np.arange(.05, 2.05, .05)
epsRng = np.array([1.])
mu = 0.

# Grid definition
dim = 2
nx0 = 200
nSTD = 5
nevPlot = 6
nPlot = 5

# Number of eigenvalues
nev = 21
#nev = 201

# Directories
resDir = '../results/numericalFP/%s' % model
plotDir = '../results/plot/numericalFP/%s' % model
os.system('mkdir %s 2> /dev/null' % plotDir)
mu += 1.e-8
if mu < 0:
    signMu = 'm'
else:
    signMu = 'p'

# Allocation
eigVal = np.empty((epsRng.shape[0], betaRng.shape[0], nev), dtype=complex)
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < epsRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))

# Read eigenvalues
for ibeta in np.arange(betaRng.shape[0]):
    beta = betaRng[ibeta]
    beta += 1.e-8
    if beta < 0:
        signBeta = 'm'
    else:
        signBeta = 'p'
    for ieps in np.arange(epsRng.shape[0]):
        eps = epsRng[ieps]
        postfix = '_%s_mu%s%02d_beta%s%03d_eps%03d_nx%d_nSTD%d_nev%d' \
                  % (model, signMu, int(round(np.abs(mu) * 10)),
                     signBeta, int(round(np.abs(beta) * 100)), int(round(eps * 100)),
                     nx0, nSTD, nev)

        # Read eigenvalues
        eigValii = np.empty((nev,), dtype=complex)
        ergoPlot.loadtxt_complex('%s/eigValForward%s.txt' \
                                 % (resDir, postfix), eigValii)
        isort = np.argsort(-eigValii.real)
        eigValii = eigValii[isort]
        eigVal[ieps, ibeta] = eigValii

# Comparison with analytics of real part of the second eigenvalue versus the noise
fig = plt.figure()
ax = fig.add_subplot(111)
xplot = np.linspace(0., epsRng[-1], 1000)
XP = np.matrix(xplot).T
evMin = []
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < betaRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
linewidth = 2
marker = 'x'
markersize = 6
steepness = np.empty((betaRng.shape[0],))
for ibeta in np.arange(betaRng.shape[0]):
    beta = betaRng[ibeta]
    ev = eigVal[:, ibeta, 1].real
    evMin.append(ev.min())
    ax.plot(epsRng, ev, '%s%s' % (marker, colors[ibeta]), markersize=markersize,
            label=r'$\Re(\lambda_1)$ for $\mu = %d$' % mu)
    X = np.matrix(epsRng).T
    B = np.matrix(ev).T
    A = (X.T * X)**(-1) * (X.T * B)
    Y = X * A
    Stot = np.var(B)
    Sres = np.sum((np.array(Y)[:, 0] - B)**2) / epsRng.shape[0]
    steepness[ibeta] = A[0, 0]
    ax.plot(xplot, np.array(XP*A)[:, 0], '-%s' % colors[ibeta], linewidth=linewidth,
            label=r'$\Re(\widehat{\lambda}_1)} = %.2f \epsilon$ for $\mu = %d$' \
            % (A[0, 0], mu))
#ax.plot(xplot, -xplot, '--k', linewidth=linewidth)
xlim = [epsRng[0], epsRng[-1]]
ylim = [np.min(evMin), 0.05]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * fig.get_figheight() / fig.get_figwidth()
# Add regression label
for ibeta in np.arange(betaRng.shape[0]):
    beta = betaRng[ibeta]
    rotation = np.arctan(steepness[ibeta] * aspect) * 180./np.pi
    txt = r'$\Re(\widehat{\lambda}_1) = %.2f \epsilon \quad \mathrm{for} \quad \beta = %d$' \
          % (steepness[ibeta], beta)
    ii = int(epsRng.shape[0] / 2)
    xcorr = (xlim[1] - xlim[0]) * 0.02
    ax.text(epsRng[ii] + xcorr, epsRng[ii] * steepness[ibeta], txt,
            rotation=rotation, fontsize = 'xx-large')
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.05, ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\mu = %d$' % mu, fontsize='xx-large')
ax.set_xlabel(r'$\epsilon$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda)$', fontsize=ergoPlot.fs_latex)
xlim = ax.set_xlim(0, epsRng[-1])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
#plt.legend(fontsize=ergoPlot.fs_default, loc='lower left', frameon=False)
fig.savefig('%s/eigVal2RealVSepsVSBeta%s.%s' \
            % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# # Fit to steepness
# X = np.ones((betaRng.shape[0], 2))
# X[:, 1] = betaRng**2
# X = np.matrix(X)
# B = np.matrix(steepness).T
# A = (X.T * X)**(-1) * (X.T * B)
# Y = X * A
# Stot = np.var(steepness)
# Sres = np.sum((np.array(Y)[:, 0] - steepness)**2) / betaRng.shape[0]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(betaRng, steepness, '%sb' % marker, markersize=markersize)
# ax.plot(betaRng, np.array(Y)[:, 0], '-b', linewidth=linewidth)

# Plot first eigenvalues versus beta
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < nevPlot:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
xplot = np.linspace(0., betaRng[-1], 1000)
XP = np.ones((xplot.shape[0], 2))
XP[:, 1] = xplot**2
XP = np.matrix(XP)

epsRngPlot = epsRng[::nPlot]
fit = np.empty((epsRngPlot.shape[0], nevPlot, 2))
for ieps in np.arange(epsRngPlot.shape[0]):
    eps = epsRngPlot[ieps]
    plotPostfix = '_%s_mu%s%02d_eps%03d_nx%d_nSTD%d_nev%d' \
                  % (model, signMu, int(round(np.abs(mu) * 10)),
                     int(round(eps * 100)), nx0, nSTD, nev)
    print 'Plot postfix = ', plotPostfix
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ev = 0
    evRank = []
    evRealMin = []
    for iev in np.arange(nevPlot):
        eigValReal = eigVal[ieps, :, ev].real
        evRealMin.append(np.min(eigValReal))
        # Linear regression
        nreg = int(betaRng.shape[0] * 0.9)
        X = np.ones((nreg, 2))
        X[:, 1] = betaRng[:nreg]**2
        X = np.matrix(X)
        B = np.matrix(eigValReal[:nreg]).T
        A = (X.T * X)**(-1) * (X.T * B)
        Y = X * A
        Stot = np.var(B)
        Sres = np.sum((np.array(Y)[:, 0] - B)**2) / nreg
        fit[ieps, iev, 0] = A[0, 0]
        fit[ieps, iev, 1] = A[1, 0]
        evRank.append(ev)
        ax.plot(betaRng, eigValReal, 'x%s' % colors[iev],
                markersize=markersize, label=r'$\Re(\lambda_{%d})$' % (ev,))
        ax.plot(xplot, np.array(XP*A)[:, 0],
                '-%s' % colors[iev], linewidth=linewidth,
                label=r'$\Re(\widehat{\lambda}_%d)} = %.2f + %.2f \beta^2$' \
                % (ev, A[0, 0], A[1, 0]))
        if np.abs(eigVal[ieps, 0, ev].imag) > 1.e-6:
            ev += 1
        ev += 1
    #plt.legend(fontsize=ergoPlot.fs_default, loc='lower left')
    xlim = [betaRng[0], betaRng[-1]]
    ymin = np.min(evRealMin)
    ylim = [ymin, -ymin / 100]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ycorr = (ylim[1] - ylim[0]) * 0.03
    xcorr = (xlim[1] - xlim[0]) * 0.02
    # Add text labels
    for iev in np.arange(1, nevPlot):
        ax.text(betaRng[0] + xcorr,
                fit[ieps, iev, 0] + fit[ieps, iev, 1] * betaRng[0]**2 + ycorr,
                r'$\Re(\widehat{\lambda}_%d) = %.2f + %.2f \beta^2$' \
                % (evRank[iev], fit[ieps, iev, 0], fit[ieps, iev, 1]),
                fontsize='xx-large', color=colors[iev])
    # Parameter labels
    ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.03,
            ylim[1] + (ylim[1] - ylim[0]) * 0.02,
            r'$\mu = %.1f$' % mu, fontsize='xx-large')
    ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.82,
            ylim[1] + (ylim[1] - ylim[0]) * 0.02,
            r'$\epsilon = %.2f$' % eps, fontsize='xx-large')
    # Axes
    ax.set_xlabel(r'$\beta$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$\Re(\lambda)$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    fig.savefig('%s/eigValRealVSbeta%s.%s' \
                % (plotDir, plotPostfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
