import os
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

# Get model
model = 'Hopf'
gam = 1.
#beta = 0.0
#beta = 0.5
beta = 1.
epsRng = np.array([0.25, 0.5, 1., 1.5, 2., 3., 4.])
#muRng = np.arange(-5, 5.05, 0.1)
muRng = np.arange(-1, 1.01, 0.1)
muZoomLim = [-1.01, 1.01]

# Grid definition
dim = 2
nx0 = 200
nSTD = 5

# Number of eigenvalues
nev = 21
#nev = 201

# Directories
beta += 1.e-8
if beta < 0:
    signBeta = 'm'
else:
    signBeta = 'p'
resDir = '../results/numericalFP/%s' % model
plotDir = '../results/plot/numericalFP/%s' % model
os.system('mkdir %s 2> /dev/null' % plotDir)
plotPostfix = '_%s_beta%s%03d_nx%d_nSTD%d_nev%d' \
              % (model, signBeta, int(round(np.abs(beta) * 100)),
                 nx0, nSTD, nev)

eigVal2 = np.empty((epsRng.shape[0], muRng.shape[0]), dtype=complex)
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < epsRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
for ieps in np.arange(epsRng.shape[0]):
    eps = epsRng[ieps]
    print 'For eps = ', eps
    for k in np.arange(muRng.shape[0]):
        mu = muRng[k]
        mu += 1.e-8
        if mu < 0:
            signMu = 'm'
        else:
            signMu = 'p'
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
        eigVal2[ieps, k] = eigVal[1]

# Comparison of the analytics with the real part of the second eigenvalue
# for varying mu and eps
fig = plt.figure()
ax = fig.add_subplot(111)
cit=0
for ieps in np.arange(epsRng.shape[0]):
    eps = epsRng[ieps]
    if eps in epsRng:
        # Plot real part of second eigenvalue
        ax.plot(muRng, eigVal2[ieps, :].real, linestyle='-', color=colors[cit],
                linewidth=ergoPlot.lw, label=r'$\epsilon = %.2f$' % eps)
        # Add analytics for mu > 0
        ax.plot(muRng[muRng > 0.], -eps**2/(2*muRng[muRng > 0.]), linewidth=ergoPlot.lw,
                linestyle='--', color=colors[cit])
        cit += 1
# Add analytics for mu < 0
ax.plot(muRng[muRng <= 0.], muRng[muRng <= 0.], '--k', linewidth=ergoPlot.lw)
# Add parameter label
xlim = (muRng[0], muRng[-1])
ylim = (muRng[0], 0.)
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.82,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\beta = %.1f$' % beta, fontsize='xx-large')
# Add legend
plt.legend(fontsize=ergoPlot.fs_default, frameon=False,
           loc=(0., 0.49))
# Configure axes
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/eigVal2RealVSmuVSeps%s.%s' \
            % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# Zoom around 0
imuZoom = (muRng >= muZoomLim[0]) & (muRng <= muZoomLim[1])
muZoom = muRng[imuZoom]
nreg = muZoom.shape[0]
X = np.ones((nreg, 2))
X[:, 1] = muZoom
X = np.matrix(X)
markersize = 6
fig = plt.figure()
ax = fig.add_subplot(111)
cit=0
evRealMin = []
ordinate = []
steepness = []
for ieps in np.arange(epsRng.shape[0]):
    eps = epsRng[ieps]
    ev = eigVal2[ieps, imuZoom].real
    evRealMin.append(np.min(ev))
    B = np.matrix(ev).T
    A = (X.T * X)**(-1) * (X.T * B)
    Y = X * A
    Stot = np.var(B)
    Sres = np.sum((np.array(Y)[:, 0] - B)**2) / nreg
    ordinate.append(A[0, 0])
    steepness.append(A[1, 0])
    ax.plot(muZoom, ev, 'x%s' % colors[cit], markersize=markersize)
    ax.plot(muZoom, np.array(Y)[:, 0], linestyle='-',
            color=colors[cit], linewidth=ergoPlot.lw)
    cit += 1
xlim = [muZoom[0], muZoom[-1]]
ylim = [np.min(evRealMin), 0.]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# Add text labels to linear regressions
aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) \
         * fig.get_figheight() / fig.get_figwidth()
ycorr = (ylim[1] - ylim[0]) * 0.09
xcorr = (xlim[1] - xlim[0]) * 0.02
cit = 0
for ieps in np.arange(epsRng.shape[0]):
    rotation = np.arctan(steepness[ieps] * aspect) * 180./np.pi
    if steepness[ieps] >= 0:
        sgn = '+'
    else:
        sgn = '-'
    ax.text(muZoom[0] + xcorr,
            ordinate[ieps] + muZoom[0] * steepness[ieps] + ycorr,
            r'$\Re(\widehat{\lambda}_1) = %.2f %s %.2f \mu$' \
            % (ordinate[ieps], sgn, np.abs(steepness[ieps])),
            fontsize='large', rotation=rotation, color=colors[cit])
    cit += 1
# Add parameter label
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.82,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02,
        r'$\beta = %.1f$' % beta, fontsize='xx-large')
# Configure axes
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/eigVal2RealVSmuVSepsZoom%s.%s' \
            % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
