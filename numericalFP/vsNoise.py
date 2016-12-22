import os
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

# Get model
model = 'Hopf'
gam = 1.
beta = 0.5
epsRng = np.arange(.05, 2.05, .05)
#muRng = np.array([0., 5.])
muRng = np.array([0.])

# Grid definition
dim = 2
nx0 = 200
nSTD = 5

# Number of eigenvalues
nev = 21
#nev = 201
nevPlot = 6

# Directories
resDir = '../results/numericalFP/%s' % model
plotDir = '../results/plot/numericalFP/%s' % model
os.system('mkdir %s 2> /dev/null' % plotDir)
beta += 1.e-8
if beta < 0:
    signBeta = 'm'
else:
    signBeta = 'p'
plotPostfix = '_%s_beta%s%03d_nx%d_nSTD%d_nev%d' \
              % (model, signBeta, int(round(np.abs(beta) * 100)),
                 nx0, nSTD, nev)
print 'Plot postfix = ', plotPostfix

# Allocation
eigVal = np.empty((epsRng.shape[0], muRng.shape[0], nev), dtype=complex)
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < epsRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))

# Read eigenvalues
for imu in np.arange(muRng.shape[0]):
    mu = muRng[imu]
    mu += 1.e-8
    if mu < 0:
        signMu = 'm'
    else:
        signMu = 'p'
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
        eigVal[ieps, imu] = eigValii

# Comparison with analytics of real part of the second eigenvalue versus the noise
fig = plt.figure()
ax = fig.add_subplot(111)
xplot = np.linspace(0., epsRng[-1], 1000)
XP = np.matrix(xplot).T
evMin = []
muLineStyle = ['-', '--']
muMarker = ['x', '+']
muMarkerSize = [6, 9]
lw = 2
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < nevPlot:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
steepness = np.empty((muRng.shape[0], nevPlot))
evRank = np.empty((muRng.shape[0], nevPlot))
for imu in np.arange(muRng.shape[0]):
    mu = muRng[imu]
    ev = 1
    for iev in np.arange(nevPlot):
        eigv = eigVal[:, imu, ev].real
        evMin.append(eigv.min())
        ax.plot(epsRng, eigv, '%s%s' % (muMarker[imu], colors[iev]),
                markersize=muMarkerSize[imu], label=r'$\Re(\lambda_1)$')
        if np.abs(mu) < 1.e-6:
            X = np.matrix(epsRng).T
            A = (X.T * X)**(-1) * (X.T * np.matrix(eigv).T)
            Y = X * A
            Stot = np.var(eigv)
            Sres = np.sum((np.array(Y)[:, 0] - eigv)**2) / epsRng.shape[0]
            steepness[imu, iev] = A[0, 0]
            evRank[imu, iev] = ev
            ax.plot(xplot, np.array(XP*A)[:, 0], '%s%s' % (muLineStyle[imu], colors[iev]), linewidth=lw,
                    label=r'$\Re(\widehat{\lambda}_{%d})} = %.2f \epsilon$' \
                    % (ev, A[0, 0]))
        else:
            label = r'$\Re(\widehat{\lambda}_{%d})} = -\epsilon^2 (1 + \beta^2) / (2 \delta)$' % ev
            ax.plot(xplot, -xplot**2 * (1 + beta**2) / (2*mu),
                    '-%s' % muColor[imu], linewidth=2, label=label)
        if np.abs(eigVal[0, imu, ev].imag) > 1.e-6:
            ev += 1
        ev += 1
#ax.plot(xplot, -xplot, '--k', linewidth=2)
xlim = [epsRng[0], epsRng[-1]]
ylim = [np.min(evMin), 0.05]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * fig.get_figheight() / fig.get_figwidth()
# Add regression label
for imu in np.arange(muRng.shape[0]):
    mu = muRng[imu]
    for iev in np.arange(nevPlot):
        if np.abs(mu) < 1.e-6:
            rotation = np.arctan(steepness[imu, iev] * aspect) * 180./np.pi
            txt = r'$\Re(\widehat{\lambda}_{%d}) = %.2f \epsilon$' \
                  % (evRank[imu, iev], steepness[imu, iev])
            ii = int(epsRng.shape[0] * 0.97)
            corr = 0.1
            xcorr = -corr *  np.sin(rotation*np.pi/180.)
            ycorr = corr / np.cos(rotation*np.pi/180.)
            ax.text(epsRng[ii] + xcorr, epsRng[ii] * steepness[imu, iev] + ycorr, txt,
                    rotation=rotation, fontsize = 'xx-large', color=colors[iev],
                    ha='right', va='bottom')
        else:
            rotation = 0.
            txt = r'$\Re(\widehat{\lambda}_{%d}) = -\epsilon^2 (1 + \beta^2) / (2 \delta)$' % evRank[iev] \
                  + '\n' \
                  + r'$\quad \quad \quad \quad \quad \quad \quad \quad \mathrm{for} \quad \delta = %d$' % mu
            xcorr = (xlim[1] - xlim[0]) * 0.01
            ycorr = (ylim[1] - ylim[0]) * (-0.05)
            ax.text(epsRng[ii] + xcorr, -epsRng[ii]**2 \
                    * (1 + beta**2) / (2*mu) + ycorr, txt,
                    rotation=rotation, fontsize = 'xx-large')
ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.05,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02, r'$\delta = %.1f$' % mu,
        fontsize='xx-large')
ax.text(xlim[1] - (xlim[1] - xlim[0]) * 0.18,
        ylim[1] + (ylim[1] - ylim[0]) * 0.02, r'$\beta = %.1f$' % beta,
        fontsize='xx-large')
ax.set_xlabel(r'$\epsilon$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda)$', fontsize=ergoPlot.fs_latex)
xlim = ax.set_xlim(0, epsRng[-1])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
#plt.legend(fontsize=ergoPlot.fs_default, loc='lower left', frameon=False)
fig.savefig('%s/eigValRealVSeps%s.%s' \
            % (plotDir, plotPostfix, ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
