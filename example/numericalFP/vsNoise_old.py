import numpy as np
import pylibconfig2
from scipy import sparse
from scipy.sparse import linalg
import ergoPlot

# Get model
omega = 1.
beta = 0.4
qRng = np.arange(.25, 2.2, .25)
qRngPlot = np.array([0.25, 0.5, 1., 1.5, 2.])
muRng = np.arange(-5, 5.05, 0.1)
#muRng = np.arange(-5, 10.1, 0.1)
#muRng = np.arange(0., 10.1, 0.1)
k0 = int(np.round((10 + muRng[0]) / (muRng[1] - muRng[0])))

# Grid definition
dim = 2
#nx0 = 100
nx0 = 200

# Plot config
#figFormat = 'png'
figFormat = 'eps'
xmin = -30.
xmax = 0.1
ymin = -10.
ymax = -ymin

eigVal2 = np.empty((qRng.shape[0], muRng.shape[0]), dtype=complex)
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < qRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))
for iq in np.arange(qRng.shape[0]):
    q = qRng[iq]
    print 'For q = ', q
    for k in np.arange(muRng.shape[0]):
        mu = muRng[k]
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
                  % (nx0, k0 + k, signMu, int(round(np.abs(mu) * 10)),
                     signBeta, int(round(np.abs(beta) * 100)), int(round(q * 100)))

        print 'Reading eigenvalues'
        srcFile = '../results/numericalFP/w_hopf_adapt%s.txt' % postfix
        fp = open(srcFile, 'r')
        eigVal = np.empty((2,), dtype=complex)
        for ev in np.arange(2):
            line = fp.readline()
            line = line.replace('+-', '-')
            eigVal[ev] = complex(line)
        
        eigVal2[iq, k] = eigVal[1]

fig = plt.figure()
ax = fig.add_subplot(111)
cit=0
for iq in np.arange(qRng.shape[0]):
    q = qRng[iq]
    if q in qRngPlot:
        ax.plot(muRng, eigVal2[iq, :].real, linestyle='-', color=colors[cit],
                linewidth=ergoPlot.lw)
        ax.plot(muRng[muRng > 0.], -q**2/(2*muRng[muRng > 0.]), linewidth=ergoPlot.lw,
                linestyle='--', color=colors[cit])
        cit += 1
ax.plot(muRng[muRng <= 0.], muRng[muRng <= 0.], '--k', linewidth=ergoPlot.lw)
ax.set_xlim(muRng[0], muRng[-1])
ax.set_ylim(muRng[0], 0.)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('../results/plot/numericalFP/w2RealVSmuVSeps_hopf_adapt%s.%s' \
            % (postfix, ergoPlot.figFormat), bbox_inches=ergoPlot.bbox_inches,
            dpi=ergoPlot.dpi)

fig = plt.figure()
ax = fig.add_subplot(111)
cit=0
for iq in np.arange(qRng.shape[0]):
    q = qRng[iq]
    if q in qRngPlot:
        ax.plot(muRng, eigVal2[iq, :].imag, linestyle='-', color=colors[cit],
                linewidth=ergoPlot.lw)
        ax.plot(muRng[muRng > 0.], np.ones((muRng.shape[0],)) * omega, linewidth=ergoPlot.lw,
                linestyle='--', color=colors[cit])
        cit += 1
ax.plot(muRng[muRng <= 0.], muRng[muRng <= 0.], '--k', linewidth=ergoPlot.lw)
ax.set_xlim(muRng[0], muRng[-1])
ax.set_ylim(muRng[0], 0.)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('../results/plot/numericalFP/w2ImagVSmuVSeps_hopf_adapt%s.%s' \
            % (postfix, ergoPlot.figFormat), bbox_inches=ergoPlot.bbox_inches,
            dpi=ergoPlot.dpi)

fig = plt.figure()
ax = fig.add_subplot(111)
xplot = np.linspace(0., qRng[-1], 1000)
mu = 0.
imu = int(np.round((10 + mu) / (muRng[1] - muRng[0]))) - k0
ev = eigVal2[:, imu].real
X = np.matrix(np.concatenate((qRng, np.ones((qRng.shape[0],)))).reshape(2, -1)).T
A = (X.T * X)**(-1) * (X.T * np.matrix(ev).T)
Y = X * A
Stot = np.var(ev)
Sres = np.sum((np.array(Y)[:, 0] - ev)**2) / qRng.shape[0]
ax.plot(qRng, ev, '+k', markersize=16, markeredgewidth=2,
        label=r'$\mu = 0 : \quad \Re(\lambda_1)$')
ax.plot(xplot, xplot*A[0, 0] + A[1,0], '--k', linewidth=2,
        label=r'$\mu = 0 : \quad %.2f \epsilon$' % A[0, 0])
#ax.plot(xplot, -xplot, '--k', linewidth=2)
ax.set_ylim(ev.min(), 0.05)

mu = 5.
imu = int(np.round((10 + mu) / (muRng[1] - muRng[0]))) - k0
ev = eigVal2[:, imu].real
ax.plot(qRng, ev, 'xk', markersize=12, markeredgewidth=2,
        label=r'$\mu = 5 : \quad \Re(\lambda_1)$')
ax.plot(xplot, -xplot**2 / (2*mu), '-.k', linewidth=2,
        label=r'$\mu = 5 : \quad -\epsilon^2 / (2 \mu)$')
ax.set_xlabel(r'$\epsilon$', fontsize=ergoPlot.fs_latex)
#ax.set_ylabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
xlim = ax.set_xlim(0, qRng[-1])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.legend(fontsize=ergoPlot.fs_default, loc='lower left')
fig.savefig('../results/plot/numericalFP/w2RealVSeps_hopf_adapt%s.%s' \
            % (postfix, ergoPlot.figFormat), bbox_inches=ergoPlot.bbox_inches,
            dpi=ergoPlot.dpi)

fig = plt.figure()
ax = fig.add_subplot(111)
xplot = np.linspace(0., qRng[-1], 1000)
mu = 0.
imu = int(np.round((10 + mu) / (muRng[1] - muRng[0]))) - k0
ev = eigVal2[:, imu].imag
X = np.matrix(np.concatenate((qRng, np.ones((qRng.shape[0],)))).reshape(2, -1)).T
A = (X.T * X)**(-1) * (X.T * np.matrix(ev).T)
Y = X * A
Stot = np.var(ev)
Sres = np.sum((np.array(Y)[:, 0] - ev)**2) / qRng.shape[0]
ax.plot(qRng, ev, '+k', markersize=16, markeredgewidth=2,
        label=r'$\mu = 0 : \quad \Re(\lambda_1)$')
ax.plot(xplot, xplot*A[0, 0] + A[1,0], '--k', linewidth=2,
        label=r'$\mu = 0 : \quad %.2f \epsilon$' % A[0, 0])
ax.set_xlabel(r'$\epsilon$', fontsize=ergoPlot.fs_latex)
#ax.set_ylabel(r'$\Re(\lambda_1)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.legend(fontsize=ergoPlot.fs_default, loc='lower left')
fig.savefig('../results/plot/numericalFP/w2ImagVSeps_hopf_adapt%s.%s' \
            % (postfix, ergoPlot.figFormat), bbox_inches=ergoPlot.bbox_inches,
            dpi=ergoPlot.dpi)

