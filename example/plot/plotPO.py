import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../cfg/')
from Hopf import *
from ergoInt import *

dim = cfg.model.dim
fileFormat = cfg.general.fileFormat

# List of continuations to plot
initContRng = [0.]
contStepRng = [0.01]
nCont = len(initContRng)

# Prepare plot
plotImag = False
#plotImag = True
lsStable = ['--', '-']
lw = 2
color = 'k'
msize = 30
cCycle = matplotlib.rcParams['axes.color_cycle']
if plotImag:
    fig = plt.figure(figsize=(6, 9))
else:
    fig = plt.figure(figsize=(6, 6))
ax = []
#nPan = 100*(1+2*nCont) + 10 + 1
if plotImag:
    nPan = 100*(1+2*1) + 10 + 1
else:
    nPan = 100*(1+1) + 10 + 1
ax.append(fig.add_subplot(nPan))
#for k in np.arange(nCont):
for k in np.arange(1):
    nPan += 1
    ax.append(fig.add_subplot(nPan))
    if plotImag:
        nPan += 1
        ax.append(fig.add_subplot(nPan))

poL = []
FloquetExpL = []
contL = []
TRngL = []
contLim = np.empty((nCont, 2))
fact = np.array([1., 1.])
for k in np.arange(nCont):
    initCont = initContRng[k]
    contStep = contStepRng[k]
    
    contAbs = np.sqrt(contStep*contStep)
    sign = contStep / contAbs
    exp = np.log10(contAbs)
    mantis = sign * np.exp(np.log(contAbs) / exp)
    contPostfix = "_cont%04d_contStep%de%d" \
                  % (int(initCont * 1000 + 0.1), int(mantis*1.01),
                     (int(exp*1.01)))
    dstPostfixFP = "%s_beta%04d_gamma%04d%s" \
                 % (srcPostfix, int(p["beta"] * 1000 + 0.1),
                    int(p["gamma"] * 1000 + 0.1), contPostfix)
    dstPostfix = "%s_dt%d" \
                 % (dstPostfixFP,
                    int(np.round(-np.log10(cfg.simulation.dt)) + 0.1))

    poFileName = '%s/poState/poState%s.%s' % (contDir, dstPostfix, fileFormat)
    FloquetExpFileName = '%s/poExp/poExp%s.%s' \
                         % (contDir, dstPostfix, fileFormat)

        
    # Read fixed point and cont
    state = readFile(poFileName)
    state = state.reshape(-1, dim+2)
    # Read FloquetExpenvalues
    FloquetExp = readFile(FloquetExpFileName)
    FloquetExp = FloquetExp.reshape(-1, dim, 2)
    FloquetExp = FloquetExp[:, :, 0] + 1j * FloquetExp[:, :, 1]
    # Remove nans
    FloquetExp[np.isnan(FloquetExp)] \
        = np.min(FloquetExp.real[~np.isnan(FloquetExp)])

    po = state[:, :dim]
    TRng = state[:, dim+1]
    contRng = state[:, dim]
    # Convert to relaxation rate in years

    # Reorder Floquet exp
    isort = np.argsort(-FloquetExp[0].real)
    FloquetExp[0] = FloquetExp[0, isort]
    for t in np.arange(1, contRng.shape[0]):
        tmp = FloquetExp[t].tolist()
        for exp in np.arange(dim):
            idx = np.argmin(np.abs(tmp - FloquetExp[t-1, exp]))
            FloquetExp[t, exp] = tmp[idx]
            tmp.pop(idx)

    poL.append(po)
    FloquetExpL.append(FloquetExp)
    contL.append(contRng)
    contLim[k, 0] = np.min(contRng)
    contLim[k, 1] = np.max(contRng)
    TRngL.append(TRng)
    
    isStable = np.max(FloquetExp.real, 1) < 1.e-6
    change = np.nonzero(isStable[1:] ^ isStable[:-1])[0] + 1
    change = np.concatenate(([0], change, [contRng.shape[0]]))

    # Plot period
    for c in np.arange(change.shape[0]-1):
        ax[0].plot(contRng[change[c]:change[c+1]],
                   TRng[change[c]:change[c+1]],
                   linestyle=lsStable[isStable[change[c]]],
                   color=color, linewidth=lw)

    # Plot real parts
    k = 0
    ax[1+2*k].plot(contRng, np.zeros((contRng.shape[0],)), '--k')
    ax[1+2*k].plot(contRng, FloquetExp.real \
                   / np.tile(fact, (FloquetExp.shape[0], 1)), linewidth=2)
    ax[1+2*k].set_ylabel(r'$\Re(\lambda_i)$ (yr$^{-1}$)',
                         fontsize=ergoPlot.fs_latex)
    plt.setp(ax[1+2*k].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax[1+2*k].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)

    # Plot imaginary parts
    if plotImag:
        ax[1+2*k+1].plot(contRng, FloquetExp.imag, linewidth=2)
        ax[1+2*k+1].set_ylabel(r'$\Im(\lambda_i)$ (rad \
        $\cdot$ yr$^{-1}$)', fontsize=ergoPlot.fs_latex)
        plt.setp(ax[1+2*k+1].get_xticklabels(),
                 fontsize=ergoPlot.fs_xticklabels)
        plt.setp(ax[1+2*k+1].get_yticklabels(),
                 fontsize=ergoPlot.fs_yticklabels)
ax[0].set_ylabel(r'$T$ (yr)', fontsize=ergoPlot.fs_latex)
plt.setp(ax[0].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax[0].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax[-1].set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
for k in np.arange(len(ax)):
    ax[k].set_xlim(np.min(contLim[:, 0]), np.max(contLim[:, 1]) * 1.001)

fig.savefig('%s/continuation/po/poCont%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')


# Fixed point
fig = plt.figure()
ax = fig.add_subplot(111)
dist = []
contFPL = []
fpL = []
for k in np.arange(nCont):
    initContFP = initContRng[k]
    contStepFP = contStepRng[k]
    contAbs = np.sqrt(contStepFP*contStepFP)
    sign = contStepFP / contAbs
    exp = np.log10(contAbs)
    mantis = sign * np.exp(np.log(contAbs) / exp)
    fpFileName = '%s/fpState/fpState%s.%s' \
                 % (contDir, dstPostfixFP, fileFormat)
    eigValFileName = '%s/fpEigVal/fpEigValCont%s.%s' \
                     % (contDir, dstPostfixFP, fileFormat)

    # Read fixed point and cont
    state = readFile(fpFileName).reshape(-1, dim+1)
    fp = state[:, :dim]
    contRngFP = state[:, dim]
    fpL.append(fp)
    contFPL.append(contRngFP)
    eigValFP = readFile(eigValFileName).reshape(-1, dim, 2)
    eigValFP = eigValFP[:, :, 0] + 1j * eigValFP[:, :, 1]
    isStable = np.max(eigValFP.real, 1) < 0
    changeFP = np.nonzero(isStable[1:] ^ isStable[:-1])[0] + 1
    changeFP = np.concatenate(([0], changeFP, [contRngFP.shape[0]]))

    # Plot fixed points
    for c in np.arange(changeFP.shape[0]-1):
        ax.plot(fp[changeFP[c]:changeFP[c+1], 0],
                fp[changeFP[c]:changeFP[c+1], 1],
                linestyle=lsStable[isStable[changeFP[c]]],
                color=color, linewidth=lw)

nOrbit = 5
for k in np.arange(nCont):
    po = poL[k]
    FloquetExp = FloquetExpL[k]
    contRng = contL[k]
    TRng = TRngL[k]
    isStable = np.max(FloquetExp.real, 1) < 1.e-6

    ic = 0
    contSel = np.linspace(np.min(contRng), np.max(contRng), nOrbit)
    for icont in np.arange(contSel.shape[0]):
        icontSel = np.argmin((contSel[icont] - contRng)**2)
        cont = contRng[icontSel]
        T = TRng[icontSel]
        x0 = po[icontSel]
        FE = FloquetExp[icontSel]
        print 'Propagating orbit of period ', T, ' at mu = ', cont, \
            ' from x(0) = ', x0
        print 'Floquet = ', FE

        # Adapt time step
	ntOrbit = int(np.ceil(T / cfg.simulation.dt) + 0.1)
	dtOrbit = T / ntOrbit
        # propagate
        p['mu'] = cont
        xt = propagate(x0, fieldHopf, p, stepRK4, dtOrbit, ntOrbit)
        
        # Plot orbit
        ax.plot(xt[:, 0], xt[:, 1], label=r'$\mu = %.2f$' % cont,
                linestyle=lsStable[isStable[icontSel]], linewidth=lw)
        
        # Plot corresponding fixed point
        mini = 1.e27
        for l in np.arange(nCont): 
            icontFPl = np.argmin((contFPL[l] - cont)**2)
            mn = contFPL[l][icontFPl]
            if mn < mini:
                mini = mn
                minil = l
                icontFP = icontFPl
        ax.scatter(fpL[minil][icontFP, 0], fpL[minil][icontFP, 1],
                   s=msize, c=cCycle[ic%len(cCycle)], edgecolor='face',
                   marker='o')
        ic += 1

ax.legend(fontsize='x-large')
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$x$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$y$', fontsize=ergoPlot.fs_latex)
fig.savefig('%s/continuation/po/poContOrbit%s.%s' \
            % (plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

    
    
