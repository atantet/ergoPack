import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../cfg/')
from Hopf import *
from ergoInt import *
from ergoCont import *
import ergoPlot

cfg = getModelParam()
dim = cfg.model.dim
fileFormat = cfg.general.fileFormat
dt = cfg.simulation.dt
if (cfg.general.fileFormat == 'bin'):
    readFile = np.fromfile
else:
    readFile = np.loadtxt


# Define diffusion matrix (constant)
# Stochastic forcing configuration
Q = np.eye(dim) * p["eps"]**2

# List of continuations to plot
initCont = 0.
contStep = 0.01

srcPostfix = "_%s" % (cfg.model.caseName,)
contDir = '%s/continuation' % cfg.general.resDir
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
vecLeftFileName = '%s/poVecLeft/poVecLeft%s.%s' \
                  % (contDir, dstPostfix, fileFormat)
vecRightFileName = '%s/poVecRight/poVecRight%s.%s' \
                   % (contDir, dstPostfix, fileFormat)
phiFileName = '%s/phase/phaseDiffusion%s.%s' \
              % (contDir, dstPostfix, fileFormat)

# Read periodic orbit
print 'Reading states...'
state = readFile(poFileName)
state = state.reshape(-1, dim+2)
poRng = state[:, :dim]
contRng = state[:, dim]
TRng = state[:, dim+1]
print 'Phase diffusion coefficient...'
data = readFile(phiFileName).reshape(-1, 2)
contRngSel = data[:, 0]
phiRngSel = data[:, 1]

# Read Floquet exponents
print 'Reading Floquet vectors...'
FloquetExp = readFile(FloquetExpFileName).reshape(-1, dim, 2)
FloquetExp = FloquetExp[:, :, 0] + 1j * FloquetExp[:, :, 1]
vecLeft = readFile(vecLeftFileName).reshape(-1, dim, dim, 2)
vecLeft = vecLeft[:, :, :, 0] + 1j * vecLeft[:, :, :, 1]
vecRight = readFile(vecRightFileName).reshape(-1, dim, dim, 2)
vecRight = vecRight[:, :, :, 0] + 1j * vecRight[:, :, :, 1]
# Select
scale = .4
for c in np.arange(contRngSel.shape[0]):
    argm = np.argmin(np.abs(contRng - contRngSel[c]))
    p['mu'] = contRng[argm]
    print 'Plotting orbit and vectors for mu = ', p['mu']
    po = poRng[argm]
    T = TRng[argm]
    FE = np.real(FloquetExp[argm])
    FVL = np.real(vecLeft[argm])
    FVR = np.real(vecRight[argm])

    # Propagate
    nt = int(np.ceil(T / dt))
    (xt, Mts) = propagateFundamentalRange(po, fieldHopf, JacFieldHopf, p,
                                          stepRK4, dt, nt)

    # Plot orbit with Floquet vectors
    comps = [0, 1]
    plotFloquetVec(xt, p, FE, FVL, FVR, comps, scale=scale)
    dstPostfixMu = "%s_mu%04d" % (dstPostfix, int(p["mu"] * 1000 + 0.1))
    plt.savefig('%s/continuation/po/orbit/orbit%s.%s' \
                % (cfg.general.plotDir, dstPostfixMu, ergoPlot.figFormat),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Analytical solution
phiAna = p["eps"]**2 * (1 + p["beta"]**2) / contRngSel
    
# Plot phase diffusion coefficient
lw=2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(contRngSel, phiRngSel, '-k', linewidth=lw)
ax.plot(contRngSel, phiAna, '--k', linewidth=lw)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Phi$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRng), np.max(contRng[-1]))
ax.set_ylim(0., phiRngSel.max() * 1.5)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusion%s.%s' \
            % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# tmp = phiRngSel / p["eps"]**2 * contRngSel
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(contRngSel, tmp, linewidth=lw)
# ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
# ax.set_ylabel(r'$\Phi$', fontsize=ergoPlot.fs_latex)
# ax.set_xlim(np.min(contRng), np.max(contRng[-1]))
# #ax.set_ylim(0., phiRngSel.max())
# plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
# plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# fig.savefig('%s/continuation/phase/phaseDiffusion%s.%s' \
#             % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
#             dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
