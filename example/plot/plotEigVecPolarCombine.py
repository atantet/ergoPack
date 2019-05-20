import os
import numpy as np
from matplotlib import cm
import pylibconfig2
from ergoPack import ergoPlot

# configFile = '../cfg/Hopf.cfg'
configFile = '../cfg/Chekroun2019.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
fileFormat = cfg.general.fileFormat
specDir = os.path.join(cfg.general.resDir, 'spectrum')

L = cfg.simulation.LCut + cfg.simulation.spinup
spinup = cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
dim = cfg.model.dim
dimObs = len(cfg.observable.components)
nProc = ''
if (hasattr(cfg.sprinkle, 'nProc')):
    nProc = '_nProc' + str(cfg.sprinkle.nProc)

nev = cfg.spectrum.nev
evPlot = np.array([0])
plotForward = True
plotBackward = False
# cbar_format = '{:.2e}'
# evPlot = np.array([1, 2, 3, 4, 5, 6])
# plotForward = False
# plotBackward = True
xmin = None
xmax = None
ymin = None
ymax = None
# cbar_format = '{:.3e}'
cbar_format = None

# ampMin = 0.
# ampMax = 0.07
# nlevAmp = 11
ampMin = None
ampMax = None
nlevAmp = None
ev_xlabel = r'$x$'
ev_ylabel = r'$y$'


def d_formatter(x, pos=None):
    fmt = '' if x % 1 > 1.e-6 else '{:.0f}'.format(x)
    return fmt


xtick_formatter = d_formatter
ytick_formatter = d_formatter

srcPostfixSim = (
    "_%s_mu%04d_alpha%04d_gamma%04d_delta%04d_beta%04d_eps%04d_sep%04d"
    "_L%d_spinup%d_dt%d_samp%d" % (
        caseName, int(cfg.model.mu * 10000 + 0.1),
        int(cfg.model.alpha * 10000 + 0.1), int(cfg.model.gamma * 10000 + 0.1),
        int(cfg.model.delta * 10000 + 0.1), int(cfg.model.beta * 10000 + 0.1),
        int(cfg.model.eps * 10000 + 0.1), int(cfg.model.sep * 10000 + 0.1),
        int(L + 0.1), int(spinup + 0.1),
        int(-np.round(np.log10(cfg.simulation.dt)) + 0.1), printStepNum))
N = np.prod(np.array(cfg.grid.nx))
gridPostfix = ""
for d in np.arange(dimObs):
    if (hasattr(cfg.grid, 'gridLimitsLow')
            & hasattr(cfg.grid, 'gridLimitsHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.gridLimitsLow[d],
                                        cfg.grid.gridLimitsHigh[d])
    else:
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.sprinkle.minInitState[d],
                                        cfg.sprinkle.maxInitState[d])
postfix = "%s_nTraj%d%s" % (srcPostfixSim, cfg.sprinkle.nTraj,
                            gridPostfix)

# Read grid
gridFile = '{}/grid/grid_{}{}.txt'.format(
    cfg.general.resDir, caseName, gridPostfix)
coord = ergoPlot.readGrid(gridFile, dimObs)
X, Y = np.meshgrid(coord[0], coord[1], indexing='ij')
coord = (X.flatten(), Y.flatten())

# tauRng = cfg.transfer.tauRng
tauRng = [cfg.stat.tauPlot]
for tau in tauRng:
    postfixTau = "%s_tau%04d" % (postfix, int(tau * 10000 + 0.1))

    # File names
    eigValForwardFile = '%s/eigval/eigValForward_nev%d%s.%s' \
                        % (specDir, nev, postfixTau, fileFormat)
    eigVecForwardFile = '%s/eigvec/eigVecForward_nev%d%s.%s' \
        % (specDir, nev, postfixTau, fileFormat)
    eigValBackwardFile = '%s/eigval/eigValBackward_nev%d%s.%s' \
        % (specDir, nev, postfixTau, fileFormat)
    eigVecBackwardFile = '%s/eigvec/eigVecBackward_nev%d%s.%s' \
        % (specDir, nev, postfixTau, fileFormat)
    maskFile = '%s/transfer/mask/mask%s.%s' \
               % (cfg.general.resDir, postfixTau, fileFormat)

    # Read mask
    if maskFile is not None:
        mask = (np.fromfile(maskFile, np.int32) if fileFormat == 'bin'
                else np.loadtxt(maskFile, np.int32))
    else:
        mask = np.arange(N)
    NFilled = np.max(mask[mask < N]) + 1

    # Read transfer operator spectrum from file and create a bi-orthonormal basis
    # of eigenvectors and backward eigenvectors:
    print('Readig spectrum for tau = {:.3f}...'.format(tau))
    (eigValForward, eigValBackward, eigVecForward, eigVecBackward) \
        = ergoPlot.readSpectrum(eigValForwardFile, eigValBackwardFile,
                                eigVecForwardFile, eigVecBackwardFile,
                                makeBiorthonormal=~cfg.spectrum.makeBiorthonormal,
                                fileFormat=fileFormat)

    print('Getting conditionning of eigenvectors...')
    eigenCondition = ergoPlot.getEigenCondition(eigVecForward, eigVecBackward)

    # Get generator eigenvalues
    eigValGen = (np.log(np.abs(eigValForward)) +
                 np.angle(eigValForward)*1j) / tau

    # Plot eigenvectors of transfer operator
    alpha = 0.05
    csfilter = 0.5
    os.makedirs(os.path.join(cfg.general.plotDir, 'spectrum', 'eigvec'),
                exist_ok=True)
    os.makedirs(os.path.join(cfg.general.plotDir, 'spectrum', 'reconstruction'),
                exist_ok=True)
    for ev in evPlot:
        cmap = cm.hot_r if ev == 0 else cm.RdBu_r

        if plotForward:
            print('Plotting polar eigenvector {:d}...'.format(ev + 1))
            fig = ergoPlot.plotEigVecPolarCombine(
                X, Y, eigVecForward[ev], mask=mask, xlabel=ev_xlabel,
                ylabel=ev_ylabel, alpha=alpha, cmap=cmap, ampMin=ampMin,
                ampMax=ampMax, nlevAmp=nlevAmp, csfilter=csfilter, xmin=xmin,
                xmax=xmax, ymin=ymin, ymax=ymax, xtick_formatter=xtick_formatter,
                ytick_formatter=ytick_formatter, cbar_format=cbar_format)

            dstFile = '%s/eigvec/eigvecForwardPolar_nev%d_ev%03d%s.%s' \
                      % (specDir, nev, ev + 1, postfixTau, ergoPlot.figFormat)
            fig.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches,
                        dpi=ergoPlot.dpi)
        if plotBackward:
            print('Plotting polar backward eigenvector {:d}...'.format(ev + 1))
            fig = ergoPlot.plotEigVecPolarCombine(
                X, Y, eigVecBackward[ev], mask=mask, xlabel=ev_xlabel,
                ylabel=ev_ylabel, alpha=alpha, cmap=cmap, ampMin=ampMin,
                ampMax=ampMax, nlevAmp=nlevAmp, csfilter=csfilter,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                xtick_formatter=xtick_formatter, ytick_formatter=ytick_formatter,
                cbar_format=cbar_format)
            dstFile = '%s/eigvec/eigvecBackwardPolar_nev%d_ev%03d%s.%s' \
                      % (specDir, nev, ev + 1, postfixTau, ergoPlot.figFormat)
            fig.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches,
                        dpi=ergoPlot.dpi)
