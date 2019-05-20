import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
from ergoPack import ergoPlot

#ergoPlot.dpi = 2000

configFile = '../cfg/Chekroun2019.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
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
readSpec = ergoPlot.readSpectrum
xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
# xlimEig = [xmineigVal, -xmineigVal/100]
# ylimEig = [ymineigVal, -ymineigVal]
# yticksPos = np.arange(0, ylimEig[1], 5.)
# yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
# yticks = np.concatenate((yticksNeg, yticksPos))
xlimEig = None
ylimEig = None
yticks = None
xticks = None
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'


srcPostfixSim = (
    "_%s_mu%04d_alpha%04d_gamma%04d_delta%04d_beta%04d_eps%04d_sep%04d"
    "_L%d_spinup%d_dt%d_samp%d" % (
        caseName, int(cfg.model.mu * 10000 + 0.1),
        int(cfg.model.alpha * 10000 + 0.1), int(cfg.model.gamma * 10000 + 0.1),
        int(cfg.model.delta * 10000 + 0.1), int(cfg.model.beta * 10000 + 0.1),
        int(cfg.model.eps * 10000 + 0.1), int(cfg.model.sep * 10000 + 0.1),
        int(L + 0.1), int(spinup + 0.1),
        int(-np.round(np.log10(cfg.simulation.dt)) + 0.1), printStepNum))
postfix = "%s_nTraj%d%s" % (srcPostfixSim, cfg.sprinkle.nTraj,
                            gridPostfix)
os.makedirs(os.path.join(cfg.general.plotDir,
                         'spectrum', 'eigval'), exist_ok=True)

# tauRng = cfg.transfer.tauRng
tauRng = [cfg.stat.tauPlot]
for tau in tauRng:
    postfixTau = "%s_tau%04d" % (postfix, int(tau * 10000 + 0.1))

    # Define file names
    eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                        % (specDir, cfg.spectrum.nev, postfixTau,
                           cfg.general.fileFormat)

    print('Readig spectrum for tau = {:.4f}...'.format(tau))
    (eigValForward,) = readSpec(eigValForwardFile)

    # Get generator eigenvalues (using the complex logarithm)
    eigValGen = np.log(eigValForward) / tau
    print('Second eigenvalue: ', eigValGen[1])

    ergoPlot.plotEig(eigValGen, xlabel=realLabel, ylabel=imagLabel,
                     xlim=xlimEig, ylim=ylimEig)
    # plt.text(xlimEig[0]*0.2, ylimEig[1]*1.05, r'$\mu = {:.2f}$'.format(mu),
    #          fontsize=ergoPlot.fs_latex)
    plt.savefig('%s/spectrum/eigval/eigVal%s.%s'
                % (cfg.general.plotDir, postfixTau, ergoPlot.figFormat),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
