import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
from ergoPack import ergoPlot

# ergoPlot.dpi = 2000

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
        gridPostfix = "{}_n{:d}l{:d}h{:d}".format(
            gridPostfix, cfg.grid.nx[d], int(cfg.grid.gridLimitsLow[d]),
            int(cfg.grid.gridLimitsHigh[d]))
    else:
        gridPostfix = "{}_n{:d}l{:d}h{:d}".format(
            gridPostfix, cfg.grid.nx[d], int(cfg.sprinkle.minInitState[d]),
            int(cfg.sprinkle.maxInitState[d]))
readSpec = ergoPlot.readSpectrum
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'
os.makedirs(os.path.join(cfg.general.plotDir,
                         'spectrum', 'eigval'), exist_ok=True)

param = {}
param['caseI'] = {
    'beta': 0., 'delta': 0., 'mu': 0.001, 'alpha': 0.056, 'gamma': 100.,
    'eps': 0.55, 'sep': [0.01, 0.0001]}
param['caseII'] = {
    'beta': 0., 'delta': 0., 'mu': 0.001, 'alpha': 1., 'gamma': 10.,
    'eps': 0.2, 'sep': [0.01, 0.0001]}
param['caseIV'] = {
    'beta': 0., 'delta': 0., 'mu': 0.001, 'alpha': 1., 'gamma': 10.,
    'eps': 0.3, 'sep': [10., 0.0001]}

# case = 'caseI'
case = 'caseII'
# case = 'caseIV'

p = param[case]
rateMax = 20.
xmineigVal = -rateMax
ymineigVal = -p['gamma'] * 10
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))
xticks = None

eigValGen = {}
postfix0 = ('_{}_mu{:04d}_alpha{:04d}_gamma{:04d}_delta{:04d}_beta{:04d}'
            '_eps{:04d}'.format(
                caseName, int(p['mu'] * 10000 + 0.1),
                int(p['alpha'] * 10000 + 0.1), int(p['gamma'] * 10000 + 0.1),
                int(p['delta'] * 10000 + 0.1), int(p['beta'] * 10000 + 0.1),
                int(p['eps'] * 10000 + 0.1)))
postfix2 = '_L{:d}_spinup{:d}_dt{:d}_samp{:d}'.format(
    int(L + 0.1), int(spinup + 0.1),
    int(-np.round(np.log10(cfg.simulation.dt)) + 0.1), printStepNum)

tau = cfg.stat.tauPlot
postfixBoth = "{}{}_nTraj{:d}{:s}_tau{:04d}".format(
    postfix0, postfix2, cfg.sprinkle.nTraj,
    gridPostfix, int(tau * 10000 + 0.1))
for isep, sep in enumerate(p['sep']):
    print('{} for sep = {:.1e}'.format(case, sep))
    postfixTau = "{}_sep{:04d}{}_nTraj{:d}{}_tau{:04d}".format(
        postfix0, int(sep * 10000 + 0.1), postfix2, cfg.sprinkle.nTraj,
        gridPostfix, int(tau * 10000 + 0.1))

    # Define file names
    eigValForwardFile = '{}/eigval/eigvalForward_nev{:d}{}.{}'.format(
        specDir, cfg.spectrum.nev, postfixTau, cfg.general.fileFormat)

    print('\tReadig spectrum for tau = {:.4f}...'.format(tau))
    (eigValForward,) = readSpec(eigValForwardFile)

    # Get generator eigenvalues (using the complex logarithm)
    eigValGen[isep] = np.log(eigValForward) / tau

    print('Second eigenvalue: ', eigValGen[isep][
        np.argsort(-eigValGen[isep].real)][1])

ergoPlot.plotEig(
    eigValGen[0], eigAna=eigValGen[1], xlabel=realLabel, ylabel=imagLabel,
    xlim=xlimEig, ylim=ylimEig)
# plt.text(xlimEig[0]*0.2, ylimEig[1]*1.05, r'$\mu = {:.2f}$'.format(mu),
#          fontsize=ergoPlot.fs_latex)
plt.savefig('{}spectrum/eigval/eigValTimeScaleSep{}.{}'.format(
    cfg.general.plotDir, postfixBoth, ergoPlot.figFormat),
    dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
