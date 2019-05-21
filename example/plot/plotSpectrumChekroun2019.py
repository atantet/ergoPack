import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
from ergoPack import ergoPlot


def main():
    configFile = '../cfg/Chekroun2019.cfg'
    cfg = pylibconfig2.Config()
    cfg.read_file(configFile)
    compName1 = 'x'
    compName2 = 'y'
    realLabel = r'$\Re(\lambda_k)$'
    imagLabel = r'$\Im(\lambda_k)$'
    corrName = 'C%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
    powerName = 'S%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
    fileFormat = cfg.general.fileFormat
    specDir = os.path.join(cfg.general.resDir, 'spectrum')

    L = cfg.simulation.LCut + cfg.simulation.spinup
    spinup = cfg.simulation.spinup
    printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
    caseName = cfg.model.caseName
    dim = cfg.model.dim
    dimObs = len(cfg.observable.components)

    N = np.prod(np.array(cfg.grid.nx))
    gridPostfix = ""
    N = 1
    for d in np.arange(dimObs):
        N *= cfg.grid.nx[d]
        if (hasattr(cfg.grid, 'gridLimitsLow')
                & hasattr(cfg.grid, 'gridLimitsHigh')):
            gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                            cfg.grid.gridLimitsLow[d],
                                            cfg.grid.gridLimitsHigh[d])
        else:
            gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                            cfg.sprinkle.minInitState[d],
                                            cfg.sprinkle.maxInitState[d])
    nTraj = cfg.sprinkle.nTraj

    # Read grid
    gridFile = '{}/grid/grid_{}{}.txt'.format(
        cfg.general.resDir, caseName, gridPostfix)
    coord = ergoPlot.readGrid(gridFile, dimObs)
    X, Y = np.meshgrid(coord[0], coord[1])
    coord = (X.flatten(), Y.flatten())

    ev_xlabel = r'$%s$' % compName1
    ev_ylabel = r'$%s$' % compName2
    corrLabel = r'$C_{%s, %s}(t)$' % (compName1[0], compName1[0])
    powerLabel = r'$S_{%s, %s}(\omega)$' % (compName1[0], compName1[0])
    xlabelCorr = r'$t$'

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
    # ylim_corr = [-2., 2.]
    # nev = 200
    # case = 'caseII'
    # ylim_corr = [-0.5, 0.5]
    # nev = 200
    case = 'caseIV'
    ylim_corr = [-0.6, 0.6]
    nev = 400

    p = param[case]
    tau = 1. / (p['gamma'] * 10)
    rateMax = 10.
    xmineigVal = -rateMax
    ymineigVal = -p['gamma'] * 10
    # plotBackward = False
    plotBackward = True
    plotImag = False
    # plotImag = True
    xlimEig = [xmineigVal, -xmineigVal/100]
    ylimEig = [ymineigVal, -ymineigVal]
    zlimEig = [cfg.stat.powerMin, cfg.stat.powerMax]
    xticks = None
    yticksPos = np.arange(0, ylimEig[1], p['gamma'] * 2)
    yticksNeg = np.arange(0, ylimEig[0], -p['gamma'] * 2)[::-1]
    yticks = np.concatenate((yticksNeg, yticksPos))
    zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                         int(np.round(np.log10(zlimEig[1]/zlimEig[0]) + 1)))
    zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                         int(np.round(np.log10(zlimEig[1]/zlimEig[0])/2 + 1)))

    postfix0 = ('_{}_mu{:04d}_alpha{:04d}_gamma{:04d}_delta{:04d}_beta{:04d}'
                '_eps{:04d}'.format(
                    caseName, int(p['mu'] * 10000 + 0.1),
                    int(p['alpha'] * 10000 + 0.1), int(p['gamma'] * 10000 + 0.1),
                    int(p['delta'] * 10000 + 0.1), int(p['beta'] * 10000 + 0.1),
                    int(p['eps'] * 10000 + 0.1)))
    postfix2 = '_L{:d}_spinup{:d}_dt{:d}_samp{:d}'.format(
        int(L + 0.1), int(spinup + 0.1),
        int(-np.round(np.log10(cfg.simulation.dt)) + 0.1), printStepNum)
    postfixBoth = "{}{}_nTraj{:d}{:s}_tau{:04d}".format(
        postfix0, postfix2, cfg.sprinkle.nTraj,
        gridPostfix, int(tau * 10000 + 0.1))

    statDist = {}
    eigValForward = {}
    eigValBackward = {}
    eigVecForward = {}
    eigVecBackward = {}
    eigenCondition = {}
    weights = {}
    eigValGen = {}
    corrRec = {}
    powerRec = {}
    for isep, sep in enumerate(p['sep']):
        print('{} for sep = {:.1e}'.format(case, sep))
        srcPostfix = "{}_sep{:04d}{}_nTraj{:d}".format(
            postfix0, int(sep * 10000 + 0.1), postfix2, cfg.sprinkle.nTraj)
        postfixTau = "{}_sep{:04d}{}_nTraj{:d}{}_tau{:04d}".format(
            postfix0, int(sep * 10000 + 0.1), postfix2, cfg.sprinkle.nTraj,
            gridPostfix, int(tau * 10000 + 0.1))

        # Define file names
        eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' % (
            specDir, nev, postfixTau, fileFormat)
        eigVecForwardFile = '%s/eigvec/eigvecForward_nev%d%s.%s' % (
            specDir, nev, postfixTau, fileFormat)
        eigValBackwardFile = '%s/eigval/eigvalBackward_nev%d%s.%s' % (
            specDir, nev, postfixTau, fileFormat)
        eigVecBackwardFile = '%s/eigvec/eigvecBackward_nev%d%s.%s' % (
            specDir, nev, postfixTau, fileFormat)
        statDistFile = '%s/transfer/initDist/initDist%s.%s' % (
            cfg.general.resDir, postfixTau, fileFormat)
        maskFile = '%s/transfer/mask/mask%s.%s' % (
            cfg.general.resDir, postfixTau, fileFormat)

        # Read stationary distribution
        if statDistFile is not None:
            statDist[isep] = (np.fromfile(statDistFile, float)
                              if fileFormat == 'bin' else
                              np.loadtxt(statDistFile, float))
        else:
            statDist[isep] = None

        # Read mask
        if maskFile is not None:
            if fileFormat == 'bin':
                mask = np.fromfile(maskFile, np.int32)
            else:
                mask = np.loadtxt(maskFile, np.int32)
        else:
            mask = np.arange(N)
        f = coord[cfg.stat.idxf][mask < N]
        g = coord[cfg.stat.idxg][mask < N]

        # Read transfer operator spectrum from file and create a bi-orthonormal basis
        # of eigenvectors and backward eigenvectors:
        print('Readig spectrum for tau = {:.3f}...'.format(tau))
        (eigValForward[isep], eigValBackward[isep], eigVecForward[isep],
         eigVecBackward[isep]) = (
            ergoPlot.readSpectrum(
                eigValForwardFile, eigValBackwardFile, eigVecForwardFile,
                eigVecBackwardFile, fileFormat=fileFormat,
                makeBiorthonormal=(not cfg.spectrum.makeBiorthonormal)))

        print('Getting conditionning of eigenvectors...')
        eigenCondition[isep] = ergoPlot.getEigenCondition(
            eigVecForward[isep], eigVecBackward[isep])

        # Get generator eigenvalues
        eigValGen[isep] = (np.log(np.abs(eigValForward[isep])) +
                           np.angle(eigValForward[isep])*1j) / tau

        print('Reading correlation function and periodogram...')
        if isep == 0:
            corrSample = np.loadtxt(os.path.join(
                cfg.general.resDir, 'correlation',
                'corrSample{}_lagMax{:d}yr.txt'.format(
                    srcPostfix, int(cfg.stat.lagMax * 1e4 + 0.1))))
            lags = np.loadtxt(os.path.join(
                cfg.general.resDir, 'correlation',
                'lags{}_lagMax{:d}yr.txt'.format(
                    srcPostfix, int(cfg.stat.lagMax * 1e4 + 0.1))))
            powerSample = np.loadtxt(os.path.join(
                cfg.general.resDir, 'power',
                'powerSample{}_chunk{:d}yr.txt'.format(
                    srcPostfix, int(cfg.stat.chunkWidth + 0.1))))
            freq = np.loadtxt(os.path.join(
                cfg.general.resDir, 'power', 'freq{}_chunk{:d}yr.txt'.format(
                    srcPostfix, int(cfg.stat.chunkWidth + 0.1))))

            # Normalize
            angFreq = freq * 2*np.pi
            if cfg.stat.norm:
                cfg0 = corrSample[lags.shape[0] // 2]
                corrSample /= cfg0
                powerSample /= 2*np.pi * cfg0
            else:
                powerSample /= 2*np.pi

        # Reconstruct correlation and power spectrum
        # Get normalized weights
        weights[isep] = ergoPlot.getSpectralWeights(
            f, g, eigVecForward[isep], eigVecBackward[isep])
        # Remove components with heigh condition number
        weights[isep][eigenCondition[isep] > cfg.stat.maxCondition] = 0.
        corrRec[isep], _ = ergoPlot.spectralRecCorrelation(
            lags, eigValGen[isep], weights[isep], norm=cfg.stat.norm)
        powerRec[isep], _ = ergoPlot.spectralRecPower(
            angFreq, eigValGen[isep], weights[isep], norm=cfg.stat.norm)

    # Plot correlation reconstruction
    ergoPlot.plotRecCorrelation(
        lags, corrSample, corrRec[0], corrAna=corrRec[1], plotPositive=True,
        ylabel=corrLabel, xlabel=xlabelCorr, ylim=ylim_corr)
    plt.savefig('%s/spectrum/reconstruction/%sRecTimeScaleSep_lag%d_nev%d%s.%s' % (
        cfg.general.plotDir, corrName, int(cfg.stat.lagMax * 1e4 + 0.1),
        nev, postfixBoth, ergoPlot.figFormat),
        dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

    # PLot spectrum, powerSampledogram and spectral reconstruction
    ergoPlot.plotEigPowerRec(
        angFreq, eigValGen[0], powerSample, powerRec[0], powerAna=powerRec[1],
        eigAna=eigValGen[1], xlabel=realLabel, ylabel=imagLabel, zlabel=powerLabel,
        xlim=xlimEig, ylim=ylimEig, zlim=zlimEig, xticks=xticks, yticks=yticks,
        zticks=zticks)
    plt.savefig('%s/spectrum/reconstruction/%sRecTimeScaleSep_chunk%d_nev%d%s.%s' % (
        cfg.general.plotDir, powerName, int(cfg.stat.chunkWidth + 0.1),
        nev, postfixBoth, ergoPlot.figFormat),
        dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


if __name__ == '__main__':
    main()
