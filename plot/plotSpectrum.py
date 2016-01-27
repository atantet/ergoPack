import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import atmath, atplot


dim = 2
#nx = 50
#nx = 100
nx = 200
#nx = 300
nSTDLow = [5, 4]
nSTDHigh = [3, 4]

#tauDimRng = np.array([1.])
tauDimRng = np.array([3.])
#tauDimRng = np.array([6.])
#tauDimRng = np.array([12.])

nev = 50
alpha = 0.0
nevPlot = 0
#nevPlot = 6
#plotAdjoint = False
plotAdjoint = True
plotImag = False
#plotImag = True
#normEig = False
normEig = True
#plotCCF = False
plotCCF = True
xminEigval = -1.2
yminEigval = -11.5
lagMax = 100

resDir = '../results/'
specDir = '%s/spectrum/' % resDir
plotDir = '%s/plot/' % resDir
srcPostfix = "%s%s_mu%04d_eps%04d" % (prefix, simType,
                                      np.round(mu * 1000, 1),
                                      np.round(eps * 1000, 1))
obsName = ''
gridPostfix = ''
N = 1
for d in np.arange(dim):
    obsName += '_%s_%s' % (fieldsDef[d][2], indicesName[d][1])
    N *= nx
    gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, nx, nSTDLow[d], nSTDHigh[d])
cpyBuffer = gridPostfix
gridPostfix = '_%s%s%s' % (srcPostfix, obsName, cpyBuffer)

# Read grid
gridFile = '%s/grid/grid%s.txt' % (resDir, gridPostfix)
gfp = open(gridFile, 'r')
bounds = []
coord = []
for k in np.arange(dim):
    bounds.append(np.array(gfp.readline().split()).astype(float))
    coord.append((bounds[k][1:] + bounds[k][:-1]) / 2)
gfp.close()
X, Y = np.meshgrid(coord[0], coord[1])
gridXlim = [coord[0].min(), coord[0].max()]
gridYlim = [coord[1].min(), coord[1].max()]

lagMaxSample = int(np.round(lagMax * sampFreq))
lags = np.arange(0, lagMax + 0.999 / sampFreq, 1. / sampFreq)
omega = np.linspace(yminEigval, -yminEigval, 0)
f = X.flatten()
g = f
obsIdx0 = 0
obsIdx1 = 0
ccfPath = '../results/%s/' % srcPostfix
ccfPostfix = '_%s_%s_%s_%s_mu%04d_eps%04d' \
             % (fieldsDef[obsIdx0][2], indicesName[obsIdx0][1],
                fieldsDef[obsIdx1][2], indicesName[obsIdx1][1],
                np.round(mu * 1000, 1), np.round(eps * 1000, 1))


for lag in np.arange(tauDimRng.shape[0]):
    tauDim = tauDimRng[lag]
    tauConv = tauDim * timeScaleConversion
    maxImagRes = np.pi / tauConv
    postfix = "%s_tau%03d" % (gridPostfix, tauDim * 1000)

    print 'Readig spectrum...'
    EigValFile = '%s/eigval/eigval_nev%d%s.txt' % (specDir, nev, postfix)
    EigVecFile = '%s/eigvec/eigvec_nev%d%s.txt' % (specDir, nev, postfix)
    EigValAdjointFile = '%s/eigval/eigvalAdjoint_nev%d%s.txt' \
                        % (specDir, nev, postfix)
    EigVecAdjointFile = '%s/eigvec/eigvecAdjoint_nev%d%s.txt' \
                        % (specDir, nev, postfix)
    statDist = np.loadtxt('%s/transitionMatrix/initDist%s.txt' % (resDir, postfix))
#     statDist[statDist < alpha] = 0.
    eigvalRaw = np.loadtxt(EigValFile)
    eigvalAdjointRaw = np.loadtxt(EigValAdjointFile)
    eigvalRaw = eigvalRaw[:, 0] + eigvalRaw[:, 1]*1j
    eigvalAdjointRaw = eigvalAdjointRaw[:, 0] + eigvalAdjointRaw[:, 1]*1j  # P = E^{-1} \Lambda E
    eigvecRaw = np.loadtxt(EigVecFile)
    eigvecAdjointRaw = np.loadtxt(EigVecAdjointFile)
    eigvecRaw = eigvecRaw[::2] + eigvecRaw[1::2]*1j                        
    eigvecAdjointRaw = eigvecAdjointRaw[::2] + eigvecAdjointRaw[1::2]*1j   #Q = F^{-1} \Lambda^* F => E^{-1} = D(\pi) F^*
    eigvec = np.zeros((nev, N), dtype=complex)
    eigvecAdjoint = np.zeros((nev, N), dtype=complex)
    eigval = np.zeros((nev,), dtype=complex)
    eigvalAdjoint = np.zeros((nev,), dtype=complex)
    nevSingle = np.min([eigvalRaw.shape[0], eigvalAdjointRaw.shape[0]])
# #     Filter
#     for count in np.arange(nevSingle):
#         ev = eigvecRaw[count]
#         evcut = np.sort(np.abs(ev))[(1. - 2*alpha) * N]
#         eigvecRaw[count][np.abs(eigvecRaw[count]) > evcut] = 0.
#         ev = eigvecAdjointRaw[count]
#         evcut = np.sort(np.abs(ev))[(1. - 2*alpha) * N]
#         eigvecAdjointRaw[count][np.abs(eigvecAdjointRaw[count]) > evcut] = 0.
#    threshold = np.sort(statDist[statDist > 0])[int(alpha*np.sum(statDist > 0))]
#    statDist[statDist <= threshold] = 0.
    
    ev = 0
    for count in np.arange(nevSingle):
        eigval[ev] = eigvalRaw[count]
        eigvec[ev] = eigvecRaw[count]
        ev += 1
        if (np.abs(eigval[ev-1].imag) > 1.e-5) & (ev < nev):
            eigval[ev] = eigvalRaw[count].real - eigvalRaw[count].imag*1j
            eigvec[ev] = eigvecRaw[count].real - eigvecRaw[count].imag*1j
            ev += 1

    ev = 0
    for count in np.arange(nevSingle):
        eigvalAdjoint[ev] = eigvalAdjointRaw[count]
        eigvecAdjoint[ev] = eigvecAdjointRaw[count]
        ev += 1
        if (np.abs(eigvalAdjoint[ev-1].imag) > 1.e-5) & (ev < nev):
            eigvalAdjoint[ev] = eigvalAdjointRaw[count].real \
                                - eigvalAdjointRaw[count].imag*1j
            eigvecAdjoint[ev] = eigvecAdjointRaw[count].real \
                                - eigvecAdjointRaw[count].imag*1j
            ev += 1
    isort = np.argsort(np.abs(eigval))[::-1]
    eigval = eigval[isort]
    eigvec = eigvec[isort]
    eigvec[0] /= eigvec[0].sum()
    isort = np.argsort(np.abs(eigvalAdjoint))[::-1]
    eigvalAdjoint = eigvalAdjoint[isort]
    eigvecAdjoint = eigvecAdjoint[isort]
    for ev in np.arange(nev):
        if np.abs(eigvalAdjoint[ev].imag - eigval[ev].imag) \
           < np.abs(eigvalAdjoint[ev].imag + eigval[ev].imag):
            eigvalAdjoint[ev] = np.conjugate(eigvalAdjoint[ev])
            eigvecAdjoint[ev] = np.conjugate(eigvecAdjoint[ev])
    for ev in np.arange(nev):
        norm = np.sum(np.conjugate(eigvecAdjoint[ev]) * statDist \
                      * eigvec[ev])
        eigvecAdjoint[ev] /= np.conjugate(norm)

    # Get generator eigenvalues
    eigvalGen = (np.log(np.abs(eigval)) + np.angle(eigval)*1j) / tauConv

    # Plot spectrum
    # print 'Plotting spectrum slowest rate ', -1. / eigvalGen[1].real
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(eigvalGen[1:].real, eigvalGen[1:].imag,
    #            c='k', s=atplot.msize, marker='o')
    # ax.scatter(eigvalGen[0].real, eigvalGen[0].imag,
    #            c='r', s=atplot.msize, marker='o')
    # ax.set_xlabel(r'$\Re(\hat{\lambda}^\mathcal{R}_k)$',
    #               fontsize=atplot.fs_latex)
    # ax.set_ylabel(r'$\Im(\hat{\lambda}^\mathcal{R}_k)$',
    #               fontsize=atplot.fs_latex)
    # plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
    # plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
    # #ax.set_title('%d-time-step spectrum for %s\nSlowest time-scale: %.1f' \
    #     #    % (tau, srcPostfix, -1. / rate[0]))
    # ax.set_xlim(xminEigval, -xminEigval / 100)
    # ax.set_ylim(yminEigval, -yminEigval)
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # plt.plot([xlim[0], xlim[1]], [maxImagRes, maxImagRes], '--k')
    # plt.plot([xlim[0], xlim[1]], [-maxImagRes, -maxImagRes], '--k')
    # plt.text(xlim[1] - (xlim[1] - xlim[0])*0.21,
    #          ylim[0] + (ylim[1] - ylim[0])*0.04,
    #          r'$\mu = %.2f$' % (mu,),
    #          fontsize=atplot.fs_latex)
    # fig.savefig('%s/spectrum/eigval/eigval_nev%d%s.%s' \
    #             % (plotDir, nev, postfix, atplot.figFormat),
    #             bbox_inches='tight', dpi=atplot.dpi)
        
    
    # Plot eigenvectors of transfer operator
    tol = 0.
    alpha = 0.01
    for k in np.arange(nevPlot):
        print 'Plotting real part of eigenvector %d...' % (k+1,)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        v2Real = eigvec[k].real.copy()
        if normEig:
            v2Real *= statDist
        vecAlpha = v2Real[v2Real != 0]
        vmax = np.sort(np.abs(vecAlpha))[int((1. - 2*alpha) \
                                             * vecAlpha.shape[0])]
        v2Real[v2Real > vmax] = vmax
        v2Real[v2Real < -vmax] = -vmax
        h = ax.contourf(X, Y, v2Real.reshape(nx, nx), atplot.levels,
                        cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax)
        ax.set_xlim(gridXlim)
        ax.set_ylim(gridYlim)
        cbar = plt.colorbar(h)
        d = 0
        ax.set_xlabel(r'%s %s (%s)' \
                      % (indicesName[d][0], fieldsDef[d][1], fieldsDef[d][3]),
                      fontsize=atplot.fs_latex)
        d = 1
        ax.set_ylabel(r'%s %s (%s)' \
                      % (indicesName[d][0], fieldsDef[d][1], fieldsDef[d][3]),
                      fontsize=atplot.fs_latex)
        # ax.set_title("Real part of the eigenvector %d" % (k+1,),
        #              fontsize=atplot.fs_default)
        plt.setp(cbar.ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
        plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
        plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
        fig.savefig('%s/spectrum/eigvec/eigvecReal_nev%d_ev%03d%s.%s' \
                    % (plotDir, nev, k+1, postfix, atplot.figFormat),
                    bbox_inches='tight', dpi=atplot.dpi)

        if plotImag & (eigval[k].imag != 0):
            print 'Plotting imaginary  part of eigenvector %d...' % (k+1,)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            v2Imag = eigvec[k].imag.copy()
            if normEig:
                v2Imag *= statDist
            vecAlpha = v2Imag[v2Imag != 0]
            vmax = np.sort(np.abs(vecAlpha))[int((1. - 2*alpha) \
                                                 * vecAlpha.shape[0])]
            v2Imag[v2Imag > vmax] = vmax
            v2Imag[v2Imag < -vmax] = -vmax
            h = ax.contourf(X, Y, v2Imag.reshape(nx, nx), atplot.levels,
                            cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax)
            ax.set_xlim(gridXlim)
            ax.set_ylim(gridYlim)
            cbar = plt.colorbar(h)
            d = 0
            ax.set_xlabel(r'%s %s (%s)' % (indicesName[d][0],
                                           fieldsDef[d][1], fieldsDef[d][3]),
                          fontsize=atplot.fs_latex)
            d = 1
            ax.set_ylabel(r'%s %s (%s)' % (indicesName[d][0], fieldsDef[d][1],
                                           fieldsDef[d][3]),
                          fontsize=atplot.fs_latex)
            # ax.set_title("Imaginary part of the eigenvector %d" % (k+1,),
            #              fontsize=atplot.fs_default)
            plt.setp(cbar.ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
            plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
            plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
            fig.savefig('%s/spectrum/eigvec/eigvecImag_nev%d_ev%03d%s.%s' \
                        % (plotDir, nev, k, postfix, atplot.figFormat),
                        bbox_inches='tight', dpi=atplot.dpi)

 
        # Plot eigenvectors of Koopman operator
        if plotAdjoint:
            print 'Plotting real part of Koopman eigenvector %d...' \
                % (k+1,)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            v2Real = eigvecAdjoint[k].real.copy()
            if normEig:
                v2Real *= statDist
            vecAlpha = v2Real[v2Real != 0]
            vmax = np.sort(np.abs(vecAlpha))[int((1. - 2*alpha) \
                                                 * vecAlpha.shape[0])]
            v2Real[v2Real > vmax] = vmax
            v2Real[v2Real < -vmax] = -vmax
            h = ax.contourf(X, Y, v2Real.reshape(nx, nx), atplot.levels,
                            cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax)
            ax.set_xlim(gridXlim)
            ax.set_ylim(gridYlim)
            cbar = plt.colorbar(h)
            d = 0
            ax.set_xlabel(r'%s %s (%s)' % (indicesName[d][0], fieldsDef[d][1],
                                           fieldsDef[d][3]),
                          fontsize=atplot.fs_latex)
            d = 1
            ax.set_ylabel(r'%s %s (%s)' % (indicesName[d][0], fieldsDef[d][1],
                                           fieldsDef[d][3]),
                          fontsize=atplot.fs_latex)
            plt.setp(cbar.ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
            plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
            plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
            # ax.set_title("Real part of the Koopman eigenvector %d" \
            #              % (k+1,),
            #              fontsize=atplot.fs_default)
            fig.savefig('%s/spectrum/eigvec/eigvecAdjointReal_nev%d_ev%03d%s.%s' \
                        % (plotDir, nev, k+1, postfix, atplot.figFormat),
                        bbox_inches='tight', dpi=atplot.dpi)

            if plotImag & (eigval[k].imag != 0):
                print 'Plotting imaginary  part of Koopman eigenvector %d...' \
                    % (k+1,)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                v2Imag = eigvecAdjoint[k].imag.copy()
                if normEig:
                    v2Imag *= statDist
                vecAlpha = v2Imag[v2Imag != 0]
                vmax = np.sort(np.abs(vecAlpha))[int((1. - 2*alpha) \
                                                     * vecAlpha.shape[0])]
                v2Imag[v2Imag > vmax] = vmax
                v2Imag[v2Imag < -vmax] = -vmax
                h = ax.contourf(X, Y, v2Imag.reshape(nx, nx), atplot.levels,
                                cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax)
                ax.set_xlim(gridXlim)
                ax.set_ylim(gridYlim)
                cbar = plt.colorbar(h)
                d = 0
                ax.set_xlabel(r'%s %s (%s)' \
                              % (indicesName[d][0], fieldsDef[d][1],
                                 fieldsDef[d][3]), fontsize=atplot.fs_latex)
                d = 1
                ax.set_ylabel(r'%s %s (%s)' \
                              % (indicesName[d][0], fieldsDef[d][1],
                                 fieldsDef[d][3]), fontsize=atplot.fs_latex)
                plt.setp(cbar.ax.get_yticklabels(),
                         fontsize=atplot.fs_yticklabels)
                plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
                plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
                fig.savefig('%s/spectrum/eigvec/eigvecAdjointImag_nev%d_ev%03d%s.%s' \
                            % (plotDir, nev, k, postfix, atplot.figFormat),
                            bbox_inches='tight', dpi=atplot.dpi)

if plotCCF:
    # Get ccf
    #    Get sample cross-correlation
    print 'Reading ccf and perio'
    ccf = np.loadtxt('%s/ccf%s_lagMax%dyr.txt' % (ccfPath, ccfPostfix, lagMax))
    perio = np.loadtxt('%s/perio%s.txt' % (ccfPath, ccfPostfix))
    perioSTD = np.loadtxt('%s/perioSTD%s.txt' % (ccfPath, ccfPostfix))
    freq = np.loadtxt('%s/freq%s.txt' % (ccfPath, ccfPostfix))
    angFreq = freq * 2*np.pi
    perioDown = perio - perioSTD / 2
    perioUp = perio + perioSTD / 2


    # Get reconstructed cross correlation 
    nComponents = comp.shape[0]+1

    componentsCCF = np.zeros((nComponents-1, lags.shape[0]), dtype=complex)
    componentsPower = np.zeros((nComponents-1, angFreq.shape[0]),
                               dtype=complex)
    weights = np.zeros((nComponents-1,), dtype=complex)
    for k in np.arange(comp.shape[0]):
        ev = comp[k]
        weights[k] = (((f * statDist * np.conjugate(eigvecAdjoint[ev])).sum() \
                     * (eigvec[ev] * statDist * np.conjugate(g)).sum())).real
        if weights[k].real < 0:
            weights[k] *= -1
        componentsCCF[k] = np.exp(eigvalGen[ev] * lags) * weights[k]
        componentsPower[k] = weights[k] * (-eigvalGen[ev].real * 2) \
                             / ((angFreq + eigvalGen[ev].imag)**2 + eigvalGen[ev].real**2)
    ccfRec = componentsCCF.sum(0).real
    ccfRec /= ccfRec[0]
    powerRec = componentsPower.sum(0).real
    powerRec /= powerRec.sum()
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lags, ccf, linewidth=2)
    ax.plot(lags, ccfRec, '--', linewidth=2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r'$t$ $\mathrm{(years)}$', fontsize=atplot.fs_latex)
    ax.set_ylabel(r'$\tilde C_{x, x}(t)$', fontsize=atplot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
    fig.savefig('%s/spectrum/ccf/ccf_xx_nev%d%s.%s' \
                % (plotDir, nev, postfix, atplot.figFormat),
                bbox_inches='tight', dpi=atplot.dpi)

    # PLot 3d
    zmin = -8
    zmax = 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    iEigVal = (eigvalGen.real >= xminEigval) \
              & (eigvalGen.real <= -xminEigval / 100) \
              & (eigvalGen.imag >= yminEigval) \
              & (eigvalGen.imag <= -yminEigval)
    ax.scatter(eigvalGen[iEigVal][1:].real,
               eigvalGen[iEigVal][1:].imag,
               np.ones((eigvalGen[iEigVal][1:].shape[0],)) * zmin,
               c='b', s=atplot.msize, marker='o', depthshade=False)
    ax.set_xlim3d(xminEigval, -xminEigval / 100)
    ax.set_ylim3d(yminEigval, -yminEigval)
    ax.set_zlim3d(zmin, zmax)
    ax.scatter(eigvalGen[0].real, eigvalGen[0].imag, zmin,
               c='r', s=atplot.msize, marker='o')
    iangFreq = (angFreq >= yminEigval) & (angFreq <= -yminEigval)
    perioPlot = np.ones((perio.shape[0],)) * zmin
    perioDownPlot = np.ones((perio.shape[0],)) * zmin
    perioUpPlot = np.ones((perio.shape[0],)) * zmin
    powerRecPlot = np.ones((powerRec.shape[0],)) * zmin
    perioPlot[perio > 0] = np.log10(perio[perio > 0])
    powerRecPlot[powerRec > 0] = np.log10(powerRec[powerRec > 0])
    perioDownPlot[perioDown > 0] = np.log10(perioDown[perioDown > 0])
    perioUpPlot[perioUp > 0] = np.log10(perioUp[perioUp > 0])
    #perioPlot = perio
    perioPlot[perioPlot < zmin] = zmin
    perioPlot[perioPlot > zmax] = zmax
    perioDownPlot[perioDownPlot < zmin] = zmin
    perioDownPlot[perioDownPlot > zmax] = zmax
    perioUpPlot[perioUpPlot < zmin] = zmin
    perioUpPlot[perioUpPlot > zmax] = zmax
    powerRecPlot[powerRecPlot < zmin] = zmin
    powerRecPlot[powerRecPlot > zmax] = zmax
    ax.plot(np.zeros((iangFreq.sum(),)), angFreq[iangFreq],
            perioPlot[iangFreq], 'k-')
    xs = np.concatenate((angFreq[iangFreq], angFreq[iangFreq][::-1]), 0)
    ys = np.concatenate((perioUpPlot[iangFreq], perioDownPlot[iangFreq]), 0)
    verts = list(zip(xs, ys))
    poly = PolyCollection([verts], facecolors=[(0,0,0)])
    poly.set_alpha(0.2)
    ax.add_collection3d(poly, zs=0, zdir='x')
    ax.plot(np.zeros((iangFreq.sum(),)), angFreq[iangFreq],
            powerRecPlot[iangFreq], '-r', linewidth=2)


    eigvalPlot = eigvalGen[iEigVal]
    iFirst = np.abs(eigvalPlot.imag / eigvalPlot.real) \
             > np.abs(eigvalPlot[1].imag / eigvalPlot[1].real) / 10
    nPlot = 2*7
    if iEigVal.sum() < nPlot:
        nPlot = iEigVal.sum()
    eigvalPlottmp = np.empty((nPlot,), dtype=complex)
    eigvalPlottmp[:np.min([iFirst.sum(), nPlot])] = eigvalPlot[iFirst][:nPlot]
    if np.sum(iFirst) < nPlot:
        eigvalPlottmp[iFirst.sum():] = eigvalPlot[:nPlot-iFirst.sum()]
    eigvalPlot = eigvalPlottmp
    for k in np.arange(eigvalPlot.shape[0]):
        ax.plot([eigvalPlot[k].real, 0.],
                [eigvalPlot[k].imag, eigvalPlot[k].imag],
                [zmin, zmin],
                '--b')
        ax.plot([0., 0.],
                [eigvalPlot[k].imag, eigvalPlot[k].imag],
                [zmin,
                 perioPlot[np.argmin(np.abs(angFreq - eigvalPlot[k].imag))]],
                '--b')
    if np.sum(~iFirst) > 1:
        ax.plot([eigvalGen[iEigVal][~iFirst][1].real, 0.],
                [eigvalGen[iEigVal][~iFirst][1].imag,
                 eigvalGen[iEigVal][~iFirst][1].imag],
                [zmin, zmin], '--g')
        ax.plot([0, 0.], [eigvalGen[iEigVal][~iFirst][1].imag,
                 eigvalGen[iEigVal][~iFirst][1].imag],
                [zmin, perioPlot[np.argmin(np.abs(angFreq - eigvalGen[iEigVal][~iFirst][1].imag))]], '--g')
    ax.set_xlabel('\n' + r'$\Re(\bar{\lambda}_k)$', fontsize=atplot.fs_default,
                  linespacing=1.5)
    ax.set_ylabel('\n' + r'$\Im(\bar{\lambda}_k)$', fontsize=atplot.fs_default,
                  linespacing=1)
    ax.set_zlabel(r'$\hat{S}_{x, x}(\omega)$', fontsize=atplot.fs_default)
    ax.set_xticks(np.arange(0., xminEigval, -0.2))[::-1]
    zticks = np.arange(zmin, zmax, 2)
    ax.set_zticks(zticks)
    zticklabels = []
    for k in np.arange(zticks.shape[0]):
        zticklabels.append(r'$10^{%d}$' % zticks[k])
    ax.set_zticklabels(zticklabels)
    ax.view_init(30, -150)
    #plt.setp(ax.get_xticklabels(), fontsize=atplot.fs_xticklabels)
    #plt.setp(ax.get_yticklabels(), fontsize=atplot.fs_yticklabels)
    #ax.set_title('%d-time-step spectrum for %s\nSlowest time-scale: %.1f' \
        #    % (tau, srcPostfix, -1. / rate[0]))
    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.))
    #ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.))
    #ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.))
    #ax.grid(False)
    fig.savefig('%s/spectrum/ccf/eigPerio%s.%s' \
                % (plotDir, postfix, atplot.figFormat),
                bbox_inches='tight', dpi=atplot.dpi)
