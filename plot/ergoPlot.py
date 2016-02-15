import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import pylibconfig2

# Default parameters
levels = 20
fs_default = 'x-large'
fs_latex = 'xx-large'
fs_xlabel = fs_default
fs_ylabel = fs_default
fs_xticklabels = fs_default
fs_yticklabels = fs_default
fs_legend_title = fs_default
fs_legend_labels = fs_default
fs_cbar_label = fs_default
figFormat = 'eps'
#figFormat = 'png'
dpi = 300
msize = 32
bbox_inches = 'tight'

#############
# Definitions
#############

def readSpectrum(EigValForwardFile, EigVecForwardFile,
                 EigValBackwardFile, EigVecBackwardFile,
                 statDist, makeBiorthonormal=False):
    """Read transfer operator spectrum from file and create a bi-orthonormal basis \
    of eigenvectors and adjoint eigenvectors"""

    # Read eigenvalues
    eigValForward = np.loadtxt(EigValForwardFile)
    eigValBackward = np.loadtxt(EigValBackwardFile)
    nev = eigValForward.shape[0]

    # Make complex
    eigValForward = eigValForward[:, 0] + eigValForward[:, 1]*1j
    eigValBackward = eigValBackward[:, 0] + eigValBackward[:, 1]*1j  # P = E^{-1} \Lambda E
    
    # Read eigenvectors
    eigVecForward = np.loadtxt(EigVecForwardFile)
    eigVecBackward = np.loadtxt(EigVecBackwardFile)
    N = eigVecForward.shape[0] / nev

    # Make complex
    eigVecForward = (eigVecForward[:, 0] + eigVecForward[:, 1]*1j).reshape(N, nev)
    #Q = F^{-1} \Lambda^* F => E^{-1} = D(\pi) F^*   
    eigVecBackward = (eigVecBackward[:, 0] + eigVecBackward[:, 1]*1j).reshape(N, nev)

    if makeBiorthonormal:
        # Sort by largest magnitude
        isort = np.argsort(np.abs(eigValForward))[::-1]
        eigValForward = eigValForward[isort]
        eigVecForward = eigVecForward[:, isort]
        eigVecForward[:, 0] = np.abs(eigVecForward[:, 0]) / np.max(np.abs(eigVecForward[:, 0]))

        # Because different eigenvalues may have the same magnitude
        # sort the adjoint eigenvalues by correspondance to the eigenvalues.
        isort = np.empty((nev,), dtype=int)
        for ev in np.arange(nev):
            isort[ev] = np.argmin(np.abs(eigValForward[ev] - np.conjugate(eigValBackward)))
        eigValBackward = eigValBackward[isort]
        eigVecBackward = eigVecBackward[:, isort]

        # Normalize adjoint eigenvectors to have a bi-orthonormal basis
        for ev in np.arange(nev):
            norm = np.sum(np.conjugate(eigVecBackward[:, ev]) * statDist \
                          * eigVecForward[:, ev])
            eigVecBackward[:, ev] /= np.conjugate(norm)

    return (eigValForward, eigVecForward, eigValBackward, eigVecBackward)


def readSpectrumCompressed(EigValForwardFile, EigVecForwardFile,
                           EigValBackwardFile, EigVecBackwardFile,
                           statDist, makeBiorthonormal=False):
    """Read transfer operator spectrum from file and create a bi-orthonormal basis \
    of eigenvectors and adjoint eigenvectors"""

    skiprows = 2
    
    # Read eigenvalues
    eigValForwardCmp = np.loadtxt(EigValForwardFile)
    eigValBackwardCmp = np.loadtxt(EigValBackwardFile)
    nev = eigValForwardCmp.shape[0]

    # Make complex
    eigValForward = eigValForwardCmp[:, 0] \
                    + 1j*eigValForwardCmp[:, 1]
    eigValBackward = eigValBackwardCmp[:, 0] \
                     + 1j*eigValBackwardCmp[:, 1]  # P = E^{-1} \Lambda E
    indexForward = eigValForwardCmp[:, 2]
    indexBackward = eigValBackwardCmp[:, 2]
    
    # Read eigenvectors
    eigVecForwardCmp = np.loadtxt(EigVecForwardFile,
                                  skiprows=skiprows).reshape(nev, -1).T
    eigVecBackwardCmp = np.loadtxt(EigVecBackwardFile,
                                   skiprows=skiprows).reshape(nev, -1).T
    N = eigVecForwardCmp.shape[0]

    # Get full forward vectors
    eigVecForward = np.empty((N, nev), dtype=complex)
    for ev in np.arange(nev):
        if indexForward[ev] == 0:
            eigVecForward[:, ev] = eigVecForwardCmp[:, ev]
        elif indexForward[ev] == 1:
            eigVecForward[:, ev] = eigVecForwardCmp[:, ev] \
                                   + 1j*eigVecForwardCmp[:, ev+1]
        elif indexForward[ev] == -1:
            eigVecForward[:, ev] = eigVecForwardCmp[:, ev - 1] \
                                   - 1j*eigVecForwardCmp[:, ev]

    # Get full forward vectors
    eigVecBackward = np.empty((N, nev), dtype=complex)
    ev = 0
    for ev in np.arange(nev):
        if indexBackward[ev] == 0:
            eigVecBackward[:, ev] = eigVecBackwardCmp[:, ev]
        elif indexBackward[ev] == 1:
            eigVecBackward[:, ev] = eigVecBackwardCmp[:, ev] \
                                   + 1j*eigVecBackwardCmp[:, ev+1]
        elif indexBackward[ev] == -1:
            eigVecBackward[:, ev] = eigVecBackwardCmp[:, ev - 1] \
                                   - 1j*eigVecBackwardCmp[:, ev]
            
    if makeBiorthonormal:
        # Sort by largest magnitude
        isort = np.argsort(np.abs(eigValForward))[::-1]
        eigValForward = eigValForward[isort]
        eigVecForward = eigVecForward[:, isort]

        # Because different eigenvalues may have the same magnitude
        # sort the adjoint eigenvalues by correspondance to the eigenvalues.
        isort = np.empty((nev,), dtype=int)
        for ev in np.arange(nev):
            isort[ev] = np.argmin(np.abs(eigValForward[ev] - np.conjugate(eigValBackward)))
        eigValBackward = eigValBackward[isort]
        eigVecBackward = eigVecBackward[:, isort]

        # Normalize adjoint eigenvectors to have a bi-orthonormal basis
        for ev in np.arange(nev):
            norm = np.sum(np.conjugate(eigVecBackward[:, ev]) * statDist \
                          * eigVecForward[:, ev])
            eigVecBackward[:, ev] /= np.conjugate(norm)

    return (eigValForward, eigVecForward, eigValBackward, eigVecBackward)


def getSpectralWeights(f, g, eigVecForward, eigVecBackward,
                       statDist, skipMean=False):
    """Calculate the spectral weights as the product of \
the scalar products of the observables on eigenvectors \
and adjoint eigenvectors w.r.tw the stationary distribution."""
    ne = eigVecForward.shape[1]
    if skipMean:
        fa = f.copy() - (f * statDist).sum()
        ga = g.copy() - (g * statDist).sum()
    else:
        fa = f
        ga = f
    weights = np.zeros((ne,), dtype=complex)
    for k in np.arange(ne):
        weights[k] = (((fa * statDist * np.conjugate(eigVecBackward[:, k])).sum() \
                     * (eigVecForward[:, k] * statDist * np.conjugate(ga)).sum()))

    return weights

def spectralRecCorrelation(lags, f, g, eigValGen, weights, statDist, skipMean=False, norm=False):
    """Calculate the reconstruction of the correlation function \
from the spectrum of the generator"""
    ne = eigValGen.shape[0]
    components = np.zeros((ne, lags.shape[0]), dtype=complex)
    for k in np.arange(ne):
        components[k] = np.exp(eigValGen[k] * lags) * weights[k]
    reconstruction = components.sum(0).real
    
    # Remove mean
    mean_f = (f * statDist).sum()
    mean_g = (statDist * np.conjugate(g)).sum()
    if not skipMean:
        reconstruction -= mean_f * mean_g / (f * statDist * np.conjugate(g)).sum()

    if norm:
        reconstruction /= ((f - mean_f) * statDist * np.conjugate(g - mean_g)).sum()


    return (reconstruction, components)

def spectralRecPower(angFreq, f, g, eigValGen, weights, statDist, norm=False):
    """Calculate the reconstruction of the power spectrum \
from the spectrum of the generator"""
    ne = eigValGen.shape[0]
    components = np.zeros((ne, angFreq.shape[0]), dtype=complex)
    for k in np.arange(1, ne): 
        if np.abs(eigValGen[k].real) > 1.e-6:
            components[k] = (-eigValGen[k].real) \
                            / ((angFreq + eigValGen[k].imag)**2 + eigValGen[k].real**2) \
                            * weights[k] / np.pi
        else:
            components[k] = 0.
    reconstruction = components.sum(0).real

    if norm:
        mean_f = (f * statDist).sum()
        mean_g = (statDist * np.conjugate(g)).sum()
        reconstruction /= ((f - mean_f) * statDist * np.conjugate(g - mean_g)).sum()
        
    return (reconstruction, components)
    
def plot2D(X, Y, vectOrig, xlabel=r'$x$', ylabel=r'y', alpha=0.):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vect = vectOrig.copy()
    vecAlpha = vect[vect != 0]
    if alpha > 0:
        vmax = np.sort(np.abs(vecAlpha))[int((1. - 2*alpha) \
                                             * vecAlpha.shape[0])]
        vect[vect > vmax] = vmax
        vect[vect < -vmax] = -vmax
    else:
        vmax = np.max(np.abs(vect))
        
    h = ax.contourf(X, Y, vect.reshape(X.shape), levels,
                    cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax)
    ax.set_xlim(X[0].min(), X[0].max())
    ax.set_ylim(Y[:, 0].min(), Y[:, 0].max())
    cbar = plt.colorbar(h)
    ax.set_xlabel(xlabel, fontsize=fs_latex)
    ax.set_ylabel(ylabel, fontsize=fs_latex)
    plt.setp(cbar.ax.get_yticklabels(), fontsize=fs_yticklabels)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)


def readGrid(gridFile, dimObs):
    """Read transfer operator grid."""
    gfp = open(gridFile, 'r')
    bounds = []
    coord = []
    for k in np.arange(dimObs):
        bounds.append(np.array(gfp.readline().split()).astype(float))
        coord.append((bounds[k][1:] + bounds[k][:-1]) / 2)
    gfp.close()
    X, Y = np.meshgrid(coord[0], coord[1])

    return (X, Y)


def plotCCF(ccf, lags=None, ls='-', lc='b', lw=2, xlim=None, ylim=None,
            xlabel=None, ylabel=None, absUnit='', plotPositive=False):
    '''Default plot for a correlation function.'''
    if lags is None:
        lags = np.arange(ccf.shape[0])
    if plotPositive:
        nLags = lags.shape[0]
        lags = lags[(nLags - 1) / 2:]
        ccf = ccf[(nLags - 1) / 2:]
    if xlim is None:
        xlim = (lags[0], lags[-1])
    if ylim is None:
        ylim = (-1.05, 1.05)
    if absUnit != '':
        absUnit = '(%s)' % absUnit
    if xlabel is None:
        xlabel = r'$t$ %s' % absUnit
    if ylabel is None:
        ylabel = r'$C_{x, x}(t)$'
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, ccf, linestyle=ls, color=lc, linewidth=lw)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=fs_latex)
    ax.set_ylabel(ylabel, fontsize=fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)

    return (fig, ax)


def plotPerio(perio, freq=None, perioSTD=None, xlim=None, ylim=None,
              xlabel=None, ylabel=None, xscale='linear', yscale='linear',
              absUnit='', absType='ang', ls='-', lc='k', lw=2,
              fc=None, ec=None, alpha=0.2, plotPositive=False):
    '''Default plot for a periodogram.'''
    nfft = perio.shape[0]
    if freq is None:
        freq = np.arange(-nfft, nfft+1)
    if plotPositive:
        freq = freq[nfft / 2 + 1:]
        perio = perio[nfft / 2 + 1:]
        perioSTD = perioSTD[nfft / 2 + 1:]
    if xlim is None:
        xlim = (freq[0], freq[-1])
    if ylim is None:
        ylim = (perio.min(), perio.max())
    if absType == 'ang':
        absName = '\omega'
        absUnit = 'rad %s' % absUnit
    elif absType == 'freq':
        absName = 'f'
        absUnit = '%s' % absUnit
    if ylabel is None:
        ylabel = r'$\hat{S}_{x,x}(%s)$' % absName
    if absUnit != '':
        absUnit = '(%s)' % absUnit
    if fc is None:
        fc = lc
    if ec is None:
        ec = fc
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if perioSTD is not None:
        perioDown = perio - perioSTD / 2
        perioUp = perio + perioSTD / 2
        ax.fill_between(freq, perioDown, perioUp,
                        facecolor=fc, alpha=alpha, edgecolor=fc)
    ax.plot(freq, perio, linestyle=ls, color=lc, linewidth=lw)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$%s$ %s' % (absName, absUnit),
                  fontsize=fs_latex)
    ax.set_ylabel(ylabel, fontsize=fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)

    return (fig, ax)

def plotRecCorrelation(lags, corrSample, corrRec,
                       lw=2, xlim=None, ylim=None,
                       xlabel=None, ylabel=None, absUnit='', plotPositive=False):
    if plotPositive:
        nLags = lags.shape[0]
        lags = lags[(nLags - 1) / 2:]
        corrSample = corrSample[(nLags - 1) / 2:]
        corrRec = corrRec[(nLags - 1) / 2:]
    if xlim is None:
        xlim = (lags[0], lags[-1])
    if ylim is None:
        ylim = (-1.05, 1.05)
    if absUnit != '':
        absUnit = '(%s)' % absUnit
    if xlabel is None:
        xlabel = r'$t$ %s' % absUnit
    if ylabel is None:
        ylabel = r'$C_{x, x}(t)$'
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lags, corrSample, linestyle='-', color='b', linewidth=lw)
    ax.plot(lags, corrRec, linestyle='--', color='g', linewidth=lw)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=fs_latex)
    ax.set_ylabel(ylabel, fontsize=fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)


def plotEigPowerRec(angFreq, eigValGen, weights, powerSample, powerSampleSTD, powerRec,
                    xlabel=None, ylabel=None, zlabel=None,
                    xlim=None, ylim=None, zlim=None):
    zlim = np.log10(zlim)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    iEigVal = (eigValGen.real >= xlim[0]) \
              & (eigValGen.real <= xlim[1]) \
              & (eigValGen.imag >= ylim[0]) \
              & (eigValGen.imag <= ylim[1])
    ax.scatter(eigValGen[iEigVal].real, eigValGen[iEigVal].imag,
               np.ones((eigValGen[iEigVal].shape[0],)) * zlim[0],
               c='k', s=msize*2, marker='+', depthshade=False)
    ax.scatter(eigValGen[iEigVal].real, eigValGen[iEigVal].imag,
               np.ones((eigValGen[iEigVal].shape[0],)) * zlim[0],
               s=weights,
               c='b', edgecolors='face', marker='o', depthshade=False)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    powerSampleDown = powerSample - powerSampleSTD / 2
    powerSampleUp = powerSample + powerSampleSTD / 2
    iangFreq = (angFreq >= ylim[0]) & (angFreq <= ylim[1])
    powerSamplePlot = np.ones((powerSample.shape[0],)) * zlim[0]
    powerSampleDownPlot = np.ones((powerSample.shape[0],)) * zlim[0]
    powerSampleUpPlot = np.ones((powerSample.shape[0],)) * zlim[0]
    powerRecPlot = np.ones((powerRec.shape[0],)) * zlim[0]
    powerSamplePlot[powerSample > 0] = np.log10(powerSample[powerSample > 0])
    powerRecPlot[powerRec > 0] = np.log10(powerRec[powerRec > 0])
    powerSampleDownPlot[powerSampleDown > 0] = np.log10(powerSampleDown[powerSampleDown > 0])
    powerSampleUpPlot[powerSampleUp > 0] = np.log10(powerSampleUp[powerSampleUp > 0])
    #powerSamplePlot = powerSample
    powerSamplePlot[powerSamplePlot < zlim[0]] = zlim[0]
    powerSamplePlot[powerSamplePlot > zlim[1]] = zlim[1]
    powerSampleDownPlot[powerSampleDownPlot < zlim[0]] = zlim[0]
    powerSampleDownPlot[powerSampleDownPlot > zlim[1]] = zlim[1]
    powerSampleUpPlot[powerSampleUpPlot < zlim[0]] = zlim[0]
    powerSampleUpPlot[powerSampleUpPlot > zlim[1]] = zlim[1]
    powerRecPlot[powerRecPlot < zlim[0]] = zlim[0]
    powerRecPlot[powerRecPlot > zlim[1]] = zlim[1]
    ax.plot(np.zeros((iangFreq.sum(),)), angFreq[iangFreq],
            powerSamplePlot[iangFreq], 'k-')
    xs = np.concatenate((angFreq[iangFreq], angFreq[iangFreq][::-1]), 0)
    ys = np.concatenate((powerSampleUpPlot[iangFreq], powerSampleDownPlot[iangFreq]), 0)
    verts = list(zip(xs, ys))
    poly = PolyCollection([verts], facecolors=[(0,0,0)])
    poly.set_alpha(0.2)
    ax.add_collection3d(poly, zs=0, zdir='x')
    ax.plot(np.zeros((iangFreq.sum(),)), angFreq[iangFreq],
            powerRecPlot[iangFreq], '-r', linewidth=2)


    eigValPlot = eigValGen[iEigVal]
    iFirst = np.abs(eigValPlot.imag / eigValPlot.real) \
             > np.abs(eigValPlot[1].imag / eigValPlot[1].real) / 10
    nPlot = 2*7
    if iEigVal.sum() < nPlot:
        nPlot = iEigVal.sum()
    eigValPlottmp = np.empty((nPlot,), dtype=complex)
    eigValPlottmp[:np.min([iFirst.sum(), nPlot])] = eigValPlot[iFirst][:nPlot]
    if np.sum(iFirst) < nPlot:
        eigValPlottmp[iFirst.sum():] = eigValPlot[:nPlot-iFirst.sum()]
    eigValPlot = eigValPlottmp
    for k in np.arange(eigValPlot.shape[0]):
        ax.plot([eigValPlot[k].real, 0.],
                [eigValPlot[k].imag, eigValPlot[k].imag],
                [zlim[0], zlim[0]],
                '--b')
        ax.plot([0., 0.],
                [eigValPlot[k].imag, eigValPlot[k].imag],
                [zlim[0],
                 powerSamplePlot[np.argmin(np.abs(angFreq - eigValPlot[k].imag))]],
                '--b')
    if np.sum(~iFirst) > 1:
        ax.plot([eigValGen[iEigVal][~iFirst][1].real, 0.],
                [eigValGen[iEigVal][~iFirst][1].imag,
                 eigValGen[iEigVal][~iFirst][1].imag],
                [zlim[0], zlim[0]], '--g')
        ax.plot([0, 0.], [eigValGen[iEigVal][~iFirst][1].imag,
                 eigValGen[iEigVal][~iFirst][1].imag],
                [zlim[0], powerSamplePlot[np.argmin(np.abs(angFreq - eigValGen[iEigVal][~iFirst][1].imag))]], '--g')
    ax.set_xlabel('\n' + xlabel, fontsize=fs_default,
                  linespacing=1.5)
    ax.set_ylabel('\n' + ylabel, fontsize=fs_default,
                  linespacing=1)
    ax.set_zlabel(zlabel, fontsize=fs_default)
    ax.set_xticks(np.arange(0., xlim[0], (xlim[0]-0.)/5))[::-1]
    zticks = np.arange(zlim[0], zlim[1], 2)
    ax.set_zticks(zticks)
    zticklabels = []
    for k in np.arange(zticks.shape[0]):
        zticklabels.append(r'$10^{%d}$' % zticks[k])
    ax.set_zticklabels(zticklabels)
    ax.view_init(30, -150)
    #plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    #plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
    #ax.set_title('%d-time-step spectrum for %s\nSlowest time-scale: %.1f' \
        #    % (tau, srcPostfix, -1. / rate[0]))
    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.))
    #ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.))
    #ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.))
    #ax.grid(False)

def getEigenCondition(eigVecForward, eigVecBackward, density=None):
    """ Return a vector containing the condition vector of each pair of eigenvectors."""
    (N, nev) = eigVecForward.shape
    
    if density is None:
        density = np.ones((N,), float) / N

    condition = np.empty((nev,))
    for ev in np.arange(nev):
        normForward = np.sqrt(np.sum(eigVecForward[:, ev] * density \
                                     * np.conjugate(eigVecForward[:, ev])).real)
        normBackward = np.sqrt(np.sum(eigVecBackward[:, ev] * density \
                                      * np.conjugate(eigVecBackward[:, ev])).real)
        inner = (np.sum(eigVecForward[:, ev] * density \
                        * np.conjugate(eigVecBackward[:, ev]))).real
        condition[ev] = normForward * normBackward / inner

    return condition
