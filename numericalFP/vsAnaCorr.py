import os
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

# Model parameters
model = 'Hopf'
gam = 1.
#beta = 0.
beta = 0.5
eps = 1.
#muRng = np.array([-5.])
#plotPoint = True
#plotOrbit = False
#muRng = np.array([0.])
#plotPoint = False
#plotOrbit = False
#muRng = np.array([3.])
#plotPoint = False
#plotOrbit = False
muRng = np.array([7.])
plotPoint = True
plotOrbit = True

mu0 = -10.
muf = 15.
dmu = 0.1

# Grid definition
dim = 2
nx0 = 200
nSTD = 5

# Number of eigenvalues
#nev = 21
nev = 201

# Indices for the analytical eigenvalues
ni = nev
i = np.arange(-ni/2, ni/2)
j = np.arange(ni)
(I, J) = np.meshgrid(i, j)

# Plot config
#figFormat = 'png'
figFormat = 'eps'
xlabel = r'$\Re(\lambda_k)$'
ylabel = r'$\Im(\lambda_k)$'
zlabel = r'$S_{f, g}$'
xmin = -30.
xmax = 0.1
xlimPlot = [xmin, xmax]
zlim = [1.e-4, 1.e2]

print 'For eps = ', eps
print 'For beta = ', beta
for k in np.arange(muRng.shape[0]):
    mu = muRng[k]
    print 'For mu = ', mu
    mu += 1.e-8
    if mu < 0:
        omega = gam
        signMu = 'm'
    else:
        omega = gam - beta * mu
        signMu = 'p'
    beta += 1.e-8
    if beta < 0:
        signBeta = 'm'
    else:
        signBeta = 'p'
    postfix = '_%s_mu%s%02d_beta%s%03d_eps%03d_nx%d_nSTD%d_nev%d' \
              % (model, signMu, int(round(np.abs(mu) * 10)),
                 signBeta, int(round(np.abs(beta) * 100)), int(round(eps * 100)),
                 nx0, nSTD, nev)
    resDir = '../results/numericalFP/%s' % model
    plotDir = '../results/plot/numericalFP/%s' % model
    os.system('mkdir %s 2> /dev/null' % plotDir)
    print 'Postfix = ', postfix

    # Get grid
    r = np.linspace(0., np.sqrt(muf)*2., 10000)
    theta = np.linspace(-np.pi, np.pi, 1000)
    (R, THETA) = np.meshgrid(r, theta)
    Ur = (-mu*R**2/2 + R**4/4)
    Ur[-2*Ur/eps**2 > 100] = 100
    rho = R * (np.exp((-2*Ur / eps**2)))
    rho /= rho.sum()
    xrt = R * np.cos(THETA)
    sigma = np.sqrt((xrt**2 * rho).sum() - (xrt * rho).sum()**2)
    xlim = np.ones((dim,)) * sigma * nSTD
    # Get grid points and steps
    x = []
    dx = np.empty((dim,))
    nx = np.ones((dim,), dtype=int) * nx0
    for d in np.arange(dim):
        x.append(np.linspace(-xlim[d], xlim[d], nx[d]))
        dx[d] = x[d][1] - x[d][0]
    N = np.prod(nx)
    idx = np.indices(nx).reshape(dim, -1)
    X, Y = np.meshgrid(*x, indexing='xy') # ! Different that for numerical
    # -> Need transpose vectors !

    # nevRec = int(np.round(np.abs(ymin / omega))) * 2 + 1
    nevRec = 21
    
    nOmega = 8
    ylim = [-nOmega * np.abs(omega), nOmega * np.abs(omega)]
    yticks = np.arange(-nOmega, nOmega*1.01, 2) * omega
    yticklabels = []
    for t in np.arange(-nOmega, nOmega * 1.01, 2):
        yticklabels.append(r'$%d \omega$' % (int(np.round(t)),))
    yticklabels[len(yticks)/2] = r'$0$'

    # Read eigenvalues
    print 'Reading backward eigenvalues'
    eigValBackward = np.empty((nev,), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigValBackward%s.txt' \
                             % (resDir, postfix), eigValBackward)

    # Read eigenvectors
    print 'Reading backward eigenvectors'
    eigVecBackward = np.empty((nx0**2, nev), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigVecBackward%s.txt' % (resDir, postfix),
                             eigVecBackward)

    # Read eigenvalues
    print 'Reading forward eigenvalues'
    eigValForward = np.empty((nev,), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigValForward%s.txt' \
                             % (resDir, postfix), eigValForward)

    # Read eigenvectors
    print 'Reading forward eigenvectors'
    eigVecForward = np.empty((nx0**2, nev), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigVecForward%s.txt' % (resDir, postfix),
                             eigVecForward)

    # Sort eigenvalues and eigenvectors
    (eigValForward, eigValBackward, eigVecForward, eigVecBackward) \
        = ergoPlot.sortEigenvectors(eigValForward, eigValBackward,
                                    eigVecForward, eigVecBackward, 'LR')

    # Make biorthonormal basis
    (eigVecForward, eigVecBackward) \
        = ergoPlot.makeBiorthonormal(eigVecForward, eigVecBackward)

    # Get stationary density (used to normalize power spectrum)
    statDist = eigVecForward[:, 0].real
    statDist /= statDist.sum()

    # Define observables
    xl, yl = X.flatten(), Y.flatten()
    rl = np.abs(xl + 1j*yl)
    phil = np.angle(xl * 1j*yl)
    if mu > 1.e-2:
        Rpo = np.sqrt(mu)
        phil[rl > 0.01] -= beta * np.log(rl / Rpo)
    # f = [xl, xl**3, rl, phil]
    # obsLabel = [r'$S_{x, x}$', r'$S_{x^3, x^3}$', r'$S_{r, r}$',
    #             r'$S_{\phi, \phi}$']
    f = [xl, xl**2, xl**3]
    obsLabel = [r'$S_{x, x}$', r'$S_{x^2, x^2}$', r'$S_{x^3, x^3}$']
    g = f
    nObs = len(f)
    angFreq = np.linspace(ylim[0], ylim[1], 1000)
    weights = []
    power = []
    cPow = rcParams['axes.prop_cycle'].by_key()['color']
    while len(cPow) < nObs:
        cPow = np.concatenate((cPow, rcParams['axes.prop_cycle'].by_key()['color']))
    lsPow = ['-'] * nObs
    lwPow = [2] * nObs
    for obs in np.arange(nObs):
        # Get weights of the spectral projections of the observables
        # let the forward eigenvectors carry the measure.
        weightsObs = ergoPlot.getSpectralWeights(f[obs], g[obs],
                                                 eigVecForward[:, :nevRec],
                                                 eigVecBackward[:, :nevRec])
        weights.append(weightsObs)
        # Get power spectrum
        powerObs, powerCompObs = ergoPlot.spectralRecPower(angFreq, f[obs], g[obs],
                                                           eigValForward[:nevRec],
                                                           weights[obs], norm=True,
                                                           statDist=statDist)
        power.append(powerObs)

        
    # Calculate analytical eigenvalues
    if mu <= 0:
        eigValAnaPoint = (J + J.T) * mu \
                         + 1j * (J - J.T) * gam
    if mu > 0:
        eigValAnaPoint = -(J + J.T + 2) * mu \
                         - 1j * (J - J.T) * gam
        D = eps**2 * (1 + beta**2) / (2 * mu)
        eigValAnaOrbit = (-I**2 * D + 1j * I * omega \
                         - 2 * mu * J).flatten()

    # Filter spectrum outside
    if mu > 0:
        eigValAnaOrbit = eigValAnaOrbit[(eigValAnaOrbit.real >= xmin) \
                                        & (eigValAnaOrbit.real <= xmax) \
                                        & (eigValAnaOrbit.imag >= ylim[0]) \
                                        & (eigValAnaOrbit.imag <= ylim[1])]
    eigValAnaPoint = eigValAnaPoint[(eigValAnaPoint.real >= xmin) \
                                    & (eigValAnaPoint.real <= xmax) \
                                    & (eigValAnaPoint.imag >= ylim[0]) \
                                    & (eigValAnaPoint.imag <= ylim[1])]

    # Plot eigenvalues
    print 'Plotting'
    msize = 32
    msizeAna = 100
    # Create axis for eigenvalue and power spectrum panels
    nullfmt = plt.NullFormatter()
    leftEig, widthEig = 0.12, 0.55
    bottom, height = 0.1, 0.85
    leftPow = leftEig + widthEig + 0.01
    widthPow = 0.3
    rectEig = [leftEig, bottom, widthEig, height]
    rectPow = [leftPow, bottom, widthPow, height]
    #    ratio = height  / (widthEig + widthPow)
    (defaultFigWidth, defaultFigHeight) = plt.rcParams['figure.figsize']
    #    fig = plt.figure(figsize=(defaultFigHeight*ratio, defaultFigHeight))
    widthFig = defaultFigWidth
    heightFig = defaultFigHeight
    fig = plt.figure(figsize=(widthFig, heightFig))
    axEig = plt.axes(rectEig)
    axPow = plt.axes(rectPow)

    # Plot numerical approximation of eigenvalues
    axEig.scatter(eigValBackward.real, eigValBackward.imag, c='k',
                  s=msize, marker='o', edgecolors='face')

    # Plot analytical eigenvalues
    # For the limit cycle
    if (mu > 0) & plotOrbit:
        axEig.scatter(eigValAnaOrbit.real, eigValAnaOrbit.imag,
                   marker='+', color='k', s=msizeAna)
    # For the fixed point
    if plotPoint:
        if mu > 0:
            pointColor = 'b'
        else:
            pointColor = 'k'
        axEig.scatter(eigValAnaPoint.real, eigValAnaPoint.imag,
                      marker='x', color=pointColor, s=msizeAna)

    # Parameter labels
    axEig.text(xmin*0.96, ylim[1]*1.03, r'$\mu = %.1f$' % mu,
               fontsize='xx-large')
    axEig.text(xmin*0.18, ylim[1]*1.03, r'$\beta = %.1f$' % beta,
               fontsize='xx-large')
    # Set axes for eigenvalues
    axEig.set_xlim(xlimPlot)
    axEig.set_ylim(ylim)
    axEig.set_xlabel(xlabel, fontsize=ergoPlot.fs_latex)
    axEig.set_ylabel(ylabel, fontsize=ergoPlot.fs_latex)
    axEig.set_yticks(yticks)
    axEig.set_yticklabels(yticklabels)
    plt.setp(axEig.get_xticklabels(), fontsize=ergoPlot.fs_latex)
    plt.setp(axEig.get_yticklabels(), fontsize=ergoPlot.fs_latex)
    # Plot power spectra
    zminObs, zmaxObs = [], []
    for obs in np.arange(nObs):
        axPow.plot(power[obs], angFreq, color=cPow[obs],
                   linewidth=lwPow[obs], linestyle=lsPow[obs],
                   label=obsLabel[obs])
        zminObs.append(np.min(power[obs]))
        zmaxObs.append(np.max(power[obs]))
    axPow.set_xscale('log')
    #axPow.set_xlim(zlim)
    axPow.set_ylim(ylim)
    axPow.set_xlabel(zlabel, fontsize=ergoPlot.fs_latex)
    nullfmt = plt.NullFormatter()
    axPow.yaxis.set_major_formatter(nullfmt)
    axPow.set_yticks(yticks)
    # Add legend
    axPow.legend(fontsize=ergoPlot.fs_default, loc=[0.43, 0.75],
                 frameon=False)
    # Set ticks
    zlim = (np.min(zminObs), np.max(zmaxObs))
    if np.log10(zlim[1]) - np.log10(zlim[0]) < 1:
        zticks = 10.**(np.arange(-5, 1.1, 0.25))
        axPow.set_xticklabels(['%.1e' % zticks[i] for i in np.arange(len(zticks))])
    else:
        zticks = 10.**(np.arange(-5, 1.1, 1))
    axPow.set_xticks(zticks)
    axPow.set_xlim(zlim)
    plt.setp(axPow.get_xticklabels(), fontsize='large')
    plt.minorticks_off()
    axPow.grid()
    fig.savefig('%s/vsAnaEigValPower%s.%s' % (plotDir, postfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
