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
muRng = np.array([0.])
plotPoint = False
plotOrbit = False
#muRng = np.array([3.])
#plotPoint = False
#plotOrbit = False
#muRng = np.array([7.])
#plotPoint = True
#plotOrbit = True

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
xmin = -30.
xmax = 0.1
ymin = -10.
ymax = -ymin

print 'For eps = ', eps
print 'For beta = ', beta
for k in np.arange(muRng.shape[0]):
    mu = muRng[k]
    print 'For mu = ', mu
    mu += 1.e-8
    if mu < 0:
        signMu = 'm'
    else:
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

    # Read eigenvalues
    print 'Reading backward eigenvalues'
    eigValBackward = np.empty((nev,), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigValBackward%s.txt' \
                             % (resDir, postfix), eigValBackward)
    isortBackward = np.argsort(-eigValBackward.real)
    eigValBackward = eigValBackward[isortBackward]

    # Read eigenvectors
    print 'Reading backward eigenvectors'
    eigVecBackward = np.empty((nx0**2, nev), dtype=complex)
    ergoPlot.loadtxt_complex('%s/eigVecBackward%s.txt' % (resDir, postfix),
                             eigVecBackward)
    eigVecBackward = eigVecBackward[:, isortBackward]

    # Convert second eigenvectors to phase and amplitude
    if eigValBackward[1].imag > 0:
        eigVec2 = eigVecBackward[:, 1]
    else:
        eigVec2 = eigVecBackward[:, 2]
    eigVec2Abs = np.abs(eigVec2)
    eigVec2Angle = np.angle(eigVec2)
    eigVec2Name = r'\psi_{%d%d}' % (0, 1)

    # Calculate analytical eigenvalues
    if mu <= 0: 
        eigValAnaPoint = (J + J.T) * mu \
                         + 1j * (J - J.T) * gam
    if mu > 0:
        eigValAnaPoint = -(J + J.T + 2) * mu \
                         - 1j * (J - J.T) * gam
        omega = gam - beta * mu
        D = eps**2 * (1 + beta**2) / (2 * mu)
        eigValAnaOrbit = (-I**2 * D + 1j * I * omega \
                         - 2 * mu * J).flatten()

    # Reshape
    eigVec2Abs = eigVec2Abs.reshape(nx0, nx0).T
    eigVec2Angle = eigVec2Angle.reshape(nx0, nx0).T

    # Filter spectrum outside
    if mu > 0:
        eigValAnaOrbit = eigValAnaOrbit[(eigValAnaOrbit.real >= xmin) \
                                        & (eigValAnaOrbit.real <= xmax) \
                                        & (eigValAnaOrbit.imag >= ymin) \
                                        & (eigValAnaOrbit.imag <= ymax)]
    eigValAnaPoint = eigValAnaPoint[(eigValAnaPoint.real >= xmin) \
                                    & (eigValAnaPoint.real <= xmax) \
                                    & (eigValAnaPoint.imag >= ymin) \
                                    & (eigValAnaPoint.imag <= ymax)]

    # Plot eigenvalues
    print 'Plotting'
    msize = 32
    msizeAna = 100
    fig = plt.figure()
    #fig.set_visible(False)
    ax = fig.add_subplot(111)
    ax.scatter(eigValBackward.real, eigValBackward.imag, c='k',
               s=msize, marker='o', edgecolors='face')
    if (mu > 0) & plotOrbit:
        ax.scatter(eigValAnaOrbit.real, eigValAnaOrbit.imag,
                   marker='+', color='k', s=msizeAna)
    if plotPoint:
        if mu > 0:
            pointColor = 'b'
        else:
            pointColor = 'k'
        ax.scatter(eigValAnaPoint.real, eigValAnaPoint.imag,
                   marker='x', color=pointColor, s=msizeAna)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Parameter labels
    ax.text(xmin*0.96, ymax*1.03, r'$\mu = %.1f$' % mu,
            fontsize='xx-large')
    ax.text(xmin*0.18, ymax*1.03, r'$\beta = %.1f$' % beta,
            fontsize='xx-large')
    ax.set_xlabel(xlabel, fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(ylabel, fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_latex)
    fig.savefig('%s/vsAnaPlaneEigVal%s.%s' % (plotDir, postfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

    # Plot eigenvectors
    levelsPhase = np.linspace(-np.pi, np.pi, 4*3+1)
    rmax = eigVec2Abs.max()
    #    levelsAmp = np.linspace(0., rmax, 6)
    levelsAmp = np.arange(0.001, 0.0071, 0.001)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot eigenvector phase
    contourPhase = ax.contourf(X, Y, eigVec2Angle,
                               levelsPhase, cmap=cm.RdBu_r)
    # Plot eigenvector amplitude
    contourAmp = ax.contour(X, Y, eigVec2Abs,
                            levelsAmp, linestyles='-', colors='k', linewidths=1)
    contourAmp = ax.contour(X, Y, eigVec2Abs,
                            levelsAmp[:1], linestyles='-', colors='k',
                            linewidths=2)
    # Set contour labels
    #plt.clabel(contourAmp, inline=1, inline_spacing=1, fontsize='x-large', fmt='%.1e')
    # Set colorbar
    cbar = plt.colorbar(contourPhase)
    plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    cbar.set_ticks(np.arange(-np.pi, np.pi*1.1, np.pi / 2))
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    #    ax.set_title(r'$|%s|$' % eigVec2Name)
    # Plot limitcycle and isochron
    theta = np.linspace(0., 2*np.pi, 1001)
    THETAXY = np.angle(X + 1j* Y)
    RXY = np.abs(X + 1j* Y)
    if mu > 1.e-2:
        Rpo = np.sqrt(mu)
        
        ax.plot(Rpo * np.cos(theta), Rpo * np.sin(theta),
                '--k', linewidth=2)
        ax.quiver(0., -Rpo, [1.], [0.], scale=50, width=0.01,
                  headwidth=20., headlength=8, color='k')
        ax.text(0., -Rpo * 1.3, r'$\Gamma$', fontsize=ergoPlot.fs_latex,
                color='k')
        factor = 2.
        xlim = [-factor * Rpo, factor * Rpo]
        ax.set_xlim(xlim)
        ylim = [-factor * Rpo, factor * Rpo]
        ax.set_ylim(ylim)
        # Plot isochron
        THETAIso = THETAXY.copy()
        THETAIso[RXY > 0] -= beta * np.log(RXY[RXY > 0] / Rpo)
        THETAIso = np.mod(THETAIso, 2*np.pi) - np.pi
        val = 0.
        idx = np.unravel_index(np.argmin(np.abs(eigVec2Angle - val)), X.shape)
        THETAIso += val - THETAIso[idx]
        THETAIso[np.abs(THETAIso - val) > np.pi/2] = None
        ciso = ax.contour(X, Y, THETAIso, [val], linestyles='-', colors='b',
                          linewidths=2)
        # ciso.clabel(v=[0], fmt={0 : r'$I(\theta = 0)$'}, fontsize=ergoPlot.fs_latex,
        #             manual=True)
        A = np.sqrt((RXY - factor * Rpo)**2 + (THETAIso - val)**2)
        A[np.isnan(A)] = 1e10
        posidx = np.unravel_index(np.argmin(A), A.shape)
        xtxt = X[posidx]
        ytxt = Y[posidx]
        xcorr = -0.1 * xtxt
        ycorr = -0.05 * ytxt
        ax.text(X[posidx] + xcorr, Y[posidx] + ycorr, r'$I(\theta = 0)$',
                fontsize=ergoPlot.fs_latex, color='b',
                bbox=dict(facecolor='w', alpha=0.9, edgecolor='none'),
                verticalalignment='center', horizontalalignment='center')
    else:
        factor = 1.
        ax.set_xlim(X[0].min()*factor, X[0].max()*factor)
        ax.set_ylim(Y[:, 0].min()*factor, Y[:, 0].max()*factor)
        if mu > -1.e-2:
            Rpo = (eps**2/2)**(1./4)
            # Plot isochron
            THETAIso = THETAXY.copy()
            THETAIso[RXY > 0] -= beta/2 * np.log((Rpo**2 + RXY[RXY > 0]**2) \
                                                 / (2*Rpo**4))
            THETAIso = np.mod(THETAIso, 2*np.pi) - np.pi
            val = 0.
            idx = np.unravel_index(np.argmin(np.abs(eigVec2Angle - val)), X.shape)
            THETAIso += val - THETAIso[idx]
            THETAIso[np.abs(THETAIso - val) > np.pi/2] = None
            ciso = ax.contour(X, Y, THETAIso, [val], linestyles='-', colors='b',
                              linewidths=2)
            # ciso.clabel(v=[0], fmt={0 : r'$I(\theta = 0)$'},
            # fontsize=ergoPlot.fs_latex, manual=True)
            A = np.sqrt((RXY - X[0].max())**2 + (THETAIso - val)**2)
            A[np.isnan(A)] = 1e10
            posidx = np.unravel_index(np.argmin(A), A.shape)
            xtxt = X[posidx]
            ytxt = Y[posidx]
            xcorr = -0.1 * xtxt
            ycorr = -0.05 * ytxt
            ax.text(X[posidx] + xcorr, Y[posidx] + ycorr, r'$I(\theta = 0)$',
                    fontsize=ergoPlot.fs_latex, color='b',
                    bbox=dict(facecolor='w', alpha=0.9, edgecolor='none'),
                    verticalalignment='center', horizontalalignment='center')
        else:
            Rpo = 0.
            # Plot isochron
            THETAIso = THETAXY.copy()
            THETAIso[RXY > 0] -= beta/2 * np.log((mu - RXY[RXY > 0]**2) / mu)
            THETAIso = np.mod(THETAIso, 2*np.pi) - np.pi
            val = 0.
            idx = np.unravel_index(np.argmin(np.abs(eigVec2Angle - val)), X.shape)
            THETAIso += val - THETAIso[idx]
            THETAIso[np.abs(THETAIso - val) > np.pi/2] = None
            ciso = ax.contour(X, Y, THETAIso, [val], linestyles='-', colors='b',
                              linewidths=2)
            # ciso.clabel(v=[0], fmt={0 : r'$I(\theta = 0)$'},
            # fontsize=ergoPlot.fs_latex, manual=True)
            A = np.sqrt((RXY - X[0].max())**2 + (THETAIso - val)**2)
            A[np.isnan(A)] = 1e10
            posidx = np.unravel_index(np.argmin(A), A.shape)
            xtxt = X[posidx]
            ytxt = Y[posidx]
            xcorr = -0.1 * xtxt
            ycorr = -0.05 * ytxt
            ax.text(X[posidx] + xcorr, Y[posidx] + ycorr, r'$I(\theta = 0)$',
                    fontsize=ergoPlot.fs_latex, color='b',
                    bbox=dict(facecolor='w', alpha=0.9, edgecolor='none'),
                    verticalalignment='center', horizontalalignment='center')
    # Set ticks and labels
    ax.set_xlabel(r'$x$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$y$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    # Add parameter indications
    pxlim = ax.get_xlim()
    pylim = ax.get_ylim()
    ax.text(pxlim[0]*0.9, pylim[1]*1.03, r'$\mu = %.1f$' % mu,
            fontsize='xx-large')
    ax.text(pxlim[1]*0.55, pylim[1]*1.03, r'$\beta = %.1f$' % beta,
            fontsize='xx-large')
    # Save
    fig.savefig('%s/vsAnaPlaneEigVec2%s.%s' % (plotDir, postfix, ergoPlot.figFormat),
                bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    plt.show()
