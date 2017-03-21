import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import ergoPlot

omega = 4.
muSub = -1.
muSup = 1.5
eps = 0.4
beta = 0.5
Rpo = np.sqrt(muSup)
D = eps**2 * (1 + beta**2)

nn = 6
nm = nn
ll = -4

msize = 32


# Get analytical eigenvalues
eigValFP = []
eigValOrbit = []
for ii in np.arange(nn):
    for jj in np.arange(nm):
        eigValFP.append((ii + jj) * muSub + 1j * (ii - jj) * omega)
        eigValOrbit.append(-ii * 2.*muSup - D * jj**2 / (2 * Rpo**2) \
                           + 1j * omega * jj)
        eigValOrbit.append(-ii * 2.*muSup - D * jj**2 / (2 * Rpo**2) \
                           - 1j * omega * jj)
eigValFP = np.array(eigValFP)
eigValOrbit = np.array(eigValOrbit)

xticksSub = np.arange((nm-1)*muSub, 0., -muSub)
xticksSup = np.arange(-(nm-1)*muSup, 0., muSup)
yticks = np.arange(-(nm-1)*omega, nm*omega, omega)

# Subcritical eigenvalues
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111)
ax.scatter(eigValFP.real, eigValFP.imag, c='k',
           s=msize, marker='o', edgecolors='face')
ax.scatter(0, 0, c='g',
           s=msize*10, marker='*', edgecolors='face')
ax.set_xlim((nn-0.5)*muSub, -muSub)
ax.set_ylim(-(nm-1./2)*omega, (nm-1./2)*omega)
ax.annotate(s='', xy=(0, 0), xytext=(0, omega),
            arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax.annotate(s='', xy=(muSub, omega), xytext=(0, omega),
            arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax.text(-muSub*0.1, omega*0.4, r'$\omega$', fontsize=ergoPlot.fs_latex)
ax.text(muSub*0.5, omega*1.2, r'$\mu$', fontsize=ergoPlot.fs_latex)
# Add indices
xoffset_yindex = -muSub*0.5
yoffset_yindex = -omega * 0.15
xoffset_xindex = muSub * 0.1
yoffset_xindex = omega * 0.13
rotation = 90
fs_index = 'x-large'
ax.text(-0.5*muSub, -2*omega + yoffset_yindex, r'$\vdots$', fontsize=fs_index)
ax.text(xoffset_yindex, -omega + yoffset_yindex, r'$n - l = -1$', fontsize=fs_index)
ax.text(xoffset_yindex, 0 + yoffset_yindex, r'$n - l = 0$', fontsize=fs_index)
ax.text(xoffset_yindex, omega + yoffset_yindex, r'$n - l = 1$', fontsize=fs_index)
ax.text(-0.5*muSub, 2*omega + yoffset_yindex, r'$\vdots$', fontsize=fs_index)
ax.text(xoffset_xindex * 2, (nn-1) * omega + yoffset_xindex,
        r'$l + n = 0$', fontsize=fs_index, rotation=rotation)
ax.text(1*muSub + xoffset_xindex, (nn-1) * omega + yoffset_xindex,
        r'$l + n = 1$', fontsize=fs_index, rotation=rotation)
ax.text(2*muSub + xoffset_xindex, (nn-1) * omega + yoffset_xindex,
        r'$l + n = 2$', fontsize=fs_index, rotation=rotation)
ax.text(3*muSub + xoffset_xindex, (nn-1) * omega - omega / 2,
        r'$\vdots$', fontsize=fs_index, rotation=rotation)
# Set ticks
ax.set_xticks(xticksSub)
ax.set_yticks(yticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.xaxis.set_tick_params(length=0)
ax.yaxis.set_tick_params(length=0)
# Grid and axes
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
ax.spines['left'].set_position(('data', 0.))
fig.savefig('plot/artMixingEigValSub.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
            
# Supercritical eigenvalues
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111)
ax.scatter(eigValOrbit.real, eigValOrbit.imag, c='k',
           s=msize, marker='o', edgecolors='face')
ax.scatter(eigValFP.real - 2*muSup, eigValFP.imag, c='b',
           s=msize, marker='^', edgecolors='face')
ax.scatter(0, 0, c='g',
           s=msize*10, marker='*', edgecolors='face')
ax.set_xlim(-(nn-0.5)*muSup, muSup)
ax.set_ylim(-(nm-1./2)*omega, (nm-1./2)*omega)
ax.annotate(s='', xy=(0, 0), xytext=(0, omega),
            arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax.text(muSup*0.1, omega*0.4, r'$\omega$', fontsize=ergoPlot.fs_latex)
ax.annotate(s='', xy=(-D/(2*muSup) - 2*muSup, 0.), xytext=(-D/(2*muSup), 0.),
            arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax.text(-2*muSup*0.5, omega*.2, r'$2\mu$', fontsize=ergoPlot.fs_latex)
ax.annotate(s='', xy=(-D*ll**2/(2*muSup), ll*omega), xytext=(0, ll*omega),
            arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax.text(-D*ll**2/(2*muSup)*1.5, ll*omega*1.2,
        r'$-\frac{n^2 \epsilon^2 (1 + \beta^2)}{2\mu}$', fontsize=ergoPlot.fs_latex)
# Add indices
xoffset_yindex = +muSup*0.5
yoffset_yindex = -omega * 0.15
xoffset_xindex =  - muSup*0.4
yoffset_xindex = omega * 0.13
fs_index = 'x-large'
ax.text(0.5*muSup, -2*omega + yoffset_yindex, r'$\vdots$', fontsize=fs_index)
ax.text(xoffset_yindex, -omega + yoffset_yindex, r'$n = -1$', fontsize=fs_index)
ax.text(xoffset_yindex, 0 + yoffset_yindex, r'$n = 0$', fontsize=fs_index)
ax.text(xoffset_yindex, omega + yoffset_yindex, r'$n = 1$', fontsize=fs_index)
ax.text(0.5*muSup, 2*omega + yoffset_yindex, r'$\vdots$', fontsize=fs_index)
ax.text(-muSup + xoffset_xindex, (nn-1) * omega + yoffset_xindex, r'$l = 0$', fontsize=fs_index)
ax.text(-3*muSup + xoffset_xindex, (nn-1) * omega + yoffset_xindex, r'$l = 1$', fontsize=fs_index)
ax.text(-5*muSup + xoffset_xindex, (nn-1) * omega + yoffset_xindex, r'$l = 2$', fontsize=fs_index)
# Set ticks
ax.set_xticks(xticksSup)
ax.set_yticks(yticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.xaxis.set_tick_params(length=0)
ax.yaxis.set_tick_params(length=0)
# Set grid and axes
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
ax.spines['left'].set_position(('data', 0.))
fig.savefig('plot/artMixingEigValSup.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

#  nx = 100
nx = 4000
rmax = 2*Rpo
l, n = 0, 1
theta = np.linspace(0., 2*np.pi, nx)
r = np.linspace(0., rmax, nx)
(THETA, R) = np.meshgrid(theta, r)
xlabel = r'$\theta$'
ylabel = r'$\theta$'
rticks = np.arange(1., 1.1, 1.) * Rpo
thetaticks = np.array([-90, 0., 90, 180])
levelsPhase = np.linspace(-np.pi, np.pi, 4*3+1)
levelsRadius = np.linspace(0., rmax, 6)
hw = 5
angDecomp = np.pi/6
trajColor = 'b'

# Simulate
dt = 0.001
L = np.abs(1.5 / muSup)
x0Sub = np.array([np.pi/3, 1.8*Rpo])
time = np.arange(0., L, dt)
xSub = np.empty((time.shape[0], x0Sub.shape[0]))
xSub[0] = x0Sub
for t in np.arange(1, time.shape[0]):
    x0Sub += dt * np.array([omega - beta*x0Sub[1]**2,
                         muSub*x0Sub[1] - x0Sub[1]**3 + eps**2/(2*x0Sub[1])]) \
                         + np.sqrt(dt)*eps*np.random.normal(size=2)/np.array([x0Sub[1], 1.])
    xSub[t] = x0Sub
x0Sup = np.array([np.pi/3, 1.8*Rpo])
xSup = np.empty((time.shape[0], x0Sup.shape[0]))
xSup[0] = x0Sup
for t in np.arange(1, time.shape[0]):
    x0Sup += dt * np.array([omega - beta*x0Sup[1]**2,
                         muSup*x0Sup[1] - x0Sup[1]**3 + eps**2/(2*x0Sup[1])]) \
    + np.sqrt(dt)*eps*np.random.normal(size=2)/np.array([x0Sup[1], 1.])
    xSup[t] = x0Sup

THETAIso = THETA.copy()
THETAIso[R > 0] -= beta * np.log(R[R > 0] / Rpo)

# Subcritical eigenvector
phaseSub = np.mod(-(l - n) * THETA, 2*np.pi) - np.pi
ampSub = R.copy()
USub = -muSub*R**2/2 + R**4/4
rhoSub = R * np.exp(-2*USub / eps**2)

# Supercritical eigenvector
phaseSup = np.mod(n * THETAIso, 2*np.pi) - np.pi
ampSup = np.ones(R.shape)
USup = -muSup*R**2/2 + R**4/4
rhoSup = R * np.exp(-2*USup / eps**2)

# Subcritical
# Distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# Plot eigenvector phase
h = ax.contourf(THETA-np.pi, R, rhoSub, 20, cmap=cm.Reds)
# # Plot limitcycle
# ax.plot(theta, np.ones(theta.shape) * Rpo, '--w', linewidth=2)
# ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8, color='w')
ax.set_rgrids([rmax], labels=[''])
# Plot trajectory
ax.plot(xSub[:, 0], xSub[:, 1], color=trajColor)
ax.scatter(xSub[0, 0], xSub[0, 1], s=20, c=trajColor,
           marker='s', edgecolors='face')
ax.scatter(xSub[-1, 0], xSub[-1, 1], s=20, c=trajColor,
           marker='>', edgecolors='face')
# # Plot isochron
# ax.plot(theta-np.pi+angDecomp, Rpo * np.exp((theta-np.pi)/beta), linewidth=2)
# ax.text(np.pi/9+angDecomp, Rpo*2.01, r'$I_{p}$', fontsize=ergoPlot.fs_latex, color='b')
# # Plot e0 (flow)
# ax.quiver([angDecomp], [Rpo], [-np.sin(angDecomp)], [np.cos(angDecomp)],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp + np.pi/6, Rpo*1.2, r'$\vec{e}_0$', fontsize=ergoPlot.fs_latex)
# # Plot e1 (transverse)
# ax.quiver([angDecomp], [Rpo], [np.cos(angDecomp + np.arctan(beta/np.sqrt(mu)))], [np.sin(angDecomp + np.arctan(beta/np.sqrt(mu)))],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp, Rpo*1.5, r'$\vec{e}_1$', fontsize=ergoPlot.fs_latex)
# # Plot state
# ax.scatter(angDecomp, Rpo, s=20, c='k', marker='o')
# ax.text(angDecomp + np.pi/24, Rpo*0.7, r'$p$', fontsize=ergoPlot.fs_latex, color='k')
# Plot fix point
ax.scatter(0., 0., s=20, c='k', marker='o')
ax.text(-np.pi*5/6, Rpo*0.3, r'$x_*$', fontsize=ergoPlot.fs_latex, color='w')
# # Colorbar
# cbar = plt.colorbar(h)
# plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# cbar.set_ticks(np.arange(-np.pi, np.pi*1.1, np.pi / 2))
# cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
# Grid
ax.set_rlim(0., rmax)
ax.set_thetagrids(thetaticks, labels=[r'$-\pi/2$', '', r'$\pi/2$', r'$\pi$'])
# Ticks
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# Save
fig.savefig('plot/artMixingDistSub.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# Second eigenvector
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# Plot eigenvector phase
h = ax.contourf(THETA-np.pi, R, phaseSub, levelsPhase, cmap=cm.RdBu_r)
# Plot eigenvector amplitude
ax.contour(THETA, R, ampSub, levelsRadius, linestyles='-', colors='k', linewidths=1)
ax.contour(THETA, R, ampSub, levelsRadius[-1:], linestyles='-', colors='k', linewidths=2)
# # Plot limit cycle
# ax.plot(theta, np.ones(theta.shape) * Rpo, '--k', linewidth=2)
# ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8)
ax.set_rgrids([rmax], labels=[''])
# # Plot e0 (flow)
# ax.quiver([angDecomp], [Rpo], [-np.sin(angDecomp)], [np.cos(angDecomp)],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp + np.pi/6, Rpo*1.2, r'$\vec{e}_0$', fontsize=ergoPlot.fs_latex)
# # Plot e1 (transverse)
# ax.quiver([angDecomp], [Rpo], [np.cos(angDecomp)], [np.sin(angDecomp)],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp - np.pi/24, Rpo*1.5, r'$\vec{e}_1$', fontsize=ergoPlot.fs_latex)
# # Plot state
# ax.scatter(angDecomp, Rpo, s=20, c='k', marker='o')
# ax.text(angDecomp + np.pi/24, Rpo*0.7, r'$p$', fontsize=ergoPlot.fs_latex, color='k')
# Plot fix point
ax.scatter(0., 0., s=20, c='k', marker='o')
ax.text(-np.pi*5/6, Rpo*0.3, r'$x_*$', fontsize=ergoPlot.fs_latex, color='k')
# Colorbar
cbar = plt.colorbar(h)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
cbar.set_ticks(np.arange(-np.pi, np.pi*1.1, np.pi / 2))
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
# Grid
ax.set_rlim(0., rmax)
ax.set_rgrids([rmax], labels='')
ax.set_thetagrids(thetaticks, labels=[r'$-\pi/2$', '', r'$\pi/2$', r'$\pi$'])
# Ticks
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# Save
fig.savefig('plot/artMixingEigVecSub.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# Supercritical
# Distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# Plot eigenvector phase
h = ax.contourf(THETA-np.pi, R, rhoSup, 20, cmap=cm.Reds)
# Plot limitcycle
ax.plot(theta, np.ones(theta.shape) * Rpo, '--w', linewidth=2)
ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8, color='w')
ax.set_rgrids(rticks, labels=[r'$\Gamma$'], angle=260, color='w')
# Plot trajectory
ax.plot(xSup[:, 0], xSup[:, 1], color=trajColor)
ax.scatter(xSup[0, 0], xSup[0, 1], s=20, c=trajColor,
           marker='s', edgecolors='face')
ax.scatter(xSup[-1, 0], xSup[-1, 1], s=20, c=trajColor,
           marker='>', edgecolors='face')
# # Plot isochron
# ax.plot(theta-np.pi+angDecomp, Rpo * np.exp((theta-np.pi)/beta), linewidth=2)
# ax.text(np.pi/9+angDecomp, Rpo*2.01, r'$I(p)$', fontsize=ergoPlot.fs_latex, color='b')
# # Plot e0 (flow)
# ax.quiver([angDecomp], [Rpo], [-np.sin(angDecomp)], [np.cos(angDecomp)],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp + np.pi/6, Rpo*1.2, r'$\vec{e}_0$', fontsize=ergoPlot.fs_latex)
# # Plot e1 (transverse)
# ax.quiver([angDecomp], [Rpo], [np.cos(angDecomp + np.arctan(beta/np.sqrt(mu)))], [np.sin(angDecomp + np.arctan(beta/np.sqrt(mu)))],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp, Rpo*1.5, r'$\vec{e}_1$', fontsize=ergoPlot.fs_latex)
# # Plot state
# ax.scatter(angDecomp, Rpo, s=20, c='k', marker='o')
# ax.text(angDecomp + np.pi/24, Rpo*0.7, r'$p$', fontsize=ergoPlot.fs_latex, color='k')
# Plot fix point
ax.scatter(0., 0., s=20, c='k', marker='o')
ax.text(-np.pi*5/6, Rpo*0.3, r'$x_*$', fontsize=ergoPlot.fs_latex, color='k')
# # Colorbar
# cbar = plt.colorbar(h)
# plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# cbar.set_ticks(np.arange(-np.pi, np.pi*1.1, np.pi / 2))
# cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
# Grid
ax.set_rlim(0., rmax)
ax.set_thetagrids(thetaticks, labels=[r'$-\pi/2$', '', r'$\pi/2$', r'$\pi$'])
# Ticks
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# Save
fig.savefig('plot/artMixingDistSup.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# Second eigenvector
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# Plot eigenvector phase
h = ax.contourf(THETA-np.pi, R, phaseSup, levelsPhase, cmap=cm.RdBu_r)
# Plot isochron
ax.plot(theta-np.pi+angDecomp, Rpo * np.exp((theta-np.pi)/beta), linewidth=2)
ax.text(np.pi/9+angDecomp, Rpo*2.01, r'$I(p)$', fontsize=ergoPlot.fs_latex, color='b')
# Plot limitcycle
ax.plot(theta, np.ones(theta.shape) * Rpo, '--k', linewidth=2)
ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8)
ax.set_rgrids(rticks, labels=[r'$\Gamma$'], angle=260)
# Plot e0 (flow)
ax.quiver([angDecomp], [Rpo], [-np.sin(angDecomp)], [np.cos(angDecomp)],
          scale=6, width=0.005, headwidth=hw, zorder=10)
ax.text(angDecomp + np.pi/6, Rpo*1.2, r'$\vec{e}_0$', fontsize=ergoPlot.fs_latex)
# Plot e1 (transverse)
ax.quiver([angDecomp], [Rpo], [np.cos(angDecomp + np.arctan(beta/np.sqrt(muSup)))],
          [np.sin(angDecomp + np.arctan(beta/np.sqrt(muSup)))],
          scale=6, width=0.005, headwidth=hw, zorder=10)
ax.text(angDecomp, Rpo*1.5, r'$\vec{e}_1$', fontsize=ergoPlot.fs_latex)
# Plot state
ax.scatter(angDecomp, Rpo, s=20, c='k', marker='o')
ax.text(angDecomp + np.pi/24, Rpo*0.7, r'$p$', fontsize=ergoPlot.fs_latex, color='k')
# Plot fix point
ax.scatter(0., 0., s=20, c='k', marker='o')
ax.text(-np.pi*5/6, Rpo*0.3, r'$x_*$', fontsize=ergoPlot.fs_latex, color='k')
# Colorbar
cbar = plt.colorbar(h)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
cbar.set_ticks(np.arange(-np.pi, np.pi*1.1, np.pi / 2))
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
# Grid
ax.set_rlim(0., rmax)
ax.set_thetagrids(thetaticks, labels=[r'$-\pi/2$', '', r'$\pi/2$', r'$\pi$'])
# Ticks
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# Save
fig.savefig('plot/artMixingEigVecSup.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)


