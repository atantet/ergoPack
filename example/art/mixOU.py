import numpy as np
import ergoPlot

mu = -2.
omega = 4.
eps = 0.4

xmax = 1.
nx = 100
x = np.linspace(-xmax, xmax, nx)
y = x
(X, Y) = np.meshgrid(x, y)

# xticks =
# yticks = xticks
trajColor = 'b'

# Simulate
dt = 0.001
L = np.abs(1.5 / mu)
time = np.arange(0., L, dt)
nMem = 3
x0Sup = np.array([[np.pi/3, 1.8*Rpo],
                  [np.pi/3, 1.6*Rpo],
                  [np.pi/3, 1.4*Rpo]))
xSup = np.empty((nMem, time.shape[0], x0Sup.shape[0]))
xSup[:, 0] = x0Sup
for mem in np.arange(nMem):
    for t in np.arange(1, time.shape[0]):
        x0Sup[ += dt * np.array([omega - beta*x0Sup[1]**2,
                                muSup*x0Sup[1] - x0Sup[1]**3 + eps**2/(2*x0Sup[1])]) \
                                + np.sqrt(dt)*eps*np.random.normal(size=2)/np.array([x0Sup[1], 1.])
        xSup[t] = x0Sup

# Second eigenvector
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# Plot isochron
# ax.plot(theta-np.pi+angDecomp, Rpo * np.exp((theta-np.pi)/beta), linewidth=2)
# ax.text(np.pi/9+angDecomp, Rpo*2.01, r'$I_{x_0}$', fontsize=ergoPlot.fs_latex, color='b')
# Plot limitcycle
ax.plot(theta, np.ones(theta.shape) * Rpo, '--k', linewidth=2)
ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8)
ax.set_rgrids(rticks, labels=[r'$\Gamma$'], angle=260)
# # Plot e0 (flow)
# ax.quiver([angDecomp], [Rpo], [-np.sin(angDecomp)], [np.cos(angDecomp)],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp + np.pi/6, Rpo*1.2, r'$\vec{e}_0$', fontsize=ergoPlot.fs_latex)
# # Plot e1 (transverse)
# ax.quiver([angDecomp], [Rpo], [np.cos(angDecomp + np.arctan(beta/np.sqrt(muSup)))],
#           [np.sin(angDecomp + np.arctan(beta/np.sqrt(muSup)))],
#           scale=6, width=0.005, headwidth=hw, zorder=10)
# ax.text(angDecomp, Rpo*1.5, r'$\vec{e}_1$', fontsize=ergoPlot.fs_latex)
# # Plot state
# ax.scatter(angDecomp, Rpo, s=20, c='k', marker='o')
# ax.text(angDecomp + np.pi/24, Rpo*0.7, r'$x_0$', fontsize=ergoPlot.fs_latex, color='k')
# # Plot fix point
# ax.scatter(0., 0., s=20, c='k', marker='o')
# ax.text(-np.pi*5/6, Rpo*0.3, r'$x_*$', fontsize=ergoPlot.fs_latex, color='k')
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
fig.savefig('plot/mixDet.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)


