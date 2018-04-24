import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg
import ergoPlot

x01 = np.array([-1.5, -1.8])
x02 = np.array([-1., -2.])
x03 = np.array([-0.7, -2.7])
simCol = '0.4'
simTailWidth = '0.08'
simMarkerSize=5

caseName = 'NA'
transferName = '\mathcal{P}_t'
#flowName = 'S(t)'
flowName = '\psi(t)'

x = np.linspace(-3, 3, 100)
nx = x.shape[0]
y = x
ny = y.shape[0]
N = nx * ny
(X, Y) = np.meshgrid(x, y)
XY = np.zeros((2, N))
XY[0] = X.flatten()
XY[1] = Y.flatten()

muf = np.array([-1, -2])
sigf = np.array([[0.25, 0], [0, 0.25]])
dsigf = np.linalg.det(sigf)
isigf = np.linalg.inv(sigf)

mug = np.array([0, 0])
sigg = np.array([[1, 0], [0, 1]])
dsigg = np.linalg.det(sigg)
isigg = np.linalg.inv(sigg)
d = muf.shape[0]

XYmMuf = XY - np.tile(muf, (N, 1)).T
XYmMug = XY - np.tile(mug, (N, 1)).T

nlev = 4

f = np.empty((N,))
for k in np.arange(N):
    f[k] = 1. / np.sqrt((2*np.pi)**d*dsigf) \
           * np.exp(-np.dot(XYmMuf[:, k], np.dot(isigf, XYmMuf[:, k]))/2)
f = f.reshape(nx, ny)
levelsf = np.linspace(f.min(), f.max(), nlev)

g = np.empty((N,))
for k in np.arange(N):
    g[k] = 1. / np.sqrt((2*np.pi)**d*dsigg) \
           * np.exp(-np.dot(XYmMug[:, k], np.dot(isigg, XYmMug[:, k]))/2)
g = g.reshape(nx, ny)
levelsg = np.linspace(g.min(), g.max(), nlev)

t = 0.5
A = np.array([[-1., 0.3],[0.3, -1.]])
mu = np.array([4., 3.])

ftr = np.empty((N,))
XYmMuftr = np.dot(scipy.linalg.expm(-t*A), XY - np.tile(mu, (N, 1)).T) \
           + np.tile(mu, (N, 1)).T \
           - np.tile(muf, (N, 1)).T
for k in np.arange(N):
    ftr[k] = 1. / np.sqrt((2*np.pi)**d*dsigf) \
           * np.exp(-np.dot(XYmMuftr[:, k], np.dot(isigf, XYmMuftr[:, k]))/2)
ftr /= np.abs(np.linalg.det(scipy.linalg.expm(-t*A)))
levelsftr = np.linspace(ftr.min(), ftr.max(), nlev)
ftr = ftr.reshape(nx, ny)

gtr = np.empty((N,))
XYmMugtr = np.dot(scipy.linalg.expm(t*A), XY - np.tile(mu, (N, 1)).T) \
           + np.tile(mu, (N, 1)).T \
           - np.tile(mug, (N, 1)).T
for k in np.arange(N):
    gtr[k] = 1. / np.sqrt((2*np.pi)**d*dsigg) \
             * np.exp(-np.dot(XYmMugtr[:, k], np.dot(isigg, XYmMugtr[:, k]))/2)
levelsgtr = np.linspace(gtr.min(), gtr.max(), nlev)
gtr = gtr.reshape(nx, ny)

idX01 = np.argmin(((XY - np.tile(x01, (XY.shape[1],1)).T)**2).sum(0))
x01 = XY[:, idX01]
xf1 = XYmMugtr[:, idX01]
idX02 = np.argmin(((XY - np.tile(x02, (XY.shape[1],1)).T)**2).sum(0))
x02 = XY[:, idX02]
xf2 = XYmMugtr[:, idX02]
idX03 = np.argmin(((XY - np.tile(x03, (XY.shape[1],1)).T)**2).sum(0))
x03 = XY[:, idX03]
xf3 = XYmMugtr[:, idX03]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(X, Y, g, np.linspace(g.min(), g.max(), nlev+2)[1:], cmap=cm.Blues)
ax.text(0.4, 0.7, r'$g(x)$', fontsize=ergoPlot.fs_latex)
ax.contour(X, Y, f, nlev, linewidths=2, colors='k', linestyles='--', label=r'$f(x)$')
ax.text(-0.1, -2.5, r'$\rho_0(x)$', fontsize=ergoPlot.fs_latex)
ax.set_axis_off()
plt.plot(x01[0], x01[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
plt.plot(x02[0], x02[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
plt.plot(x03[0], x03[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
fig.savefig('plot/spinupTransfer%s.%s' % (caseName, ergoPlot.figFormat), bbox_inches='tight', dpi=ergoPlot.dpi)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(X, Y, g, np.linspace(g.min(), g.max(), nlev+2)[1:], cmap=cm.Blues)
ax.text(0.4, 0.7, r'$g(x)$', fontsize=ergoPlot.fs_latex)
ax.contour(X, Y, f, nlev, linewidths=2, colors='k', linestyles='--', label=r'$f(x)$')
ax.text(-0.1, -2.5, r'$\rho_0(x)$', fontsize=ergoPlot.fs_latex)
ax.contour(X, Y, ftr, nlev, linewidths=2, colors='k', label=r'$%s \rho_0(x)$' % transferName)
ax.text(1.05, -0.8, r'$%s \rho_0(x)$' % transferName, fontsize=ergoPlot.fs_latex)
ax.arrow(0.3, -2.15, 1.15 - 0.3, -1.05 + 2.15, head_width=0.1, width = 0.02, head_length=0.1,
         fc='k', ec='k')
#ax.text(0.9, -1.1, r'$\mathcal{L}_t \rho_0(x)$', fontsize=ergoPlot.fs_latex)
ax.set_axis_off()
plt.plot(x01[0], x01[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
plt.plot(xf1[0], xf1[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
#ax.text(x01[0]-0.45, x01[1] + 0.2, r'$x_0^{(1)}$', color=simCol, fontsize=ergoPlot.fs_latex)
ax.annotate('', xy=xf1, xycoords='data',
            xytext=(x01[0], x01[1]), textcoords='data',
            size=ergoPlot.fs_latex, color=simCol,
            arrowprops=dict(arrowstyle='simple, tail_width=' + simTailWidth,
                            fc=simCol, ec="none",
                            connectionstyle="arc3,rad=-0.3"))
#ax.text(xf1[0]-0.65, xf1[1]+0.2, r'$\phi(t) x_0^{(1)}$', color=simCol, fontsize=ergoPlot.fs_latex)
plt.plot(x02[0], x02[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
plt.plot(xf2[0], xf2[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
#ax.text(x02[0]-0.45, x02[1] + 0.2, r'$x_0^{(2)}$', color=simCol, fontsize=ergoPlot.fs_latex)
ax.annotate('', xy=xf2, xycoords='data',
            xytext=(x02[0], x02[1]), textcoords='data',
            size=ergoPlot.fs_latex, color=simCol,
            arrowprops=dict(arrowstyle="simple, tail_width=" + simTailWidth,
                            fc=simCol, ec="none",
                            connectionstyle="arc3,rad=-0.2"))
#ax.text(xf2[0]-0.65, xf2[1]+0.2, r'$\phi(t) x_0^{(2)}$', color=simCol, fontsize=ergoPlot.fs_latex)
plt.plot(x03[0], x03[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
plt.plot(xf3[0], xf3[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
#ax.text(x03[0]-0.45, x03[1] + 0.2, r'$x_0^{(3)}$', color=simCol, fontsize=ergoPlot.fs_latex)
ax.annotate('', xy=xf3, xycoords='data',
            xytext=(x03[0], x03[1]), textcoords='data',
            size=ergoPlot.fs_latex, color=simCol,
            arrowprops=dict(arrowstyle="simple, tail_width=" + simTailWidth,
                            fc=simCol, ec="none",
                            connectionstyle="arc3,rad=-0.1"))
#ax.text(xf3[0]-0.65, xf3[1]+0.2, r'$\phi(t) x_0^{(3)}$', color=simCol, fontsize=ergoPlot.fs_latex)
fig.savefig('plot/spinupTransfer%s.%s' % (caseName, ergoPlot.figFormat), bbox_inches='tight', dpi=ergoPlot.dpi)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(X, Y, gtr, np.linspace(gtr.min(), gtr.max(), nlev+2)[1:], cmap=cm.Blues)
ax.text(-2.85, 0.75, r'$g(%s x)$' % flowName, fontsize=ergoPlot.fs_latex)
ax.contour(X, Y, g, np.linspace(g.min(), g.max(), nlev+2)[1:],
           linewidths=1.5, colors='b', linestyles='--', label=r'$g(x)$')
ax.text(0.5, 1.2, r'$g(x)$', fontsize=ergoPlot.fs_latex)
ax.contour(X, Y, f, nlev, linewidths=2, colors='k', label=r'$f(x)$')
ax.text(-0.1, -2.5, r'$\rho_0(x)$', fontsize=ergoPlot.fs_latex)
ax.arrow(0.4, 1.275, -1.7-0.4, 0.85-1.275, head_width=0.1, width = 0.02, head_length=0.1,
         fc='k', ec='k')
ax.set_axis_off()
plt.plot(x01[0], x01[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
# plt.plot(xf1[0], xf1[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
# #ax.text(x01[0]-0.45, x01[1] + 0.2, r'$x_0^{(1)}$', color=simCol, fontsize=ergoPlot.fs_latex)
# ax.annotate('', xy=xf1, xycoords='data',
#             xytext=(x01[0], x01[1]), textcoords='data',
#             size=ergoPlot.fs_latex, color=simCol,
#             arrowprops=dict(arrowstyle='simple, tail_width=' + simTailWidth,
#                             fc=simCol, ec="none",
#                             connectionstyle="arc3,rad=-0.3"))
#ax.text(xf1[0]-0.65, xf1[1]+0.2, r'$\phi(t) x_0^{(1)}$', color=simCol, fontsize=ergoPlot.fs_latex)
plt.plot(x02[0], x02[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
# plt.plot(xf2[0], xf2[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
# #ax.text(x02[0]-0.45, x02[1] + 0.2, r'$x_0^{(2)}$', color=simCol, fontsize=ergoPlot.fs_latex)
# ax.annotate('', xy=xf2, xycoords='data',
#             xytext=(x02[0], x02[1]), textcoords='data',
#             size=ergoPlot.fs_latex, color=simCol,
#             arrowprops=dict(arrowstyle="simple, tail_width=" + simTailWidth,
#                             fc=simCol, ec="none",
#                             connectionstyle="arc3,rad=-0.2"))
#ax.text(xf2[0]-0.65, xf2[1]+0.2, r'$\phi(t) x_0^{(2)}$', color=simCol, fontsize=ergoPlot.fs_latex)
plt.plot(x03[0], x03[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
# plt.plot(xf3[0], xf3[1], marker='o', markerfacecolor=simCol, markersize=simMarkerSize, markeredgecolor="none")
# #ax.text(x03[0]-0.45, x03[1] + 0.2, r'$x_0^{(3)}$', color=simCol, fontsize=ergoPlot.fs_latex)
# ax.annotate('', xy=xf3, xycoords='data',
#             xytext=(x03[0], x03[1]), textcoords='data',
#             size=ergoPlot.fs_latex, color=simCol,
#             arrowprops=dict(arrowstyle="simple, tail_width=" + simTailWidth,
#                             fc=simCol, ec="none",
#                             connectionstyle="arc3,rad=-0.1"))
#ax.text(xf3[0]-0.65, xf3[1]+0.2, r'$\phi(t) x_0^{(3)}$', color=simCol, fontsize=ergoPlot.fs_latex)
fig.savefig('plot/spinupKoopman%s.%s' % (caseName, ergoPlot.figFormat), bbox_inches='tight', dpi=ergoPlot.dpi)
