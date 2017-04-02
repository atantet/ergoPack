import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fs_latex = 'xx-large'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.broken_barh([(15, 20)], (0, 100),
               facecolors=[0, 0, 1, 0.3], linewidth=0)
ax.broken_barh([(15, 20)], (0, 3),
               facecolors='b', linewidth=0)
ax.text(20, -6, r'$\chi_E(y)$', fontsize=fs_latex, color='b')
ax.text(15, 19, r'$\chi_{\mathcal{R}^{-1}(E)}(x)$',
        fontsize=fs_latex, color='b')
ax.annotate('',
            xytext=(25, 1),      # theta, radius
            xy=(25, 16),   # theta, radius
            xycoords='data',
            textcoords='data',
            arrowprops=dict(arrowstyle='simple, tail_width=0.2, head_width=0.7', fc='k', ec="none"))

ax.broken_barh([(60, 20)], (0, 100),
               facecolors=[1, 0, 0, 0.5], linewidth=0)
ax.broken_barh([(60, 20)], (0, 3),
               facecolors='r', linewidth=0)
ax.text(65, -6, r'$\chi_F(y)$', fontsize=fs_latex, color='r')
ax.text(60, 19, r'$\chi_{\mathcal{R}^{-1}(F)}(x)$',
        fontsize=fs_latex, color='r')
ax.annotate('',
            xytext=(70, 1),      # theta, radius
            xy=(70, 16),   # theta, radius
            xycoords='data',
            textcoords='data',
            arrowprops=dict(arrowstyle='simple, tail_width=0.2, head_width=0.7', fc='k', ec="none"))


#el = Ellipse((25, 55), 30, 60, angle=-20, facecolor='r', alpha=0.3)
#ax.add_artist(el)
ax.annotate('',
            xytext=(60.5, 78),      # theta, radius
            xy=(40, 78),   # theta, radius
            xycoords='data',
            textcoords='data',
            arrowprops=dict(arrowstyle='simple, tail_width=0.3, head_width=1',
                            fc='k', ec="none",
                            connectionstyle="arc3,rad=0.2"))
ax.text(26, 85, r'$E[\chi_{\mathcal{R}^{-1}(F)}(\phi(t, \cdot)x)]$',
        fontsize=fs_latex)
#ax.text(15, 50, r'$\mathbb{P}_\mu(X(t, x) \in \mathcal{R}^{-1}(F) | x \in \mathcal{R}^{-1}(E))$', fontsize=fs_latex)
ax.text(23, 69, r'$\left<\mathfrak{L}_t^\mathcal{R} \chi_E, \chi_F \right>_\mathfrak{m}$', fontsize=fs_latex, rotation=90, color=[0.3, 0, 0.6])

x = np.linspace(0, 100, 1000)
nx = x.shape[0]
y = x
ny = y.shape[0]
N = nx * ny
(X, Y) = np.meshgrid(x, y)
XY = np.zeros((2, N))
XY[0] = X.flatten()
XY[1] = Y.flatten()

nlev = 5
theta = np.pi / 3
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
muf = np.array([25, 55])
sigf = np.array([[30, 0], [0, 60]])
sigf = np.dot(sigf, R)
dsigf = np.linalg.det(sigf)
isigf = np.linalg.inv(sigf)
d = muf.shape[0]
XYmMuf = XY - np.tile(muf, (N, 1)).T
f = np.empty((N,))
for k in np.arange(N):
    f[k] = 1. / np.sqrt((2*np.pi)**d*dsigf) \
           * np.exp(-np.dot(XYmMuf[:, k], np.dot(isigf, XYmMuf[:, k]))/2)
f = f.reshape(nx, ny)
vmin = 5.e-4
f[f < vmin] = np.nan
levelsf = np.linspace(f.min(), f.max(), nlev)
ax.contourf(X, Y, f, nlev, cmap=cm.Reds, alpha=0.4)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$Y$', fontsize=fs_latex)
ax.set_ylabel(r'$H / Y$', fontsize=fs_latex)
ax.xaxis.set_label_coords(0.93, -0.03)
ax.yaxis.set_label_coords(-0.01, 0.88)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axhline(linewidth=3, color="k")
ax.axvline(linewidth=3, color="k")
ax.text(90, 85, r'$H$', fontsize=fs_latex)
plt.show()
fig.savefig('coarseTransition.png', dpi=300, bbox_inches='tight')
