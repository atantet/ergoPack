import numpy as np
import ergoPlot

omega = 4.
mu = 1.
eps = 0.4
beta = 0.8
Rpo = np.sqrt(mu)
D = eps**2 * (1 + beta**2)

# Polar coordinates
nt = 1000
nr = 1000
rmax = 3 * Rpo
theta = np.linspace(0., 2*np.pi, nt)
r = np.linspace(0.01, rmax, nr)

# Limit cycle
Gamma = np.concatenate((theta, np.ones((nt,))*Rpo), axis=0).reshape(2, nt)
# Flow of points on isochron
def flow(x, t):
    (phi, r) = (x[0], x[1])
    return (phi + omega * t,
            Rpo * (1 + (Rpo**2 / r**2 - 1) * e**(-2*mu*t))**(-1./2))
# Asymptotic phase
def polar2Phase(x):
    (theta, r) = (x[0], x[1])
    return np.array([theta - beta * np.log(r / Rpo), r])
def phase2Polar(x):
    (phi, r) = (x[0], x[1])
    return np.array([phi + beta * np.log(r / Rpo), r])
# Point at cross-section isochron 1
phi0 = np.pi/3
riso = np.array([Rpo, Rpo * 1.8, Rpo * 2.5])
nPoints = riso.shape[0]
xiso = np.empty((nPoints, 2))
for p in np.arange(nPoints):
    xiso[p] = phase2Polar([phi0, riso[p]])
# Trajectories
(tmin, tmax) = (phi0 / omega, (4*np.pi/3 + phi0) / omega)
time = np.linspace(tmin, tmax, 120)
nt = time.shape[0]
traj = np.empty((nPoints, 2, nt))
for p in np.arange(nPoints):
    traj[p, :, 0] = xiso[p]
    for k in np.arange(1, nt):
        traj[p, :, k] = phase2Polar(flow(polar2Phase(traj[p, :, 0]), time[k] - time[0]))
# Isochrons
folio = np.empty((nr, 2, nt))
for k in np.arange(nt):
    folio[:, :, k] = (phase2Polar(np.concatenate((omega * time[k] * np.ones((nr,)), r), 0).reshape(2, nr))).T
# Isochrons for beta = 0
folioBeta0 = np.empty((nr, 2, nt))
for k in np.arange(nt):
    folioBeta0[:, :, k] = (np.concatenate((omega * time[k] * np.ones((nr,)), r), 0).reshape(2, nr)).T
# Plotted isochrons
niso = 3
kisoPlot = np.round(np.linspace(0, nt-1, 3)).astype(int)
tisoPlot = time[k]
labelIso = [r'$I^\beta(p)$', r'$I^\beta(S_{t_1}p)$', r'$I^\beta(S_{t_2}p)$']
    
# Tangent vectors to isochron 1
e0 = np.array([[xiso[0, 0]], [xiso[0, 1]], [-np.sin(xiso[0, 0])], [np.cos(xiso[0, 1])]])
e1rot = np.matrix([[np.cos(xiso[0, 0]), -np.sin(xiso[0, 0])], [np.sin(xiso[0, 0]), np.cos(xiso[0, 0])]]) \
        * np.matrix([[beta/ Rpo], [1.]]) / np.sqrt(1 + (beta/Rpo)**2)
#e1 = np.array([[xiso[0, 0]], [xiso[0, 1]], e1rot[0], e1rot[1]])
#e1 = np.array([[xiso[0, 0]], [xiso[0, 1]],
#               [np.cos(xiso[0, 0] * beta/np.sqrt(mu))],
#               [np.sin(xiso[0, 0] * beta/np.sqrt(mu))]])
e1 = np.array([[xiso[0, 0]], [xiso[0, 1]], [np.cos(xiso[0, 0] + np.arctan(beta/np.sqrt(mu)))],
               [np.sin(xiso[0, 0] + np.arctan(beta/np.sqrt(mu)))]])
e10 = np.array([[xiso[0, 0]], [xiso[0, 1]], [np.cos(xiso[0, 0])], [np.sin(xiso[0, 0])]])

# Plot
# Configure
rticks = np.arange(Rpo, rmax, Rpo)
thetaticks = np.array([-90, 0., 90, 180])
hw = 5
c_iso = 'r'
ls_iso = '-'
lw_iso = 1
c_isobeta0 = 'b'
ls_isobeta0 = '-'
vecScale = 3. * rmax
fs_point = 'x-large'
fs_iso = 'x-large'
fs_vec = 'x-large'
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# Plot limitcycle
ax.plot(Gamma[0], Gamma[1], '-k', linewidth=1)
ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8)
# Plot isochrons
for p in np.arange(niso):
    iso = kisoPlot[p]
    ax.plot(folio[:, 0, iso], folio[:, 1, iso],
            linestyle=ls_iso, color=c_iso, linewidth=lw_iso)
p = 0
iso = kisoPlot[p]
ax.text(folio[-1, 0, iso]-np.pi/20, folio[-1, 1, iso]*0.9, labelIso[p],
        fontsize=fs_iso, color=c_iso)
p = 1
iso = kisoPlot[p]
ax.text(folio[-1, 0, iso], folio[-1, 1, iso]*1., labelIso[p],
        fontsize=fs_iso, color=c_iso)
p = 2
iso = kisoPlot[p]
ax.text(folio[-1, 0, iso], folio[-1, 1, iso]*0.65, labelIso[p],
        fontsize=fs_iso, color=c_iso)
# Plot isochron 1 with beta = 0
ax.plot(folioBeta0[:, 0, 0], folioBeta0[:, 1, 0],
        linestyle=ls_isobeta0, color=c_isobeta0, linewidth=1)
ax.text(folioBeta0[-1, 0, 0] - np.pi/40, folioBeta0[-1, 1, 0]*0.75, r'$I^0(p)$',
        fontsize=fs_iso, color=c_isobeta0)
# Plot e0 (flow)
ax.quiver(e0[0], e0[1], e0[2], e0[3], scale=vecScale,
          width=0.005, headwidth=hw, zorder=10)
ax.text(xiso[0, 0] + np.pi/5, xiso[0, 1]*1.3, r'$\vec{e}_0$', fontsize=fs_vec)
# Plot e1 (transverse)
ax.quiver(e1[0], e1[1], e1[2], e1[3], scale=vecScale,
          width=0.005, headwidth=hw, zorder=10, color=c_iso)
ax.text(xiso[0, 0] + np.pi/10, xiso[0, 1]*1.6, r'$\vec{e}^\beta_1$', fontsize=fs_vec, color=c_iso)
# Plot e1 (transverse) for beta = 0
ax.quiver(e10[0], e10[1], e10[2], e10[3], scale=vecScale,
          width=0.005, headwidth=hw, zorder=10, color=c_isobeta0)
ax.text(xiso[0, 0] - np.pi/15, xiso[0, 1]*1.5, r'$\vec{e}^0_1$',
        fontsize=fs_vec, color=c_isobeta0)
# Plot points on first isochron
for p in np.arange(nPoints):
    ax.scatter(xiso[p, 0], xiso[p, 1], s=20, c='k', marker='o')
ax.text(xiso[0, 0] - np.pi/25, xiso[0, 1]*1.05, r'$p$' % k,
        fontsize=fs_point, color='k')
# Plot trajectories
for p in np.arange(nPoints):
    plt.plot(traj[p, 0], traj[p, 1], '--k', linewidth=1)
    # Plot trajectory intersection with isochrons
    for iso in kisoPlot:
        ax.scatter(traj[p, 0, iso], traj[p, 1, iso], s=20, c='k', marker='o')

# Set ticks
ax.set_rlim(0., rmax)
ax.set_rgrids(rticks, labels=[r'$\Gamma$'], angle=260)
ax.set_thetagrids(thetaticks, labels=[r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('plot/isochrons.%s' % ergoPlot.figFormat,
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
