import numpy as np
import ergoPlot

# Control parameters
gam = 4.8
mu = 1.
#beta = 0.
beta = 0.8
Rpo = np.sqrt(mu)
omega = gam - beta * mu
T = 2*np.pi / omega
dim = 2

# Constant radial forcing field
sr = 1.
V1 = np.array([0, sr])

# Polar coordinates
nth = 1000
nr = 1000
rmax = 2 * Rpo
theta = np.linspace(0., 2*np.pi, nth)
r = np.linspace(0.01, rmax, nr)

# Limit cycle
Gamma = np.concatenate((theta, np.ones((nth,))*Rpo), axis=0).reshape(dim, nth)
# Vector field in cartesian coordinates
def V0Cart(x):
    (x, y) = (x[0], x[1])
    return np.array([(mu - (x**2 + y**2)) * x - (gam - beta * (x**2 + y**2)) * y,
                     (gam - beta * (x**2 + y**2)) * x + (mu - (x**2 + y**2)) * y])
def integrateSHE(x0, dt, V1):
    (theta0, r0) = (x0[0], x0[1])
    theta = np.mod(theta0 + dt * (gam - beta * r0**2) \
                   + np.sqrt(dt) * V1[0] * np.random.normal(0, 1), 2*np.pi)
    r = r0 + dt * (mu*r0 - r0**3) \
        + np.sqrt(dt) * V1[1] * np.random.normal(0, 1)
    return (theta, r)
# Flow in asymptotic phase phi coordinate
def flow(x, t):
    (phi, r) = (x[0], x[1])
    return (np.mod(phi + omega * t, 2*np.pi),
            Rpo * (1 + (Rpo**2 / r**2 - 1) * e**(-2*mu*t))**(-1./2))
# Flow of the forcing field in polar coordinates
def flowV1(x, t):
    (theta, r) = (x[0], x[1])
    return (np.mod(theta + V1[0] * t, 2* np.pi), r + V1[1] * t)

# r, theta -> x, y
def polar2Cart(x):
    (theta, r) = (x[0], x[1])
    return r * np.array([np.cos(theta), np.sin(theta)])
# r, theta -> r, phi
def polar2Phase(x):
    (theta, r) = (x[0], x[1])
    if type(theta) == np.ndarray:
        phi = np.empty(theta.shape)
        phi[r < 1.e-6] = theta[r < 1.e-6]
        phi[r >= 1.e-6] = theta[r >= 1.e-6] - beta * np.log(r[r >= 1.e-6] / Rpo)
    else:
        if r < 1.e-6:
            phi = theta
        else:
            phi = theta - beta * np.log(r / Rpo)
    return  np.array([phi, r])
# r, phi -> r, theta
def phase2Polar(x):
    (phi, r) = (x[0], x[1])
    if type(phi) == np.ndarray:
        theta = np.empty(phi.shape)
        theta[r < 1.e-6] = phi[r < 1.e-6]
        theta[r >= 1.e-6] = phi[r >= 1.e-6] + beta * np.log(r[r >= 1.e-6] / Rpo)
    else:
        if r < 1.e-6:
            theta = phi
        else:
            theta = phi + beta * np.log(r / Rpo)
    return np.array([theta, r])

# Length of flow integration
eps = np.pi / 3 / omega
# Phase at of first isochron
phi0 = np.pi/3
# Initial point (on the limit cycle)
x0 = np.array([phi0, Rpo])

# Allocate time and trajectory arrays
# Number of vertices, including initial condition
nPoints = 5
t0 = 0.
nt = 1000
tnt = (nPoints - 1) * nt + 1
time = np.linspace(t0, 4*eps, tnt)
dt = time[1] - time[0]
traj = np.empty((dim, tnt))
traj[:, 0] = x0

# Apply Lie bracket by composition of the flows
## Apply the flow forward
for k in np.arange(1, nt+1):
    traj[:, k] = phase2Polar(flow(polar2Phase(traj[:, k-1]), dt))
## Apply the forcing flow forward
for k in np.arange(nt+1, 2*nt+1):
    traj[:, k] = flowV1(traj[:, k-1], dt)
## Apply the flow backward
for k in np.arange(2*nt+1, 3*nt+1):
    traj[:, k] = phase2Polar(flow(polar2Phase(traj[:, k-1]), -dt))
## Apply the forcing flow backward
for k in np.arange(3*nt+1, 4*nt+1):
    traj[:, k] = flowV1(traj[:, k-1], -dt)

# Get isochrons at images of flows
isochron = np.empty((nr, 2, nPoints-1))
for p in np.arange(nPoints-1):
    point = traj[:, p*nt]
    isochron[:, :, p] = (phase2Polar(np.concatenate((polar2Phase(point)[0]*np.ones((nr,)), r), 0).reshape(dim, nr))).T

# Get forcing integral curves at images of flows
curvesV1 = np.empty((nr, 2, nPoints-1))
for p in np.arange(nPoints-1):
    point = traj[:, p*nt]
    curvesV1[:, :, p] = (np.concatenate((point[0]*np.ones((nr,)), r), 0).reshape(dim, nr)).T
    
# Plot
# Configure
rticks = np.arange(Rpo, rmax, Rpo)
thetaticks = np.array([-90, 0., 90, 180])
hw = 5
c_iso = 'r'
ls_iso = '-'
lw_iso = 1
c_isobeta0 = 'b'
c_curvesV1 = 'b'
ls_curvesV1 = '--'
lw_curvesV1 = 1
ls_isobeta0 = '-'
ls_traj = '-'
c_traj = 'k'
fs_point = 'x-large'
fs_iso = 'x-large'
fs_vec = 'x-large'
figsize = [10, 10]
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='polar')
# Plot limit cycle
ax.plot(Gamma[0], Gamma[1], '--k', linewidth=2)
#ax.quiver([-np.pi/2], [Rpo], [1.], [0.], scale=50, width=0.01, headwidth=20., headlength=8)
# Plot isochrons
for p in np.arange(nPoints - 1):
    plt.plot(isochron[:, 0, p], isochron[:, 1, p],
             linestyle=ls_iso, color=c_iso, linewidth=lw_iso)
# Plot integral curves of forcing
for p in np.arange(nPoints - 1):
    plt.plot(curvesV1[:, 0, p], curvesV1[:, 1, p],
             linestyle=ls_curvesV1, color=c_curvesV1, linewidth=lw_curvesV1)
# Plot trajectory
plt.plot(traj[0], traj[1], linestyle=ls_traj, color=c_traj, linewidth=2)
# Plot arrows on trajectories
vect = np.array([V0Cart(polar2Cart(traj[:, nt/2])),
                 [V1[1] * np.cos(traj[0, nt+nt/2]), V1[1] * np.sin(traj[0, nt+nt/2])],
                 -V0Cart(polar2Cart(traj[:, 2*nt + nt/2])),
                 [-V1[1] * np.cos(traj[0, 3*nt+nt/2]), -V1[1] * np.sin(traj[0, 3*nt+nt/2])]])
for p in np.arange(nPoints - 1):
    point = traj[:, p*nt + nt/2]
    v = vect[p].copy()
    v /= np.sqrt(np.sum(v**2))
    v *= 120
    ax.quiver(point[0], point[1], v[0], v[1], units='dots',
              scale=10., width=10, headwidth=20., headlength=10)
# Label flows
p = 0
point = traj[:, p*nt + nt/2]
plt.text(point[0]+np.pi/25, point[1]*0.82, r'$S_\epsilon$',
         fontsize=fs_point, color='k')
p = 1
point = traj[:, p*nt + nt/2]
plt.text(point[0]+np.pi/15, point[1]*1.07, r'$S^{V_1}_\epsilon$',
         fontsize=fs_point, color='k')
p = 2
point = traj[:, p*nt + nt/2]
plt.text(point[0] + np.pi/40, point[1]*1.1, r'$S_{-\epsilon}$',
         fontsize=fs_point, color='k')
p = 3
point = traj[:, p*nt + nt/2]
plt.text(point[0] - np.pi/40, point[1]*0.95, r'$S^{V_1}_{-\epsilon}$',
         fontsize=fs_point, color='k')
    
# Plot Lie bracket
p0 = traj[:, 0]
pf = traj[:, (nPoints-1)*nt]
ax.annotate('', xy=pf, xytext=p0,
            arrowprops=dict(facecolor='line', color=c_iso, width=2, headwidth=10),)
# Plot forcing vector at initial point
pf = flowV1(p0, eps)
ax.annotate('', xy=pf, xytext=p0,
            arrowprops=dict(facecolor='line', color=c_curvesV1, width=2, headwidth=10),)
# Plot legend
ax.annotate('', xy=[np.pi/5, 1.45], xytext=[np.pi/5, 1.2],
            arrowprops=dict(facecolor='line', color=c_curvesV1, width=2, headwidth=10),)
plt.text(np.pi/6, 1.5, r'$V_1(p)$', fontsize=ergoPlot.fs_latex)
ax.annotate('', xy=[np.pi/6.2, 1.35], xytext=[np.pi/6.5, 1.11],
            arrowprops=dict(facecolor='line', color=c_iso, width=2, headwidth=10),)
plt.text(np.pi/8, 1.38, r'$[V_0, V_1](p)$', fontsize=ergoPlot.fs_latex)

# Plot points
for p in np.arange(nPoints):
    point = traj[:, p * nt]
    ax.scatter(point[0], point[1], s=20, c='k', marker='o')
# Label first point
ax.text(traj[0, 0] - np.pi/25, traj[0, 1]*0.85, r'$p$' % k,
        fontsize=fs_point, color='k')

# Set ticks
ax.set_rlim(0., rmax)
ax.set_rgrids(rticks, labels=[r'$\Gamma$'], angle=260)
#ax.set_thetagrids(thetaticks, labels=[r'$-\pi/2$', '', r'$\pi/2$', r'$\pi$'])
ax.set_thetagrids([], labels=[])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('plot/isochronsLie_beta%02d.%s' % (int(beta * 10), ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

# Plot time series
L = 10 * T
dts = 1.e-2
time_ts = np.arange(0, L + dts, dts)
nts = time_ts.shape[0]
ts = np.empty((dim, nts))
ts[:, 0] = [0., 0.]
for k in np.arange(1, nts):
    ts[:, k] = integrateSHE(ts[:, k - 1], dts, V1*1)
tsPhase = polar2Phase(ts)

# Plot time series
ls_ts = '-'
c_ts = 'k'
xticks = np.arange(0, L*1.001, T)
nxticks = xticks.shape[0]
xticklabels = ['']*nxticks
xticklabels[0] = r'$0$'
xticklabels[1] = r'$T$'
for tk in np.arange(1, nxticks):
    xticklabels[tk] = r'$%dT$' % tk
yticks = [-np.pi, -np.pi/2, 0., np.pi/2, np.pi]
yticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(time_ts, tsPhase[0] - np.pi, linestyle=ls_ts, color=c_ts)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(yticks[0], yticks[-1])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('plot/phase_beta%02d.%s' % (int(beta * 10), ergoPlot.figFormat),
            bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
