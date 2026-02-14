import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import trig
import tqdm

# Save the state at every moment in time, and then plot at end
# caluclate a geodesic at each moment, see how it changes with the curvature
# Jost 2.3 up to curvature?? Fubini-Study metric

# beta in [0, 1]
BETA = 0.5

# rho determines the target scalar curvature (though currently in a confusing way)
RHO = 0.5

yMIN = -1.0
yMAX = 1.0
tMAX = 10.0
a = 2.0
ySAMPLES = 40
thetaSAMPLES = 2

yVals = np.linspace(yMIN, yMAX, ySAMPLES)

# Get theta from [-pi, pi]
theta = np.linspace(-np.pi, np.pi, thetaSAMPLES, endpoint=True)

# Calculate radial and angular steps
dy = yVals[1] - yVals[0]
dtheta = theta[1] - theta[0]

# Stable dt for no diffusion dependent on theta
#FIXME will update if we extend to non-radially symmetric
dt = (dy**2) / (4*a) / 2
tSamples = int(tMAX / dt) + 1

# Initial metric coefficient -- 1 gives rise to flat conical metric
def ff(x):
    return 1

# Background metric
muBackground = (ff(yVals) * np.exp((BETA - 1) * yVals))[:,np.newaxis] * np.full((thetaSAMPLES),1)

# Calculates laplacian.  Output is missing both y endpoints.
# This version uses broadcasting to perform the operation in C, which is about twice as fast as the old version below.
def standard_laplacian(f):
    f_yy = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dy ** 2)
    return np.exp(-2 * yVals[1:-1])[:, np.newaxis] * f_yy

# OLD VERSION
# def standard_laplacian(f):
#     lapf=[]
#     for i in range(1, ySAMPLES - 1):
#         radius = yVals[i]
#         #Second derivative
#         f_yy = (f[i+1, :] - 2*f[i, :] + f[i-1, :]) / (dy ** 2)

#         # Apply to next slice
#         lapf.append(np.exp(-2 * yVals[i]) * f_yy)
#     return np.array(lapf)

# Calculates laplacian with respect to a metric coefficient mu
def metric_laplacian(f, mu):
    return standard_laplacian(f) / (mu[1:-1] ** 2)

# Calculates curvature
def measure_curvature(mu):
    return -metric_laplacian(np.log(mu), mu)

# Integrates a function
def integrate(f, mu):
    return np.sum(
        np.array([mu[i] ** 2 * np.exp(2 * BETA * yVals[i]) * f[i] * yVals[i] * dy * dtheta for i in range(0,ySAMPLES-2)])
        )

# First order extrapolation on the r=0 end
def left_extrapolate(f):
    firstRow = f[0] + (f[0] - f[1])
    return np.concatenate((firstRow[None,:], f), axis=0)

# Measures average curvature
def average_curvature(mu):
    curvature = measure_curvature(mu)
    return integrate(curvature, mu) / integrate(np.full((ySAMPLES-2),1), mu)

def sim_in_polar(t=tMAX):
    # Background scalar curvature
    KBackground = measure_curvature(muBackground)

    # Initialize u, where u[t, i, j] is the temp at time t, radius i, angle j
    u = np.ones((tSamples, ySAMPLES, thetaSAMPLES), dtype=float)


    # Set init condition
    u[0, :, :] = 1.0
    u[:, -1, :] = 1.1


    # The model: updates for each time step t
    for n in tqdm.tqdm(range(tSamples-1)):
        KAvg = average_curvature(u[n] * muBackground)

        # Update excludes y endpoints.  I'm struggling to actually uniformize curvature here.  I'll probably need to work with honest Ricci flow rather than separating muBacground from u.
        u[n+1, 1:-1] = u[n, 1:-1] + dt * (metric_laplacian(u[n], muBackground) + KAvg * muBackground[1:-1] - KBackground)

        #u[n+1, 1:-1] = u[n, 1:-1] + dt * (metric_laplacian(u[n], muBackground) + 0.988 * KAvg * u[n,1:-1] - KBackground)

        # First order extrapolation to update r=0 values
        u[n+1, 0, :] = u[n+1, 1, :]# + (u[n+1, 1, :] - u[n+1, 2, :])

    #print(u[-1])
    print(measure_curvature(u[-1] * muBackground))

    Y, TH = np.meshgrid(yVals, theta, indexing="ij")

    #Multiply muBackground by u to get the metric g_t
    gMetric = muBackground * u * np.exp(- yVals)[:, np.newaxis]

    #Remove singularity by duplicating next value
    gMetric[:,0,:] = gMetric[:,1,:] + (gMetric[:,1,:] - gMetric[:,2,:])

    # Plot the sim
    plot_side_by_side_with_slider(u, gMetric, Y, TH, dt)


def set_boundary(w:np.ndarray):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = 1.0
    return l


def plot_side_by_side_with_slider(u1, u2, X, Y, dt, title1="u", title2="g"):
    tSamples = u1.shape[0]
    fig = plt.figure(figsize=(12, 6))
    
    # Subplot 1
    ax1 = fig.add_subplot(121, projection="3d")
    zmin1 = float(u1.min())
    zmax1 = float(u1.max())
    ax1.set_zlim(zmin1, zmax1)
    ax1.set_xlabel("y")
    ax1.set_ylabel("theta")
    ax1.set_zlabel(title1)

    # Subplot 2
    ax2 = fig.add_subplot(122, projection="3d")
    zmin2 = float(u2.min())
    zmax2 = float(u2.max())
    ax2.set_zlim(zmin2, zmax2)
    ax2.set_xlabel("y")
    ax2.set_ylabel("theta")
    ax2.set_zlabel(title2)

    fig.subplots_adjust(bottom=0.18)

    k0 = 0
    surf1 = ax1.plot_surface(X, Y, u1[k0], cmap="jet", vmin=zmin1, vmax=zmax1, shade=True)
    ax1.set_title(f"{title1} at t = {k0*dt:.4f} s")
    
    surf2 = ax2.plot_surface(X, Y, u2[k0], cmap="jet", vmin=zmin2, vmax=zmax2, shade=True)
    ax2.set_title(f"{title2} at t = {k0*dt:.4f} s")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(slider_ax, "time index k", 0, tSamples - 1, valinit=k0, valstep=1)

    def update(val):
        nonlocal surf1, surf2
        k = int(s.val)
        
        surf1.remove()
        surf1 = ax1.plot_surface(X, Y, u1[k], cmap="jet", vmin=zmin1, vmax=zmax1, shade=True)
        ax1.set_title(f"{title1} at t = {k*dt:.4f} s")
        
        surf2.remove()
        surf2 = ax2.plot_surface(X, Y, u2[k], cmap="jet", vmin=zmin2, vmax=zmax2, shade=True)
        ax2.set_title(f"{title2} at t = {k*dt:.4f} s")
        
        fig.canvas.draw_idle()

    s.on_changed(update)
    plt.show()


if __name__ == "__main__":
    sim_in_polar()