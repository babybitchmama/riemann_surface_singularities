import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import trig

# Save the state at every moment in time, and then plot at end
# caluclate a geodesic at each moment, see how it changes with the curvature
# Jost 2.3 up to curvature?? Fubini-Study metric

# beta in [0, 1]
BETA = 0.5

# rho determines the target scalar curvature (though currently in a confusing way)
RHO = 0.5

rMAX = 1.0

tMAX = 10.0

# Get radii in [0,1]
a=0.1
Nr=60
Ntheta=6

rVals = np.linspace(0.0, rMAX, Nr)

# Get theta from [-pi, pi]
theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=True)

# Calculate radial and angular steps
dr = rVals[1] - rVals[0]
dtheta = theta[1] - theta[0]

# Stable dt for no diffusion dependent on theta
#FIXME will update if we extend to non-radially symmetric
dt = (dr**2) / (4*a) / 2
tVals = int(tMAX / dt) + 1

# Background metric
uBackground = []
for r in rVals:
    uBackground.append(np.full(shape=(Ntheta), fill_value=np.exp(r ** RHO) * r ** (2 * BETA - 2)))
uBackground = np.array(uBackground)

def metric_laplacian(f,g):
    lapf=[]
    for i in range(1, Nr - 1):
        radius = rVals[i]
        #Second derivative
        f_rr = (f[i+1, :] - 2*f[i, :] + f[i-1, :]) / (dr ** 2)

        # First derivative
        f_r = (f[i, :] - f[i-1, :]) / dr

        # Apply to next slice
        lapf.append((f_rr + f_r / rVals[i]) / g[i,:])
    return np.array(lapf)

# First order extrapolation on the r=0 end
def left_extrapolate(f):
    firstRow = f[0] + (f[0] - f[1])
    return np.concatenate((firstRow[None,:], f), axis=0)

# Calculates curvature
def measure_curvature(g):
    return metric_laplacian(np.log(g) / 2, g)

# Integrates a function with respect to metric g
def integrate(f, g):
    return np.sum(
        np.array([g[i] * f[i] * rVals[i] * dr * dtheta for i in range(1,Nr-2)])
        )

# Measures average curvature
def average_curvature(g):
    curvature = measure_curvature(g)
    curvature = left_extrapolate(curvature[1:])
    return integrate(curvature, g) / integrate(np.full((Nr-2),1), g)

def sim_in_polar(t=tMAX):
    # Background scalar curvature
    KBackground = measure_curvature(uBackground)
    KBackground = left_extrapolate(KBackground[1:])

    # Initialize u, where u[t, i, j] is the temp at time t, radius i, angle j
    u = np.zeros((tVals, Nr, Ntheta), dtype=float)


    # Set init condition
    u[0, :, :] = 1

    # Set boundary condition
    u[0] = set_boundary(u[0])

    # The model: updates for each time step t
    for n in range(tVals - 1):
        KAvg = average_curvature(u[n] * uBackground)
        #print(KAvg)

        # Update excludes endpoints in rVals
        u[n+1, 1:-1] = u[n, 1:-1] + dt * (metric_laplacian(u[n,:,:],uBackground) + KAvg * u[n,1:-1,:] - KBackground)

        # Expensive...can we update in place?
        u[n+1] = set_boundary(u[n+1])

        # First order extrapolation to update r=0 values
        u[n+1, 0, :] = u[n+1, 1, :] + (u[n+1, 1, :] - u[n+1, 2, :])

    print(measure_curvature(u[-1] * uBackground))

    R, TH = np.meshgrid(rVals, theta, indexing="ij")

    phi = TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)


    #Multiply uBackground by u to get the metric g_t
    gMetric = uBackground * u

    #Remove singularity by duplicating next value
    gMetric[:,0] = gMetric[:,1] + (gMetric[:,1] - gMetric[:,2])

    # Plot the sim
    plot_side_by_side_with_slider(u, gMetric, X, Y, dt)


def set_boundary(w:np.ndarray):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = 1
    return l


def plot_side_by_side_with_slider(u1, u2, X, Y, dt, title1="u", title2="g"):
    tVals = u1.shape[0]
    fig = plt.figure(figsize=(12, 6))
    
    # Subplot 1
    ax1 = fig.add_subplot(121, projection="3d")
    zmin1 = float(u1.min())
    zmax1 = float(u1.max())
    ax1.set_zlim(zmin1, zmax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel(title1)

    # Subplot 2
    ax2 = fig.add_subplot(122, projection="3d")
    zmin2 = float(u2.min())
    zmax2 = float(u2.max())
    ax2.set_zlim(zmin2, zmax2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel(title2)

    fig.subplots_adjust(bottom=0.18)

    k0 = 0
    surf1 = ax1.plot_surface(X, Y, u1[k0], cmap="jet", vmin=zmin1, vmax=zmax1, shade=True)
    ax1.set_title(f"{title1} at t = {k0*dt:.4f} s")
    
    surf2 = ax2.plot_surface(X, Y, u2[k0], cmap="jet", vmin=zmin2, vmax=zmax2, shade=True)
    ax2.set_title(f"{title2} at t = {k0*dt:.4f} s")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(slider_ax, "time index k", 0, tVals - 1, valinit=k0, valstep=1)

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