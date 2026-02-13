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

tMAX = 2.0

def sim_in_polar(a=1.0, t=tMAX, Nr=80, Ntheta=16):

    # Get radii in [0,1]
    r = np.linspace(0.0, rMAX, Nr)
    # Get theta from [-pi, pi]
    theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=True)

    # Calculate radial and angular steps
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    
    # Stable dt for no diffusion dependent on theta
    #FIXME will update if we extend to non-radially symmetric
    dt = (dr**2) / (4*a) / 2
    t_nodes = int(t / dt) + 1

    # Background metric
    uBackground = []
    for rInd in r:
        uBackground.append(np.full(shape=(Ntheta), fill_value=np.exp(rInd ** (RHO)) * rInd ** (2 * BETA - 2)))
    uBackground = np.array(uBackground)

    def laplacianBackground(f):
        lapf=[]
        for i in range(1, Nr - 1):
            radius = r[i]
            #Second derivative
            f_rr = (f[i+1, :] - 2*f[i, :] + f[i-1, :]) / dr ** 2

            # First derivative
            f_r = (f[i+1, :] - f[i-1, :]) / (2 * dr)

            # Apply to next slice
            lapf.append(1 / uBackground[i,:] * (f_rr + 1 / r[i] * f_r))
        return np.array(lapf)


    # Background scalar curvature
    KBackground = laplacianBackground(np.log(uBackground))

    # Fix divide by 0 error by using first order extrapolation (sloppy: need to fix this)
    KBackground[0,:] = KBackground[1,:] + dr * (KBackground[1,:] - KBackground [2,:])

    # Calculate average curvature
    KAverage = 0
    for i in range(1,Nr-2):
        for j in range(0,Ntheta):
            KAverage = KAverage + KBackground[i,j] * r[i] * dr * dtheta
    KAverage = KAverage / (2 * np.pi)
    print(KAverage)


    # Initialize u, where u[t, i, j] is the temp at time t, radius i, angle j
    u = np.zeros((t_nodes, Nr, Ntheta), dtype=float)


    # Set init condition
    u[0, :, :] = 1

    # Set boundary condition
    u[0] = set_boundary(u[0])

    # The model: updates for each time step t
    for n in range(t_nodes - 1):
        w = u[n]
        #print(np.log(w))
        log_w = np.log(w)


        # Update excludes endpoints in r
        u[n+1, 1:-1, :] = u[n, 1:-1, :] + dt * a * (laplacianBackground(u[n,:,:]) + KAverage * u[n,1:-1,:] - KBackground[:,:])

        # Expensive...can we update in place?
        u[n+1] = set_boundary(u[n+1])

        # set r = 0 to the temp of the closest point
        u[n+1, 0, :] = u[n+1, 1, 0]


    R, TH = np.meshgrid(r, theta, indexing="ij")

    phi = TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)


    #Multiply uBackground by u to get the metric g_t
    gMetric = uBackground[None,:,:] * u[:,:]

    #Remove singularity by duplicating next value
    gMetric[:,0,:] = gMetric[:,1,:]

    # Plot the sim
    plot_u_with_slider(u, X, Y, dt)
    plot_u_with_slider(gMetric, X, Y, dt)


def set_boundary(w:np.ndarray):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = 1
    return l


def plot_u_with_slider(u, X, Y, dt):
    t_nodes = u.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)

    zmin = float(u.min())
    zmax = float(u.max())
    ax.set_zlim(zmin, zmax)

    k0 = 0
    surf = ax.plot_surface(X, Y, u[k0], cmap="jet", vmin=zmin, vmax=zmax, shade=True)
    ax.set_title(f"t = {k0*dt:.4f} s (k={k0})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temp")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(slider_ax, "time index k", 0, t_nodes - 1, valinit=k0, valstep=1)

    def update(val):
        nonlocal surf
        k = int(s.val)
        surf.remove()
        surf = ax.plot_surface(X, Y, u[k], cmap="jet", vmin=zmin, vmax=zmax, shade=True)
        ax.set_title(f"t = {k*dt:.4f} s (k={k})")
        fig.canvas.draw_idle()

    s.on_changed(update)
    plt.show()



def initialize(u, dx, dy):
    Nt, Nx, Ny = u.shape

    # Get coordinate arrays
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy

    # f(0, y) = 0
    u[:, 0, :] = 0

    # f(1, y) = 1-2y
    u[:, -1, :] = 1 - (2 * y)

    # f(x, 0) = sin((pi * x) /2)
    u[:, :, 0] = np.sin(((np.pi / 2) * x))

    # f(x, 1) = -sin((pi * x) /2)
    u[:, :, -1] = -np.sin(((np.pi / 2) * x))

    return u


if __name__ == "__main__":
    sim_in_polar(t=2)