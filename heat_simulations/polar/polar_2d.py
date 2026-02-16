import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Save the state at every moment in time, and then plot at end
# caluclate a geodesic at each moment, see how it changes with the curvature
# Jost 2.3 up to curvature?? Fubiini-Study metric

BETA = 0

def sim_in_polar(a=1.0, t=1.0, Nr=80, Ntheta=80):

    # Get radii in [0,1]
    r = np.linspace(0.0, 1.0, Nr)
    # Get theta from [-pi, pi]
    theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False)

    # Calculate radial and angular steps
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    
    # Stable dt for no diffusion dependent on theta
    #FIXME will update if we extend to non-radially symmetric
    dt = (dr**2) / (4*a) / 2
    t_nodes = int(t / dt) + 1

    # Initialize u, where u[t, i, j] is the temp at time t, radius i, angle j
    u = np.zeros((t_nodes, Nr, Ntheta), dtype=float)



    # ------ Stopping point ------ #
    # Set init condition
    print(u.shape)
    for p, q, theta_i in (u.shape):
        angle = theta[theta_i]
        u[0, -1, angle] = np.cos(10*angle)

        

    # Set boundary condition
    u[0] = set_boundary(u[0])

    # The model: updates for each time step t
    for n in range(t_nodes - 1):
        w = u[n]

        # Update excludes r = 0
        for i in range(1, Nr - 1):
            radius = r[i]
            for j in range(1, Ntheta - 1):
            
                #Second derivative of r
                u_rr = (w[i+1, j] - 2*w[i, j] + w[i-1, j]) / dr ** 2

                # First derivative of r
                u_r = (w[i+1, j] - w[i-1, j]) / (2 * dr)

                # Second derivative of theta
                u_theta_theta = (w[i, j+1] - 2*w[i, j] + w[i, j - 1])/ dtheta**2

                # Apply to next slice
                u[n+1, i, j] = u[n, i, j] + dt * a * (u_rr + (1/radius * u_r) + 1/(radius**2)(u_theta_theta))

        # Expensive...can we update in place?
        u[n+1] = set_boundary(u[n+1])


        # set r = 0 to the temp of the closest point
        #FIXME will need to change with 2d
        u[n+1, 0, :] = u[n+1, 1, 0]


    R, TH = np.meshgrid(r, theta, indexing="ij")

    phi = (1-BETA) * TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    # Plot the sim
    plot_u_with_slider(u, X, Y, dt)


def set_boundary(w:np.ndarray):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = 2
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


if __name__ == "__main__":
    sim_in_polar()