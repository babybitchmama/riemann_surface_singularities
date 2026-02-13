import numpy as np
import matplotlib.pyplot as plt

# Save the state at every moment in time, and then plot at end
# caluclate a geodesic at each moment, see how it changes with the curvature
# Jost 2.3 up to curvature, Fubiini-Study metric

def set_boundary_conditions(u: np.array, dx, dy):
    #----boundary conditions----#

    # Get number of points on each axis
    Nx, Ny = u.shape

    # Get coordinate arrays
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy

    # f(0, y) = 0
    u[0, :] = 0

    # f(1, y) = 1-2y
    u[-1, :] = 1 - (2 * y)

    # f(x, 0) = sin((pi * x) /2)
    u[:, 0] = np.sin(((np.pi / 2) * x))

    # f(x, 1) = -sin((pi * x) /2)
    u[:, -1] = -np.sin(((np.pi / 2) * x))

    return u

def sim_u(a, length, width, time, nodes):

    # conditions
    a = 1
    length = 1
    width = 1
    nodes = 20
    time = 1 # seconds for the simulation

    # calculate additional parameters
    dx = length / (nodes - 1)
    dy = width / (nodes - 1)
    dt = min(dx ** 2 / (4 * a), dy ** 2 / (4 * a)) / 2

    t_nodes = int(time / dt)

    #-----Init conditions------#

    # u(x, y, 0) = 0
    u = np.zeros((nodes, nodes))
    u[5, 5] = 2


    #----boundary conditions----#
    set_boundary_conditions(u, dx, dy)


    # visualization
    x = np.linspace(0, length, nodes)
    y = np.linspace(0, length, nodes)
    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # set z limits
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("Temp (°C)")

    surf = ax.plot_surface(X, Y, u, cmap="jet", vmin=-1, vmax=1)

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)



    # model
    counter = 0
    data = {}

    plot_every = 1
    for step in range(t_nodes):
        interior = u[1:-1, 1:-1]
        print(f"t={counter:.3f}  interior std={interior.std():.4f}  center={u[nodes//2, nodes//2]:.4f}")

        w = u.copy()
        set_boundary_conditions(w, dx, dy)

        for i in range(1, nodes - 1):
            for j in range(1, nodes - 1):

                dd_ux = (w[i - 1, j] - 2*w[i, j] + w[i + 1, j]) / dx ** 2
                dd_uy = (w[i, j - 1] - 2*w[i, j] + w[i, j + 1]) / dy ** 2

                # set u[t_idx + 1]
                u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]
        # Reinforce boundary
        set_boundary_conditions(u, dx, dy)


        counter += dt
        data[counter] = u
        print (step)

        if step % plot_every == 0 or step == 1:
            surf.remove()  # remove old surface
            surf = ax.plot_surface(X, Y, u, cmap="jet", vmin=-1, vmax=1, shade=True)

            ax.set_title(f"t = {counter:.3f} s, avg = {u.mean():.2f} °C")

            plt.pause(0.01)


    #print(data)
    plt.ioff()
    plt.show()

sim_u(1, 1, 1, 3, 40)