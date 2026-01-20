import numpy as np
import matplotlib.pyplot as plt

def sim_u(a, length, width, time, nodes, BCx, BCy):

    # Boundary conditions and Initial conditins

    dx = length / (nodes - 1)
    dy = width / (nodes - 1)

    dt = min(dx ** 2 / (4 * a), dy ** 2 / (4 * a))

    t_nodes = int(time / dt)

    # Init conditions
    u = np.zeros((nodes, nodes)) + 20 # plate initially at 20 degrees c

    # boundary conditions
    u[0, :] = BCx
    u[-1, :] = BCy

    # visualization
    x = np.linspace(0, length, nodes)
    y = np.linspace(0, length, nodes)
    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # set fixed z limits so the scale doesn't jump around
    ax.set_zlim(0, 100)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("Temp (°C)")

    surf = ax.plot_surface(X, Y, u, cmap="jet", vmin=0, vmax=100)

    # optional: a colorbar for the surface colors (height is still z)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

    data = {}
    # simulation

    counter = 0

    for step in range(t_nodes):
        w = u.copy()

        for i in range(1, nodes - 1):
            for j in range(1, nodes - 1):
                dd_ux = (w[i - 1, j] - 2*w[i, j] + w[i + 1, j]) / dx ** 2
                dd_uy = (w[i, j - 1] - 2*w[i, j] + w[i, j + 1]) / dy ** 2
            
                u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]


        u[0, :] = BCx
        u[-1, :] = BCy

        counter += dt
        data[counter] = u

        print('t: {:.3f} [s], Average temp: {:.2f} Celcius'.format(counter, np.average(u)))
        surf.remove()  # remove old surface
        surf = ax.plot_surface(X, Y, u, cmap="jet", vmin=0, vmax=100)

        ax.set_title(f"t = {counter:.3f} s, avg = {u.mean():.2f} °C")

        plt.pause(0.01)

    #print(data)
    plt.ioff()
    plt.show()

sim_u(110, 50, 35, 3, 40, 100, 100)