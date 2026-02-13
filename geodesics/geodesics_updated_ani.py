import numpy as np
import matplotlib.pyplot as plt
import doctest
from matplotlib.animation import FuncAnimation

"""Store pos and v, simulate, then plot at the end (optionally animate)."""

def cmplx_div(z: tuple, w: tuple) -> tuple:
    a, b = z
    c, d = w
    v1 = (a*c + b*d) / (c**2 + d**2)
    v2 = (b*c - a*d) / (c**2 + d**2)
    return (v1, v2)

def cmplx_mult(z: tuple, w: tuple):
    a, b = z
    c, d = w
    v1 = a*c - b*d
    v2 = a*d + b*c
    return (v1, v2)

def plus(z, w):
    z1, z2 = z
    w1, w2 = w
    new_1 = w1 + z1
    new_2 = z2 + w2
    return (new_1, new_2)

def simulate(dt=0.001, steps=1000, pos0=(1, 0.1), v0=(-1, 0), beta=0.1):
    # store history
    pos_hist = np.empty((steps + 1, 2), dtype=float)
    v_hist = np.empty((steps + 1, 2), dtype=float)

    pos = pos0
    v = v0
    a = (0.0, 0.0)

    pos_hist[0] = pos
    v_hist[0]   = v

    for k in range(steps):
        new_pos = plus(pos, cmplx_mult((dt, 0), v))
        a = cmplx_div(cmplx_mult((1 - beta, 0), cmplx_mult(v, v)), pos)
        v = plus(v, cmplx_mult((dt, 0), a))
        pos = new_pos

        pos_hist[k + 1] = pos
        v_hist[k + 1] = v

    return pos, v, a, pos_hist, v_hist

def plot_path(pos_hist):
    x = pos_hist[:, 0]
    y = pos_hist[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=1)
    ax.scatter([x[0]], [y[0]], marker="o", label="start")
    ax.scatter([x[-1]], [y[-1]], marker="x", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory (all points)")
    ax.legend()
    plt.show()

def animate_path(pos_hist, interval_ms=16, trail=300):
    x = pos_hist[:, 0]
    y = pos_hist[:, 1]

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory (animation)")

    pad = 0.05
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    (line,) = ax.plot([], [], linewidth=1)
    (dot,)  = ax.plot([], [], marker="o")

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def update(i):
        j0 = max(0, i - trail)
        line.set_data(x[j0:i+1], y[j0:i+1])
        dot.set_data([x[i]], [y[i]])
        return line, dot

    ani = FuncAnimation(
        fig, update,
        frames=len(x),
        init_func=init,
        interval=interval_ms,
        blit=True
    )
    plt.show()

def main():
    dt = 0.001
    steps = 10000
    pos0 = (1, 0.1)
    v0 = (-1, 1)
    beta = 0.2

    pos, v, a, pos_hist, v_hist = simulate(dt=dt, steps=steps, pos0=pos0, v0=v0, beta=beta)

    print("pos:", tuple(pos))
    print("v:", tuple(v))
    print("a:", tuple(a))

    # plot after sim
    plot_path(pos_hist)

    #animate_path(pos_hist, interval_ms=16, trail=300)

main()
#print(doctest.testmod())