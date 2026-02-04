import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import doctest

def cmplx_div(z: tuple, w: tuple) -> tuple:
    a, b = z
    c, d = w
    denom = c**2 + d**2
    return ((a*c + b*d) / denom, (b*c - a*d) / denom)

def cmplx_mult(z: tuple, w: tuple) -> tuple:
    a, b = z
    c, d = w
    return (a*c - b*d, a*d + b*c)

def plus(z: tuple, w: tuple) -> tuple:
    return (z[0] + w[0], z[1] + w[1])

def simulate(dt=0.001, steps=1000, pos0=(1, 0.1), v0=(-1, 0), beta=0.1):
    """
    Returns:
      t: (steps+1,)
      pos_hist: (steps+1, 2)
      v_hist:   (steps+1, 2)
      a_hist:   (steps+1, 2)
    """
    t = np.arange(steps + 1) * dt

    pos_hist = np.empty((steps + 1, 2), dtype=float)
    v_hist   = np.empty((steps + 1, 2), dtype=float)
    a_hist   = np.empty((steps + 1, 2), dtype=float)

    pos = pos0
    v   = v0

    pos_hist[0] = pos
    v_hist[0]   = v
    a_hist[0]   = (0.0, 0.0)

    for k in range(steps):
        # compute acceleration from current state
        a = cmplx_div(
            cmplx_mult((1 - beta, 0), cmplx_mult(v, v)),
            pos
        )

        # Euler updates
        pos = plus(pos, cmplx_mult((dt, 0), v))
        v   = plus(v,   cmplx_mult((dt, 0), a))

        pos_hist[k + 1] = pos
        v_hist[k + 1]   = v
        a_hist[k + 1]   = a

    return t, pos_hist, v_hist, a_hist

def plot_all_at_once(pos_hist):
    x = pos_hist[:, 0]
    y = pos_hist[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=1)
    ax.scatter([x[0]], [y[0]], marker="o", label="start")
    ax.scatter([x[-1]], [y[-1]], marker="x", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Trajectory (all points)")
    plt.show()

def animate_from_history(pos_hist, interval_ms=16, trail=400):
    x = pos_hist[:, 0]
    y = pos_hist[:, 1]

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory (animation)")

    # set limits with padding so the motion stays in frame
    pad = 0.05
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    (line,) = ax.plot([], [], linewidth=1)   # trail
    (dot,)  = ax.plot([], [], marker="o")    # current point

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
    steps = 1000
    beta = 0.1

    t, pos_hist, v_hist, a_hist = simulate(dt=dt, steps=steps, beta=beta)

    print("pos:", tuple(pos_hist[-1]))
    print("v:",   tuple(v_hist[-1]))
    print("a:",   tuple(a_hist[-1]))

    # 1) Plot the whole path at once
    plot_all_at_once(pos_hist)

    # 2) Animate using the stored history (fast + simple)
    animate_from_history(pos_hist, interval_ms=16, trail=300)

if __name__ == "__main__":
    main()
    print(doctest.testmod())