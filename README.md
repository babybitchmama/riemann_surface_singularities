
# Heat Diffusion and Geodesics
We're modeling heat diffusion, starting in 1d and working to 2d and then eventually to curvature on Riemann surfaces. 

Recently updated to model more interesting boundary conditions f(0, y) = 0, f(1, y) = 1-2y, f(x, 0) = sin(pi * x / 2), f(x, 1) = -sin(pi * x / 2)

## Installation
To run the simulation, you'll need to install `python`, `numpy` and `matplotlib`.

In `cool_boundaries.py` check out the more interesting conditions.

In `heat_eq_2d.py` you can edit the initial conditions and boundary conditions to create a different environment for the simulation. Our model solves the heat equation to model diffusion across a surface.

In `geodesics.py` use the initial conditions to see how the geodesic behaves under different cone angles, velocites, and initial positions.


This project will eventually be used to study conic singularities on Riemann surfaces, but for now it's a simple simulation of heat diffusion and a simulation of a geodesic on a conical surface.
