
# Heat Diffusion and Geodesics
We're modeling heat diffusion and investigating curvature. This process began with simple simulations of heat diffusion on a 1D rod, and then a 2D surface. Then we messed with different boundary conditions, and transitioned to simulations in polar. Now we're working on extending to non-radially symmetric systems.

Our long-term goal is to model curvature flow on Riemann surfaces with conical singularities, and study the conditions for uniform curvature as a function of the cone angle.

### Installation
To run the simulation, you'll need to install `python`, `numpy` and `matplotlib`.

## `heat_simulations`
### `first_passes`
In `first_passes`, you'll find the first simulations we created, like the initial 1D and 2D models, as well as experiments with more interesting boundary conditions. With `bounds_time_stack` we made our first attempt at storing all values and plotting all at once.

### `polar`
In `polar`, you'll find our more advanced simulations in polar coordinates, including the one involving curvature ove a cone in `cone.py`. `polar.py` was my first attempt at simulating in polar coordinates.

In `polar_2d.py`, I just made my first attempt at using the angular derivative. Update: different from past models, we don't save the entire data set. We just store the current and next ones. This has been updated to include the theta = 0 endpoint. 

## `geodesics`
In the `geodesics` directory, you can see some attempts at modeling the trajectory of a geodesic near a cone structure. The animations are rudimentary (very ugly).


## Log
Updated runtime around 4x speed increase with Arya's slice method replacing the nested loops. Need to apply runtime increase and angular derivative to the conformal cone.