################################################################################
## GLOBAL PARAMETERS NEEDED THROUGHOUT #########################################
################################################################################



## DEFINITIONS ##

# Basic
const _dim = 1              # space dimension of the problem
const _n = 100              # total number of particles for simulation

# Initialisation
const _initM = [Interval([0.],1.75,"Int")]     # initial manifold
# (must be vector)
const _randomness = "RPUG"              # type of initial particle distribution:
# UG (Uniform Grid), RPUG (Randomly Perturbed Uniform Grid), RS (Random Sample)
const _amp = 0.01                       # amplitude of perturbation for RPUG

# Optimisation (or gradient flow)
const _scheme = GFeps(2.0e-10,0.1)    # optim scheme/gradflow, tolerance (and ε)
const _finalT = 25.         # final time, only for gradient flow schemes GFeps
# GFP; simulation ends whenever tolerance or final time is reached, whichever
# is first.
const _corrdt = 1.        # for gradient flow schemes GFeps and GFP only:
# correction of time step if too large (sometimes needed when looking at the
# boundary of a manifold, when particles "oscillate" around the boundary before
# stabilising on the boundary); can also be used to increase time step.
const _pert = false          # for schemes GFeps and GFP only: perturb solution
# or not to help escape local minima; the amplitude _amp above is used as
# reference and is decreased with time.
const _sampling = "Linear"   # way the distance to the manifolds (other than
# those the exact distance is exactly known for) is computed using the sample of
# the manifolds' boundaries; can be "Pointwise" (where the distance is obtained
# by taking the closest among the points of the sample) or "Linear" (where the
# distance is given by the projection to the piecewise linear approximation of
# the boundary obtained via the points of the sample)

# Interaction potential and manifolds
const _W = Linear(2.,1.)                     # interaction potential
const _M = [Interval([0.0],1.,"Int"),Point1d([1.5])]     # manifolds (must be vector)
const _union = true                         # is the resulting manifold the
# union or the intersection of all the manifolds in manifold vector _M?

# Plotting
const _plots = true     # choose whether or not to plot results (when true, a
# file named time_energy.txt is generated together with the plots; it is
# at each run)
const _log = false      # choose whether or not the dynamics should be plotted
# with a log time scale
const _arrows = false    # choose whether the initial gradient field should be
# plotted
const _title = false     # choose whether plots should have titles

################################################################################



## CHECK PARAMETERS ARE WITHIN ACCEPTED VALUES ##

# Basic
@assert _dim == 1 || _dim == 2      # eventually extend to dimension 3
@assert _n > 0

# Initialisation
if _n % length(_initM) != 0
    error("The number of particles must be divisible by the number of components
        in the initial manifold.")
end
n_component = Int64(_n/length(_initM))  # number of sampling points per
# component of initial manifold
if _dim == 2 && !iszero(sqrt(n_component)-floor(sqrt(n_component)))
    error("The number of particles in each initial manifold component must be
        a perfect square.")
end
for i = 1:length(_initM)
    @assert _initM[i].dimension == _dim # assert dimensions match
    @assert _initM[i].side == "Int"     # we can only sample manifold's interior
    @assert _randomness == "UG" || _randomness == "RPUG" || _randomness == "RS"
    if (_initM[i].name != "Interval" && _initM[i].name != "Rectangle") &&
            (_randomness == "UG" || _randomness == "RPUG")
        error("We can only sample uniformly an interval or rectangle.")
    end
end

# Optimisation
@assert _sampling == "Pointwise" || _sampling == "Linear"

# Interaction potential and manifolds
for i = 1:length(_M)
    @assert _M[i].dimension == _dim     # assert dimensions match
end
if !_union
    println("WARNING: When taking the manifolds' intersection, the union of the
        considered manifolds should cover the whole space. Check.")
    if length(_M) > 2
        error("If taking the manifolds' intersection, must restrict to up to two
            manifolds.")
    end
end
