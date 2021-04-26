################################################################################
## MAIN FILE ###################################################################
################################################################################

# Authors: Francesco S. Patacchini and Dejan Slepcev (2019--2021)

# Description:
#   We approximate of the interaction equation on a manifold M (or union, or
# intersection of manifolds), that is, the equation
#               ∂ₜρ = div(ρ P_M(∇W⋆ρ))      (*)
# where P_M is the vector field orthogonal projection on M.
# We approximate this equation with particles via the following equations:
#               ∂ₜρ = div(ρ ∇W⋆ρ + (1/ε)ρ ∇d_M²),                           (1)
#   ρ(kτ + t) = Π_M(μ(τ)), for t ∈ (0,τ], ∂ₜμ = div(μ ∇W⋆μ), μ(0) = ρ(kτ)   (2)
# where d_M is the distance to the manifold M, Π_M is the measure orthogonal
# projection and k is any integer. From our paper, we know that (1) converges to
# (*) as ε goes to 0 and that (2) converges to (*) as τ goes to 0.
#   For (1) we use the corresponding energy
#               E_ε(ρ) = (1/2)∫∫W(x-y)dρ(x)dρ(y) + (1/ε)∫d_M(x)² dρ(x),
# where the distance part of the energy plays the role of a penalising
# confinement term (whose strength is quantified by the parameter ε). We study
# the problem from two points of view: optimisation (minimisation), in which
# case we only look at the minimisers of the energy, and dynamics, in which case
# we follow the particles in time according to gradient flow for E_ε.
#   For (2) we use the energy
#               E(ρ) = (1/2)∫∫W(x-y)dρ(x)dρ(y).
# Here, we follow the dynamics according to the gradient flow for E while
# projecting on M the particles at the end of each time step.

# To use the code: change the parameters in parameters.jl and run main.jl

# Possible extensions:
#   - Space dimension: extend the code to dimension 3. Should not be difficult,
#   as the code is already well laid out to be extended. Although plotting may
#   be cumbersome in Julia...
#   - Changing masses: at the moment, the code only accepts uniform masses of
#   particles. We could envisage to add nonuniform masses and also optimise the
#   energy on the masses as well (in the case of nondymanics) via a constrained
#   optimisation on the "positive" sphere (since masses are positive and sum
#   to 1.
#   - More manifolds: we could include more manifolds by accepting nonsymmetric
#   ones around the x-axis (and y-axis in 3d), and also make the intersection
#   of manifolds hold for more than 2 manifolds and for manifolds whose union is
#   not the whole space.
#   - Extend the cases we know the exact distance for: at the moment, only for
#   points, intervals, discs and whole spaces we use the fact that we know exact
#   formulas for the squared distance (see distanceSqToManifold function). We
#   could extend this by adding two fields in the manifold structure: one for
#   squared distance and another for its gradient. Then, in distanceSqToManifold
#   we would call those fields and, if they are not set to "nothing", we would
#   use them rather than sample the manifold's boundary to get the distance.
#   - More initial conditions: we could include uniform sampling on manifolds
#   other than intervals or rectangles (via a map from the interval/rectangle to
#   the manifold, or by arclength, as already approximately done to sample the
#   boundary of the manifold). We could also allow to start on the boundary of a
#   manifold (maybe by orthogonally projecting points on the manifold).



## INCLUDE PACKAGES AND FILES

using Plots; pyplot()
using DelimitedFiles
using LaTeXStrings, Measures, LinearAlgebra, NearestNeighbors, Parameters
using FiniteDiff, QuadGK, Optim, LineSearches

include("./toolbox.jl")
include("./parameters.jl")
include("./energy.jl")
include("./initialiser.jl")
include("./optimiser.jl")
include("./plotter.jl")

################################################################################



## INITIALISE, OPTIMISE AND PLOT

x0 = initialise()

@time resultOpt = optimise(x0)

if _plots
    if _scheme.name != "GFeps" && _scheme.name != "GFP"
        plotSol(x0,resultOpt)
    elseif _scheme.name == "GFeps" || _scheme.name == "GFP"
        plotSol(x0,resultOpt...)    # the ... dots unpack the tuple resultOpt

        # Below we also print in a file the vectors containing time and energy.
        # Warning: this file gets replaced everytime a new simulation is run, so
        # it needs to be saved elsewhere between two simulations.
        ntime = length(resultOpt[2])
        open("./plots/time_energy.txt","w") do io
            writedlm(io,["Time" "Energy"],"   ")
            for i = 1:ntime
                writedlm(io,[resultOpt[2][i] resultOpt[3][i]],"   ")
            end
        end
        close("./plots/time_energy.txt")
    end
end
