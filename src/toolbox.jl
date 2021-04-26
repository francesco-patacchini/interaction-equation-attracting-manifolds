################################################################################
## USEFUL NOTATION, FUNCTIONS AND STRUCTURES TO BE USED THROUGHOUT #############
################################################################################



## NOTATION ##

Particle = Vector{Float64}     # Abbreviation for a particle

################################################################################



## VECTOR AND MATRIX MANIPULATION ##

# Function vvm takes in a vector x of vectors (all of same length n1) and
# converts it into a matrix of size n1xn2, where n2 is the length of x.
function vvm(x::Vector{Particle})::Matrix{Float64}
    n1 = length(x[1]); n2 = length(x)
    res = Matrix{Float64}(undef,n1,n2)
    for i = 1:n2
        res[:,i] = x[i]
    end
    return res
end


# Function vvm_inv is the inverse of the function vvm above.
function vvm_inv(X::Matrix{Float64})::Vector{Particle}
    n = length(X[1,:])
    res = Vector{Particle}(undef,n)
    for i = 1:n
        res[i] = X[:,i]
    end
    return res
end


# Function mva takes in a matrix X of vectors (all of length n1) and converts it
# into an array of size n1xn2xn3, where n2xn3 is the size of X. (We use this
# later to plot surfaces and contours.)
function mva(X::Matrix{Particle})::Array{Float64,3}
    n1 = length(X[1,1]); n2 = length(X[1,:]); n3 = length(X[:,1])
    res = Array{Float64,3}(undef,n1,n2,n3)
    for j = 1:n3
        res[:,:,j] = vvm(X[j,:])
    end
    return res
end


# Function perturb takes in a vector of particles x and an float amp, and
# updates x by perturbing it randomly with amplitude amp.
function perturb!(x::Vector{Particle},amp::Float64)

    n = length(x)
    dim = length(x[1])

    pert = Vector{Vector{Float64}}(undef,dim)
    for d = 1:dim
        r = rand(n)
        pert[d] = amp*(-1 .+ 2*r)   # fill in perturbation between -amp and +amp
    end

    pert_mat = vvm(pert)    # transform pert vector into an nxd matrix
    for i = 1:n
        x[i] .+= pert_mat[i,:]
    end
end

################################################################################



## DEFINITION AND EXAMPLES OF INTERACTION POTENTIALS ##


# POTENTIAL STRUCTURE #

@with_kw struct Potential
    name::String
    power::Float64; @assert power >= 2
    coefficient::Float64;
    expression::Function
    derivative::Function
end
# The field power gives the dependence of potential on the distance between each
# pair of particles, i.e., the value of the potential at x is
# expression(coeff*norm(x)^power). The condition power >= 2 ensures smoothness
# of potential at 0. (See potential function in energy.jl.)


# EXAMPLES #

function Inverse(power::Float64,coeff::Float64)::Potential
    Potential("Inverse",power,coeff,r -> 1/(1 + r),r -> -1/(1 + r)^2)
end
function InverseAttr(power::Float64,coeff::Float64)::Potential
    Potential("InverseAttr",power,coeff,r -> -1/(1 + r),r -> 1/(1 + r)^2)
end
function Exponential(power::Float64,coeff::Float64)::Potential
    Potential("Exponential",power,coeff,r -> exp(-r),r -> -exp(-r))
end
function Linear(power::Float64,coeff::Float64)::Potential
    Potential("Linear",power,coeff,r -> r,r -> 1.0)
end
function Constant()::Potential
    Potential("Constant",2.0,0.0,r -> 0.0,r -> 0.0)
end

################################################################################



## DEFINITION AND EXAMPLE OF MANIFOLDS ##


# MANIFOLD STRUCTURE #

@with_kw struct Manifoldd
    dimension::Int8; @assert dimension == 1 || dimension == 2
    name::String
    parameters::Union{Vector{Float64},Nothing}
    centre::Particle; @assert length(centre) == dimension
    xboundary::Function
    yboundary::Union{Function,Nothing}
    side::String; @assert side == "Int" || side == "Bdry" || side == "Ext"
end
# To adapt the manifold structure to space dimension 3, add a field called
# zboundary::{Function,Nothing}, which stores the graph (x,y) -> g(x,y) of the
# surface. So, xboundary is a function of no arguments giving out a constant,
# yboundary of x giving out a number f(x) and zboundary of (x,y) giving out a
# number g(x,y).
# The function xboundary gives the maximum of the x-domain and yboundary() the
# the maximum of the y-domain (and zboundary of the z-domain). They must all be
# positive functions, since we then suppose symmetry of the manifold. (This
# assumption could be relaxed by adding additional functions to the manifold
# structure giving the negative boundaries, i.e., the domain minimums.)


# EXAMPLES #
# (Note that Point1d, Point2d, R1, R2, Interval, Disc and Rectangle are reserved
# names and should not be changed as they are treated differently by the code.)

# 1-dimensional
function Point1d(centre::Particle)::Manifoldd
    Manifoldd(1,"Point1d",nothing,centre,() -> 0.0,nothing,"Bdry")
end

function R1()::Manifoldd
    Manifoldd(1,"R1",nothing,[0.0],() -> 0.0,nothing,"Ext")
end

function Interval(centre::Particle,radius::Float64,side::String)::Manifoldd
    Manifoldd(1,"Interval",[radius],centre,() -> radius,nothing,side)
end


# 2-dimensional
function Point2d(centre::Particle)::Manifoldd
    Manifoldd(2,"Point2d",nothing,centre,() -> 0.0,x -> 0.0,"Bdry")
end

function R2()::Manifoldd
    Manifoldd(2,"R2",nothing,[0.0,0.0],() -> 0.0,x -> 0.0,"Ext")
end

function Rectangle(centre::Particle,xmax::Float64,ymax::Float64,
        side::String)::Manifoldd
    Manifoldd(2,"Rectangle",[xmax,ymax],centre,() -> xmax,x -> ymax,side)
end

function Disc(centre::Particle,radius::Float64,side::String)::Manifoldd
    Manifoldd(2,"Disc",[radius],centre,
        () -> radius,x -> sqrt(radius^2 - x^2),side)
end

function Ellipse(centre::Particle,xRadius::Float64,yRadius::Float64,
        side::String)::Manifoldd
    @assert xRadius > 0
    if xRadius == yRadius
        error("x-radius = y-radius; use Disc instead, since distance is computed
            exactly.")
    end
    Manifoldd(2,"Ellipse",[xRadius,yRadius],centre,() -> xRadius,
        x -> yRadius*sqrt(1 - (x/xRadius)^2),side)
end

function Bean(centre::Particle,neck::Float64,side::String)::Manifoldd
    @assert neck >= 1
    Manifoldd(2,"Bean",[neck],centre,() -> 1.0,
        x -> 0.4*sqrt(1 - x^2)*(neck - cos(3*x)),side)
end

function Plectrum(centre::Particle,α::Float64,side::String)::Manifoldd
    Manifoldd(2,"Plectrum",[α],centre,
        () -> 1.0,x -> sqrt(1 - x^2)*exp(α*x-1),side)
end



## TOOLS FOR MANIFOLDS ##

# Function revolve takes in a function f and floats x and y, and outputs the
# the height of the surface at (x,y) obtained by revolving the function f around
# the x-axis. (This would be used to generate manifolds in space dimension 3.)
# function revolve(f::Function,x::Float64,y::Float64)::Float64
#     @assert f(x) >= 0.0 && abs(y) <= f(x)
#     if f(x) == 0.0
#         return 0.0
#     else
#         return f(x)* cos(asin(y/f(x)))
#     end
# end


# Functions diffBdryManifold, lengthBdryManifold, xSampleBdryManifold and
# bdrySampleManifold are functions used to generate a sample of the boundary of
# a manifold. diffBdryManifold computes the numerical derivative of the boundary
# of a manifold M at a point x. (FiniteDiff package cannot be used because we
# need to ensure manually not to evaluate the boundary function M.yboundary
# outside of its domain.) lengthBdryManifold gives the length of the positive
# boundary of M between the left-most domain point and a point x.
# xSampleBdryManifold generates a sample of the x-axis (x-domain of yboundary)
# such that the sample of the bdry is approximately uniform (according to
# arclength). Finally, bdrySampleManifold takes in a manifold M and a float
# density and outputs two particle vectors: one is a sample of the boundary of M
# for positive y's and the other its mirror part for negative y's. The input
# density gives the number of points per unit length that is going to be used to
# get the sample size of the boundary; if yboundary(±xboundary()) > 0 we add a
# sampling of the y-axis as well depending on how large yboundary(±xboundary())
# is. We use this in distanceSqToManifold to compute the squared distance to the
# boundary of a manifold we do not know the exact distance formula for, and
# later to plot the boundary of the manifold in plotter.jl.
function diffBdryManifold(M::Manifoldd,x::Float64)::Float64

    xmax = M.xboundary()
    yb(x) = M.yboundary(x)

    if x < -xmax || x > xmax
        return 0.0
    else
        h = sqrt(eps()) * abs(x)    # take h as small as possible such that
        # machine precision is still good
        xph = min(x + h,xmax)   # makes sure we do not evaluate yb outside its
        # domain
        xmh = max(x - h,-xmax)  # same
        dh = xph - xmh

        return (yb(xph) - yb(xmh))/dh
    end
end

function lengthBdryManifold(M::Manifoldd,x::Float64)::Float64

    xmax = M.xboundary()
    g(y::Float64)::Float64 = sqrt(1 + (diffBdryManifold(M,y))^2)

    if x > -xmax && x < xmax
        return quadgk(y -> g(y),-xmax,x,rtol=1e-9)[1]
    elseif x <= -xmax
        return 0.0
    elseif x >= xmax
        return quadgk(y -> g(y),-xmax,0.0,rtol=1e-9)[1] +
            quadgk(y -> g(y),0.0,xmax,rtol=1e-9)[1]     # need to split the
            # integral as somehow quadgk does not always work on symmetric
            # domains
    end
end

function xSampleBdryManifold(M::Manifoldd,density::Float64)::Vector{Float64}
    xmax = M.xboundary()

    densityInv = 1000   # density of points used to compute inverse of
    # lengthBdryManifold
    NInv = Int64(ceil(densityInv* 2*xmax))  # corresponding number of points

    sampleLength = Matrix{Float64}(undef,NInv+1,2)  # to contain x values
    # (uniformly spread) and respective length values
    for i = 1:NInv+1
        sampleLength[i,1] = -xmax + (i-1)/NInv * 2*xmax
        sampleLength[i,2] = lengthBdryManifold(M,sampleLength[i,1])
    end

    lenM = sampleLength[NInv+1,2]     # length of the boundary of M
    N = Int64(ceil(density * lenM)) # number of points to be taken on x-axis for
    # the sample

    xSample = Vector{Float64}(undef,N+1)
    for i = 1:N+1
        l = (i-1)/N * lenM  # uniform length (arclength)
        if l > sampleLength[NInv+1,2]       # if length l not reached
            xSample[i] = Inf
        elseif l == sampleLength[NInv+1,2]
            xSample[i] = xmax
        else
            k = findfirst(j -> sampleLength[j,2] >= l,1:NInv+1)
            if k == 1
                xSample[i] = -xmax
            else    # take the average of x-values around the actual inverse
                xSample[i] = (sampleLength[k-1,1] + sampleLength[k,1])/2
            end
        end
    end

    return unique(xSample)  # unique() removes duplicates in xSample
end

function bdrySampleManifold(M::Manifoldd,
        density::Float64)::Tuple{Vector{Particle},Vector{Particle}}

    centre = M.centre

    if M.dimension == 1
        error("No need to call bdrySampleManifold in dimension 1.")
    end

    xboundary, yboundary = M.xboundary, M.yboundary
    xmax = xboundary()

    # Get the sample in the y-direction if yboundary is not 0 at the x-extremes
    d1 = yboundary(-xmax); d2 = yboundary(xmax)
    lLeftSample = Int64(ceil(density* d1))  # length of left-side sample
    lRightSample = Int64(ceil(density* d2)) # length of right-side sample
    yLeftSample = Float64[]                 # initialise left-side sample
    yRightSample = Float64[]                # initialise right-side sample
    if lLeftSample >= 2     # do not add any points if length is >= 2
        yLeftSample = collect(range(0.0,d1,length=lLeftSample))[1:end-1]
    end
    if lRightSample >= 2                # same
        yRightSample = collect(range(0.0,d2,length=lRightSample))[1:end-1]
    end
    leftSample = map(y -> [-xmax,y],yLeftSample)    # store sample as particles
    rightSample = map(y -> [xmax,y],yRightSample)   # same for right side
    minus_leftSample = map(y -> [-xmax,-y],yLeftSample)     # store the negative
    # symmetric left sample as particles
    minus_rightSample = map(y -> [xmax,-y],yRightSample)    # same for right

    xSample = xSampleBdryManifold(M,density)    # get the sample of the x-axis
    # according to the approximated arclength of ∂M given by xSampleManifold
    centreSample = map(x -> [x,yboundary(x)],xSample)   # store as particles
    minus_centreSample = map(x -> [x,-yboundary(x)],xSample)    # same for sym

    # Get the samples by concatenating and shifting by centre
    sample = vcat(leftSample,centreSample,rightSample) .+ [centre]
    minus_sample = vcat(minus_leftSample,minus_centreSample,
        minus_rightSample) .+ [centre]

    return sample, minus_sample

end


# Function distanceSqToManifold takes a particle p, a manifold M and a vector
# sample of particles as inputs outputs either the squared distance from p to M
# (or to the set sample of points in case the exact distance to M is not known)
# or its gradient at p, according to whether the optional input grad is false or
# true. The vector sample should be computed using the function
# bdrySampleManifold defined above (see the parameter _sample in energy.jl).
function distanceSqToManifold(p::Particle,M::Manifoldd,sample::Vector{Particle};
    grad::Bool=false)::Union{Float64,Particle}

    centre = M.centre
    p_shifted = p .- centre   # shift particle by centre to compute dist
    dim, name, side = M.dimension, M.name, M.side

    if !grad    # only compute squared distance
        d = 0.0

        # We first treat the cases we know the exact distance for.
        if name == "Point1d" || name == "Point2d" ||
                name == "Interval" || name == "Disc"

            radius = M.xboundary()
            if side == "Int"
                d = max(0, norm(p_shifted) - radius)
            elseif side == "Bdry"
                d = abs(norm(p_shifted) - radius)
            elseif side == "Ext"
                d = min(0,norm(p_shifted) - radius)
            end

        # We now treat the other cases. If whole spaces R1 and R2, then no need
        # to update distance since it is 0.
        elseif name != "R1" && name != "R2"

            xboundary, yboundary = M.xboundary, M.yboundary

            # Set true/false parameters for computations below.
            compare = !=; yn = !    # compare and yes/no parameters
            if side == "Int"
                compare = >
            elseif side == "Ext"
                compare = <; yn = !!
            end

            if dim == 1
                p1 = p_shifted[1]
                if compare(abs(p1),xboundary())
                    d = abs(abs(p1)-xboundary())    # no need to sample in 1d
                end

            elseif dim == 2
                p1 = p_shifted[1]; p2 = p_shifted[2]
                xdomain(x::Float64)::Bool = abs(x) <= xboundary()   # to check
                # if we are within the x-domain. (In 3d, we would have an
                # additional ydomain depending on x,y.)
                if ( yn(xdomain(p1)) ||
                        (xdomain(p1) && compare(abs(p2),yboundary(p1))) )
                    bdrySample = vvm(sample .- [centre])    # we take abs(p2) by
                    # symmetry around the x-axis of the manifold

                    # Below, use NearestNeighbors package to get nearest sample
                    # point to p; nearest_p contains the index of nearest point
                    # and the distance to nearest point. We take abs
                    nearest_p = knn(BruteTree(bdrySample),[p1,abs(p2)],1,true)

                    if _sampling == "Pointwise"
                        d = nearest_p[2][1]     # get distance to nearest point
                    elseif _sampling == "Linear"
                        bdrySampleVec = sample .- [centre]  # the sample as a
                        # vector of vectors
                        i_nearest = nearest_p[1][1] # get index of nearest point
                        xi = bdrySampleVec[i_nearest]    # get nearest point
                        xir = Particle(undef,_dim)  # initialise right neighbour
                        xil = Particle(undef,_dim)  # initialise left neightbour
                        p_xi = [p1,abs(p2)] .- xi   # vector p - xi

                        if i_nearest != 1 && i_nearest != length(sample)
                            xir = bdrySampleVec[i_nearest+1]
                            xil = bdrySampleVec[i_nearest-1]

                            norm_xir_xi = norm(xir .- xi)   # norm of vector
                            # xir - xi
                            tr = (xir .- xi)/norm_xir_xi
                            cr = max(min(dot(p_xi,tr),norm_xir_xi),0)
                            zr = xi .+ cr*tr    # projection on segment between
                            # xi and xir; equals xi if xi is closer than zr to p
                            # since in this case we have cr = 0

                            norm_xil_xi = norm(xil .- xi)   # norm of vector
                            # xim - xi
                            tl = (xil.-xi)/norm_xil_xi
                            cl = max(min(dot(p_xi,tl),norm_xil_xi),0)
                            zl = xi .+ cl*tl    # projection on segment between
                            # xi and xil; equals xi if xi is closer than zl to p
                            # since in this case we have cl = 0

                            d = min(norm([p1,abs(p2)] .- zr),
                                norm([p1,abs(p2)] .- zl)) # we take the minimum
                            # between the projections zr and zl, and xi

                        elseif nearest_p[1][1] == 1 # case sample point is first
                            # and therefore has no left neighbour
                            xir = bdrySampleVec[i_nearest+1]

                            norm_xir_xi = norm(xir .- xi)
                            tr = (xir .- xi)/norm_xir_xi
                            cr = max(min(dot(p_xi,tr),norm_xir_xi),0)
                            zr = xi .+ cr*tr

                            d = norm([p1,abs(p2)] .- zr)

                        elseif nearest_p[1][1] == length(sample) # case sample
                            # point is last and therefore has no right neighbour
                            xil = bdrySampleVec[i_nearest-1]

                            norm_xil_xi = norm(xil .- xi)
                            tl = (xil.-xi)/norm_xil_xi
                            cl = max(min(dot(p_xi,tl),norm_xil_xi),0)
                            zl = xi .+ cl*tl

                            d = norm([p1,abs(p2)] .- zl)
                        end

                    end
                end
            end
        end

        return d^2

    elseif grad
        d_sq(p::Particle)::Float64 = distanceSqToManifold(p,M,sample)

        if d_sq(p) == 0.0
            ∇d_sq = zeros(Float64,dim)  # when inside the manifold, set gradient
            # to zero
        else
            if name == "Point1d" || name == "Point2d" ||
                    name == "Interval" || name == "Disc"

                radius = M.xboundary()
                if (side == "Int" && norm(p_shifted) > radius) ||
                    (side == "Bdry" && norm(p_shifted) > 0) ||
                    (side == "Ext" && norm(p_shifted) < radius)

                    ∇d_sq = 2*p_shifted/norm(p_shifted)*
                        (norm(p_shifted) - radius)
                end

            elseif name != "R1" && name != "R2"
                ∇d_sq = FiniteDiff.finite_difference_gradient(d_sq,p)   # use
                # package FiniteDiff to compute numerical gradient
            end
        end

        return ∇d_sq
    end

end

################################################################################



## DEFINITION AND EXAMPLES OF OPTIMISATION/DYNAMICS SCHEMES ##


# SCHEME STRUCTURE #

@with_kw struct Scheme
    name::String
    nameOptim::Union{String,Nothing}
    tolerance::Float64
    epsilon::Union{Float64,Nothing}
    @assert epsilon == nothing || (epsilon > 0 && epsilon < Inf)
end
# The field nameOptim refers to the name of the optimiser as known in the Julia
# package Optim.jl. We have several options: any hessian-free, but
# gradient-dependent optimisation method from Optim.jl to optimise the penalised
# energy E_ε or get the dynamics of the particles by following the gradient flow
# for E_ε or the projected gradient flow for the unpenalised energy E. In the
# latter cases we will have access to the whole time evolution.


# EXAMPLES OF SCHEMES #

# Below, _Optim stands for Optim package. _Optim schemes solve (1) via
# minimisation of the energy E_ε. GFeps solves (1) via the dynamics (gradient
# flow) and GFP solves (2) via the dynamics.
function LBFGS_Optim(tol::Float64,ε::Float64)::Scheme
    Scheme("LBFGS","LBFGS",tol,ε)
end

function BFGS_Optim(tol::Float64,ε::Float64)::Scheme
    Scheme("BFGS","BFGS",tol,ε)
end

function CG_Optim(tol::Float64,ε::Float64)::Scheme
    Scheme("CG","ConjugateGradient",tol,ε)
end

function GD_Optim(tol::Float64,ε::Float64)::Scheme
    Scheme("GD","GradientDescent",tol,ε)
end

function GFeps(tol::Float64,ε::Float64)::Scheme # gradient flow scheme
    Scheme("GFeps",nothing,tol,ε)
end

function GFP(tol::Float64)::Scheme              # projected gradient flow scheme
    Scheme("GFP",nothing,tol,nothing)
end


# OPTIMISATION TOOLS #

# Function linesearch_dt is used for the GF scheme to get the right time step dt
# using a linesearch defined in optimiser.jl (a good choice being a backtracking
# of order 3 from the LineSearches Juila package). It takes in a function f,
# its gradient g!, a combination fg!, a vector xx and a linesearch lineSearch,
# and ouptuts the time step obtained by lineSearch from xx according to f. (This
# function was taken from a website.)
function lineSearch_dt(f::Function,g!::Function,fg!::Function,
        xx::Vector{Float64},lineSearch)::Float64

    x = copy(xx)
    gvec = similar(x)
    g!(gvec,x)
    fx = f(x)
    s = similar(gvec) # Step direction

    # Univariate line search functions
    ϕ(dt) = f(x .+ dt.*s)
    function dϕ(dt)
        g!(gvec,x .+ dt.*s)
        return dot(gvec,s)
    end
    function ϕdϕ(dt)
        phi = fg!(gvec,x .+ dt.*s)
        dphi = dot(gvec,s)
        return phi, dphi
    end

    s .= -gvec
    dϕ_0 = dot(s,gvec)

    return lineSearch(ϕ,dϕ,ϕdϕ,1.0,fx,dϕ_0)[1]  # returns time step dt
end
