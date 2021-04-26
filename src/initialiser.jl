################################################################################
## INITIALISE POSITIONS ########################################################
################################################################################


## PROJECTION ROUTINE FOR THE GFP SCHEME ##

# Function project takes a vector of particles as an input and updates it by
# orthogonally projecting the particles on the manifold _M. Needed for GFP
# scheme only. (Note that if the gradient of the square distance is zero at a
# given particle and that this particle is outside the manifold, then we choose
# not to project. One should make sure initially that all the particles are
# close enough to the manifold and to take dt small enough to avoid this issue.)
function project!(x::Vector{Particle})
    for i = 1:_n
        distSq_i = distanceSq(x[i])[1]
        if distSq_i > 0
            ∇distSq_i = distanceSq(x[i],grad=true)
            ∇distSq_norm_i = norm(∇distSq_i,2)
            if ∇distSq_norm_i > 0
                x[i] = x[i] - sqrt(distSq_i)*∇distSq_i/∇distSq_norm_i
            end
        end
    end
end

################################################################################



## INITIALISE PARTICLES ##

# Function initialise builds the initial positions on a given manifold initM for
# a given number of particles n. (Could be generalised to include nonuniform
# masses, i.e., could include the construction of initial masses alongside the
# initial positions.)
function initialise(initM::Manifoldd,n::Int64)::Vector{Particle}

    x0 = Vector{Particle}(undef,n)

    centre = initM.centre
    xboundary, yboundary = initM.xboundary, initM.yboundary
    xmax = xboundary()

    if _dim == 1

        if _randomness == "UG" || _randomness == "RPUG"
            xSample = collect(range(-xmax,xmax,length=n))
            x0 = map(i -> [xSample[i]],collect(1:n)) .+ [centre]   # store as
            # a vector of particles, and shift by centre

            if _randomness == "RPUG"
                perturb!(x0,_amp)    # perturb the UG
            end
        elseif _randomness == "RS"
            r = rand(n); xSample = -(1 .-r)*xmax + r*xmax
            x0 = map(i -> [xSample[i]],collect(1:n)) .+ [centre]
        end

    elseif _dim == 2

        n_sqrt = Int64(sqrt(n))
        if _randomness == "UG" || _randomness == "RPUG"
            xSample = collect(range(-xmax,xmax,length=n_sqrt))
            ymax = maximum(yboundary.(xSample))
            ySample = collect(range(-ymax,ymax,length=n_sqrt))
            xySample = reshape([[x,y] for x in xSample, y in ySample],n)
            # xySample is stored as a vector of particles of length n
            ydomain(x::Float64,y::Float64)::Bool = abs(y) <= yboundary(x)
            # ydomain needed to filter out particles in xySample which do not
            # belong to the interior of the manifold
            x0 = filter(xy -> ydomain(xy[1],xy[2]),xySample) .+ [centre]

            if _randomness == "RPUG"
                perturb!(x0,_amp)    # perturb the UG
            end
        elseif _randomness == "RS"
            r = rand(n); xSample = -(1 .-r)*xmax + r*xmax
            ymax = initM.yboundary.(xSample)
            r = rand(n); ySample = -(1 .-r).*ymax + r.*ymax
            x0 = map(i -> [xSample[i],ySample[i]],collect(1:n)) .+ [centre]
        end
    end

    return x0

end


# Function initialise builds the initial positions on the union of all the
# manifolds present in _initM.
function initialise()::Vector{Particle}

    n_initM = length(_initM)
    x0 = Vector{Particle}(undef,_n)

    n_component = Int64(_n/n_initM)   # number of sampling points per component
    # of initial manifold
    for i = 1:n_initM
        x0[1+(i-1)*n_component:i*n_component] =
            initialise(_initM[i],n_component)
    end

    if _scheme.name == "GFP"
        project!(x0)    # In the GFP case, we project the initial particles so
        # that the initial particles are plotted within the manifold
    end

    return x0

end
