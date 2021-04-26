################################################################################
## COMPUTE THE INTERACTION-DISTANCE ENERGY #####################################
################################################################################



## STORE A SAMPLE OF THE BOUNDARY OF EACH MANIFOLD OF INTEREST ##

_sample = Vector{Tuple{Vector{Particle},Vector{Particle}}}(undef,length(_M))
for i = 1:length(_M)
    M = _M[i]
    if M.dimension == 1
        _sample[i] = ([[Inf]],[[Inf]])  # not actually used since the distance to a
        # one-dimensional manifold can be computed exactly
    else
        _sample[i] = bdrySampleManifold(M,500.)  # _sample contains a sample
        # of the boundary of each manifold in the vector _M. Needed below to
        # compute the squared distance of a set of particles to the manifold
        # _M[i] using the function distanceSqToManifold in toolbox.jl. Only used
        # in dimension 2 for manifolds for which distances are not eactly known.
        # The density of sampling points can be changed here.
    end
end

################################################################################



## PREPARE FUNCTIONS FOR ENERGY COMPUTATION ##

# Function potential takes a particle p as an input and outputs either the value
# of the potential or its gradient at p, according to whether the optional input
# grad is false or true.
function potential(p::Particle;grad::Bool=false)::Union{Float64,Vector{Float64}}

    power = _W.power
    coeff = _W.coefficient
    if !grad
        _W.expression(coeff*norm(p)^power)

    elseif grad
        _W.derivative(coeff*norm(p)^power) *coeff*power*norm(p)^(power-2)* p
    end

end


# Function distanceSq is used to compute the distance between a particle p and
# the union or intersection (depending on the value of _union) of the manifolds
# in the vector _M.
function distanceSq(p::Particle;
        grad::Bool=false)::Union{Tuple{Float64,Vector{Int64}},Vector{Float64}}

    nM = length(_M)

    if !grad

        d_sq = zeros(Float64,nM)
        for i = 1:nM    # take the distance to every manifold in _M
            d_sq[i] = distanceSqToManifold(p,_M[i],_sample[i][1])
        end

        d_sq_all = 0.0
        if !_union
            d_sq_all = maximum(d_sq)    # take maximum if intersection (only
            # works for two manifolds whose union is the whole space, as for
            # example an annulus which is the intersection of the exterior of a
            # disc and the interior of another larger concentric disc)
        elseif _union
            d_sq_all = minimum(d_sq)    # take minimum if union
        end

        d_sq_all_ind = filter(j -> d_sq[j] == d_sq_all,1:nM)    # store the
        # indices in case there are two or more equally-distant closest
        # manifolds, in which case the gradient should be set to 0 below (only
        # relevant when _union is true)

        return d_sq_all, d_sq_all_ind

    elseif grad

        if length(distanceSq(p)[2]) > 1
            return zeros(Float64,_dim)  # set gradinet to 0 if more than one
            # manifold is equally distant to the closest one.
        else
            i_ref = distanceSq(p)[2][1]
            return distanceSqToManifold(p,_M[i_ref],_sample[i_ref][1],grad=true)
        end
    end

end

# Functions potential and distanceSq below take a vector x of particles as input
# and output the matrix and vector containing W(x_i-x_j) and d(x_i)^2 or their
# gradients, necessary to the computation of the energy later, according to
# whether the optional input grad is false or true.
function potential(x::Vector{Particle};
        grad::Bool=false)::Union{Matrix{Float64},Matrix{Particle}}

    if !grad

        W_mat = Matrix{Float64}(undef,_n,_n)
        for i = 1:_n
            W_mat[i,:] = map(y -> potential(x[i]-y),x)
        end
        return W_mat

    elseif grad

        ∇W_mat = Matrix{Particle}(undef,_n,_n)
        for i = 1:_n
            ∇W_mat[i,:] = map(y -> potential(x[i]-y,grad=true),x)
        end
        return ∇W_mat
    end

end

function distanceSq(x::Vector{Particle};
        grad::Bool=false)::Union{Vector{Float64},Vector{Particle}}

    if !grad
        d_sq_all(p::Particle)::Float64 = distanceSq(p)[1]
        return d_sq_all.(x)
    elseif grad
        ∇d_sq_all(p::Particle)::Vector{Float64} = distanceSq(p,grad=true)
        return ∇d_sq_all.(x)
    end

end

################################################################################



## ENERGY COMPUTATION ##

# Function energy takes a vector x of particles as inputs and outputs the value
# of the energy or of its gradients for the configuration x, according to
# whether the optional input grad is false or true.
function energy(x::Vector{Particle};
        grad::Bool=false)::Union{Float64,Vector{Particle}}

    m = 1/_n * ones(_n)     # we choose uniform masses. We could envisage
    # to generalise this, in particular to optimise on the masses too. In this
    # case we would need to add the term
    # ∇_mE = 1/ε* d_sq_vec + W_mat * m to the gradient of the energy below. We
    # would also need to be careful with the linesearch for the gradient flow
    # schemes in optimiser.jl to ensure that we include the masses in the
    # forward Euler update formula.

    if !grad
        W_mat = potential(x)
        if _scheme.name == "GFP"
            return 0.5*dot(W_mat * m,m)
        else
            ε = _scheme.epsilon
            d_sq_vec = distanceSq(x)
            return 1/ε* dot(d_sq_vec,m) + 0.5*dot(W_mat * m,m)
        end

    elseif grad
        ∇W_mat = potential(x,grad=true)
        if _scheme.name == "GFP"
            return m .* ∇W_mat * m
        else
            ε = _scheme.epsilon
            ∇d_sq_vec = distanceSq(x,grad=true)
            return m .* (1/ε* ∇d_sq_vec + ∇W_mat * m)
        end
    end

end
