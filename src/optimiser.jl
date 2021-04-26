################################################################################
## FIND SOLUTION BY OPTIMISING OR USING GRADIENT FLOW ##########################
################################################################################



## REDEFINE ENERGY AND GRADIENT OF ENERGY TO BE USED BY OPTIM AND LINESEARCHES
## PACKAGES ##

# Function E_obj gives the energy as a function of 2*_dim arguments (or a vector
# of 2*_dim elements). It is the objective function to optimise.
function E_obj(xx::Vector{Float64})::Float64
    xx_reshape = vvm_inv(reshape(xx,(_dim,_n)))     # xx_reshape is a vector
    # of points
    return energy(xx_reshape)
end


# Function ∇E_obj! gives the gradient of the energy as a function of 2_dim
# arguments. It takes a container as an input.
function ∇E_obj!(G::Vector{Float64},xx::Vector{Float64})
    xx_reshape = vvm_inv(reshape(xx,(_dim,_n))) # xx_reshape is a vector
    # of points
    ∇E = energy(xx_reshape,grad=true)
    for i = 1:_dim:_dim*_n
        j = Int64((i+_dim-1)/_dim)
        for d = 0:(_dim-1)
            G[i+d] = ∇E[j][1+d]     # store gradient as a vector of floats
        end
    end
end


# Function E_obj∇E_obj! is needed for the linesearch in the Dynamics scheme.
function E_obj∇E_obj!(G::Vector{Float64},xx::Vector{Float64})
    ∇E_obj!(G,xx)
    E_obj(xx)
end

################################################################################



## OPTIMISATION ##

# Function optimise takes a vector x0 as an input and outputs the result
# from the function optimize from package Optim or using a the gradient flow
# scheme for the energy, according to the parameter _scheme. It is used to find
# a local minimiser to the energy (in case of Optim) or determine the dynamics
# (in case of the GF).
function optimise(x0::Vector{Particle})::
        Union{Vector{Particle},
        Tuple{Matrix{Particle},Vector{Float64},Vector{Float64},Vector{Float64}}}

    tol = _scheme.tolerance

    if _scheme.name != "GFeps" && _scheme.name != "GFP"

        method_Optim = getfield(Main,Symbol(_scheme.nameOptim)) # transform the
        # string of the scheme's name into a symbol to be used in the optimize
        # function below of the Optim package

        xx = reshape(vvm(x0),_dim*_n)   # convert from particles to floats
        opt = optimize(E_obj,∇E_obj!,xx,method_Optim(),
            Optim.Options(g_tol=tol,show_trace=true,iterations = 10000))
        # tolerance set on the gradient of the energy, we display results at end
        # of each optimisation step and we abort computation after 10000
        # iterations.
        display(opt)    # display final result

        xx = Optim.minimizer(opt)   # store the optimised positions
        x = vvm_inv(reshape(xx,(_dim,_n)))  # convert back to particles

        return x

    elseif _scheme.name == "GFeps" || _scheme.name == "GFP"

        x = Vector{Particle}(undef,_n)
        x = copy(x0)    # initilise x by making sure x0 won't change (with copy)
        xt = reshape(x0,1,:)    # reshape x so it is an Array{Particle,2},
        # otherwise it won't work with vcat later

        t = 0       # initialise time
        tt = t      # used later for vcat to store all times

        Et = energy(x)    # store initial energy to be stored with all times
        ∇Ex = energy(x,grad=true)       # initialise gradient of energy

        Dt = norm(distanceSq(x),Inf)    # store initial squared distance of all
        # particles according to infinity norm

        if _scheme.name == "GFP"
            cStop = tol + 1.0   # We make sure to enter the time loop below
            println("E = ",round(Et,digits=10), "  d_sq = ",round(Dt,digits=10),
                "  t = ",round(t,digits=4))
        else
            cStop = norm(norm.(∇Ex,Inf),Inf)    # stopping criterion (as in
            # Optim package, we take the infinity norm)
            println("E = ",round(Et,digits=10),
                "  |∇E| = ",round(cStop,digits=10),"  t = ",round(t,digits=4))
        end

        while cStop > tol && t < _finalT # Stop simulation when the wanted
            # tolerance of final time is reached

            if _pert
                perturb!(x,_amp/(1+t))  # perturb particles with decreasing
                # amplitude
            end

            xx = reshape(vvm(x),_dim*_n)    # convert from particles to floats

            dt = lineSearch_dt(E_obj,∇E_obj!,E_obj∇E_obj!,xx,
                BackTracking(order=3))      # linesearch with backtracking
            dt /= _n                        # correct dt so that it gives a
            # physical time. Indeed, the gradient flow is not exactly the
            # gradient descent of the energy since we divide the energy by
            # the masses in the forward Euler formula below. This approach
            # would need to be checked if we decide to consider nonuniform
            # masses.
            dt *= _corrdt                   # correction of dt if too large (can
            # also be used to increase time step)

            x .= x .- (_n*dt)* ∇Ex         # update x via gradient flow with
            # forward Euler in time (here m = 1/_n * ones(_n))

            if _scheme.name == "GFP"
                project!(x)     # In the GFP case, we project the particles
            end

            t = t + dt                      # update time

            Ex = energy(x)                  # update energy
            ∇Ex = energy(x,grad=true)       # update energy gradient

            D = norm(distanceSq(x),Inf)     # update squared distance

            xt = vcat(xt,reshape(x,1,:))    # add updated positions to old ones
            tt = vcat(tt,t)                 # add new time to old ones
            Et = vcat(Et,Ex)                # add new energy to old ones
            Dt = vcat(Dt,D)                 # add new distance to old ones

            # Below we update stopping criterion. In case the scheme is GFP we
            # look at the time derivative of the energy rather than its gradient
            # since the gradient does not necessarily converge to 0 because of
            # the aposteriori projection of the particles on the manifold.
            if _scheme.name == "GFP"
                cStop = abs(Et[end] - Et[end-1])/dt
                println("E = ",round(Ex,digits=10),
                    "  d_sq = ",round(D,digits=10),
                    "  |dE/dt| = ",round(cStop,digits=10),
                    "  t = ",round(t,digits=4),"  dt = ",round(dt,digits=4))
            else
                cStop = norm(norm.(∇Ex,Inf),Inf)
                println("E = ",round(Ex,digits=10),
                    "  d_sq = ",round(D,digits=10),
                    "  |∇E| = ",round(cStop,digits=10),
                    "  t = ",round(t,digits=4),
                    "  dt = ",round(dt,digits=4))
            end
        end

        return xt, tt, Et, Dt
    end

end
