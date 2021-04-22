################################################################################
## PLOT SOLUTION ###############################################################
################################################################################



## FILE NAMING ##

# Function nameFigure takes a string as an input and outputs a string used later
# to name figures, starting with the input string prefix.
function nameFigure(prefix::String)::String

    nM = length(_M)   # get number of manifolds

    nameW, powerW, coeffW = _W.name, _W.power, _W.coefficient
    figNamePos = ["./plots/$prefix", "$_dim", "d", "_$nameW($powerW,$coeffW)_"]
    for i = 1:nM
        M = _M[i]; name = M.name
        centre, parameters, side = M.centre, M.parameters, M.side
        if parameters == nothing
            if !(name == "Point1d" || name == "Point2d")
                figNamePos = vcat(figNamePos, ["$name($centre,$side)"])
            else
                figNamePos = vcat(figNamePos, ["$name($centre)"])
            end
        else
            figNamePos = vcat(figNamePos, ["$name($centre,$parameters,$side)"])
        end
    end

    if !_union
        figNamePos = vcat(figNamePos, ["-Inters"])
    end

    figNamePos = vcat(figNamePos, ["_$_n", "_"])

    for i = 1:length(_initM)
        initM = _initM[i]; initName = initM.name
        initCentre, initParameters = initM.centre, initM.parameters
        if initParameters == nothing
            figNamePos = vcat(figNamePos, ["$initName($initCentre)"])
        else
            figNamePos = vcat(figNamePos,
                ["$initName($initCentre,$initParameters)"])
        end
    end

    nameScheme, tol, ε = _scheme.name, _scheme.tolerance, _scheme.epsilon
    if nameScheme == "GFP"
        figNamePos = join(vcat(figNamePos, ["-$_randomness",
            "_$nameScheme($tol)", "-$_finalT"]))
    else
        figNamePos = join(vcat(figNamePos, ["-$_randomness",
            "_$nameScheme($tol,$ε)", "-$_finalT"]))
    end

    for i = 1:nM
        name = _M[i].name
        if name == "Point1d" || name == "Point2d" ||
                name == "Interval" || name == "Disc" ||
                name == "R1" || name == "R2"
            figNamePos = join(vcat(figNamePos, [".pdf"]))
        else
            figNamePos = join(vcat(figNamePos, ["-$_sampling", ".pdf"]))
        end
    end

    return figNamePos
end

################################################################################



## PLOTS WITHOUT DYNAMICS ##

# Function plotSol takes in vectors x0 and x of particles, and outputs plots for
# the initial profile x0 and the final profile x. This function does not need
# dynamics to be used.
function plotSol(x0::Vector{Particle},x::Vector{Particle})

    nM = length(_M)   # get number of manifolds

    x0_mat = vvm(x0)        # transform x0 into a matrix _dim x _n
    x_mat = vvm(x)          # transform x into a matrix _dim x _n

    # Below we set the default settings for plotting
    default(markersize=3.5,markerstrokewidth=0.3,linewidth=2,linealpha=0.5,
        xlab = L"x",ylab = L"y",zlab = L"z",legend=false)


    figNamePos = nameFigure("positions")    # set the name of the file to be
    # output (for positions)

    if _dim == 1

        scatter(x0_mat[1,:],0.25*ones(_n),color=:blue)  # for better visibility
        # on the plot, place the initial particles at y = 0.25
        scatter!(x_mat[1,:],zeros(_n),color=:red)

        # Below we plot the manifold(s)
        for i = 1:nM
            M = _M[i]
            name, centre = M.name, M.centre

            if name == "Point1d"    # if point, only scatter one point
                scatter!(centre,[0.0],color=:gray,aspect_ratio=:equal,
                    markersize = 4.5,markeralpha = 0.5,markerstrokewidth = 0)

            elseif name != "R1"
                xboundary = M.xboundary
                bdryShifted = xboundary() + centre[1]   # shift the boundary
                minus_bdryShifted = -xboundary() + centre[1]    # shift the
                # symmetric boundary around 0
                scatter!([bdryShifted],[0.0],color=:gray,
                    markersize = 4.5,markeralpha=0.5,markerstrokewidth = 0)
                scatter!([minus_bdryShifted],[0.0],color=:gray,
                    markersize = 4.5,markeralpha=0.5,markerstrokewidth = 0)
            end
        end

        if _title
            plot!(title=L"Particle\ positions")
        end
        display(plot!(yaxis=false,ylims=(-0.25,0.5)))     # switch of y-axis
        savefig(figNamePos)

    elseif _dim == 2

        if _arrows
            ∇E0 = energy(x0,grad=true)  # get the gradient of the energy to
            # plot the initial gradient field using quiver
            ∇E0_mat = vvm(∇E0)          # convert ∇E0 into a matrix _dim x _n
            ∇E0_max = 1.0
            ∇E0_norm = maximum(norm(norm.(∇E0,Inf),Inf))
            if ∇E0_norm > 0     # if the norm of the energy > 0, rescale the max
                ∇E0_max = 2.5*maximum(norm(norm.(∇E0,Inf),Inf))
            end
            quiver(x0_mat[1,:],x0_mat[2,:],color=:blue,linealpha = 1,
                linewidth = 1,
                quiver=(-∇E0_mat[1,:]/∇E0_max,-∇E0_mat[2,:]/∇E0_max))
                # above, we normalise the gradient field by the max of the
                # gradient to have shorter arrows
            scatter!(x0_mat[1,:],x0_mat[2,:],color=:blue)
        else
            scatter(x0_mat[1,:],x0_mat[2,:],color=:blue)
        end

        scatter!(x_mat[1,:],x_mat[2,:],color=:red)

        # Below we plot the manifold(s)
        for i = 1:nM
            M = _M[i]
            name, centre = M.name, M.centre

            if name == "Point2d"    # if point, only scatter one point
                scatter!([centre[1]],[centre[2]],color=:gray,
                    markersize = 4.5,markeralpha = 0.5,markerstrokewidth = 0,
                    aspect_ratio=:equal)

            elseif name != "R2"
                xboundary, yboundary = M.xboundary, M.yboundary
                bdrySampleM = _sample[i]    # sample boundary of manifold M
                # (already shifted by centre); see energy.jl
                bdry = vvm(bdrySampleM[1])          # convert to particle matrix
                minus_bdry = vvm(bdrySampleM[2])    # same for the x-symmetry
                plot!(bdry[1,:],bdry[2,:],color=:gray)
                plot!(minus_bdry[1,:],minus_bdry[2,:],color=:gray,
                    aspect_ratio=:equal)
            end
        end

        if _title
            plot!(title = L"Particle\ positions")
        end
        display(plot!())
        savefig(figNamePos)
    end

end

################################################################################



## PLOTS WITH DYNAMICS ##

# Function plotSol (second method) takes in the initial vector x0 of particles,
# the matrices xt, tt, Et and Dt, and outputs plots for all times coming from
# the gradient flow schemes, i.e., the dynamics.  It still calls function
# plotSol above.
function plotSol(x0::Vector{Particle},xt::Matrix{Particle},tt::Vector{Float64},
    Et::Vector{Float64},Dt::Vector{Float64})

    nM = length(_M)   # get number of manifolds

    x = xt[end,:]
    plotSol(x0,x)   # call previous method to plot final results

    if _log
        zoom = 1e-2 # how much to "enlarge" small times; the smaller zoom,
        # the more enlargement
        ttlog = log.(tt .+ zoom) # rescale time with log (shift it by zoom
        # to avoid zeros in log scale
    end


    figNameTraj = nameFigure("trajectories")    # set the name of the file to be
    # output (for trajectories)

    if _dim == 1

        xt_arr = mva(xt)    # convert the matrix of vectors xt to an array
        # (to be able to plot the trajectories)

        if !_log
            plot(transpose(xt_arr[1,:,:]),tt)   # need transpose to move time
            # dimension from the second (columns) to the first dimension (rows)
            if _title
                plot!(title = "Particle trajectories")
            end
            display(plot!(xlab = L"x",ylab = L"t"))
        elseif _log
            y_scale(a) = round(exp(a) - zoom,sigdigits=3) # inverse of above
            # operation in ttlog; round it to avoid too many digits in plot
            plot(transpose(xt_arr[1,:,:]),ttlog)   # start from second
            # element to avoid t = 0 for the log scale
            if _title
                plot!(title = "Particle trajectories")
            end
            display(plot!(xlab = L"x",ylab = L"t\ (log)",yaxis = y_scale))
        end
        savefig(figNameTraj)

    elseif _dim == 2

        xt_arr = mva(xt)

        angle = (40,30)     # camera angle for 2d trajectory plots
        if !_log
            plot(transpose(xt_arr[1,:,:]),transpose(xt_arr[2,:,:]),tt)
            if _title
                plot!(title = "Particle trajectories")
            end
            display(plot!(xlab = L"x",ylab = L"y",zlab = L"t",camera = angle))
        elseif _log
            z_scale(a) = round(exp(a) - zoom,sigdigits=3)
            plot(transpose(xt_arr[1,:,:]),transpose(xt_arr[2,:,:]),ttlog)
            if _title
                plot!(title = "Particle trajectories")
            end
            display(plot!(xlab = L"x",ylab = L"y",zlab = L"t\ (log)",
                zaxis = z_scale,camera = angle))
        end
        savefig(figNameTraj)
    end


    figNameEn = nameFigure("energy")    # set the name of the file to be output
    # (for energy)

    if !_log
        plot(tt,Et,label = "Energy")
        plot!(tt,Dt,label = "Squared distance")
        if _title
            plot!(title = "Energy and squared distance evolution")
        end
        display(plot!(xlab = L"t",ylab = L"E(t),d_M^2(t)",legend = true))

    elseif _log
        x_scale(a) = round(exp(a) - zoom,sigdigits=3)
        plot(ttlog,Et,label = "Energy")
        plot!(ttlog,Dt,label = "Squared distance")
        if _title
            plot!(title = "Energy and squared distance evolution")
        end
        display(plot!(xlab = L"t\ (log)",ylab = L"E(t),d_M^2(t)",
            xaxis = x_scale,legend = true))
    end
    savefig(figNameEn)

end
