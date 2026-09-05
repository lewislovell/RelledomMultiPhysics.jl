module MMSIsoviscStokes

using ..StokesSolvers
using Plots

"""
    BCs2D{L, R, T, B}
Contains the boundary conditions for the 2D domain
"""
@kwdef struct BCs2D{L, R, T, B}
    left::L
    right::R
    top::T
    bottom::B
end
"""
    StokesVars{VX, VY, P, F1, F2}
Contains analytical solutions for Stokes MMS
"""
@kwdef struct StokesVars{VX, VY, P, F1, F2}
    vx::VX
    vy::VY
    p::P
    xstokes::F1
    ystokes::F2
end
"""
    StokesBCs{BC1, BC2, BC3}}
Contains analytical BCs for Stokes MMS
"""
@kwdef struct StokesBCs{BC1, BC2, BC3}
    bc_vx::BC1
    bc_vy::BC2
    bc_p::BC3
end
"""
    MMSParameters{FloatType}
Configuration parameters for the Mehod of Manufactured Solutions (MMS)

Coordinate System          x
     x=0            x=w  +--->
 y=0 +---------------+   | 
     |               |   |y
     |               |   V
  h  |               |
     |               |
     |               |
     |               |
 y=h +---------------+
             w
Based on Exercise 20.3 in Gerya (2019)
"""
@kwdef struct MMSParameters{T <: AbstractFloat, I <:Integer}
    # Physics
    w::T = T(1_000_000)     # Model width [m]
    h::T = T(1_500_000)     # Model hight [m]
    vx0::T = T(1e-9)        # Horizontal velocity [m/s]
    vy0::T = T(3e-9)        # Vertical velocity [m/s]
    p0::T = T(1e5)          # Initial pressure [Pa]
    dp0dy::T = T(33_000.0)  # Hydrostatic pressure [Pa/m]
    deltap::T = T(1e6)      # Dynamic pressure scaling [Pa]
    eta::T = T(1e19)        # Viscosity [Pa s]
    gy::T = T(10.0)     # Gravitational acceleration [m/s^2]
    # Numerics
    nx::I = I(51)     # Number of gridpoints in x
    ny::I = I(101)     # Number of gridpoints in y
    # Derived Numerics
    dx::T = T(w/(nx-1))
    dy::T = T(h/(ny-1)) # Grid spacings [m]
    dn::I = I(3*(ny+1)) # Number of elements in linear system column
end

struct VldData{FNUM,FANLYT}
    fnum::FNUM
    fanlyt::FANLYT
end

function validate(outputdata::VldData)
    (; fnum, fanlyt) = outputdata
    res_mat = fnum .- fanlyt # Residuals matrix
    n = length(res_mat)
    l_inf = maximum(abs.(res_mat))
    l1_norm = sum(abs.(res_mat)) / n
    l2_norm = sqrt(sum(map(r -> r^2,res_mat))/n)
    println("L_inf: $l_inf, L1-Norm: $l1_norm, L2-Norm: $l2_norm")
    return res_mat
end

function compute_anlyt(sol_func, coord_struct)
    (; xvec, yvec) = coord_struct
    nx = length(xvec)
    ny = length(yvec)
    sol_mat = zeros(ny,nx)
    for i in 1:ny
        for j = 1:nx
            sol_mat[i,j] = sol_func(xvec[j],yvec[i])
        end
    end
    return sol_mat
end

function analyt_compare(mms_sol, vx_num, vy_num, p_num, coordinates,
    ::DirectMonolithic, plot_anlyt::Bool = true, plot_error::Bool = true)
    # Compute analytical solutions
    vx_anlyt = compute_anlyt(mms_sol.vx, coordinates.vx_coords)
    vy_anlyt = compute_anlyt(mms_sol.vy, coordinates.vy_coords)
    p_anlyt  = compute_anlyt(mms_sol.p,  coordinates.p_coords)

    ### Validation Results ###
    println("Output for Vx:\n")
    res_vx = validate(VldData(vx_num[:,1:end-1], vx_anlyt[:,1:end-1]))
    println("\n Output for Vy:\n")
    res_vy = validate(VldData(vy_num[1:end-1,:], vy_anlyt[1:end-1,:]))
    println("\n Output for P:\n")
    res_p = validate(VldData(p_num[2:end-1,2:end-1], p_anlyt[2:end-1,2:end-1]))
    if plot_anlyt
        plot_output(coordinates.vx_coords.xvec[1:end-1],
            coordinates.vx_coords.yvec,
            vx_anlyt[:,1:end-1]*1e9,
            "Vx-Velocity Analytic *10^-9, [m/s]"
        )
        plot_output(coordinates.vy_coords.xvec,
            coordinates.vy_coords.yvec[1:end-1],
            vy_anlyt[1:end-1,:]*1e9,
            "Vy-Velocity Analytic *10^-9, [m/s]"
        )
        plot_output(coordinates.p_coords.xvec[2:end-1],
            coordinates.p_coords.yvec[2:end-1],
            p_anlyt[2:end-1,2:end-1]*1e-9,
            "Pressure Analytic, [GPa]"
        )
    end
    if plot_error
        plot_output(coordinates.vx_coords.xvec[1:end-1],
            coordinates.vx_coords.yvec,
            res_vx*1e9,
            "Error in vx-Velocity *10^-9, [m/s]"; 
            cbar=:vik
        )
        plot_output(coordinates.vy_coords.xvec,
            coordinates.vy_coords.yvec[1:end-1],
            res_vy*1e9,
            "Error in vy-Velocity *10^-9, [m/s]";
            cbar=:vik
        )
        plot_output(coordinates.p_coords.xvec[2:end-1],
            coordinates.p_coords.yvec[2:end-1],
            res_p,
            "Error in Pressure, [Pa]";
            cbar=:vik
        )
    end
end

function analyt_compare(mms_sol, vx_num, vy_num, p_num, coordinates,
    ::PseudoTransient, plot_anlyt::Bool = true, plot_error::Bool = true)
    # Compute analytical solutions
    vx_anlyt = compute_anlyt(mms_sol.vx, coordinates.vx_coords)
    vy_anlyt = compute_anlyt(mms_sol.vy, coordinates.vy_coords)
    p_anlyt  = compute_anlyt(mms_sol.p, coordinates.p_coords)

    ### Validation Results ###
    println("Output for Vx:\n")
    res_vx = validate(VldData(vx_num, vx_anlyt))
    println("\n Output for Vy:\n")
    res_vy = validate(VldData(vy_num, vy_anlyt))
    println("\n Output for P:\n")
    res_p = validate(VldData(p_num, p_anlyt))
    if plot_anlyt
        plot_output(coordinates.vx_coords.xvec,
            coordinates.vx_coords.yvec,
            vx_anlyt*1e9,
            "Vx-Velocity Analytic *10^-9, [m/s]"
        )
        plot_output(coordinates.vy_coords.xvec,
            coordinates.vy_coords.yvec,
            vy_anlyt*1e9,
            "Vy-Velocity Analytic *10^-9, [m/s]"
        )
        plot_output(coordinates.p_coords.xvec,
            coordinates.p_coords.yvec,
            p_anlyt*1e-9,
            "Pressure Analytic, [GPa]"
        )
    end
    if plot_error
        plot_output(coordinates.vx_coords.xvec,
            coordinates.vx_coords.yvec,
            res_vx*1e9,
            "Error in vx-Velocity *10^-9, [m/s]"; 
            cbar=:vik
        )
        plot_output(coordinates.vy_coords.xvec,
            coordinates.vy_coords.yvec,
            res_vy*1e9,
            "Error in vy-Velocity *10^-9, [m/s]";
            cbar=:vik
        )
        plot_output(coordinates.p_coords.xvec,
            coordinates.p_coords.yvec,
            res_p,
            "Error in Pressure, [Pa]";
            cbar=:vik
        )
    end
end

function plot_output(xvec, yvec, var, title_str; cbar = :berlin)
    varplot = heatmap(
        xvec, yvec, var,
        xlabel = "Distance [m]",
        ylabel = "Depth [m]",
        title = title_str,
        # colorbar_title = "Velocity (m/s)",
        yflip = true,
        # aspect_ratio = :equal,
        c = cgrad(cbar, rev = false),
        right_margin=15Plots.mm
    )
    display(varplot)
end


"""
    mms_2d_validation(config::MMSParameters{T}) where {T})

Generate MMS Solution for 2D Isoviscous Incompressible Stokes Equation.
Returns a `StokesVars` object containing analytical functions and boundary
conditions
"""
function mms_2d_validation(config::MMSParameters{T}) where {T}
    (; w, h, vx0, vy0, p0, dp0dy, deltap, eta) = config
    z = zero(T)
    π = T(pi)
    # Construct Boundaries
    bc_vx = BCs2D(
        left = z,
        right = z,
        top = (x) -> -vx0*sin(2π*x/w),
        bottom = (x) -> vx0*sin(2π*x/w)
    )
    bc_vy = BCs2D(
        left = (y) -> vy0*sin(π*y/h),
        right = (y) -> vy0*sin(π*y/h),
        top = z,
        bottom = z
    )
    bc_p = BCs2D(
        left = (y) -> p0 + y*dp0dy + deltap*sin(π*y/h),
        right = (y) -> p0 + y*dp0dy + deltap*sin(π*y/h),
        top = p0,
        bottom = p0 + h*dp0dy
    )

    d2vxdx2 = (x,y) -> vx0*4π^2/w^2 * sin(2π*x/w)*cos(π*y/h)
    d2vxdy2 = (x,y) -> vx0*π^2/h^2 * sin(2π*x/w)*cos(π*y/h)
    d2vydx2 = (x,y) -> -vy0*4π^2/w^2 * cos(2π*x/w)*sin(π*y/h)
    d2vydy2 = (x,y) -> -vy0*π^2/h^2 * cos(2π*x/w)*sin(π*y/h)
    dpdx = (x,y) -> -deltap*2π/w * sin(2π*x/w)*sin(π*y/h)
    dpdy = (x,y) -> dp0dy + deltap*π/h * cos(2π*x/w)*cos(π*y/h)

    xstokes = (x,y) -> eta*(d2vxdx2(x,y)+d2vxdy2(x,y))-dpdx(x,y)
    ystokes = (x,y) -> eta*(d2vydx2(x,y)+d2vydy2(x,y))-dpdy(x,y)
    
    # Analytical Solutions
    vx = (x, y) -> -vx0*sin(2π*x/w)*cos(π*y/h)
    vy = (x, y) -> vy0*cos(2π*x/w)*sin(π*y/h)
    p = (x, y) -> p0 + y*dp0dy + deltap*cos(2π*x/w)*sin(π*y/h)
    
    return StokesVars(vx, vy, p, xstokes, ystokes),
        StokesBCs(bc_vx, bc_vy, bc_p)
end
end

# @code_warntype mms_2d_validation(gpu_config)

