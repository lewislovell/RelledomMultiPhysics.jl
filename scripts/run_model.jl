module ModelRunner

include("../src/StokesSolvers.jl")
include("../src/Validations/mms_isovisc_stokes.jl")

using Plots
using .StokesSolvers
import .MMSIsoviscStokes as MMS

@kwdef struct BCConfig{BC1, BC2, BC3}
    bc_vx::BC1
    bc_vy::BC2
    bc_p::BC3
end

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

bc_vx = BCs2D(
        left = 0.0,
        right = 0.0,
        top = (x) -> sin(x*pi/1_000_000)*1e-8,
        bottom = (x) -> -sin(x*pi/1_000_000)*1e-8
    )
bc_vy = BCs2D(
        left = (y) -> sin(y*pi/1_500_000)*1e-8,
        right = (y) -> -sin(y*pi/1_500_000)*1e-8,
        top = 0.0,
        bottom = 0.0
    )
bc_p = BCs2D(
        left = 0.0,
        right = 0.0,
        top = 0.0,
        bottom = 0.0
    )

@kwdef struct ModelParams{T <: AbstractFloat, I <:Integer}
    # Physics
    w::T = T(1_000_000) # Domain width [m]
    h::T = T(1_500_000) # Domain height [m]
    gy::T = T(10.0)     # Gravitational acceleration [m/s^2]
    eta::T = T(1e21)    # Viscosity [Pa s]
    # Numerics
    nx::I = I(101)     # Number of gridpoints in x
    ny::I = I(151)     # Number of gridpoints in y
    # Derived Numerics
    dx::T = T(w/(nx-1))
    dy::T = T(h/(ny-1)) # Grid spacings [m]
    dn::I = I(3*(ny+1)) # Number of elements in linear system column
    rho::Matrix{T} = let
        mat = zeros(T, nx, ny)
        mid = I(round(nx/2))
        mat[1:mid,:] .= 3400
        mat[(mid+1):end,:] .= 3300
        mat
    end
end

cpu_config = MMS.MMSParameters{Float64, Int64}()
gpu_config = MMS.MMSParameters{Float32, Int32}()

function plot_output(xvec, yvec, var, title_str; cbar=:berlin)
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

function main()
    Params = MMS.MMSParameters{Float64, Int64}()
    # Params = gpu_config
    # Params = ModelParams{Float32, Int32}
    # BCCParams = BCConfig(bc_vx, bc_vy, bc_p)

    MMSSol, MMSBcs = MMS.mms_2d_validation(Params)

    (vx_num, vy_num, p_num), Coordinates = isoviscous_stokes(Params, MMSBcs, 
        VelP(), DirectMonolithic(), CPUSingle(), false, true, MMSSol = MMSSol,
        VldParams=Params)

    # (vx_num, vy_num, p_num), Coordinates = isoviscous_stokes(Params(), BCCParams, 
    #     VelP(), DirectMonolithic(), CPUSingle(), false, false)


    plot_output(Coordinates.vx_coords.xvec,
        Coordinates.vx_coords.yvec,
        vx_num*1e9,
        "Vx-Velocity Numeric *10^-9, [m/s]"
    )
    plot_output(Coordinates.vy_coords.xvec,
        Coordinates.vy_coords.yvec,
        vy_num*1e9,
        "Vy-Velocity Numeric *10^-9, [m/s]"
    )
    plot_output(Coordinates.p_coords.xvec,
        Coordinates.p_coords.yvec,
        p_num*1e-9,
        "Pressure Numeric, [GPa]"
    )

    MMS.analyt_compare(MMSSol, vx_num, vy_num, p_num, Coordinates, 
        DirectMonolithic(), true, true)
end

if isinteractive() || abspath(PROGRAM_FILE) == abspath(@__FILE__)
    Base.invokelatest(() -> getfield(@__MODULE__, :main)())
end
end