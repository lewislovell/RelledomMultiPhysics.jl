include("../src/StokesSolvers.jl")

using.StokesSolvers

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
        top = 0.0,
        bottom = 0.0
    )
bc_vy = BCs2D(
        left = 0.0,
        right = 0.0,
        top = 0.0,
        bottom = 0.0
    )
bc_p = BCs2D(
        left = 0.0,
        right = 0.0,
        top = 0.0,
        bottom = 0.0
    )

bc_params = BCConfig(bc_vx, bc_vy, bc_p)

@kwdef struct ModelParams{T <: AbstractFloat, I <:Integer}
    # Physics
    w::T = T(1_000_000) # Domain width [m]
    h::T = T(1_500_000) # Domain height [m]
    gy::T = T(10.0)     # Gravitational acceleration [m/s^2]
    eta::T = T(1e21)    # Viscosity [Pa s]
    # Numerics
    nx::I = I(51)     # Number of gridpoints in x
    ny::I = I(101)     # Number of gridpoints in y
    # Derived Numerics
    dx::T = T(w/(nx-1))
    dy::T = T(h/(ny-1)) # Grid spacings [m]
    dn::I = I(3*(ny+1)) # Number of elements in linear system column
end

function main()
    params = ModelParams{Float32, Int32}()
    isoviscous_stokes(params, bc_params, VelP(), DirectMonolithic(), 
        CPUSingle(), true, false)
end

main()