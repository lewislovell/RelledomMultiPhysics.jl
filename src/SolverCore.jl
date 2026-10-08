module SolverCore

### Exports ###
export ModelParams, PerformanceData, Coords, VarCoords, BCs2D, BCConfig
export coord_allocation, coord_allocation_direct
export @d_xa, @d_ya, @d_xb, @d_yb, @d2_xa, @d2_ya
export eval_forcing!

### Shared Data Structures ###
@kwdef mutable struct ModelParams{T <: AbstractFloat, I <:Integer}
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
        mat[1:mid,:] .= 3375
        mat[(mid+1):end,:] .= 3300
        mat
    end
end

"""
    PerformanceData
Struct to pass solver performance outputs to the Benchmarks module for 
evaluation and comparisons.
"""
@kwdef struct PerformanceData
    method::Symbol
    float_type::DataType
    nx::Int
    ny::Int
    a_eff::Float64
    t_toc::Float64
    t_eff::Float64
end

# @kwdef struct BenchmarkData
#     n::Int
#     median_time::DataType
#     min_time::Int
#     max_time::Int
#     median_bandwidth::Float64
#     min_bandwidth::Float64
#     max_bandwidth::Float64
# end
"""
    Coords{XVEC, YVEC}
Basic structure containing x and y vector coordinates
"""
@kwdef struct Coords{XVEC, YVEC}
    xvec::XVEC
    yvec::YVEC
end
"""
    VarCoords{VX, VY, P}
Structure of structures containing the x and y coordinates for variables vx,
vy and pressure.
"""
@kwdef struct VarCoords{VX, VY, P}
    vx_coords::VX
    vy_coords::VY
    p_coords::P
end
"""
    BCs2D{L, R, T, B}
Structure containing the boundary conditions for the 2D domain. Can be single
values or functions.
"""
@kwdef struct BCs2D{L, R, T, B}
    left::L
    right::R
    top::T
    bottom::B
end
"""
    BCConfig{BC1, BC2, BC3}
Structure containing the boundary condition structures for each variable vx,
vy and pressure.
"""
@kwdef struct BCConfig{BC1, BC2, BC3}
    bc_vx::BC1
    bc_vy::BC2
    bc_p::BC3
end

### Finite Difference Macros ###
# Forward differences
macro d_xa(A) esc(:($A[ix+1,iy]-$A[ix,iy])) end
macro d_ya(A) esc(:($A[ix,iy+1]-$A[ix,iy])) end
# Backward differences
macro d_xb(A) esc(:($A[ix,iy]-$A[ix-1,iy])) end
macro d_yb(A) esc(:($A[ix,iy]-$A[ix,iy-1])) end
# Central second derivatives
macro d2_xa(A) esc(:($A[ix+1,iy]-2*$A[ix,iy]+$A[ix-1,iy])) end
macro d2_ya(A) esc(:($A[ix,iy+1]-2*$A[ix,iy]+$A[ix,iy-1])) end

### Function Utilities ###
"""
    coord_allocation(dx,dy,xsize,ysize)
Struct to store coordinate vectors for x-y Stokes and Continuity equation nodal
points.
"""
function coord_allocation(dx,dy,xsize,ysize)
    nx = round(Int, xsize/dx) + 1
    ny = round(Int, ysize/dy) + 1
    vx_coords = Coords(
        xvec = range(0; step=dx, length=nx),
        yvec = range(dy/2; step=dy, length=ny-1)
    )
    vy_coords = Coords(
        xvec = range(dx/2; step=dx, length=nx-1),
        yvec = range(0; step=dy, length=ny)
    )
    p_coords = Coords(
        xvec = range(dx/2; step=dx, length=nx-1),
        yvec = range(dy/2; step=dy, length=ny-1)
    )
    return VarCoords(vx_coords,vy_coords,p_coords)
end
"""
    coord_allocation_direct(dx,dy,xsize,ysize)
Struct to store coordinate vectors for x-y Stokes and Continuity equation nodal
points using the DirectMonolithis approach
"""
function coord_allocation_direct(dx,dy,xsize,ysize)
    vx_coords = Coords(
        xvec = 0.0:dx:(xsize+dx),
        yvec =  (-dy/2):dy:(ysize+dy/2)
    )
    vy_coords = Coords(
        xvec = (-dx/2):dx:(xsize+dx/2),
        yvec =  0.0:dy:(ysize+dy)
    )
    p_coords = Coords(
        xvec = (-dx/2):dx:(xsize+dx/2),
        yvec = (-dy/2):dy:(ysize+dy/2)
    )
    return VarCoords(vx_coords,vy_coords,p_coords)
end

"""
    eval_forcing!(fy, rho, gy)
Function to unpack density array into forcing term in y-Stokes.
"""
function eval_forcing!(fy, rho, gy)
    # fy = -gy (rho1+rho2)/2
    nx, ny = size(fy)[1]+1, size(fy)[2]
    for iy = 1:ny
        for ix = 1:nx-1
            fy[ix, iy] = gy*(rho[ix,iy] + rho[ix+1,iy])/2
        end
    end
    return nothing
end

end