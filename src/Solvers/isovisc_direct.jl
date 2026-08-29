using SparseArrays
using TimerOutputs
"""
    function sfd_stokes_const_eta(ModelParams, BCConfig, benchmark::bool = false,
    validate::Bool = false; mm_sol= nothing, mms_params= nothing)
Run isoviscous, incompressible stokes equations using a staggered-finite-
difference numerical formulation. Includes benchmarking and validation.
"""
function isoviscous_stokes(ModelParams, BCConfig, ::VelP, ::DirectMonolithic,
    ::CPUSingle, benchmark::Bool = false, is_validate::Bool = false; 
    mms_sol = nothing, VldParams = nothing)
    
    # Input ModelParams
    w, h   = ModelParams.w, ModelParams.h    # Domain size [m]
    nx, ny = ModelParams.nx, ModelParams.ny  # Model gridpoints
    dx, dy = ModelParams.dx, ModelParams.dy  # Grid spacings [m]
    dn     = ModelParams.dn          # Number of elements in linear system column
    eta    = ModelParams.eta         # Viscosity [Pa s]
    gy     = ModelParams.gy

    if is_validate && !isnothing(VldParams)
        w, h   = VldParams.w, VldParams.h # Domain size [m]
        eta    = VldParams.eta             # Viscosity [Pa s]
    end

    # Derived ModelParams
    idx, idy = 1/dx, 1/dy        # Inverse grid spacings [1/m]
    idx2, idy2 = 1/dx^2, 1/dy^2  # Inverse grid spacings squared [1/m^2]
    idxdy = 1/(dx*dy)            # Inverse grid spacings product [1/m^2]
    kconst = 2*eta/(dx+dy)       # Linar system scaling constant [Pa s/m]
    # Density setup at basic nodal points [kg/m^3]
    rho_Mat = zeros(ny+1, nx+1)
    rho_Mat[:, 1:Int(round(nx / 2))] .= 3300     # Left Side
    rho_Mat[:, (Int(round(nx/2))+1):end] .= 3300 # Right Side (inc ghost)
    # Linear System Setup
    l_mat = spzeros(3*(nx+1)*(ny+1), 3*(nx+1)*(ny+1))
    r_vec = zeros(3*(nx+1)*(ny+1), 1)
    # Results array initialisations
    vx_num = zeros(ny+1, nx+1)
    vy_num = zeros(ny+1, nx+1)
    p_num  = zeros(ny+1, nx+1)
    # Generate and store coordinates into a single struct
    coordinates = coord_allocation(dx, dy, w, h)
    # Benchmark timer toggle
    to = TimerOutput()
    if !benchmark
        disable_timer!(to)
    end
    # Stokes Linear System Allocation Loop
    @timeit to "SFD Stokes Const Eta" begin
    for i in 2:ny
        for j in 2:nx
            # Global indexing
            glb = (j-1)*(ny+1) + i
            gvx, gvy, gp, = 3*glb-2, 3*glb-1, 3*glb

            # Right Side Boundary Condition for vx
            if j == nx
                l_mat[gvx, gvx] = 1*eta
                r_vec[gvx] = BCConfig.bc_vx.right #mms_sol.bc_vx.right
            else
                # X-Stokes
                l_mat[gvx, gvx - dn] = 2*eta*idx2   # vx1
                l_mat[gvx, gvx - 3] = eta*idy2    # vx2
                l_mat[gvx, gvx] = -2*eta*idx2-2*eta*idx2-eta*idy2-eta*idy2 # vx3
                l_mat[gvx, gvx + 3] = eta*idy2    # vx4
                l_mat[gvx, gvx + dn] = 2*eta*idx2   # vx5
                l_mat[gvx, gvy - 3] = eta*idxdy    # vy1
                l_mat[gvx, gvy] = -eta*idxdy      # vy2
                l_mat[gvx, gvy + dn - 3] = -eta*idxdy # vy3
                l_mat[gvx, gvy + dn] = eta*idxdy    # vy4
                l_mat[gvx, gp] = idx*kconst       # P1
                l_mat[gvx, gp + dn] = -idx*kconst   # P2
                if is_validate
                    r_vec[gvx] = mms_sol.xstokes(coordinates.vx_coords.xvec[j],
                    coordinates.vx_coords.yvec[i])
                else
                    r_vec[gvx] = 0.0
                end
            end
            # Lower Boundary Condition for vy
            if i == ny
                l_mat[gvy, gvy] = 1*eta
                r_vec[gvy] = BCConfig.bc_vy.bottom 
            else
                # Y-Stokes
                l_mat[gvy, gvx - dn] = eta*idxdy    # vx1
                l_mat[gvy, gvx - dn + 3] = -eta*idxdy # vx2
                l_mat[gvy, gvx] = -eta*idxdy      # vx3
                l_mat[gvy, gvx + 3] = eta*idxdy     # vx4
                l_mat[gvy, gvy - dn] = eta*idx2     # vy1
                l_mat[gvy, gvy - 3] = 2*eta*idy2    # vy2
                l_mat[gvy, gvy] = -eta*idx2 - eta*idx2 - 2*eta*idy2 - 2*eta*idy2 # vy3
                l_mat[gvy, gvy + 3] = 2*eta*idy2    # vy4
                l_mat[gvy, gvy + dn] = eta*idx2     # vy5
                l_mat[gvy, gp] = idy*kconst       # P1
                l_mat[gvy, gp + 3] = -idy*kconst    # P2
                if is_validate
                    r_vec[gvy] = mms_sol.ystokes(coordinates.vy_coords.xvec[j],
                    coordinates.vy_coords.yvec[i])
                else
                    r_vec[gvy] = -gy*(rho_Mat[i, j] + rho_Mat[i, j + 1])/2
                end
            end
            # Boundary Condition for P
            if i == 2 && j==2
                l_mat[gp, gp] = 1*kconst
                if is_validate
                    r_vec[gp] = mms_sol.p(dx/2, dy/2)
                else
                    r_vec[gp] = 0.0
                end
            else
                # Continuity Equation
                l_mat[gp, gvx - dn] = -idx*kconst  # vx1
                l_mat[gp, gvx] = idx*kconst      # vx2
                l_mat[gp, gvy - 3] = -idy*kconst   # vy1
                l_mat[gp, gvy] = idy*kconst      # vy2
                r_vec[gp] = 0
            end
        end
    end

    # LHS BC for vx and vy, ghost for P
    for i in 1:(ny + 1)
        j = 1
        # Global indexing
        glb = (j-1)*(ny+1) + i
        gvx, gvy, gp, = 3*glb-2, 3*glb-1, 3*glb
        # vx, Free-slip, vx = 0
        l_mat[gvx, gvx] = 1*eta
        r_vec[gvx] = BCConfig.bc_vx.left
        # vy, Free-slip, dvy/dx = 0, vy1 = vy2
        if is_validate
            l_mat[gvy, gvy] = 0.5*eta
            l_mat[gvy, gvy + dn] = 0.5*eta
            r_vec[gvy] = BCConfig.bc_vy.left((i-1)*dy)*eta
        else
            l_mat[gvy, gvy] = 1*eta
            l_mat[gvy, gvy + dn] = -1*eta
            r_vec[gvy] = 0.0
        end
        # P, Ghost
        l_mat[gp, gp] = 1*kconst
        r_vec[gp] = 0
    end
    # RHS BC for vy, ghost for vx and P
    for i in 1:(ny + 1)
        j = nx+1
        # Global indexing
        glb = (j-1)*(ny+1) + i
        gvx, gvy, gp, = 3*glb-2, 3*glb-1, 3*glb
        # vx, Ghost
        l_mat[gvx, gvx] = 1*eta
        r_vec[gvx] = 0
        # vy, Free-slip, dvy/dx = 0, vy1 = vy2
        if is_validate 
            l_mat[gvy, gvy - dn] = 0.5*eta # vy1
            l_mat[gvy, gvy] = 0.5*eta   # vy2
            r_vec[gvy] = BCConfig.bc_vy.right((i-1)*dy)*eta
        else
            l_mat[gvy, gvy - dn] = 1*eta # vy1
            l_mat[gvy, gvy] = -1*eta   # vy2
            r_vec[gvy] = 0.0
        end
        # P, Ghost
        l_mat[gp, gp] = 1*kconst
        r_vec[gp] = 0
    end
    # Top BC for vx and vy, ghost for P
    for j in 2:nx
        i = 1
        # Global indexing
        glb = (j-1)*(ny+1) + i
        gvx, gvy, gp, = 3*glb-2, 3*glb-1, 3*glb
        # vx, Free-slip, dvx/dy = 0, vx1 = vx2
        if is_validate
            l_mat[gvx, gvx] = 0.5*eta # vx1
            l_mat[gvx, gvx + 3] = 0.5*eta # vx2
            r_vec[gvx] = BCConfig.bc_vx.top((j-1)*dx)*eta
        else
            l_mat[gvx, gvx] = 1*eta # vx1
            l_mat[gvx, gvx + 3] = -1*eta # vx2
            r_vec[gvx] = 0.0
        end
        # vy, Free-slip, vy = 0
        l_mat[gvy, gvy] = 1*eta
        r_vec[gvy] = BCConfig.bc_vy.top*eta
        # P, Ghost
        l_mat[gp, gp] = 1*kconst
        r_vec[gp] = 0
    end
    # Bottom BC for vx, ghost for vy and P
    for j in 2:nx
        i = ny+1
        # Global indexing
        glb = (j-1)*(ny+1) + i
        gvx, gvy, gp, = 3*glb-2, 3*glb-1, 3*glb
        # vx, Free-slip, dvx/dy = 0, vx1 = vx2
        if is_validate
            l_mat[gvx, gvx - 3] = 0.5*eta # vx1
            l_mat[gvx, gvx] = 0.5*eta # vx2
            r_vec[gvx] = BCConfig.bc_vx.bottom((j-1)*dx)*eta
        else
            l_mat[gvx, gvx - 3] = 1*eta # vx1
            l_mat[gvx, gvx] = -1*eta # vx2
            r_vec[gvx] = 0.0
        end
        # vy, Ghost, vy = 0
        l_mat[gvy, gvy] = 1*eta
        r_vec[gvy] = 0
        # P, Ghost
        l_mat[gp, gp] = 1*kconst
        r_vec[gp] = 0
    end
    # Corner Nodes
    CN = [(1, 1), (1, nx+1), (ny+1, 1), (ny+1, nx+1)]
    for (i, j) in CN
        # Global indexing
        glb = (j-1)*(ny+1) + i
        gvx, gvy, gp, = 3*glb-2, 3*glb-1, 3*glb
        l_mat[gvx, gvx] = 1*eta
        l_mat[gvy, gvy] = 1*eta
        l_mat[gp, gp] = 1*kconst
        r_vec[gvx] = 0
        r_vec[gvy] = 0
        r_vec[gp] = 0
    end

    s_vec = l_mat \ r_vec

    # Unpack Pressure Velocity Solutions Vector
    for i in 1:ny+1
        for j in 1:nx+1
            glb = (j-1)*(ny+1) + i # Global index
            gvx = 3*glb-2
            gvy = 3*glb-1
            gp = 3*glb
            vx_num[i, j] = s_vec[gvx]
            vy_num[i, j] = s_vec[gvy]
            p_num[i, j]  = s_vec[gp]*kconst
        end
    end
    end
    print_timer(to)

    return (vx_num, vy_num, p_num), coordinates
end

@kwdef struct Coords{XVEC, YVEC}
    xvec::XVEC
    yvec::YVEC
end
@kwdef struct VarCoords{VX, VY, P}
    vx_coords::VX
    vy_coords::VY
    p_coords::P
end

function coord_allocation(dx,dy,xsize,ysize)
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
    VarCoords(
        vx_coords = vx_coords,
        vy_coords = vy_coords,
        p_coords = p_coords
    )
    return VarCoords(vx_coords,vy_coords,p_coords)
end

function plot_output(xvec, yvec, var, title_str)
    varplot = heatmap(
        xvec, yvec, var,
        xlabel = "Distance [m]",
        ylabel = "Depth [m]",
        title = title_str,
        # colorbar_title = "Velocity (m/s)",
        yflip = true,
        # aspect_ratio = :equal,
        c = :viridis
    )
    display(varplot)
end

