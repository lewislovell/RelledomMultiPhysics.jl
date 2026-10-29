using TimerOutputs
using Plots, Plots.PlotMeasures
using oneAPI, KernelAbstractions
using ..SolverCore

"""
    gpu_x_stokes_kernel!(p, vx, vx_old, _dx, _dx2, _dy2, eta, dtau, _rho_pt, fx,
    vxtop, vxbottom)
2D x-Stokes equation with top and bottom boundary conditions.
"""
@kernel function gpu_x_stokes_kernel!(@Const(p), vx, @Const(vx_old), _dx, _dx2,
    _dy2, eta, dtau, _rho_pt, @Const(fx), @Const(vxtop), @Const(vxbottom))
    ix, iy = @index(Global, NTuple)
    nx, ny = size(p)[1]+1, size(p)[2]+1

    if (iy >= 2 && iy <= ny-2 && ix >= 2 && ix <= nx-1)
        @inbounds vx[ix,iy] = vx_old[ix,iy] + _rho_pt*dtau*(
                @d2_xa(vx_old)*_dx2*eta + @d2_ya(vx_old)*_dy2*eta -
                 @d_xb(p)*_dx - fx[ix,iy])
    end

    if (iy == 1 && ix >= 2 && ix <= nx-1)
        @inbounds vx[ix,iy] = vx_old[ix,iy] + _rho_pt*dtau*(
            @d2_xa(vx_old)*_dx2*eta + (2*vxtop[ix]-3*vx_old[ix,iy]+
            vx_old[ix,iy+1])*_dy2*eta - @d_xb(p)*_dx - fx[ix,iy])
    end

    if (iy == ny-1 && ix >= 2 && ix <= nx-1)
        @inbounds vx[ix,iy] = vx_old[ix,iy] + _rho_pt*dtau*(
            @d2_xa(vx_old)*_dx2*eta +(vx_old[ix,iy-1]-3*vx_old[ix,iy]+
            2*vxbottom[ix])*_dy2*eta - @d_xb(p)*_dx - fx[ix,iy])
    end
end

"""
    gpu_y_stokes_kernel!(p, vy, vy_old, _dy, _dx2, _dy2, eta, dtau, _rho_pt, fy,
    vyleft, vyright)
2D y-Stokes equation with left and right boundary conditions.
"""
@kernel function gpu_y_stokes_kernel!(@Const(p), vy, @Const(vy_old), _dy, _dx2,
    _dy2, eta, dtau, _rho_pt, @Const(fy), @Const(vyleft), @Const(vyright))
    ix, iy = @index(Global, NTuple)
    nx, ny = size(p)[1]+1, size(p)[2]+1

    if (iy >= 2 && iy <= ny-1 && ix >= 2 && ix <=nx-2)
        @inbounds vy[ix,iy] = vy_old[ix,iy] + _rho_pt*dtau*(
                (@d2_xa(vy_old))*_dx2*eta + @d2_ya(vy_old)*_dy2*eta -
                @d_yb(p)*_dy - fy[ix,iy])
    end

    if (ix == 1 && iy >= 2 && iy <= ny-1)
        @inbounds vy[ix,iy] = vy_old[ix,iy] + _rho_pt*dtau*(
            (2*vyleft[iy]-3*vy_old[ix,iy]+vy_old[ix+1,iy])*_dx2*eta +
                @d2_ya(vy_old)*_dy2*eta - @d_yb(p)*_dy - fy[ix,iy])
    end

    if (ix == nx-1 && iy >= 2 && iy <= ny-1)
        @inbounds vy[ix,iy] = vy_old[ix,iy] + _rho_pt*dtau*(
            (vy_old[ix-1,iy]-3*vy_old[ix,iy]+2*vyright[iy])*_dx2*eta +
                @d2_ya(vy_old)*_dy2*eta - @d_yb(p)*_dy - fy[ix,iy])
    end
end

"""
    gpu_continuity_kernel!(p, vx, vy, _dx, _dy, _beta, dtau)
2D continuity equation with artificial compressibility for pseudo-transient
time integration.
"""
@kernel function gpu_continuity_kernel!(p, @Const(vx), @Const(vy), _dx, _dy,
    _beta, dtau)
    ix, iy = @index(Global, NTuple)
    @inbounds p[ix,iy] -= dtau*_beta*(@d_xa(vx)*_dx+@d_ya(vy)*_dy)
end

"""
    coord_to_gpu(backend, range, T)
Helper function to create gpu arrays for BCs
"""
function coord_to_gpu(backend, range, T)
    # Collect ranges into arrays
    cpu_array = collect(range)
    # Allocate memory on the GPU
    gpu_array = KernelAbstractions.allocate(backend, T, size(cpu_array))
    # Copy CPU data to GPU
    KernelAbstractions.copyto!(backend, gpu_array, cpu_array)
    return gpu_array
end

"""
    function isoviscous_stokes(ModelParams, BCConfig, ::VelP, ::PseudoTransient,
    ::GPU, benchmark::Bool = false, is_validate::Bool = false;
    MMSSol = nothing, VldParams = nothing)
Run isoviscous, incompressible stokes equations using a staggered-finite-
difference numerical formulation. Includes benchmarking and validation.
Note: for PseudoTransient solves rows correspond to x and columns correspond
to y.
"""
function isoviscous_stokes(ModelParams, BCConfig, ::VelP, ::PseudoTransient,
    ::GPU, benchmark::Bool = false, is_validate::Bool = false;
    MMSSol = nothing, VldParams = nothing)
    # Obtain float and int types
    T = typeof(ModelParams.w); I = typeof(ModelParams.nx)
    # Input ModelParams
    w, h   = ModelParams.w, ModelParams.h    # Domain size [m]
    nx, ny = ModelParams.nx, ModelParams.ny  # Model gridpoints
    dx, dy = ModelParams.dx, ModelParams.dy  # Grid spacings [m]
    eta    = ModelParams.eta         # Viscosity [Pa s]
    gy     = ModelParams.gy          # Gravitational acceleration [m/s]
    if !is_validate
        rho    = ModelParams.rho     # Density array [kg/m^3]
        @assert size(rho) == (nx,ny)
    end
    # Derived ModelParams
    _dx, _dy = 1/dx, 1/dy        # Inverse grid spacings [1/m]
    _dx2, _dy2 = 1/dx^2, 1/dy^2  # Inverse grid spacings squared [1/m^2]
    # Pseudo-transient parameters.
    maxiter = I(40_000) # Maximum number of iterations
    n_check = I(1000)   # Number of iterations between each residual calculation
    nvis    = I(1000)   # Number of iterations between each display plot
    h_eff   = T(1/sqrt(_dx2 + _dy2))   # Effective grid spacing [m]
    tol     = T(max(1e-6, 1000*eps(T))) # Float-Type dependant tolerance
    cfl     = T(1.0)    # CFL for pseudo-transients
    # Derived PT parameters
    lc      = max(w,h)       # Domain scale
    c       = T(lc/(2*pi))   # Artificial wave velocity
    dtau    = T(cfl*h_eff/(c*sqrt(2))) # Pseudo-transient time-step
    rho_pt  = T(eta*pi/(c*h_eff))      # Artificial density
    _rho_pt = T(1/rho_pt)              # Inverse artificial density
    beta    = T(1/(rho_pt*c^2))        # Artificial compressibility
    _beta   = T(1/beta)                # Inverse artificial compressibility

    # Generate and store coordinates into a single struct
    coordinates = coord_allocation(dx, dy, w, h)
    # Results array initialisations
    vx_num = oneAPI.zeros(T, nx, ny-1)
    vy_num = oneAPI.zeros(T, nx-1, ny)
    vx_old = oneAPI.zeros(T, nx, ny-1)
    vy_old = oneAPI.zeros(T, nx-1, ny)
    p_old  = oneAPI.zeros(T, nx-1, ny-1)
    # Initialise pressure with CPU array for simpler pre-estimate update
    p_num_cpu  = zeros(T, nx-1, ny-1)
    # Boost convergence with pre-estimate of pressure
    if is_validate
        for iy = 1:ny-1
            y = coordinates.p_coords.yvec[iy]
            p_num_cpu[:, iy] .= VldParams.p0 + y*VldParams.dp0dy
        end
    end
    # Initialise driving forces with CPU arrays
    fx_cpu = zeros(T, nx, ny-1)
    fy_cpu = zeros(T, nx-1, ny)

    if is_validate
        fx_cpu .= MMSSol.xstokes.(
            coordinates.vx_coords.xvec, coordinates.vx_coords.yvec')
        fy_cpu .= MMSSol.ystokes.(
            coordinates.vy_coords.xvec, coordinates.vy_coords.yvec')
    else
        eval_forcing!(fy_cpu, rho, gy)
    end
    # Allocate CPU arrays to GPU device
    backend = KernelAbstractions.get_backend(vx_num)
    threads = (16,16)
    p_num = coord_to_gpu(backend, p_num_cpu, T)
    fx = coord_to_gpu(backend, fx_cpu, T)
    fy = coord_to_gpu(backend, fy_cpu, T)
    # Boundary vector initialisations
    vxleft, vxright = zeros(T, ny - 1), zeros(T, ny - 1)
    vxtop, vxbottom = zeros(T, nx), zeros(T, nx)
    vyleft, vyright = zeros(T, ny), zeros(T, ny)
    vytop, vybottom = zeros(T, nx - 1), zeros(T, nx - 1)
    # Multiple dispatch for unpacking BCs as functions versus scalars
    eval_bc(bc::Number, coords) = bc
    eval_bc(bc::Function, coords) = bc.(coords)
    # Unpack BCs into vectors depending on functions or scalars
    vxtop[:]    .= eval_bc(BCConfig.bc_vx.top,    coordinates.vx_coords.xvec)
    vxbottom[:] .= eval_bc(BCConfig.bc_vx.bottom, coordinates.vx_coords.xvec)
    vxleft[:]   .= eval_bc(BCConfig.bc_vx.left,   coordinates.vx_coords.yvec)
    vxright[:]  .= eval_bc(BCConfig.bc_vx.right,  coordinates.vx_coords.yvec)
    vyleft[:]   .= eval_bc(BCConfig.bc_vy.left,   coordinates.vy_coords.yvec)
    vyright[:]  .= eval_bc(BCConfig.bc_vy.right,  coordinates.vy_coords.yvec)
    vytop[:]    .= eval_bc(BCConfig.bc_vy.top,    coordinates.vy_coords.xvec)
    vybottom[:] .= eval_bc(BCConfig.bc_vy.bottom, coordinates.vy_coords.xvec)
    # Create GPU arrays for BCs
    vxtop_gpu    = coord_to_gpu(backend, vxtop, T)
    vxbottom_gpu = coord_to_gpu(backend, vxbottom, T)
    vxleft_gpu   = coord_to_gpu(backend, vxleft, T)
    vxright_gpu  = coord_to_gpu(backend, vxright, T)
    vyleft_gpu   = coord_to_gpu(backend, vyleft, T)
    vyright_gpu  = coord_to_gpu(backend, vyright, T)
    vytop_gpu    = coord_to_gpu(backend, vytop, T)
    vybottom_gpu = coord_to_gpu(backend, vybottom, T)
    # Apply BC vectors to collocated boundaries
    vx_num[1,:]   .= vxleft_gpu
    vx_num[end,:] .= vxright_gpu
    vy_num[:,1]   .= vytop_gpu
    vy_num[:,end] .= vybottom_gpu
    # Setup solvers for GPUs
    gpu_x_stokes! = gpu_x_stokes_kernel!(backend, threads)
    gpu_y_stokes! = gpu_y_stokes_kernel!(backend, threads)
    gpu_continuity! = gpu_continuity_kernel!(backend, threads)
    # Benchmark timer toggle
    to = TimerOutput()
    if !benchmark
        disable_timer!(to)
    end
    # Residual initialisations and vector constructions
    r_infc, r_infvx, r_infvy = 2*tol, 2*tol, 2*tol
    r_inf_vec = T[]; nt_vec = I[];
    f_scale = max(maximum(abs, fx), maximum(abs, fy), eps(T))
    # While loop for Pseudo-Transient iterations
    anim = Animation(); nframes = 0; iter = 1
    @timeit to "SFD Stokes Const Eta" begin
    while max(r_infc, r_infvx, r_infvy) >= tol && iter <= maxiter
        # Update previous arrays
        copyto!(p_old, p_num)
        copyto!(vx_old, vx_num)
        copyto!(vy_old, vy_num)
        # Compute Stokes updates
        gpu_x_stokes!(p_num, vx_num, vx_old, _dx, _dx2, _dy2, eta, dtau, 
        _rho_pt, fx, vxtop_gpu, vxbottom_gpu; ndrange=size(vx_num))
        gpu_y_stokes!(p_num, vy_num, vy_old, _dy, _dx2, _dy2, eta, dtau,
        _rho_pt, fy, vyleft_gpu, vyright_gpu; ndrange=size(vy_num))
        # Update continuity utilising updated velocities
        gpu_continuity!(p_num, vx_num, vy_num, _dx, _dy, _beta, dtau;
            ndrange=size(p_num))
        if !benchmark
            # Compute residuals
            if mod(iter, n_check) == 0
                KernelAbstractions.synchronize(backend)
                r_infc  = maximum(abs, p_num.-p_old)*beta/dtau/f_scale/h_eff
                r_infvx = maximum(abs, vx_num.- vx_old)/(_rho_pt*dtau)/f_scale
                r_infvy = maximum(abs, vy_num.-vy_old)/(_rho_pt*dtau)/f_scale
                push!(r_inf_vec, max(r_infc,r_infvx,r_infvy))
                push!(nt_vec, iter)
            end
            # Visualise PT iterations
            if mod(iter,nvis) == 0
                KernelAbstractions.synchronize(backend)
                p1 = heatmap(Array(vy_num)'*1e9;
                title="Iter=$iter",
                            yflip=true,
                            c=cgrad(:berlin, rev=false),
                            right_margin=15Plots.mm)
                p2 = plot(nt_vec./nx, r_inf_vec; 
                            xlabel="iter/nx", ylabel="err",
                            yscale=:log10, grid=true, markershape=:circle, markersize=4)
                display(plot(p1, p2; layout=(2, 1)))
                frame(anim)
                nframes += 1
            end
        end
        iter += 1
    end
    KernelAbstractions.synchronize(backend)
    end
    if !benchmark && max(r_infc, r_infvx, r_infvy) >= tol
        @warn "PT solver did not converge" iter r_infc r_infvx r_infvy
    end

    if benchmark
        t_toc = TimerOutputs.time(to["SFD Stokes Const Eta"])*1e-9
        niter = iter - 1
        n_x_updates = (nx-2)*(ny-1)
        n_y_updates = (nx-1)*(ny-2)
        n_continuity_updates = (nx-1)*(ny-1)
        n_reads = 10*(n_x_updates+n_y_updates) + 5*n_continuity_updates
        n_writes = n_x_updates + n_y_updates + n_continuity_updates
        a_eff = (n_reads+n_writes)*sizeof(T)*niter*1e-9
        t_eff = a_eff/t_toc
        return (Array(vx_num)', Array(vy_num)', Array(p_num)'), coordinates, 
            PerformanceData(
            method = :PseudoTransient,
            float_type = T,
            nx = nx,
            ny = ny,
            a_eff = a_eff,
            t_toc = t_toc,
            t_eff = t_eff
            )
    else
        if nframes > 0
            gif(anim, "isovisc_pt_gpu.gif", fps = 15)
        end
        return (Array(vx_num)', Array(vy_num)', Array(p_num)'), coordinates
    end
end

