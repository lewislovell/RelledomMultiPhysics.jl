module StokesSolvers

export isoviscous_stokes, VelP, DirectMonolithic, PseudoTransient, CPUSingle,
    CPULV, GPU, GPUSM

abstract type Formulation end
struct VelP <: Formulation end

abstract type Method end
struct DirectMonolithic <: Method end
struct PseudoTransient <: Method end

abstract type Hardware end
struct CPUSingle <: Hardware end
struct CPULV <: Hardware end
struct GPU <: Hardware end
struct GPUSM <: Hardware end

function isoviscous_stokes end

include("Solvers/isovisc_direct.jl")
include("Solvers/isovisc_pseudotransient.jl")
include("Solvers/isovisc_pseudotransient_lv.jl")
include("Solvers/isovisc_pseudotransient_gpu.jl")
# include("Solvers/isovisc_pseudotransient_gpu_sm.jl")

end