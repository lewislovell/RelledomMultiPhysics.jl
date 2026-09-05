module StokesSolvers

export isoviscous_stokes, VelP, DirectMonolithic, PseudoTransient, CPUSingle,
    GPU

abstract type Formulation end
struct VelP <: Formulation end

abstract type Method end
struct DirectMonolithic <: Method end
struct PseudoTransient <: Method end

abstract type Hardware end
struct CPUSingle <: Hardware end
struct GPU <: Hardware end

function isoviscous_stokes end

include("Solvers/isovisc_direct.jl")
include("Solvers/isovisc_pseudotransient.jl")
include("Solvers/isovisc_pseudotransient_gpu.jl")

end