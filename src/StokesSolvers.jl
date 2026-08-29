module StokesSolvers

export isoviscous_stokes, VelP, DirectMonolithic, CPUSingle

abstract type Formulation end
struct VelP <: Formulation end

abstract type Method end
struct DirectMonolithic <: Method end

abstract type Hardware end
struct CPUSingle <: Hardware end

function isoviscous_stokes end

include("Solvers/isovisc_direct.jl")

end