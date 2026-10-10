using Documenter, RelledomMultiPhysics

makedocs(
    sitename = "RelledomMultiPhysics.jl Documentation",
    modules  = [RelledomMultiPhysics],
    pages    = [
        "Home" => "index.md",
        "Psuedo Transient Stokes Solvers" => "psuedotransient.md",
        "Solver Core" => "solvercore.md"
    ],
    checkdocs = :none
    )

deploydocs(
    repo = "github.com/lewislovell/RelledomMultiPhysics.jl.git",
    devbranch = "main"
)