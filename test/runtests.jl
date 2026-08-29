using RelledomMultiPhysics
using Test
using Aqua

@testset "RelledomMultiPhysics.jl" begin
    # Write your tests here.
    @testset "Aqua" begin
        @info "Tesing quality assurance via Aqua"
        Aqua.test_all(RelledomMultiPhysics)
    end
end
