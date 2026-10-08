module Benchmarks

using Statistics
using ..SolverCore
using Plots, Plots.PlotMeasures

export run_benchmark, scaling_benchmark, plot_summary

"""
    run_benchmark(run_model; nsamples=10, warmup=true)

Benchmark a zero-argument model runner. The runner must return
`(solution, coordinates, performance)`.
"""

function run_benchmark(run_model::Function, Params; nsamples=10, warmup=true)
    if warmup
        run_model(Params)
    end
    # Initialise vectors
    it_times = zeros(nsamples) # Iteration times
    oh_times = zeros(nsamples) # Overhead times
    t_eff    = zeros(nsamples) # Effective throughputs

    for this in 1:nsamples
        tot_time = @elapsed begin
            _, _, Performance = run_model(Params)
        end
        print("Sample ",this," Runtime: ", tot_time, "\n")
        it_times[this] = Performance.t_toc
        oh_times[this] = tot_time - Performance.t_toc
        t_eff[this]    = Performance.t_eff
    end

    return(
        med_it_time = median(it_times),
        max_it_time = maximum(it_times),
        min_it_time = minimum(it_times),
        med_oh_time = median(oh_times),
        max_oh_time = maximum(oh_times),
        min_oh_time = minimum(oh_times),
        med_t_eff = median(t_eff),
        min_t_eff = minimum(t_eff),
        max_t_eff = maximum(t_eff)
    )
end

function scaling_benchmark(model, Params::Type{<:ModelParams}, min_order::Real,
    max_order::Real; nsamples=5, warmup=true)
    print("### Running Scaling Benchmark ###\n")
    order_vec     = Float64[]
    nx_vec        = Int[]
    med_it_vec    = Float64[]
    med_oh_vec    = Float64[]
    med_t_eff_vec = Float64[]

    for n = min_order:0.5:max_order
        println("Benchmarking order: ", n,"\n")
        order = 10.0^n
        nx = round(Int, sqrt(order))

        push!(order_vec, order)
        push!(nx_vec, nx)

        params = Params(nx=nx, ny=nx)

        summary = run_benchmark(model, params;
            nsamples=nsamples, warmup=warmup)

        push!(med_it_vec, summary.med_it_time)
        push!(med_oh_vec, summary.med_oh_time)
        push!(med_t_eff_vec, summary.med_t_eff)
    end
    return (
        order_vec     = order_vec,
        nx_vec        = nx_vec,
        med_it_vec    = med_it_vec,
        med_oh_vec    = med_oh_vec,
        med_t_eff_vec = med_t_eff_vec,
        nsamples      = nsamples
    )
end

function plot_summary(scaling_output)
    scaling_plot = plot(
        scaling_output.order_vec,
        scaling_output.med_t_eff_vec,
        xaxis = :log10,
        xlabel = "Model size",
        ylabel = "Median effective throughput [GB/s]",
        label = "Pseudo-transient GPU",
        color = :black,
        left_margin = 12Plots.mm,
    )
    timing_plot = plot(
        scaling_output.order_vec,
        scaling_output.med_it_vec,
        xaxis = :log10,
        xlabel = "Model size",
        ylabel = "Iteration Time [s]",
        label = "Pseudo-transient GPU",
        color = :black,
        left_margin = 12Plots.mm,
    )
    overhead_plot = plot(
        scaling_output.order_vec,
        scaling_output.med_oh_vec,
        xaxis = :log10,
        xlabel = "Model size",
        ylabel = "Overhead Time [s]",
        label = "Pseudo-transient GPU",
        color = :black,
        left_margin = 12Plots.mm,
    )
    display(plot(
        scaling_plot,
        timing_plot,
        overhead_plot;
        layout = (3, 1),
        size = (800, 1000),
    ))
end
end