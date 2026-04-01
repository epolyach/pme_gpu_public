# src/distributed/DistributedPME.jl
"""
Distributed PME calculation module for multi-node computing.

Architecture:
1. Steps 1-5 (model setup through 2D/3D arrays) can run on any GPU node
2. Step 6 (Pi elements) is distributed across gpu1, gpu2, gpu3 (1/3 each)
3. Steps 7-8 (K matrix construction + eigenvalues) run on haruka (high memory)

Node configuration:
- gpu1 (10.186.114.34): A6000 GPU
- gpu2 (10.186.114.28): A6000 GPU  
- gpu3 (10.186.115.14): A6000 GPU
- haruka (83.149.230.221): 256GB RAM, CPU-only

Usage:
julia run_pme_distributed.jl config.toml model.toml
"""

module DistributedPME

using Distributed
using SharedArrays
using LinearAlgebra
using Printf
using Dates
using TOML

# Import PME module and PMEConfig to avoid type conflicts
import PME
using PME: PMEConfig

# Import standalone distributed configuration (no complex dependencies)
include("DistributedConfig.jl")
using .DistributedConfig: NodeInfo, DistributedConfigData, create_distributed_config_from_toml

# We'll reference other PME functions through qualified names
# Note: CUDA is not loaded here - only on worker nodes

# Define a simplified OrbitData for workers, containing only the necessary arrays.
# The `grids` object is problematic for serialization, so we extract what we need.
mutable struct OrbitDataWorker{T<:Real}
    ra::Array{T,3}
    pha::Array{T,3}
    w1::Array{T,3}
    L_m::Matrix{T}
    SGNL::Matrix{T}
    Omega_1::Matrix{T}
    Omega_2::Matrix{T}
end

export DistributedConfigData, NodeInfo, create_distributed_config_from_toml, run_distributed_pme_calculation, setup_distributed_workers, test_distributed_pme

# Setup distributed workers on remote nodes via SSH.
# This function ONLY adds the procs. It does not load any code on them.
function setup_distributed_workers(dist_config::DistributedConfigData; julia_path="julia", verbose=true)
    if verbose
        println("Setting up distributed workers...")
    end
    
    worker_pids = Int[]
    nodes_to_connect = vcat(dist_config.gpu_nodes, [dist_config.cpu_node])
    unique!(nodes_to_connect)

    for node_name in nodes_to_connect
        # Skip if node is the local host and not explicitly configured for remote connection
        if node_name == gethostname() && dist_config.nodes[node_name].ssh_address == "localhost"
            if verbose
                println("  Skipping local node setup: $node_name")
            end
            continue
        end
        
        node_info = dist_config.nodes[node_name]
        if verbose
            println("  Connecting to $(node_info.hostname) ($(node_info.ssh_address))...")
        end
        
        try
            # Use Julia path from node configuration
            actual_julia_path = node_info.julia_path
            
            # Create the remote directory if it doesn't exist
            ssh_cmd = `ssh $(node_info.ssh_address) mkdir -p $(node_info.remote_dir)/src/distributed`
            run(ssh_cmd)
            
            # Copy only the files needed by the workers - suppress scp output
            if verbose
                run(`scp src/distributed/WorkerFunctions.jl $(node_info.ssh_address):$(node_info.remote_dir)/src/distributed/`)
                run(`scp src/distributed/DistributedPiElementsGPU.jl $(node_info.ssh_address):$(node_info.remote_dir)/src/distributed/`)
            else
                run(`scp -q src/distributed/WorkerFunctions.jl $(node_info.ssh_address):$(node_info.remote_dir)/src/distributed/`)
                run(`scp -q src/distributed/DistributedPiElementsGPU.jl $(node_info.ssh_address):$(node_info.remote_dir)/src/distributed/`)
            end
            
            # Add worker using proper addprocs SSH syntax
            new_pids = addprocs([(node_info.ssh_address, 1)];
                                     exename=actual_julia_path,
                                     dir=node_info.remote_dir,
                                     exeflags="--project",
                                     tunnel=true)
            pid = new_pids[1]
            push!(worker_pids, pid)
            if verbose
                println("    Worker $pid started on $(node_info.hostname)")
            end
        catch e
            println("    Failed to connect to $(node_info.hostname): $e")
        end
    end
    
    if verbose
        println("  Distributed workers setup complete: $(length(worker_pids)) workers")
    end
    return worker_pids
end

# Phase 1: This runs on the main process, so it's safe here.
function run_phase_1_setup(config::PMEConfig, dist_config::DistributedConfigData)
    println("  Running setup on $(dist_config.nodes[dist_config.gpu_nodes[1]].hostname)")
    model, grids, orbit_data, z, psi_z = PME.run_shared_pipeline_steps_1_to_5(config)
    
    # Always save z and psi_z for GPU workers (not just in debug mode)
    println("  Saving z and psi_z for distributed GPU workers...")
    data_dir = config.io.data_path
    binary_dir = joinpath(data_dir, "binary")
    mkpath(binary_dir)
    
    if config.io.single_precision
        PME.BinaryIO.write_binary_float32(joinpath(binary_dir, "z.bin"), z)
        PME.BinaryIO.write_binary_float32(joinpath(binary_dir, "psi_z.bin"), psi_z)
    else
        PME.BinaryIO.write_binary_float64(joinpath(binary_dir, "z.bin"), z)
        PME.BinaryIO.write_binary_float64(joinpath(binary_dir, "psi_z.bin"), psi_z)
    end
    println("  ✅ z and psi_z saved to binary files")
    
    println("  Phase 1 complete: Model and grids ready for distribution")
    return model, grids, orbit_data, z, psi_z
end

# NOTE: compute_pi_chunk_on_gpu function moved to GPUWorkerFunctions.jl
# to avoid DistributedPME module references on remote workers

# Helper function to summarize results
function summarize_results(pi4)
    min_val = minimum(pi4)
    max_val = maximum(pi4)
    mean_val = sum(pi4) / length(pi4)
    return "Range: $(round(min_val, digits=4)) to $(round(max_val, digits=4)), Mean: $(round(mean_val, digits=6))"
end

# Test wrapper function for debugging
function test_distributed_pme(config::PMEConfig, dist_config::DistributedConfigData)
    println("🔧 Testing distributed PME calculation...")
    println("Config type: $(typeof(config))")
    println("DistConfig type: $(typeof(dist_config))")
    
    try
        return run_distributed_pme_calculation(config, dist_config)
    catch e
        println("❌ Error in test function: $e")
        rethrow(e)
    end
end

end # module DistributedPME
