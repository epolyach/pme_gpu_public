# src/distributed/DistributedConfig.jl
"""
Standalone distributed configuration for PME calculations.
This file contains only configuration data and can be loaded independently
without any DistributedPME module dependencies.
"""

module DistributedConfig

using TOML

export NodeInfo, DistributedConfigData, create_distributed_config_from_toml

# Node configuration structure - standalone, no module dependencies
struct NodeInfo
    hostname::String
    ssh_address::String  # How to SSH to this node
    role::Symbol  # :gpu_worker, :cpu_master, :coordinator
    gpu_devices::Vector{Int}
    memory_gb::Int
    remote_dir::String
    julia_path::String  # Path to Julia executable on this node
end

struct DistributedConfigData
    nodes::Dict{String, NodeInfo}
    coordinator_node::String
    gpu_nodes::Vector{String}
    cpu_node::String
end

# Node configuration from TOML only - no fallbacks
function create_distributed_config(julia_path_gpu, julia_path_cpu, remote_dir_gpu, remote_dir_cpu)
    nodes = Dict(
        "gpu1" => NodeInfo("sedan-gpu1", "gpu1", :gpu_worker, [0], 64, remote_dir_gpu, julia_path_gpu),
        "gpu2" => NodeInfo("sedan-gpu2", "gpu2", :gpu_worker, [0], 64, remote_dir_gpu, julia_path_gpu), 
        "gpu3" => NodeInfo("sedan-gpu3", "gpu3", :gpu_worker, [0], 64, remote_dir_gpu, julia_path_gpu),
        "haruka" => NodeInfo("haruka", "epolyach@haruka", :cpu_master, Int[], 256, remote_dir_cpu, julia_path_cpu)
    )
    
    return DistributedConfigData(
        nodes,
        "haruka",  # coordinator and final computation node
        ["gpu1", "gpu2", "gpu3"],
        "haruka"
    )
end

# Create distributed config from TOML configuration file
function create_distributed_config_from_toml(config_file::String)
    config_data = TOML.parsefile(config_file)
    
    # Get distributed section - must exist
    dist_section = config_data["distributed"]
    
    # Extract required configuration values - no fallbacks
    julia_path_gpu = dist_section["julia_path_gpu"]
    julia_path_cpu = dist_section["julia_path_cpu"]
    remote_dir_gpu = dist_section["remote_dir_gpu"]
    remote_dir_cpu = dist_section["remote_dir_cpu"]
    
    return create_distributed_config(julia_path_gpu, julia_path_cpu, remote_dir_gpu, remote_dir_cpu)
end

end # module DistributedConfig
