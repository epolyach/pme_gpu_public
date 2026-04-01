#!/usr/bin/env julia

"""
PME GPU Production Runner - Multi-Backend Version (NVIDIA/AMD)

Supports automatic detection of GPU type:
- NVIDIA GPUs via CUDA.jl
- AMD GPUs via AMDGPU.jl

Usage: julia run_pme_gpu.jl config.toml model.toml [--gpu=0|--gpu=1|--gpu=01] [--threads=N]

Arguments:
  config.toml    Configuration file path
  model.toml     Model file path
  --gpu=X        GPU device selection (0, 1, or 01 for both)
  --threads=N    Number of BLAS threads (optional, max 4, default 4)

Examples:
  julia run_pme_gpu.jl configs/miyamoto.toml models/miyamoto.toml --gpu=0 --threads=4
  julia run_pme_gpu.jl configs/isochrone.toml models/isochrone.toml --gpu=1 --threads=2
  julia run_pme_gpu.jl configs/toomre.toml models/toomre.toml --gpu=01 --threads=4
"""

using Pkg
Pkg.activate(".")

using PME
using TOML
using LinearAlgebra
using Printf
using Dates

# Global log file
const LOG_FILE = Ref{Union{IOStream, Nothing}}(nothing)
const LOG_PATH = Ref{String}("")

function log_print(args...)
    print(args...)
    if LOG_FILE[] !== nothing
        print(LOG_FILE[], args...)
        flush(LOG_FILE[])
    end
end

function log_println(args...)
    println(args...)
    if LOG_FILE[] !== nothing
        println(LOG_FILE[], args...)
        flush(LOG_FILE[])
    end
end

function start_global_logging(config)
    log_dir = config.io.data_path
    mkpath(log_dir)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    beta_str = replace(string(config.physics.beta), "." => "")
    log_filename = @sprintf("run_%dx%dx%d_beta%s_gr%d_gv%d_%s.log",
                           config.grid.NR, config.grid.Nv, config.grid.NW,
                           beta_str, config.grid.radial_grid_type, 
                           config.grid.circulation_grid_type, timestamp)
    LOG_PATH[] = joinpath(log_dir, log_filename)
    LOG_FILE[] = open(LOG_PATH[], "w")
end

function stop_global_logging()
    if LOG_FILE[] !== nothing
        flush(LOG_FILE[])
        close(LOG_FILE[])
        LOG_FILE[] = nothing
    end
end


# Include GPUBackend for detection
include("src/utils/GPUBackend.jl")
using .GPUBackend

# Detect GPU type and load appropriate package
const GPU_TYPE = GPUBackend.detect_gpu_type()

if GPU_TYPE == GPUBackend.NVIDIA
    using CUDA
    const GPU_PACKAGE = CUDA
    println("🎮 Detected NVIDIA GPU - using CUDA.jl")
elseif GPU_TYPE == GPUBackend.AMD
    using AMDGPU
    const GPU_PACKAGE = AMDGPU
    println("🎮 Detected AMD GPU - using AMDGPU.jl")
else
    println("❌ No GPU detected")
    println("   Ensure nvidia-smi (NVIDIA) or rocm-smi (AMD) is available")
    exit(1)
end

# Backend-agnostic GPU helper functions
function gpu_ndevices()
    if GPU_TYPE == GPUBackend.NVIDIA
        return CUDA.ndevices()
    elseif GPU_TYPE == GPUBackend.AMD
        return length(AMDGPU.devices())
    end
    return 0
end

function gpu_device!(id::Int)
    if GPU_TYPE == GPUBackend.NVIDIA
        CUDA.device!(id)
    elseif GPU_TYPE == GPUBackend.AMD
        AMDGPU.device!(AMDGPU.devices()[id + 1])  # 0-indexed to 1-indexed
    end
end

function gpu_functional()
    if GPU_TYPE == GPUBackend.NVIDIA
        return CUDA.functional()
    elseif GPU_TYPE == GPUBackend.AMD
        return AMDGPU.functional()
    end
    return false
end

function gpu_total_memory()
    if GPU_TYPE == GPUBackend.NVIDIA
        return CUDA.total_memory()
    elseif GPU_TYPE == GPUBackend.AMD
        try
            # Try HIP properties API for memory info
            dev = AMDGPU.device()
            props = AMDGPU.HIP.properties(dev)
            return props.totalGlobalMem
        catch
            # Fallback: default 24GB for RX 7900 XTX
            return 24 * 1024^3
        end
    end
    return 0
end

function gpu_device_name()
    if GPU_TYPE == GPUBackend.NVIDIA
        return CUDA.name(CUDA.device())
    elseif GPU_TYPE == GPUBackend.AMD
        try
            dev = AMDGPU.device()
            return string(dev)
        catch
            return "AMD GPU"
        end
    end
    return "Unknown GPU"
end

function parse_arguments(cpu_config)
    """Parse command line arguments"""
    if length(ARGS) < 2
        println("Error: Insufficient arguments")
        println("Usage: julia run_pme_gpu.jl config.toml model.toml [--gpu=0|--gpu=1|--gpu=01] [--threads=N]")
        exit(1)
    end

    config_file = ARGS[1]
    model_file = ARGS[2]
    
    # Default values - use detected GPU count
    n_gpus = gpu_ndevices()
    gpu_devices = collect(0:(n_gpus-1))  # Default to all GPUs
    gpu_id = -1  # Multi-GPU mode by default
    blas_threads = cpu_config.max_threads  # Use config default

    # Parse optional arguments
    for arg in ARGS[3:end]
        if startswith(arg, "--gpu=")
            gpu_spec = split(arg, "=")[2]
            if gpu_spec == "0"
                gpu_devices = [0]
                gpu_id = 0
            elseif gpu_spec == "1"
                gpu_devices = [1]
                gpu_id = 1
            elseif gpu_spec == "01"
                gpu_devices = [0, 1]
                gpu_id = -1  # Multi-GPU mode
            else
                println("Error: Invalid GPU specification: $gpu_spec")
                println("Valid options: --gpu=0, --gpu=1, --gpu=01")
                exit(1)
            end
        elseif startswith(arg, "--threads=")
            requested_threads = parse(Int, split(arg, "=")[2])
            blas_threads = min(requested_threads, cpu_config.max_threads)
            if requested_threads > cpu_config.max_threads
                println("⚠️  BLAS threads capped at $(cpu_config.max_threads) (requested: $requested_threads)")
            end
        end
    end
    
    return config_file, model_file, gpu_devices, gpu_id, blas_threads
end

function main()
    if length(ARGS) < 2
        println("Error: Insufficient arguments")
        println("Usage: julia run_pme_gpu.jl config.toml model.toml [--gpu=0|--gpu=1|--gpu=01] [--threads=N]")
        exit(1)
    end
    
    config_file = ARGS[1]
    model_file = ARGS[2]
    
    # Validate input files
    if !isfile(config_file)
        println("Error: Configuration file not found: $config_file")
        exit(1)
    end

    if !isfile(model_file)
        println("Error: Model file not found: $model_file")
        exit(1)
    end
    
    # Load config first to get cpu settings
    config = PME.load_config(config_file)
    println("DEBUG: gpu.precision_double=$(config.gpu.precision_double), cpu.precision_double=$(config.cpu.precision_double)")
    model_data = TOML.parsefile(model_file)
    
    # Now parse full arguments with cpu config
    config_file, model_file, gpu_devices, gpu_id, blas_threads = parse_arguments(config.cpu)
    
    # Start logging
    start_global_logging(config)
    
    # Compact initial header
    log_println("\n--- PME GPU Calculation ---")
    log_println("Config: $config_file")
    log_println("Model:  $model_file")
    
    # GPU configuration - backend-agnostic
    if !gpu_functional()
        backend_name = GPU_TYPE == GPUBackend.NVIDIA ? "CUDA" : "ROCm"
        log_println("Error: $backend_name not functional")
        exit(1)
    end

    # Verify all requested GPUs are available
    n_available = gpu_ndevices()
    for device in gpu_devices
        if device >= n_available
            log_println("Error: GPU $device not available. Available GPUs: 0-$(n_available-1)")
            exit(1)
        end
    end

    # Set primary GPU and display compact info
    gpu_device!(gpu_devices[1])
    backend_str = GPU_TYPE == GPUBackend.NVIDIA ? "CUDA" : "ROCm"
    
    if length(gpu_devices) == 1
        dev = gpu_devices[1]
        gpu_device!(dev)
        mem_gb = round(gpu_total_memory() / 1e9, digits=1)
        log_println("GPU:    Device $dev ($(gpu_device_name()), $(mem_gb) GB) [$backend_str]")
    else
        log_println("GPU:    Multi-GPU mode, devices: $(gpu_devices) [$backend_str]")
        for device in gpu_devices
            gpu_device!(device)
            mem_gb = round(gpu_total_memory() / 1e9, digits=1)
            log_println("        Device $device: $(gpu_device_name()), $(mem_gb) GB")
        end
    end
    log_println("BLAS:   $blas_threads threads (max: $(config.cpu.max_threads))")
    
    # GPU precision
    gpu_precision = config.gpu.precision_double ? "Float64 (double)" : "Float32 (single)"
    log_println("GPU Pi:  $gpu_precision precision")

    # Set BLAS threads
    LinearAlgebra.BLAS.set_num_threads(blas_threads)
    
    # Apply model configuration - now using flat structure
    if haskey(model_data, "model")
        for (key, value) in model_data["model"]
            sym = Symbol(key)
            if hasfield(typeof(config.model), sym)
                setfield!(config.model, sym, value)
            elseif key == "L_star" && hasfield(typeof(config.model), :L_0)
                # Map L_star from model file to legacy L_0 field in ModelConfig
                setfield!(config.model, :L_0, value)
            end
        end
    end
    
    if haskey(model_data, "physics")
        # Save config values to preserve user overrides
        config_m_original = config.core.m
        config_beta_original = config.physics.beta
        config_selfgravity_original = config.physics.selfgravity
        
        physics = model_data["physics"]
        haskey(physics, "m") && (config.core.m = physics["m"])
        haskey(physics, "beta") && (config.physics.beta = physics["beta"])
        haskey(physics, "selfgravity") && (config.physics.selfgravity = physics["selfgravity"])
        
        # Restore config values (config file overrides model file)
        if config_m_original != 2
            config.core.m = config_m_original
        end
        if config_beta_original != 0.1
            config.physics.beta = config_beta_original
        end
        if config_selfgravity_original != 1.0
            config.physics.selfgravity = config_selfgravity_original
        end
    end
    
    # Apply model type
    model_type = get(get(model_data, "model", Dict()), "type", "ExpDisk")
    config.model.type = model_type
    
    # Print FINAL parameters - this is the key section
    log_println("\n--- FINAL PARAMETERS ---")
    log_println("Physics:")
    log_println("  m (azimuthal mode): $(config.core.m)")
    log_println("  beta (softening):   $(config.physics.beta)")
    log_println("  selfgravity:        $(config.physics.selfgravity)")
    
    log_println("Model-specific:")
    if haskey(model_data["model"], "n_M")
        log_println("  n_M (Miyamoto):     $(model_data["model"]["n_M"])")
    end
    if haskey(model_data["model"], "mk")
        log_println("  mk (Isochrone):     $(model_data["model"]["mk"])")
    end
    # Add taper information
    if occursin("Taper", config.model.type)
        taper_type = if occursin("TaperTanh", config.model.type)
            "Tanh"
        elseif occursin("TaperPoly3", config.model.type)
            "Poly3"
        elseif occursin("TaperExp", config.model.type)
            "Exp"
        else
            "Unknown"
        end
        log_println("  Taper type:         $(taper_type)")
        if haskey(model_data["model"], "v_0")
            log_println("  v_0:                $(model_data["model"]["v_0"])")
        elseif haskey(model_data["model"], "L_0")
            log_println("  L_0:                $(model_data["model"]["L_0"])")
        end
    end
    
    log_println("Grid:")
    log_println("  NR × Nv × kres:     $(config.grid.NR) × $(config.grid.Nv) × $(config.core.kres)")
    log_println("  Radial grid:        type=$(config.grid.radial_grid_type)" * 
            (config.grid.radial_grid_type <= 1 ? " (alpha=$(config.grid.alpha))" : 
             " (Rmin=$(config.grid.R_min), Rmax=$(config.grid.R_max))"))
    log_println("  Circulation grid:   type=$(config.grid.circulation_grid_type)")
    if config.grid.circulation_grid_type in [6, 7, 8] && config.grid.Nvt !== nothing
        log_println("  Nvt (taper nodes):  $(config.grid.Nvt)")
    end
    log_println("  k0:                 $(-((config.core.kres - 1) ÷ 2))")
    log_println("  NW (phase):         $(config.grid.NW)")
    log_println("  NW_orbit:           $(config.grid.NW_orbit)")
    
    
    log_println("GPU Precision:")
    gpu_prec_str = config.gpu.precision_double ? "Float64 (double)" : "Float32 (single)"
    log_println("  Pi elements:        $gpu_prec_str")
    
    log_println("CPU Precision:")
    cpu_prec_str = config.cpu.precision_double ? "Float64 (double)" : "Float32 (single)"
    println("  Pi elements, K:     $cpu_prec_str")
    
    if config.eigenvectors.iterative
        num_eig = length(config.eigenvectors.shift_Omega_p)
        println("Eigenvalue Solver:")
        println("  Mode:               Iterative (Arpack shift-invert)")
        println("  Num eigenvalues:    $num_eig")
        println("  Krylov dimension:   $(config.eigenvectors.krylov_dim)")
        println("  Tolerance:          $(config.eigenvectors.tol)")
    end
    
    log_println("Sharp DF:")
    log_println("  sharp_df:           $(config.phase_space.sharp_df)")
    if config.phase_space.sharp_df
        log_println("  Omega_2_lim:        $(config.phase_space.Omega_2_lim)")
    end
    println()
    

    # Run with GPU(s)
    start_time = time()

    try
        eigenvalues, txt = run_complete_pme_calculation(
            config,
            force_recalculate = true,
            save_intermediate = true,
            gpu_id = gpu_id,
            gpu_devices = gpu_devices,
            blas_threads = blas_threads
        )

        
        elapsed_time = time() - start_time
        
        # Completion messages printed by PMEWorkflow, just add Runtime and header
        log_println("Runtime: $(round(elapsed_time, digits=1))s ($(round(elapsed_time/60, digits=2)) min)")
        log_println("\n--- Calculation Complete ---")
        log_println(txt)
        
    catch e
        elapsed_time = time() - start_time
        println("\nError after $(round(elapsed_time/60, digits=2)) minutes:")
        println("Results: $(config.io.data_path)/")
        println(typeof(e), ": ", sprint(showerror, e)[1:min(500, end)])
        exit(1)
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
