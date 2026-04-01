# src/GPUWorkflow.jl
"""
GPU-enabled PME workflow wrapper.
Provides the same interface as the CPU version but uses GPU for Pi elements calculation.
"""

using TOML

"""
    run_pme_analysis(config_file, model_file; gpu_id="0")

Run PME analysis with GPU acceleration for Pi elements.
This function provides a simplified interface for GPU-enabled PME calculations.
"""
function run_pme_analysis(config_file::String, model_file::String; gpu_id::String="0")
    # Load configuration
    config = load_config(config_file)
    
    # Load model configuration data directly from TOML
    model_data = TOML.parsefile(model_file)
    
    # Merge model parameters into main config
    if haskey(model_data, "model")
        config.model.type = get(model_data["model"], "type", "ExpDisk")
        # Copy model parameters
        for (key, value) in model_data["model"]
            config.model.parameters[key] = value
        end
    end
    
    # Override settings from model physics section
    if haskey(model_data, "physics")
        physics = model_data["physics"]
        haskey(physics, "m") && (config.core.m = physics["m"])
        haskey(physics, "beta") && (config.physics.beta = physics["beta"])
    end
    
    # Store GPU ID in config for later use
    setfield!(config, :gpu_id, gpu_id)
    
    # Run the PME calculation with GPU support
    return run_complete_pme_calculation_gpu(
        config,
        force_recalculate = true,
        save_intermediate = true,
        gpu_id = gpu_id
    )
end

"""
    run_complete_pme_calculation_gpu(config; gpu_id="0", kwargs...)

GPU-enabled version of run_complete_pme_calculation.
Uses GPU acceleration for Pi elements calculation.
"""
function run_complete_pme_calculation_gpu(config::PMEConfig; 
                                         gpu_id::String="0",
                                         force_recalculate::Bool=false,
                                         resonance_selection::Vector{Int}=[1,1,1,1,1,1],
                                         save_intermediate::Bool=true,
                                         export_matlab::Bool=false)
    
    try
        # Override Pi elements calculation with GPU version
        # We'll temporarily replace the PiElements module
#         println("🔥 Initializing GPU-accelerated PME calculation...")
        
        # Store GPU ID in global variable for access by Pi elements calculation
        global _gpu_id = gpu_id
        
        # Load GPU Pi elements module and override the function
        include("matrix/PiElementsGPUSimple.jl")
        
        # Replace the Pi elements function in the PiElements module
        # We'll use a trick: define the function in the right scope
        eval(:(PiElements.calculate_pi_elements = calculate_pi_elements))
        
        # Run the standard PME workflow
        return run_complete_pme_calculation(
            config,
            force_recalculate = force_recalculate,
            resonance_selection = resonance_selection,
            save_intermediate = save_intermediate,
            export_matlab = export_matlab
        )
        
    catch e
#         println("❌ GPU calculation failed: $e")
#         println("🔄 Falling back to CPU calculation...")
        
        # Fallback to CPU calculation
        return run_complete_pme_calculation(
            config,
            force_recalculate = force_recalculate,
            resonance_selection = resonance_selection,
            save_intermediate = save_intermediate,
            export_matlab = export_matlab
        )
    end
end

