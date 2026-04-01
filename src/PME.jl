# src/PME.jl - Main module file
"""
The PME module provides a complete framework for calculating galactic normal modes
using the Polyachenko Matrix Equation method. It includes functionality for model setup,
grid construction, orbit calculation, matrix construction, and eigenvalue analysis.
"""
module PME

# Version info
const VERSION = v"0.1.0"

# Include all submodules that constitute the PME framework
# Order is important - dependencies must be loaded first
include("utils/Progress.jl")
using .ProgressUtils

include("utils/CUDAUtils.jl")
using .CUDAUtils

include("config/Configuration.jl")
using .Configuration

include("models/AbstractModel.jl")
using .AbstractModel

include("models/ExpDisk.jl")
include("models/Isochrone.jl")
include("models/IsochroneTaperJH.jl")
include("models/IsochroneTaperZH.jl")
include("models/IsochroneTaperTanh.jl")
include("models/IsochroneTaperExp.jl")
include("models/IsochroneTaperPoly3.jl")
using .ExpDisk
using .Isochrone
using .IsochroneTaperJH
using .IsochroneTaperZH
using .IsochroneTaperTanh
using .IsochroneTaperExp
using .IsochroneTaperPoly3

include("models/Miyamoto.jl")
include("models/MiyamotoTaperExp.jl")
include("models/MiyamotoTaperTanh.jl")
include("models/MiyamotoTaperPoly3.jl")
using .Miyamoto
using .MiyamotoTaperExp
using .MiyamotoTaperTanh
using .MiyamotoTaperPoly3

include("models/Toomre.jl")
using .Toomre

include("models/Kuzmin.jl")
include("models/KuzminTaperPoly3.jl")
include("models/KuzminTaperPoly3L.jl")
using .Kuzmin
using .KuzminTaperPoly3
using .KuzminTaperPoly3L

include("grids/GridConstruction.jl")
using .GridConstruction

include("orbits/OrbitCalculator.jl")
using .OrbitCalculator

include("matrix/PiElements.jl")
using .PiElements

# Conditionally load GPU modules
if pme_has_cuda()
    include("matrix/PiElementsGPU.jl")
    include("matrix/PiElementsMultiGPU.jl")
    using .PiElementsGPU, .PiElementsMultiGPU
end

include("matrix/MatrixCalculator.jl")
using .MatrixCalculator

include("matrix/KEigenvalues.jl")
using .KEigenvalues

include("matrix/ProgressiveKRES.jl")
using .ProgressiveKRES

include("io/BinaryIO.jl")
using .BinaryIO

include("PMEWorkflow.jl")

# Conditionally load GPU workflow
if pme_has_cuda()
    include("GPUWorkflow.jl")
end

# --- Public API Exports ---

# Configuration: Load, save, and manage calculation parameters
export PMEConfig, CoreConfig, GridConfig, PhysicsConfig, PhaseSpaceConfig, IOConfig, ModelConfig, ToleranceConfig, EllipticConfig, load_config, save_config, get_k0

# Models: Define the galactic potential and distribution function
export AbstractGalacticModel, ExpDiskModel, create_expdisk_model
export MiyamotoModel, create_miyamoto_model
export MiyamotoTaperExpModel, create_miyamoto_taper_exp_model
export MiyamotoTaperTanhModel, create_miyamoto_taper_tanh_model
export MiyamotoTaperPoly3Model, create_miyamoto_taper_poly3_model
export IsochroneModel, create_isochrone_model
export IsochroneTaperJHModel, create_isochrone_taper_jh_model
export IsochroneTaperZHModel, create_isochrone_taper_zh_model
export IsochroneTaperTanhModel, create_isochrone_taper_tanh_model
export IsochroneTaperExpModel, create_isochrone_taper_exp_model
export IsochroneTaperPoly3Model, create_isochrone_taper_poly3_model
export ToomreModel, create_toomre_model
export KuzminModel, create_kuzmin_model
export KuzminTaperPoly3Model, create_kuzmin_taper_poly3_model
export KuzminTaperPoly3LModel, create_kuzmin_taper_poly3L_model

# Grids and Orbits: Construct computational grids and calculate orbital data
export PMEGrids, OrbitData, create_pme_grids, calculate_orbits, evaluate_distribution_function

# Workflow: High-level functions to run the entire calculation pipeline
export run_complete_pme_calculation

# CUDA utilities
export pme_has_cuda, cuda_functional, gpu_info

# --- Module Initialization ---

function __init__()
    # Only show initialization messages if PME_VERBOSE environment variable is set
    # This allows suppressing messages on workers while keeping them on the main process
    show_init_messages = get(ENV, "PME_VERBOSE", "false") == "true"

    if show_init_messages
        @info "PME.jl v$(VERSION) loaded. Welcome to the Polyachenko Matrix Equation solver."

        # Show GPU status on load (suppress warnings on coordinator nodes)
        if cuda_functional()
            @info "GPU acceleration available"
        elseif pme_has_cuda()
            # Only show CUDA warning if we're likely on a worker node that should have GPU
            # Skip warning on coordinator nodes (like MacBooks) that don't need GPU
            if !Sys.isapple() || get(ENV, "PME_SHOW_CUDA_WARNING", "false") == "true"
                @info "CUDA installed but not functional - using CPU only"
            end
        else
            @info "CPU-only mode (to enable GPU: using Pkg; Pkg.add(\"CUDA\"))"
        end
    end
end

end # module PME
