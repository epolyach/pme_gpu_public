# src/config/Configuration.jl
"""
Handles loading, parsing, and saving of PME configurations.
Supports modern TOML format and legacy C-style .cfg files.
"""
module Configuration

using TOML
using Dates

export PMEConfig, CoreConfig, GridConfig, PhysicsConfig, PhaseSpaceConfig, IOConfig, ModelConfig, ToleranceConfig, EllipticConfig, CPUConfig, EigenvectorConfig, EtaSweepConfig,
       load_config, save_config, get_k0,
       setup_data_directory, get_data_path, check_data_file

# Define substructs for better organization, matching the TOML structure
@kwdef mutable struct CoreConfig
    m::Int = 2
    kres::Int = 7  # Must be odd; resonance numbers l ∈ {-(kres-1)/2, ..., +(kres-1)/2}
end

@kwdef mutable struct GridConfig
    NR::Int = 71
    Nv::Int = 15
    Nvt::Union{Int, Nothing} = nothing  # Number of circulation points in taper region (for grid types 6, 7, 8)
    NW::Int = 101
    NW_orbit::Int = 10001
    radial_grid_type::Int = 0
    circulation_grid_type::Int = 2
    
    # Radial grid parameters
    alpha::Float64 = 3.0      # For grid types 0 (exponential) and 1 (rational)
    R_min::Float64 = 0.1      # For grid types 2 (linear) and 3 (logarithmic)
    R_max::Float64 = 16.0     # For grid types 2 (linear) and 3 (logarithmic)
    # Circulation grid parameters
    alpha_v::Float64 = 3.0    # Power/stretch exponent for circulation grids (types 4,5); >1 clusters near lower end
    v_jump_scale::Float64 = 0.05  # Scale for v in s = tanh(v / v_jump_scale) (type 5)
end

@kwdef mutable struct PhysicsConfig
    beta::Float64 = 1e-5
    unit_mass::Float64 = 1.0      # Total mass in model units
    unit_length::Float64 = 1.0    # Scale length in model units
    selfgravity::Float64 = 1.0    # Self-gravity scaling factor (0.0 to 1.0)
end

@kwdef mutable struct PhaseSpaceConfig
    optimization_on::Bool = true
    full_space::Bool = true
    sharp_df::Bool = false
    Omega_2_lim::Int = 1      # Omega_2 at L=0: +1=L->0+, 0=Omega_2(J,0)=0, -1=L->0-
    delta_F0::Float64 = 1e-3          # Circulation grid parameter
end

@kwdef mutable struct IOConfig
    data_path::String = "data"
    overwrite_data::Bool = false
    verbose::Bool = true
    debug::Bool = false
    single_precision::Bool = false
    binary_output::Bool = true
    max_display_modes::Int = 10
end

@kwdef mutable struct ModelConfig
    type::String = "ExpDisk"
    # Model parameters are stored as fields in this struct
    mk::Int = 12
    unit_mass::Float64 = 1.0
    unit_length::Float64 = 1.0
    # ExpDisk parameters
    RC::Float64 = 1.0
    N::Int = 6
    lambda::Float64 = 0.625
    alpha::Float64 = 0.34
    L0::Float64 = 0.1
    # Toomre-Zang parameters
    n_zang::Int = 4           # Exponent for Zang taper
    q1::Int = 7               # Controls radial velocity dispersion
    # Isochrone taper parameters
    Jc::Union{Float64,Nothing} = nothing
    Rc::Union{Float64,Nothing} = nothing
    eta::Float64 = 1.0
    # Miyamoto parameters
    n_M::Int = 3
    L_0::Float64 = 0.2          # MiyamotoTaperExp: exponential taper parameter
    v_0::Float64 = 0.2          # MiyamotoTaperTanh: tanh taper parameter (circulation scale)
end

@kwdef mutable struct ToleranceConfig
    instability_threshold::Float64 = 1e-3
    orbital_tolerance::Float64 = 1e-8
end

@kwdef mutable struct EllipticConfig
    NZ::Int = 10001                     # Number of points for z-grid calculation
    z_precision::Float64 = 1e-6
    psi_integration_points::Int = 10001
end

@kwdef mutable struct CPUConfig
    precision_double::Bool = true  # true = Float64, false = Float32 for Pi-elements and K-matrix
    max_threads::Int = 8           # Hard cap on BLAS threads
end

@kwdef mutable struct GPUConfig
    precision_double::Bool = true  # true = Float64, false = Float32
end

@kwdef mutable struct EtaSweepConfig
    eta_start::Float64 = 0.5          # Starting eta value
    eta_end::Float64 = 0.005          # Ending eta value
    Neta::Int = 11                    # Number of eta points
    # eta values are computed as reverse(logspace(-3, 0, Neta))
end

@kwdef mutable struct EigenvectorConfig
    compute::Bool = true              # Whether to compute eigenvectors along with eigenvalues
    num_output::Int = 6               # Number of eigenvectors to output for analysis
    iterative::Bool = false           # Use iterative solver (Arpack) - recommended for matrices > 20000x20000
    krylov_dim::Int = 50              # Krylov subspace dimension (larger = more accurate but slower)
    tol::Float64 = 1e-8               # Convergence tolerance for iterative solver
    # Progressive KRES mode
    progressive::Bool = false         # Enable progressive KRES expansion
    kres_start::Int = 7               # Starting KRES (must be odd)
    kres_max::Int = 21                # Maximum KRES to expand to
    kres_step::Int = 2                # Step size for KRES expansion (must be even to keep KRES odd)
    convergence_tol::Float64 = 1e-4   # Stop when eigenvalue change < this
    # Shift values for shift-invert: Ωₚ (pattern speed) and γ (growth rate)
    # Number of eigenvalues to find = length(shift_Omega_p)
    # Code multiplies Ωₚ by m to get ω_real
    shift_Omega_p::Vector{Float64} = [0.4397, 0.3916, 0.3607, 0.3309, 0.4284, 0.3035, 0.2786, 0.2558, 0.2348, 0.2154]
    shift_gamma::Vector{Float64} = [0.1256, 0.0516, 0.0483, 0.0441, 0.0420, 0.0412, 0.0384, 0.0355, 0.0327, 0.0300]
end

"""
    PMEConfig

Main configuration structure for PME calculations, composed of organized substructs.
"""
@kwdef mutable struct PMEConfig
    core::CoreConfig = CoreConfig()
    grid::GridConfig = GridConfig()
    physics::PhysicsConfig = PhysicsConfig()
    phase_space::PhaseSpaceConfig = PhaseSpaceConfig()
    eigenvectors::EigenvectorConfig = EigenvectorConfig()
    io::IOConfig = IOConfig()
    model::ModelConfig = ModelConfig()
    gpu::GPUConfig = GPUConfig()
    tolerances::ToleranceConfig = ToleranceConfig()
    elliptic::EllipticConfig = EllipticConfig()
    cpu::CPUConfig = CPUConfig()
    eta_sweep::EtaSweepConfig = EtaSweepConfig()
end

"""
    load_config(filename::String) -> PMEConfig

Universal config loader that detects the file format (TOML or legacy .cfg)
and returns a populated PMEConfig struct.
"""
function load_config(filename::String)::PMEConfig
    if !isfile(filename)
        @error "Configuration file '$filename' not found!"
        return PMEConfig() # Return defaults
    end
    
    ext = lowercase(splitext(filename)[2])
    
    config = if ext == ".toml"
        parse_toml_config(filename)
    elseif ext == ".cfg"
        @warn "Loading legacy .cfg format. Consider converting to .toml for more features."
        parse_cfg_file(filename)
    else
        @error "Unknown config format '$ext'. Please use .toml or .cfg."
        PMEConfig()
    end
    
    # Only show loading message if verbose is enabled
    # Check PME_VERBOSE environment variable first (used by workers to suppress output)
    pme_verbose = get(ENV, "PME_VERBOSE", "true")
    if ext == ".toml" && pme_verbose != "false" && config.io.verbose
        # println("Loading TOML format: $filename")
    
    end
    # Setup data directory after loading
    setup_data_directory(config)
    
    return config
end

"""
    parse_toml_config(filename::String) -> PMEConfig

Parses a TOML configuration file into a PMEConfig struct.
"""
function parse_toml_config(filename::String)::PMEConfig
    data = TOML.parsefile(filename)
    config = PMEConfig()
    
    # Helper to merge a dictionary into a struct
    function merge_dict_to_struct!(s, d)
        for (key, value) in d
            if hasfield(typeof(s), Symbol(key))
                field_type = fieldtype(typeof(s), Symbol(key))
                setfield!(s, Symbol(key), convert(field_type, value))
            end
        end
    end

    haskey(data, "core") && merge_dict_to_struct!(config.core, data["core"])
    haskey(data, "grid") && merge_dict_to_struct!(config.grid, data["grid"])
    haskey(data, "physics") && merge_dict_to_struct!(config.physics, data["physics"])
    haskey(data, "phase_space") && merge_dict_to_struct!(config.phase_space, data["phase_space"])
    haskey(data, "gpu") && merge_dict_to_struct!(config.gpu, data["gpu"])
    haskey(data, "eigenvectors") && merge_dict_to_struct!(config.eigenvectors, data["eigenvectors"])
    haskey(data, "io") && merge_dict_to_struct!(config.io, data["io"])
    haskey(data, "tolerances") && merge_dict_to_struct!(config.tolerances, data["tolerances"])
    haskey(data, "elliptic") && merge_dict_to_struct!(config.elliptic, data["elliptic"])
    haskey(data, "cpu") && merge_dict_to_struct!(config.cpu, data["cpu"])
    
    if haskey(data, "model")
        merge_dict_to_struct!(config.model, data["model"])
    end
    
    return config
end

"""
    parse_cfg_file(filename::String) -> PMEConfig

Parses a legacy .cfg file (C-style #define) into a PMEConfig struct.
"""
function parse_cfg_file(filename::String)::PMEConfig
    config = PMEConfig()
    # Mapping from CFG keys to new struct fields
    key_map = Dict(
        "m" => (:core, :m), "KRES" => (:core, :kres),
        "NL" => (:grid, :NR), "NI" => (:grid, :Nv), "NZ" => (:tolerances, :NZ), "NW" => (:grid, :NW),
        "BETA" => (:physics, :beta), "PATH" => (:io, :data_path)
    )

    open(filename, "r") do file
        for line in eachline(file)
            line = strip(line)
            if startswith(line, "#define")
                parts = split(line)
                if length(parts) >= 3
                    key, value_str = parts[2], join(parts[3:end], " ")
                    value = tryparse(Int, value_str)
                    if value === nothing
                        value = tryparse(Float64, value_str)
                    end
                    if value === nothing
                        value = strip(value_str, '"') # As string
                    end

                    if haskey(key_map, key)
                        section, field = key_map[key]
                        setfield!(getfield(config, section), field, value)
                    else
                        @warn "Unknown legacy config key: $key"
                    end
                end
            end
        end
    end
    return config
end

"""
    save_config(config::PMEConfig, filename::String)

Saves a PMEConfig struct to a file in TOML format.
"""
function save_config(config::PMEConfig, filename::String)
    # Convert config to a nested dictionary for serialization
    config_dict = Dict(
        "core" => Dict(fieldnames(CoreConfig) .=> getfield.(Ref(config.core), fieldnames(CoreConfig))),
        "grid" => Dict(fieldnames(GridConfig) .=> getfield.(Ref(config.grid), fieldnames(GridConfig))),
        "physics" => Dict(fieldnames(PhysicsConfig) .=> getfield.(Ref(config.physics), fieldnames(PhysicsConfig))),
        "phase_space" => Dict(fieldnames(PhaseSpaceConfig) .=> getfield.(Ref(config.phase_space), fieldnames(PhaseSpaceConfig))),
        "eigenvectors" => Dict(fieldnames(EigenvectorConfig) .=> getfield.(Ref(config.eigenvectors), fieldnames(EigenvectorConfig))),
        "io" => Dict(fieldnames(IOConfig) .=> getfield.(Ref(config.io), fieldnames(IOConfig))),
        "tolerances" => Dict(fieldnames(ToleranceConfig) .=> getfield.(Ref(config.tolerances), fieldnames(ToleranceConfig))),
        "elliptic" => Dict(fieldnames(EllipticConfig) .=> getfield.(Ref(config.elliptic), fieldnames(EllipticConfig))),
        "cpu" => Dict(fieldnames(CPUConfig) .=> getfield.(Ref(config.cpu), fieldnames(CPUConfig))),
        "model" => Dict(fieldnames(ModelConfig) .=> getfield.(Ref(config.model), fieldnames(ModelConfig)))
    )
    
    ext = lowercase(splitext(filename)[2])
    if ext == ".toml"
        open(filename, "w") do io
            TOML.print(io, config_dict)
        end
    else
        @error "Unsupported save format: $ext. Please use .toml."
        return
    end
#     println("Configuration saved to: $filename")
end

"""
    setup_data_directory(config::PMEConfig) -> String

Ensures the data output directory and its subdirectories exist.
Prompts for overwrite confirmation if the directory is not empty.
"""
function setup_data_directory(config::PMEConfig)
    data_dir = config.io.data_path
    
    # Check if directory exists and overwrite is not already configured
    if isdir(data_dir) && !config.io.overwrite_data
        # Check if we're in a non-interactive environment (e.g., automated run)
        if isinteractive()
            print("Data directory '$data_dir' already exists. Overwrite? [y/N]: ")
            response = readline()
            if lowercase(strip(response)) == "y"
                config.io.overwrite_data = true
#                 println("✓ Overwriting enabled for this session.")
            end
        else
            # In non-interactive mode, default to not overwriting
#             println("Data directory '$data_dir' already exists. Use overwrite_data=true in config to overwrite.")
        end
    end
    
    mkpath(data_dir)
    for subdir in ["binary", "results", "plots", "logs"]
        mkpath(joinpath(data_dir, subdir))
    end
    
    return data_dir
end

# Other utility functions remain the same
function get_data_path(config::PMEConfig, filename::String, subdir::String="binary")::String
    return joinpath(config.io.data_path, subdir, filename)
end

function check_data_file(config::PMEConfig, filepath::String)::Bool
    if isfile(filepath) && !config.io.overwrite_data
        config.io.verbose && println("Skipping existing: $filepath")
        return false
    end
    return true
end


end # module Configuration
