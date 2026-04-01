# src/PMEWorkflow.jl
# Test DF evaluation for L < 0 points
# Insert this into PMEWorkflow.jl after model creation

function export_distribution_to_csv(orbit_data, config)
    data_dir = config.io.data_path
    csv_dir = joinpath(data_dir, "csv")
    mkpath(csv_dir)
    
    NR, Nv = size(orbit_data.F0)
    
    # println("\n" * "="^80)
    # println("EXPORTING CSV FILES")
    # println("="^80)
    
    # Export F0
    # println("Writing F0.csv...")
    open(joinpath(csv_dir, "F0.csv"), "w") do io
        # Header with v indices
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        # Data rows
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.F0[iR, iv]))
            end
            write(io, "\n")
        end
    end
    println("─"^50)
    
    # Export FE
    # println("Writing FE.csv...")
    open(joinpath(csv_dir, "FE.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.FE[iR, iv]))
            end
            write(io, "\n")
        end
    end
    
    # Export FL
    # println("Writing FL.csv...")
    open(joinpath(csv_dir, "FL.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.FL[iR, iv]))
            end
            write(io, "\n")
        end
    end
    
    # Export t1
    # println("Writing t1.csv...")
    open(joinpath(csv_dir, "t1.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.t1[iR, iv]))
            end
            write(io, "\n")
        end
    end
    
    # Export t2
    # println("Writing t2.csv...")
    open(joinpath(csv_dir, "t2.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.t2[iR, iv]))
            end
            write(io, "\n")
        end
    end
    
    # Export Omega_1
    # println("Writing Omega_1.csv...")
    open(joinpath(csv_dir, "Omega_1.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.Omega_1[iR, iv]))
            end
            write(io, "\n")
        end
    end
    
    # Export Omega_2
    # println("Writing Omega_2.csv...")
    open(joinpath(csv_dir, "Omega_2.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.Omega_2[iR, iv]))
            end
            write(io, "\n")
        end
    end

    # Export jacobian
    # println("Writing jacobian.csv...")
    open(joinpath(csv_dir, "jacobian.csv"), "w") do io
        write(io, "Rc")
        for iv in 1:Nv
            write(io, ",v$iv")
        end
        write(io, "\n")
        
        for iR in 1:NR
            write(io, string(orbit_data.grids.Rc[iR, 1]))
            for iv in 1:Nv
                write(io, ",", string(orbit_data.jacobian[iR, iv]))
            end
            write(io, "\n")
        end
    end
    
    println("✓ CSV files exported to: $csv_dir")
    # println("="^80)
end


function test_df_negative_L(model, grids)
    println("\n" * "="^80)
    println("TESTING DF FOR NEGATIVE L")
    println("="^80)
    
    # Find a radial point with negative L values
    NR, Nv = size(grids.L_m)
    
    for iR in 1:min(5, NR)
        # Count negative L at this radius
        neg_L_count = sum(grids.L_m[iR, :] .< 0)
        pos_L_count = sum(grids.L_m[iR, :] .> 0)
        
        if neg_L_count > 0
            println("\nRadial point iR=$iR (Rc=$(grids.Rc[iR,1])):")
            println("  Negative L points: $neg_L_count")
            println("  Positive L points: $pos_L_count")
            
            # Test first few negative L points
            for iv in 1:Nv
                L_val = grids.L_m[iR, iv]
                if L_val < 0
                    E_val = grids.E[iR, iv]
                    r1_val = grids.R1[iR, iv]
                    r2_val = grids.R2[iR, iv]
                    Rc_val = grids.Rc[iR, iv]
                    v = grids.v[iR, iv]
                   
                    # Call DF directly
                    F0 = model.distribution_function(E_val, L_val, iR, iv, grids)
                    
                    println("  iv=$iv: L=$L_val, E=$E_val, v=$v, F0=$F0")
                    
                    if F0 == 0.0
                        println("    WARNING: F0=0 for L<0!")
                        println("    r1=$r1_val, r2=$r2_val, Rc=$Rc_val")
                    elseif F0 > 0
                        println("    ✓ SUCCESS: F0 > 0 for L < 0")
                    end
                    
                    # Only test first 3 negative L points
                    if iv >= 3
                        break
                    end
                end
            end
            
            # Only test first radial point with negative L
            break
        end
    end
    
    println("="^80)
end


function get_radial_grid_name(grid_type::Int)
    grid_names = Dict(
        0 => "Exponential",
        1 => "Rational", 
        2 => "Linear",
        3 => "Logarithmic"
    )
    return get(grid_names, grid_type, "Unknown")
end
function get_radial_grid_parameters(grid_type::Int, parameters::Dict{String,Any})
    if grid_type == 0
        # Exponential grid: Rc = -α log(u)
        return "α=$(parameters["alpha"])"
    elseif grid_type == 1  
        # Rational grid: Rc = α(1-u)/u
        return "α=$(parameters["alpha"])"
    elseif grid_type == 2
        # Linear grid
        return "R_min=$(parameters["R_min"]), R_max=$(parameters["R_max"])"
    elseif grid_type == 3
        # Logarithmic grid  
        return "R_min=$(parameters["R_min"]), R_max=$(parameters["R_max"])"
    else
        return "Unknown grid type"
    end
end

function get_eccentricity_grid_name(grid_type::Int)
    grid_names = Dict(
        1 => "Linear circulation",
        2 => "Trapezoidal circulation", 
        3 => "Adaptive",
        4 => "Velocity dispersion optimized"
    )
    return get(grid_names, grid_type, "Unknown")
end
"""
High-performance PME calculation workflow optimized for 64-core AMD systems
Integrates model setup, orbit calculation, and matrix eigenvalue solving
Replaces the entire MATLAB + C + MATLAB workflow with unified Julia
Features advanced progress monitoring, NUMA optimization, and automatic performance tuning
"""

using Printf
using ProgressMeter
using Dates

# Import local modules
# Import local modules will be accessed through PME namespace

# Modules are already included in PME.jl and will be accessed through PME

"""
Complete PME calculation from configuration to eigenvalues
This is the unified function for all PME calculations (tests and production)
Accepts either a config file path (String) or a config object (PMEConfig)
"""
function run_complete_pme_calculation(config_input::Union{String,PMEConfig}="configs/pme_default.toml";
                                    force_recalculate::Bool=false,
                                    resonance_selection::Vector{Int}=[1,1,1,1,1,1],
                                    save_intermediate::Bool=true,
                                    export_matlab::Bool=false,
                                    gpu_id::Int=-1, 
                                    gpu_devices::Vector{Int}=Int[],
                                    blas_threads::Union{Int,String,Nothing}=nothing)
    
    # Load or use provided configuration
    if isa(config_input, String)
        config = load_config(config_input)
    else
        config = config_input
    end
    
    # Generate timestamp for this calculation run
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    # Verbosity control
    verbose = config.io.verbose
    progressive_log_path = nothing  # Will be set if progressive mode is used
    
    
    # Create model, grids, calculate orbits and distribution function
    model = if config.model.type == "ExpDisk"
        create_expdisk_model(config)
    elseif config.model.type == "Isochrone"
        create_isochrone_model(config)
    elseif config.model.type == "IsochroneTaperJH"
        create_isochrone_taper_jh_model(config)
    elseif config.model.type == "IsochroneTaperZH"
        create_isochrone_taper_zh_model(config)
    elseif config.model.type == "IsochroneTaperTanh"
        create_isochrone_taper_tanh_model(config)
    elseif config.model.type == "Miyamoto"
        create_miyamoto_model(config)
    elseif config.model.type == "IsochroneTaperExp"
        create_isochrone_taper_exp_model(config)
    elseif config.model.type == "MiyamotoTaperExp"
        create_miyamoto_taper_exp_model(config)
    elseif config.model.type == "MiyamotoTaperTanh"
        create_miyamoto_taper_tanh_model(config)
    elseif config.model.type == "MiyamotoTaperPoly3"
        create_miyamoto_taper_poly3_model(config)
    elseif config.model.type == "KuzminTaperPoly3L"
        create_kuzmin_taper_poly3L_model(config)
    elseif config.model.type == "KuzminTaperPoly3"
        create_kuzmin_taper_poly3_model(config)
    elseif config.model.type == "Kuzmin"
        create_kuzmin_model(config)
    elseif config.model.type == "IsochroneTaperPoly3"
        create_isochrone_taper_poly3_model(config)
    elseif config.model.type == "Toomre"
        create_toomre_model(config)
    else
        error("Unknown model type: $(config.model.type). Supported types: ExpDisk, Isochrone, IsochroneTaperJH, IsochroneTaperZH, IsochroneTaperTanh, IsochroneTaperExp, IsochroneTaperPoly3, Miyamoto, MiyamotoTaperExp, MiyamotoTaperTanh, MiyamotoTaperPoly3, KuzminTaperPoly3L, Kuzmin, KuzminTaperPoly3, Toomre")
    end
    
    grids = create_pme_grids(config, model)
    # test_df_negative_L(model, grids)
    orbit_data = calculate_orbits(grids, model, config)
    orbit_data = evaluate_distribution_function(orbit_data, model)

    verbose && println("\n" * "="^50)
    verbose && println("ARRAY RANGES")
    verbose && println("="^50)
    verbose && println("✓ Phase space:")
    verbose && println("  Guiding center radius range: $(round(minimum(grids.Rc), digits=4)) - $(round(maximum(grids.Rc), digits=4))")
    verbose && println("  R1 range: $(round(minimum(orbit_data.grids.R1), digits=4)) - $(round(maximum(orbit_data.grids.R1), digits=4))")
    verbose && println("  R2 range: $(round(minimum(orbit_data.grids.R2), digits=4)) - $(round(maximum(orbit_data.grids.R2), digits=4))")
    verbose && println("  Energy range: $(round(minimum(grids.E), digits=4)) - $(round(maximum(grids.E), digits=4))")
    verbose && println("  Angular momentum range: $(round(minimum(grids.L_m), digits=4)) - $(round(maximum(grids.L_m), digits=4))")
    verbose && println("")
    verbose && println("✓ Orbital trajectories:")
    verbose && println("  Frequency range Ω₁: $(round(minimum(orbit_data.Omega_1), digits=4)) - $(round(maximum(orbit_data.Omega_1), digits=4))")
    verbose && println("  Frequency range Ω₂: $(round(minimum(orbit_data.Omega_2), digits=4)) - $(round(maximum(orbit_data.Omega_2), digits=4))")
    verbose && println("  Action range Iᵣ: $(round(minimum(orbit_data.Ir), digits=4)) - $(round(maximum(orbit_data.Ir), digits=4))")
    verbose && println("")
    verbose && println("✓ Distribution function:")
    verbose && println("  DF range: $(round(minimum(orbit_data.F0), digits=6)) - $(round(maximum(orbit_data.F0), digits=6))")
    verbose && println("  ∂DF/∂E range: $(round(minimum(orbit_data.FE), digits=6)) - $(round(maximum(orbit_data.FE), digits=6))")
    verbose && println("  ∂DF/∂L range: $(round(minimum(orbit_data.FL), digits=6)) - $(round(maximum(orbit_data.FL), digits=6))")
    
    # Save intermediate results
    if save_intermediate && config.io.binary_output
        verbose && println("")
        # Export CSV files before binary
        export_distribution_to_csv(orbit_data, config)        
        save_orbit_data_binary(orbit_data, config)

        verbose && println("")
    end
    
    # Calculate elliptic function ψ(z)
    z, psi_z = calculate_z_and_psi_z(config)
    # println("✓ Elliptic function ψ(z) calculated and saved")

    # Matrix Element Calculation
    verbose && println("\n" * "="^50)
    verbose && println("MATRIX ELEMENT CALCULATION")
    verbose && println("="^50)
    
    # Variable to track if we use chunked mode (which returns K directly)
    use_chunked = false
    K = nothing
    pi4 = nothing
    
    # Unified GPU path: always use chunked GPU K build when any GPUs are available
    if !isempty(gpu_devices)
        println("  Using unified chunked GPU path (devices: $(gpu_devices))")
        K = PiElementsMultiGPU.calculate_k_matrix_chunked_gpu(config, orbit_data, model, psi_z, z, gpu_devices)
        use_chunked = true
    elseif gpu_id >= 0
        # Backward-compatibility if a single gpu_id is provided
        println("  Using unified chunked GPU path (device: $gpu_id)")
        K = PiElementsMultiGPU.calculate_k_matrix_chunked_gpu(config, orbit_data, model, psi_z, z, [gpu_id])
        use_chunked = true
    else
        # CPU
        verbose && println("  Using CPU multi-threading for Pi elements")
        pi4 = PiElements.calculate_pi_elements(config, orbit_data, z, psi_z)
        verbose && println("✓ Pi matrix elements calculated")
    end
    
    # Debug output: save Pi elements if debug mode is enabled (only if not chunked)
    if !use_chunked && config.io.debug && config.io.binary_output
        verbose && println("Debug mode: saving Pi matrix elements...")
        data_dir = config.io.data_path
        binary_dir = joinpath(data_dir, "binary")
        mkpath(binary_dir)
        
        single_precision = config.io.single_precision
        if single_precision
            BinaryIO.write_binary_float32(joinpath(binary_dir, "pi_elements.bin"), pi4)
        else
            BinaryIO.write_binary_float64(joinpath(binary_dir, "pi_elements.bin"), pi4)
        end
        
        verbose && println("✓ Pi matrix elements saved to binary file (dimensions: $(size(pi4)))")
    end

    # K Matrix Construction (skip if already computed via chunked mode)
    if !use_chunked
        K = MatrixCalculator.construct_k_matrix(config, model, orbit_data, pi4)
        verbose && println("✓ K matrix constructed")
    end
    
    # Calculate eigenvalues - always use progressive pipeline (with logging)
    # If progressive=false, use single KRES value (kres_min = kres_max = kres)
    if !config.eigenvectors.progressive
        # Override kres_min and kres_max to use single kres value
        config.eigenvectors.kres_start = config.core.kres
        config.eigenvectors.kres_max = config.core.kres
        verbose && println("  Single KRES mode: kres = $(config.core.kres)")
    else
        verbose && println("  Progressive KRES mode: $(config.eigenvectors.kres_start) → $(config.eigenvectors.kres_max)")
    end
    
    # Create Pi block computation function that captures context
    compute_pi_block = (l_i, l_j) -> PiElementsMultiGPU.compute_pi_block_gpu(
        config, orbit_data, z, psi_z, l_i, l_j; device_id=gpu_devices[1]
    )
    
    # Pass pre-computed K when not in progressive mode (single KRES, no recomputation needed)
    K_for_progressive = config.eigenvectors.progressive ? nothing : K
    eigenvalues, eigenvectors, final_kres, progressive_log_path = ProgressiveKRES.progressive_eigenvalue_solve(
        config, orbit_data, model, psi_z, z, compute_pi_block; K_precomputed=K_for_progressive
    )
    
    # Update config.core.kres to final value for output
    config.core.kres = final_kres
    if config.eigenvectors.compute
        verbose && println("✓ Eigenvalues and eigenvectors calculated ($(length(eigenvalues)) total)")
        KEigenvalues.save_eigenvectors_and_grid(eigenvalues, eigenvectors, orbit_data, config, config.model.type, timestamp)
    end

    # Final Summary
    verbose && println("\n" * "="^80)
    # println("PME CALCULATION COMPLETE")
    # println("="^80)

    # Find unstable modes
    instability_threshold = config.tolerances.instability_threshold
    unstable_modes = eigenvalues[imag(eigenvalues) .> instability_threshold]
    if !isempty(unstable_modes)
        # Display formatted eigenvalue table (top 10 by γ, regardless of threshold)
        display_eigenvalue_table(eigenvalues, config, timestamp)

        # Print footer information
        verbose && println("──────────────────────────────────────────────────")
        verbose && println("NW = $(config.grid.NW)           # Orbital phase grid")
        verbose && println("NW_orbit = $(config.grid.NW_orbit)   # Orbit integration grid")
        if config.phase_space.sharp_df
            println("sharp_df = true    # With g_l(E,L=0) extension")
        end

        
        unstable_eigenvalue_file = joinpath(config.io.data_path, "results", "unstable_eigenvalues_$(timestamp).csv")
        open(unstable_eigenvalue_file, "w") do io
            # Write metadata header
            println(io, "# Run: $(timestamp)")
            println(io, "# Grid: $(config.grid.NR)x$(config.grid.Nv)x$(config.core.kres), r_gr=$(config.grid.radial_grid_type), v_gr=$(config.grid.circulation_grid_type)")
            println(io, "# beta=$(config.physics.beta), NW=$(config.grid.NW), NW_orbit=$(config.grid.NW_orbit), sharp_df=$(config.phase_space.sharp_df)")
            println(io, "Ωₚ,γ")
            for eig in unstable_modes
                println(io, "$(real(eig)),$(imag(eig))")
            end
        end
        txt = "✓ Unstable eigenvalues written to CSV file: $unstable_eigenvalue_file"
        # println(txt)
    else
        verbose && println("✅ GALACTIC DISK IS STABLE")
        verbose && println("──────────────────────────────────────────────────")
        verbose && println("NW = $(config.grid.NW)           # Orbital phase grid")
        verbose && println("NW_orbit = $(config.grid.NW_orbit)   # Orbit integration grid")
        if config.phase_space.sharp_df
            println("sharp_df = true    # With g_l(E,L=0) extension")
        end
        txt = "✓ No unstable modes found - disk is stable"
    end
        if progressive_log_path !== nothing
            txt *= "\nLog saved to: " * (isdefined(Main, :LOG_PATH) ? Main.LOG_PATH[] : progressive_log_path)
        end
    
    return eigenvalues, txt
end

"""
Shared pipeline: Steps 1-5 (Model setup through elliptic integrals)
This is the SINGLE implementation used everywhere.
Returns: model, grids, orbit_data, z, psi_z
"""
function run_shared_pipeline_steps_1_to_5(config::PMEConfig; save_intermediate::Bool=true)
    # Step 1: Model Setup
    println("\n" * "="^50)
    println("STEP 1: Galactic Model Setup")
    println("="^50)
    
    # Create model based on type from config
    if config.model.type == "ExpDisk"
        model = create_expdisk_model(config)
    elseif config.model.type == "IsochroneTaperJH"
        model = create_isochrone_taper_jh_model(config)
    elseif config.model.type == "IsochroneTaperZH"
        model = create_isochrone_taper_zh_model(config)
    elseif config.model.type == "IsochroneTaperTanh"
        model = create_isochrone_taper_tanh_model(config)
    elseif config.model.type == "IsochroneTaperExp"
        model = create_isochrone_taper_exp_model(config)
    elseif config.model.type == "Miyamoto"
        model = create_miyamoto_model(config)
    elseif config.model.type == "MiyamotoTaperExp"
        model = create_miyamoto_taper_exp_model(config)
    elseif config.model.type == "MiyamotoTaperTanh"
        model = create_miyamoto_taper_tanh_model(config)
    elseif config.model.type == "MiyamotoTaperPoly3"
        model = create_miyamoto_taper_poly3_model(config)
    elseif config.model.type == "KuzminTaperPoly3L"
        model = create_kuzmin_taper_poly3L_model(config)
    elseif config.model.type == "IsochroneTaperPoly3"
        model = create_isochrone_taper_poly3_model(config)
    elseif config.model.type == "Toomre"
        model = create_toomre_model(config)
    else
        error("Unknown model type: $(config.model.type). Supported types: ExpDisk, Isochrone, IsochroneTaperJH, IsochroneTaperZH, IsochroneTaperTanh, IsochroneTaperExp, IsochroneTaperPoly3, Miyamoto, MiyamotoTaperExp, MiyamotoTaperTanh, MiyamotoTaperPoly3, KuzminTaperPoly3L, Kuzmin, KuzminTaperPoly3, Toomre")
    end
    println("✓ Model created: $(model.model_type)")
    
    # Step 2: Grid Construction
    println("\n" * "="^50)
    println("STEP 2: Computational Grid Construction")
    println("="^50)
    
    grids = create_pme_grids(config, model)
    println("✓ Grids created:")
    println("  Radial: $(length(grids.radial.points)) points, type $(grids.radial.grid_type) ($(get_radial_grid_name(grids.radial.grid_type)))")
    println("    Parameters: $(get_radial_grid_parameters(grids.radial.grid_type, grids.radial.parameters))")
    println("  Eccentricity: $(size(grids.eccentricity.points, 2)) points per radius, type $(grids.eccentricity.grid_type) ($(get_eccentricity_grid_name(grids.eccentricity.grid_type)))")
    
    # Step 3: Orbital Motion Calculation
    println("\n" * "="^50)
    println("STEP 3: Orbital Motion Calculation")
    println("="^50)
    
    orbit_data = calculate_orbits(grids, model, config)
    println("✓ Orbital trajectories calculated")
    
    # Step 4: Distribution Function Evaluation
    println("\n" * "="^50)
    println("STEP 4: Distribution Function Evaluation")
    println("="^50)
    
    orbit_data = evaluate_distribution_function(orbit_data, model)
    println("✓ Distribution function evaluated")
    
    # Save intermediate results
    if save_intermediate && config.io.binary_output
        save_orbit_data_binary(orbit_data, config)
    end
    
    # Step 5: Elliptic integral psi_m(z)
    println("\n" * "="^50)
    println("STEP 5: Elliptic integral psi_m(z)")
    println("="^50)
    z, psi_z = calculate_z_and_psi_z(config)
    println("✓ Elliptic integral psi_m(z) calculated")
    
    # Save z and psi_z in debug mode
    if config.io.debug && save_intermediate && config.io.binary_output
        data_dir = config.io.data_path
        binary_dir = joinpath(data_dir, "binary")
        mkpath(binary_dir)
        
        if config.io.single_precision
            BinaryIO.write_binary_float32(joinpath(binary_dir, "z.bin"), z)
            BinaryIO.write_binary_float32(joinpath(binary_dir, "psi_z.bin"), psi_z)
        else
            BinaryIO.write_binary_float64(joinpath(binary_dir, "z.bin"), z)
            BinaryIO.write_binary_float64(joinpath(binary_dir, "psi_z.bin"), psi_z)
        end
        println("✓ z and psi_z saved for debug analysis")
    end
    
    return model, grids, orbit_data, z, psi_z
end

"""
Calculate z grid and psi(z) function using standard PME parameters.
This is the SINGLE implementation used everywhere.
"""
function calculate_z_and_psi_z(config::PMEConfig)
    Nz = config.elliptic.NZ
    z_precision = config.elliptic.z_precision
    z_range_exp = log10(z_precision)
    z = 1 .- 10 .^ range(z_range_exp, 0, length=Nz)
    z = reverse(z)
    psi_integration_points = config.elliptic.psi_integration_points
    psi_z = calculate_psi_z(config.core.m, z, psi_integration_points=psi_integration_points)
    return z, psi_z
end

"""
Calculate psi(z) function
"""
function calculate_psi_z(m::Int, z_values::AbstractVector{T}; psi_integration_points::Int) where T<:Real
    Nz = length(z_values)
    psi_z = zeros(T, Nz)
    
    Nth = psi_integration_points
    th = range(0, π, length=Nth)
    Sth = GridConstruction.simpson_coef(collect(th))
    costh = cos.(th)
    cos_mth = cos.(m .* th)
    
    for j in 1:Nz
        zj = z_values[j]
        # denominator = sqrt.(1 .+ beta^2 .+ zj^2 .- 2*zj .* costh)
        denominator = sqrt.(1 .- zj .* costh)
        integrand = cos_mth ./ denominator
        psi_z[j] = sum(Sth .* integrand)
    end
    
    psi_z = psi_z / π
    
    return psi_z
end

function display_eigenvalue_table(modes, config, timestamp)
    m = config.core.m
    # Use max_display_modes from config (default 10)
    max_modes = config.io.max_display_modes
    
    # Extract and sort by growth rate (descending)
    eigenvalue_data = [(real(mode)/m, imag(mode)) for mode in modes]
    sort!(eigenvalue_data, by=x->x[2], rev=true)  # Sort by γ (descending)
    
    # Take top modes
    top_modes = eigenvalue_data[1:min(max_modes, length(eigenvalue_data))]
    
    # Format beta value
    beta_str = @sprintf("%.0e", config.physics.beta)
    beta_str = replace(beta_str, r"e\+?0?(\d+)" => s"e-\\1")
    if occursin("e-0", beta_str)
        beta_str = replace(beta_str, "e-0" => "e-")
    end
    
    # Get grid type descriptions
    r_gr = config.grid.radial_grid_type
    v_gr = config.grid.circulation_grid_type
    
    # Radial grid parameter description
    radial_params = r_gr <= 1 ?
        " (alpha=$(config.grid.alpha))" :
        " (Rmin=$(config.grid.R_min), Rmax=$(config.grid.R_max))"
    
    # Header with grid info
    println(timestamp)
    println("")
    println("$(config.grid.NR)x$(config.grid.Nv)x$(config.core.kres), r_gr=$(r_gr)$(radial_params), v_gr=$(v_gr), beta=$(beta_str)")
    
    # Detailed grid description
    println("Radial grid:        type=$(r_gr)" * radial_params)
    println("Circulation grid:   type=$(v_gr)")
    
    # Taper information (if applicable)
    if occursin("Taper", config.model.type)
        taper_type = occursin("TaperTanh", config.model.type) ? "Tanh" :
                     occursin("TaperPoly3", config.model.type) ? "Poly3" :
                     occursin("TaperExp", config.model.type) ? "Exp" : "Unknown"
        println("Taper type:         $(taper_type)")
        if occursin("Miyamoto", config.model.type)
            if taper_type == "Exp"
                println("L_0:                $(config.model.L_0)")
            else
                println("v_0:                $(config.model.v_0)")
            end
        end
    end
    
    println("─"^50)
    println(@sprintf("%4s │ %12s │ %12s", "Rank", "Ωₚ", "γ"))
    # Eigenvalue rows
    for (i, (Ωₚ, γ)) in enumerate(top_modes)
        if i == 1
            println(@sprintf("%4d │ %12.4f │ %12.4f  ← Dominant", i, Ωₚ, γ))
        else
            println(@sprintf("%4d │ %12.4f │ %12.4f", i, Ωₚ, γ))
        end
    end
    println("─"^50)
end



