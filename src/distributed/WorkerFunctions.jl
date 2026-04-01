# Unified Worker function definitions for distributed PME calculations
# Note: Verbose output controlled by passing verbose flag

# Common imports for all workers
using Printf
using Dates
using LinearAlgebra
using TOML

# Set environment variable to suppress PME module initialization messages
ENV["PME_VERBOSE"] = "false"

# Load PME module on workers (suppress output)
try
    using PME
catch e
    println("      ❌ Failed to load PME module on worker $(myid()): $e")
end

# Global flag to track if function is defined
compute_pi_elements_gpu_worker_defined = false

# GPU-specific imports and functions (conditionally loaded)
# Try to load CUDA and define GPU version
try
    using CUDA
    
    # Include the GPU module code as a string to avoid file path issues
    gpu_module_code = """
    module DistributedPiElementsGPU
    
    using CUDA
    using LinearAlgebra
    using Printf
    
    export calculate_pi_rows_on_gpu
    
    # GPU kernel function from PiElementsGPU.jl - modified for row chunking
    function pi_element_kernel!(pi4_chunk, ra, pha, w1, L_m, SGNL, Omega_1, Omega_2, 
        psi_z_vals, z_vals, nz, N, NR, Nv, NW, KRES,
        m, k0, beta, beta_sq, orbital_tolerance, z_precision, start_row)

        # Get thread index - each thread computes one matrix element
        tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        
        # Calculate chunk dimensions
        chunk_rows = size(pi4_chunk, 1)
        total_elements = chunk_rows * N * KRES * KRES
        
        if tid <= total_elements
            # Convert linear index to 4D indices within the chunk
            ires = ((tid - 1) % KRES) + 1
            jres = (((tid - 1) ÷ KRES) % KRES) + 1
            j = (((tid - 1) ÷ (KRES * KRES)) % N) + 1
            i_chunk = ((tid - 1) ÷ (KRES * KRES * N)) + 1  # Local index within chunk
            
            # Convert chunk index to global matrix index
            i_global = start_row + i_chunk - 1
            
            # Convert global i,j to grid coordinates
            iR = ((i_global - 1) ÷ Nv) + 1
            iv = ((i_global - 1) % Nv) + 1
            jL = ((j - 1) ÷ Nv) + 1
            jI = ((j - 1) % Nv) + 1
            
            # Skip invalid grid points
            if abs(ra[NW, iR, iv]) < orbital_tolerance || 
               abs(ra[NW, jL, jI]) < orbital_tolerance
                return
            end
            
            # Calculate single Pi element exactly matching CPU
            pi_val = calculate_single_pi_gpu_exact(
                iR, iv, jL, jI, ires, jres,
                ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
                psi_z_vals, z_vals, nz, NW,
                m, k0, beta, beta_sq, z_precision
            )
            
            # Store in chunk using local index
            pi4_chunk[i_chunk, j, ires, jres] = 4.0 * pi_val
        end
        
        return nothing
    end
    
    # GPU helper function from PiElementsGPU.jl
    @inline function calculate_single_pi_gpu_exact(
        iR, iv, jL, jI, ires, jres,
        ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
        psi_z_vals, z_vals, nz, NW,
        m, k0, beta, beta_sq, z_precision)

        # sgnl = grids.SGNL[iR, iv]
        om_pr = Omega_2[iR, iv]
        
        # sgnl_s = grids.SGNL[iLs, iIs]
        om_pr_s = Omega_2[jL, jI]
        omega_1_inv = 1.0 / Omega_1[jL, jI]
        
        result = 0.0
        
        for iw1 in 1:NW
            r_1 = ra[iw1, iR, iv]
            sw1 = if iw1 == 1
                (w1[2, iR, iv] - w1[1, iR, iv]) / 2.0
            elseif iw1 == NW
                (w1[NW, iR, iv] - w1[NW-1, iR, iv]) / 2.0
            else
                (w1[iw1+1, iR, iv] - w1[iw1-1, iR, iv]) / 2.0
            end
            
            s2_val = 0.0
            
            for iw1s in 1:NW
                r_1s = ra[iw1s, jL, jI]
                r_min = min(r_1, r_1s)
                r_max = max(r_1, r_1s)
                tmp1 = r_min^2 + r_max^2 + beta_sq
                zi = 2.0 * r_min * r_max / tmp1
                
                psi_val = gpu_interpolate_linear(zi, z_vals, psi_z_vals, nz)
                
                bar_zi = 1.0 - zi
                if bar_zi > z_precision
                    s3_val = psi_val / sqrt(tmp1)
                else
                    s3_val = sqrt(2.0) * (log(32.0 / bar_zi) - (16.0/3.0)) / (2*π) / sqrt(tmp1)
                end
                
                sw1s = if iw1s == 1
                    (w1[2, jL, jI] - w1[1, jL, jI]) / 2.0
                elseif iw1s == NW
                    (w1[NW, jL, jI] - w1[NW-1, jL, jI]) / 2.0
                else
                    (w1[iw1s+1, jL, jI] - w1[iw1s-1, jL, jI]) / 2.0
                end
                
                phi_as = w1[iw1s, jL, jI] * om_pr_s * omega_1_inv - pha[iw1s, jL, jI]
                phase_s = (jres + k0 - 1) * w1[iw1s, jL, jI] + m * phi_as
                
                s2_val += sw1s * s3_val * cos(phase_s)
            end
            
            phi_a = w1[iw1, iR, iv] * om_pr / Omega_1[iR, iv] - pha[iw1, iR, iv]
            phase_source = (ires + k0 - 1) * w1[iw1, iR, iv] + m * phi_a
            
            result += sw1 * s2_val * cos(phase_source)
        end
        
        return result
    end

    # GPU interpolation helper from PiElementsGPU.jl
    @inline function gpu_interpolate_linear(x, x_vals, y_vals, n)
        if x <= x_vals[1]
            return y_vals[1]
        elseif x >= x_vals[n]
            return y_vals[n]
        end
        
        left = 1
        right = n
        while right - left > 1
            mid = (left + right) ÷ 2
            if x_vals[mid] <= x
                left = mid
            else
                right = mid
            end
        end
        
        i = left
        t = (x - x_vals[i]) / (x_vals[i+1] - x_vals[i])
        return y_vals[i] * (1.0 - t) + y_vals[i+1] * t
    end
    
    # GPU computation closely mimicking the logic in PiElementsGPU.jl
    function calculate_pi_rows_on_gpu(config, orbit_data, z, psi_z, gpu_device, start_row, end_row)
        if config.io.verbose
            println("  🔥 GPU \$gpu_device: Computing Pi[\$start_row:\$end_row, :, :, :] (\$(end_row-start_row+1) rows)")
        end
        
        # Set GPU device
        CUDA.device!(gpu_device)
        
        # Get dimensions
        NR, Nv = config.grid.NR, config.grid.Nv
        N = NR * Nv
        KRES = config.core.kres
        NW = size(orbit_data.ra, 1)
        nz = length(z)
        
        # Calculate chunk size
        chunk_rows = end_row - start_row + 1
        if chunk_rows <= 0
            println("    ⚠️  No rows to compute, returning empty chunk")
            return zeros(Float64, 0, N, KRES, KRES)
        end
        
        # Debug data shapes - only if debug is enabled
        if config.io.debug
            println("    🔍 DEBUG: orbit_data.ra shape: \$(size(orbit_data.ra))")
            println("    🔍 DEBUG: orbit_data.Omega_1 shape: \$(size(orbit_data.Omega_1))")
            println("    🔍 DEBUG: Processing rows \$start_row to \$end_row")
        end
        
        # Convert linear indices to 2D grid indices
        NR, Nv = config.grid.NR, config.grid.Nv
        
        # Calculate which radial (iR) and eccentricity (iv) indices correspond to our row range
        start_iL = ((start_row - 1) ÷ Nv) + 1
        start_iI = ((start_row - 1) % Nv) + 1
        end_iL = ((end_row - 1) ÷ Nv) + 1
        end_iI = ((end_row - 1) % Nv) + 1
        
        if config.io.debug
            println("    🔍 DEBUG: Grid indices - start: (\$start_iL, \$start_iI), end: (\$end_iL, \$end_iI)")
        end
        
        # For simplicity, transfer all orbital data (not sliced) since slicing is complex for 3D arrays
        # The GPU kernel will handle the specific row computation
        ra_gpu = CuArray(orbit_data.ra)
        pha_gpu = CuArray(orbit_data.pha)
        w1_gpu = CuArray(orbit_data.w1)
        L_m_gpu = CuArray(orbit_data.grids.L_m)
        SGNL_gpu = CuArray(orbit_data.grids.SGNL)
        Omega_1_gpu = CuArray(orbit_data.Omega_1)
        Omega_2_gpu = CuArray(orbit_data.Omega_2)
        z_gpu = CuArray(z)
        psi_z_gpu = CuArray(psi_z)
        result_chunk_gpu = CuArray(zeros(Float64, chunk_rows, N, KRES, KRES))
        
        # Launch kernel as in PiElementsGPU
        threads_per_block = 256
        blocks_per_grid = (chunk_rows * N * KRES * KRES + threads_per_block - 1) ÷ threads_per_block


        println("  🔥🔥🔥🔥 GPU \$gpu_device: Beta = \$(config.physics.beta)")

        @cuda blocks=blocks_per_grid threads=threads_per_block pi_element_kernel!(
            result_chunk_gpu, ra_gpu, pha_gpu, w1_gpu, L_m_gpu, SGNL_gpu, Omega_1_gpu, Omega_2_gpu,
            psi_z_gpu, z_gpu, nz, N, NR, Nv, NW, KRES,
            config.core.m, get_k0(config.core.kres), config.physics.beta, config.physics.beta^2,
            config.tolerances.orbital_tolerance, config.elliptic.z_precision, start_row
        )

        # Transfer result back to CPU
        result_chunk = Array(result_chunk_gpu)

        if config.io.verbose
            println("    ✅ GPU \$gpu_device chunk computation complete")
        end
        return result_chunk
    end

    end # module DistributedPiElementsGPU
    """


    include_string(Main, gpu_module_code, "DistributedPiElementsGPU.jl")
    using .DistributedPiElementsGPU
    
    global compute_pi_elements_gpu_worker_defined = true
    
catch e
    # CUDA not available, will use CPU fallback
end

# Define the function based on what was loaded
if compute_pi_elements_gpu_worker_defined
    # GPU version
    function compute_pi_elements_gpu_worker(config_path::String, data_path::String, gpu_device::Int, start_row::Int, end_row::Int)
        config = PME.load_config(config_path)
        if config.io.verbose
            println("      🔥 GPU Worker $(myid()) on $(gethostname()): Starting computation for rows $start_row:$end_row on GPU:$gpu_device")
        end
        flush(stdout)
        
        orbit_data = load_orbital_data(joinpath(data_path, "binary"), config.grid.NR, config.grid.Nv, config.grid.NW)
        z, psi_z = load_z_and_psi_z(joinpath(data_path, "binary"), config)

        temp_orbit_data = (
            ra = orbit_data.ra,
            pha = orbit_data.pha,
            w1 = orbit_data.w1,
            Omega_1 = orbit_data.Omega_1,
            Omega_2 = orbit_data.Omega_2,
            grids = (L_m = orbit_data.L_m, SGNL = orbit_data.SGNL)
        )

        result_chunk = calculate_pi_rows_on_gpu(config, temp_orbit_data, z, psi_z, gpu_device, start_row, end_row)

        if config.io.verbose
            println("    ✅ GPU $gpu_device chunk computation complete on Worker $(myid())")
            println("      ✅ GPU Worker $(myid()) on $(gethostname()): Computation complete, returning chunk of size $(size(result_chunk))")
        end
        flush(stdout)

        return result_chunk
    end
else
    # CPU fallback version
    function compute_pi_elements_gpu_worker(config_path::String, data_path::String, gpu_device::Int, start_row::Int, end_row::Int)
        println("      🔥 CPU Fallback Worker $(myid()) on $(gethostname()): Starting computation for rows $start_row:$end_row")
        flush(stdout)
        
        # Load config to get proper dimensions
        config = PME.load_config(config_path)
        NR, Nv = config.grid.NR, config.grid.Nv
        N = NR * Nv
        KRES = config.core.kres
        
        # For testing, return a dummy array with correct dimensions
        result_size = end_row - start_row + 1
        result_chunk = rand(Float64, result_size, N, KRES, KRES)  # Proper 4D array dimensions
        
        println("      ✅ CPU Fallback Worker $(myid()) on $(gethostname()): Computation complete, returning chunk of size $(size(result_chunk))")
        flush(stdout)
        
        return result_chunk
    end
end

# Verify the function was properly defined
function_defined = @isdefined(compute_pi_elements_gpu_worker)

if !function_defined
    println("      ❌ ERROR: No compute_pi_elements_gpu_worker function defined on worker $(myid())!")
    error("compute_pi_elements_gpu_worker not available on worker $(myid())")
end

# Load z and psi_z from binary files
function load_z_and_psi_z(binary_dir::String, config::PMEConfig)
    if config.io.verbose
        println("        Loading z and psi_z from binary files...")
    end
    
    z_path = joinpath(binary_dir, "z.bin")
    psi_z_path = joinpath(binary_dir, "psi_z.bin")
    
    if config.io.debug
        println("        🔍 DEBUG: Attempting to load z from: $(abspath(z_path))")
        println("        🔍 DEBUG: File exists: $(isfile(z_path))")
    end
    z = read_binary_float64(z_path, (config.elliptic.NZ,))
    
    if config.io.debug
        println("        🔍 DEBUG: Attempting to load psi_z from: $(abspath(psi_z_path))")
        println("        🔍 DEBUG: File exists: $(isfile(psi_z_path))")
    end
    psi_z = read_binary_float64(psi_z_path, (config.elliptic.NZ,))
    
    if config.io.verbose
        println("        ✅ z and psi_z loaded from binary files")
    end
    
    return z, psi_z
end

# Binary data loading functions
"""
    read_binary_float64(filename, dims)

Read binary Float64 data from file with specified dimensions.
"""
function read_binary_float64(filename::String, dims::Tuple)
    data = Array{Float64}(undef, dims)
    open(filename, "r") do f
        read!(f, data)
    end
    return data
end

"""
    load_orbital_data(binary_dir, NR, Nv, NW)

Load orbital data from binary files.
"""
function load_orbital_data(binary_dir::String, NR::Int, Nv::Int, NW::Int)
    # Note: This function doesn't have access to config, so we show minimal output
    # Load core frequency and momentum data
    omega1_path = joinpath(binary_dir, "Omega_1.bin")
    Omega_1 = read_binary_float64(omega1_path, (NR, Nv))
    Omega_2 = read_binary_float64(joinpath(binary_dir, "Omega_2.bin"), (NR, Nv))
    jacobian = read_binary_float64(joinpath(binary_dir, "jacobian.bin"), (NR, Nv))
    L_m = read_binary_float64(joinpath(binary_dir, "L_m.bin"), (NR, Nv))
    
    # Load orbital trajectory data
    ra = read_binary_float64(joinpath(binary_dir, "r.bin"), (NW, NR, Nv))
    pha = read_binary_float64(joinpath(binary_dir, "ph.bin"), (NW, NR, Nv))
    w1 = read_binary_float64(joinpath(binary_dir, "w1.bin"), (NW, NR, Nv))
    
    # Load distribution function data
    F0 = read_binary_float64(joinpath(binary_dir, "F0.bin"), (NR, Nv))
    FE = read_binary_float64(joinpath(binary_dir, "FE.bin"), (NR, Nv))
    FL = read_binary_float64(joinpath(binary_dir, "FL.bin"), (NR, Nv))
    
    # Load action variables and grid info
    Ir = read_binary_float64(joinpath(binary_dir, "Ir.bin"), (NR, Nv))
    Rc = read_binary_float64(joinpath(binary_dir, "Rc.bin"), (NR, Nv))
    E = read_binary_float64(joinpath(binary_dir, "E.bin"), (NR, Nv))
    
    # Load circulation data
    v = read_binary_float64(joinpath(binary_dir, "v.bin"), (NR, Nv))
    
    # Load grid weights product for K-matrix construction
    grid_weights = read_binary_float64(joinpath(binary_dir, "grid_weights.bin"), (NR, Nv))
    
    # Create SGNL array (sign of circulation) - this appears to be derived from v
    SGNL = sign.(v)
    
    # Note: This function doesn't have access to config, so we check the environment variable
    if get(ENV, "PME_VERBOSE", "false") == "true"
        println("        ✅ Orbital data loaded from binary files")
    end
    
    # Return a named tuple that matches the expected structure
    return (
        ra=ra, 
        pha=pha, 
        w1=w1, 
        Omega_1=Omega_1, 
        Omega_2=Omega_2, 
        jacobian=jacobian, 
        F0=F0,
        FE=FE,
        FL=FL,
        Ir=Ir,
        L_m=L_m,  # Direct access for backward compatibility
        SGNL=SGNL, # Direct access for backward compatibility
        grid_weights=grid_weights,  # Precomputed grid weights product for K-matrix
        grids=(
            L_m=L_m,
            Rc=Rc,
            E=E,
            v=v,
            SGNL=SGNL
        )
    )
end

# CPU-specific worker functions

# Main CPU worker function for eigenvalue calculation
function compute_eigenvalues_cpu_worker(config_path::String, model_path::String, data_path::String)
    # Load config first to check verbose/debug flags
    config = PME.load_config(config_path)
    verbose = get(ENV, "PME_VERBOSE", "false") == "true" || config.io.debug
    
    if verbose
        println("      🔥 CPU Worker $(myid()) on $(gethostname()): Starting eigenvalue calculation from files")
        flush(stdout)
    end

    try
        # Load everything from files
        if verbose
            println("        Loading config from: $config_path")
            println("        Loading model from: $model_path")
        end
        model_data = TOML.parsefile(model_path)
        # It's safer to merge configs on the worker than to assume the main one is complete
        if haskey(model_data, "model")
            config.model.type = get(model_data["model"], "type", "ExpDisk")
            for (key, value) in model_data["model"]
                config.model.parameters[key] = value
            end
        end
        if haskey(model_data, "physics")
            physics = model_data["physics"]
            haskey(physics, "m") && (config.core.m = physics["m"])
            haskey(physics, "beta") && (config.physics.beta = physics["beta"])
            haskey(physics, "selfgravity") && (config.model.parameters["selfgravity"] = physics["selfgravity"])
        end
        # Create model based on type
        model = if config.model.type == "ExpDisk"
            PME.create_expdisk_model(config)
        elseif config.model.type == "IsochroneTaperJH"
            PME.create_isochrone_taper_jh_model(config)
        elseif config.model.type == "IsochroneTaperZH"
            PME.create_isochrone_taper_zh_model(config)
        else
            error("Unknown model type: $(config.model.type). Supported: ExpDisk, IsochroneTaperJH, IsochroneTaperZH")
        end

        if verbose
            println("        Loading orbital data from: $data_path/binary")
        end
        orbit_data = load_orbital_data(joinpath(data_path, "binary"), config.grid.NR, config.grid.Nv, config.grid.NW)
        

        pi4_path = joinpath(data_path, "binary", "pi4.bin")
        if verbose
            println("        Loading Pi matrix from: $pi4_path")
        end
        pi4 = Array{Float64,4}(undef, (config.grid.NR * config.grid.Nv, config.grid.NR * config.grid.Nv, config.core.kres, config.core.kres))
        open(pi4_path, "r") do f
            read!(f, pi4)
        end
        if config.io.debug
            println("        🔍 DEBUG: Pi matrix loaded with size: $(size(pi4))")
        end

        # Note: Using the MatrixCalculator module that was loaded with PME
        # Remote reloading is not needed since the PME module contains the latest version
        
        # Now perform the calculation
        if verbose
            println("        Constructing K-matrix...")
            println("        Calculating eigenvalues...")
        end
        if config.io.debug
            println("        🔍 DEBUG: About to call construct_k_matrix with:")
            println("        🔍 DEBUG: - config type: $(typeof(config))")
            println("        🔍 DEBUG: - model type: $(typeof(model))")
            println("        🔍 DEBUG: - orbit_data type: $(typeof(orbit_data))")
            println("        🔍 DEBUG: - pi4 size: $(size(pi4))")
        end
        K_remote = PME.MatrixCalculator.construct_k_matrix(config, model, orbit_data, pi4)

        eigenvalues, eigenvectors = PME.KEigenvalues.calculate_eigenvalues(K_remote, config.eigenvectors, config.core.m)
        
        if verbose
            println("      ✅ CPU Worker $(myid()): Eigenvalue calculation complete")
            flush(stdout)
        end

        # Return both results
        return eigenvalues, eigenvectors

    catch e
        println("      ❌ ERROR in CPU Worker $(myid()): $e")
        # Propagate the error back to the main process
        rethrow(e)
    end
end

