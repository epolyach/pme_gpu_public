"""
Chunked GPU Pi Elements calculation for PME.
Computes Pi elements in memory-efficient chunks and accumulates directly to K matrix.
This allows processing grids larger than GPU memory can hold at once.
"""

module PiElementsChunkedCUDA

using CUDA
using LinearAlgebra
using Printf

# Import get_k0
const get_k0 = (kres::Int) -> -(kres - 1) ÷ 2

# Import the kernel function from PiElementsMultiGPUCUDA
using ..PiElementsMultiGPUCUDA: pi_element_chunk_kernel!, calculate_single_pi_element_gpu

export calculate_k_matrix_chunked_cuda, estimate_chunk_size_cuda


"""
Estimate the optimal chunk size based on available GPU memory.
Returns the maximum number of rows that can be processed per chunk.

Parameters:
- N: Total number of grid points (NR * Nv)
- KRES: Number of resonance points
- T: Element type (Float32 or Float64)
- gpu_devices: List of GPU device IDs
- target_utilization: Fraction of GPU memory to use (default 0.80)
"""
function estimate_chunk_size_cuda(config, orbit_data, gpu_devices::Vector{Int}; target_utilization::Float64=0.80)
    NR, Nv, KRES = config.grid.NR, config.grid.Nv, config.core.kres
    N = NR * Nv
    NW = size(orbit_data.ra, 1)
    nz = config.elliptic.NZ
    
    T = config.gpu.precision_double ? Float64 : Float32
    sizeof_T = sizeof(T)
    
    # Find minimum available memory across all GPUs
    min_memory = typemax(Int)
    for device_id in gpu_devices
        CUDA.device!(device_id)
        mem = CUDA.available_memory()
        min_memory = min(min_memory, mem)
    end
    
    available_bytes = Int(floor(min_memory * target_utilization))
    
    # Calculate fixed array memory requirements
    # ra, pha: NW × N each
    # w1, L_m, SGNL, Omega_1, Omega_2: N each  
    # psi_z, z: nz each
    fixed_bytes = (2 * NW * N + 5 * N + 2 * nz) * sizeof_T
    
    # Memory available for Pi4 chunk
    chunk_bytes_available = available_bytes - fixed_bytes
    
    if chunk_bytes_available <= 0
        error("Insufficient GPU memory even for fixed arrays. Need at least $(fixed_bytes / 1e9) GB")
    end
    
    # Pi4 chunk: chunk_rows × N × KRES × KRES
    bytes_per_row = N * KRES * KRES * sizeof_T
    max_chunk_rows = floor(Int, chunk_bytes_available / bytes_per_row)
    
    # Ensure at least 1 row
    max_chunk_rows = max(max_chunk_rows, 1)
    
    return max_chunk_rows, fixed_bytes, chunk_bytes_available
end


"""
Calculate K matrix using chunked GPU computation.
Computes Pi elements in chunks and accumulates directly to K matrix,
allowing processing of grids larger than GPU memory.

This function:
1. Pre-computes fs and mu1 arrays needed for K matrix
2. Processes Pi elements in chunks that fit in GPU memory
3. Accumulates each chunk's contribution to K matrix immediately
4. Frees chunk memory before computing next chunk

Returns the fully constructed K matrix.
"""
function calculate_k_matrix_chunked_cuda(config, orbit_data, model, psi_z, z, gpu_devices::Vector{Int})
    NR, Nv, KRES = config.grid.NR, config.grid.Nv, config.core.kres
    N = NR * Nv
    k0 = get_k0(config.core.kres)
    m = config.core.m
    selfgravity = config.physics.selfgravity
    sharp_df = config.phase_space.sharp_df
    
    # Pre-compute weight_J for sharp_df (if enabled)
    weight_J = nothing
    if sharp_df
        weight_J = zeros(Float64, NR)
        for ir in 1:NR
            if ir == 1
                weight_J[ir] = (orbit_data.Ir[2, 1] - orbit_data.Ir[1, 1]) / 2
            elseif ir == NR
                weight_J[ir] = (orbit_data.Ir[NR, 1] - orbit_data.Ir[NR-1, 1]) / 2
            else
                weight_J[ir] = (orbit_data.Ir[ir+1, 1] - orbit_data.Ir[ir-1, 1]) / 2
            end
        end
        println("  Sharp DF mode enabled")
    end
    
    # Estimate optimal chunk size
    max_chunk_rows, fixed_bytes, chunk_bytes = estimate_chunk_size_cuda(config, orbit_data, gpu_devices)
    num_chunks = ceil(Int, N / max_chunk_rows)
    
    T = config.gpu.precision_double ? Float64 : Float32
    println("  Chunked GPU mode: $(num_chunks) chunks of up to $(max_chunk_rows) rows each")
    println("  Fixed arrays: $(round(fixed_bytes / 1e9, digits=2)) GB, chunk: $(round(chunk_bytes / 1e9, digits=2)) GB")
    println("  Precision: $T")
    
    # Pre-compute fs coefficients (same as MatrixCalculator.jl)
    fs = zeros(Float64, KRES, N)
    for ires = 1:KRES
        l = ires + k0 - 1
        for iR in 1:NR
            for iv in 1:Nv
                i_ri = (iR - 1) * Nv + iv
                fs[ires, i_ri] = (l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]) * orbit_data.FE[iR, iv] +
                                 m * orbit_data.FL[iR, iv]
            end
        end
    end
    
    # Pre-compute mu1 (same as MatrixCalculator.jl)
    jac = orbit_data.jacobian
    if hasfield(typeof(orbit_data), :grid_weights)
        grid_weights = orbit_data.grid_weights
    else
        radial_weights = orbit_data.grids.radial.weights
        eccentricity_weights = orbit_data.grids.eccentricity.weights
        grid_weights = zeros(Float64, NR, Nv)
        for iR in 1:NR
            for iv in 1:Nv
                grid_weights[iR, iv] = radial_weights[iR] * eccentricity_weights[iR, iv]
            end
        end
    end
    SS = grid_weights .* jac
    mu1 = reshape(SS', 1, N)
    
    # Initialize K matrix
    K_size = KRES * N
    println("K matrix size = $(K_size) x $(K_size)")
    K = zeros(Float64, K_size, K_size)
    
    # Process chunks
    num_gpus = length(gpu_devices)
    
    for chunk_idx in 1:num_chunks
        chunk_start = (chunk_idx - 1) * max_chunk_rows + 1
        chunk_end = min(chunk_idx * max_chunk_rows, N)
        chunk_rows = chunk_end - chunk_start + 1
        
        print("  Chunk $(chunk_idx)/$(num_chunks) (rows $(chunk_start)-$(chunk_end))... ")
        
        # Distribute this chunk across GPUs
        rows_per_gpu = div(chunk_rows, num_gpus)
        remainder = chunk_rows % num_gpus
        
        gpu_row_ranges = Vector{Tuple{Int,Int}}()
        local_start = chunk_start
        
        for i in 1:num_gpus
            local_end = local_start + rows_per_gpu - 1
            if i <= remainder
                local_end += 1
            end
            # Skip if no rows for this GPU
            if local_start <= local_end
                push!(gpu_row_ranges, (local_start, local_end))
            end
            local_start = local_end + 1
        end
        
        # Launch GPU tasks for this chunk
        tasks = Task[]
        for (i, (start_row, end_row)) in enumerate(gpu_row_ranges)
            device_id = gpu_devices[min(i, num_gpus)]
            task = Threads.@spawn calculate_pi_rows_for_chunk(
                config, orbit_data, z, psi_z, device_id, start_row, end_row
            )
            push!(tasks, task)
        end
        
        # Collect results and accumulate to K matrix
        for (i, task) in enumerate(tasks)
            pi4_chunk = fetch(task)  # Shape: (chunk_rows_this_gpu, N, KRES, KRES)
            start_row, end_row = gpu_row_ranges[i]
            
            # Accumulate this chunk's contribution to K
            accumulate_to_k_matrix!(K, pi4_chunk, start_row, fs, mu1, selfgravity, N, KRES)
            
            # Accumulate sharp_df terms (if enabled)
            if sharp_df
                accumulate_sharp_df_to_k_matrix!(K, pi4_chunk, start_row, orbit_data.F0, weight_J, selfgravity, m, N, KRES, NR, Nv)
            end
            
            # Free chunk memory
            pi4_chunk = nothing
        end
        
        # Force garbage collection to free GPU memory
        GC.gc(false)
        
        println("done")
    end
    
    # Add diagonal frequency terms (NOT scaled by selfgravity)
    for ires in 1:KRES
        l = ires + k0 - 1
        for iR in 1:NR
            for iv in 1:Nv
                i_ri = (iR - 1) * Nv + iv
                is = (ires - 1) * N + i_ri
                K[is, is] += l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]
            end
        end
    end
    
    # Sharp DF extensions (if enabled)
    if sharp_df && weight_J !== nothing
        println("✓ Sharp DF terms accumulated during chunk processing")
    end
    
    println("✓ K matrix constructed via chunked GPU computation")
    
    return K
end


"""
Accumulate a Pi4 chunk's contribution to the K matrix.
pi4_chunk has shape (chunk_rows, N, KRES, KRES) and corresponds to
rows chunk_start:(chunk_start + chunk_rows - 1) of the full pi4 array.
"""
function accumulate_to_k_matrix!(K, pi4_chunk, chunk_start::Int, fs, mu1, selfgravity, N, KRES)
    chunk_rows = size(pi4_chunk, 1)
    
    # K[(ires-1)*N + j, (jres-1)*N + js] = selfgravity * pi4[j, js, ires, jres] * fs[jres, js] * mu1[js]
    # Here, j = chunk_start + (chunk_row - 1), js = 1:N
    
    @inbounds for jres in 1:KRES
        for ires in 1:KRES
            for js in 1:N
                coeff = selfgravity * fs[jres, js] * mu1[js]
                col_idx = (jres - 1) * N + js
                
                for chunk_row in 1:chunk_rows
                    j = chunk_start + chunk_row - 1
                    row_idx = (ires - 1) * N + j
                    K[row_idx, col_idx] += pi4_chunk[chunk_row, js, ires, jres] * coeff
                end
            end
        end
    end
end


"""
Calculate Pi elements for a range of rows on a single GPU.
This is similar to the function in PiElementsMultiGPUCUDA but can be
called independently for chunking.
"""
function calculate_pi_rows_for_chunk(config, orbit_data, z, psi_z, device_id::Int, start_row::Int, end_row::Int)
    # Set GPU device
    CUDA.device!(device_id)
    
    NR, Nv, KRES = config.grid.NR, config.grid.Nv, config.core.kres
    N = NR * Nv
    NW = size(orbit_data.ra, 1)
    nz = length(z)
    
    chunk_rows = end_row - start_row + 1
    
    if chunk_rows <= 0
        return zeros(Float64, 0, N, KRES, KRES)
    end
    
    # Select precision type
    T = config.gpu.precision_double ? Float64 : Float32
    
    # Initialize result chunk
    pi4_chunk = zeros(T, chunk_rows, N, KRES, KRES)
    
    # Transfer data to GPU
    ra_gpu = CuArray(T.(orbit_data.ra))
    pha_gpu = CuArray(T.(orbit_data.pha))
    w1_gpu = CuArray(T.(orbit_data.w1))
    L_m_gpu = CuArray(T.(orbit_data.grids.L_m))
    SGNL_gpu = CuArray(T.(orbit_data.grids.SGNL))
    Omega_1_gpu = CuArray(T.(orbit_data.Omega_1))
    Omega_2_gpu = CuArray(T.(orbit_data.Omega_2))
    psi_z_gpu = CuArray(T.(psi_z))
    z_gpu = CuArray(T.(z))
    pi4_chunk_gpu = CuArray(pi4_chunk)
    
    # Get parameters
    m = config.core.m
    k0 = get_k0(config.core.kres)
    beta = config.physics.beta
    beta_sq = beta * beta
    orbital_tolerance = config.tolerances.orbital_tolerance
    z_precision = config.elliptic.z_precision
    
    # Launch kernel
    chunk_elements = chunk_rows * N * KRES * KRES
    threads_per_block = 256
    blocks_per_grid = cld(chunk_elements, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks_per_grid pi_element_chunk_kernel!(
        pi4_chunk_gpu, ra_gpu, pha_gpu, w1_gpu, L_m_gpu, SGNL_gpu, Omega_1_gpu, Omega_2_gpu,
        psi_z_gpu, z_gpu, chunk_rows, N, NR, Nv, NW, KRES, nz, start_row,
        m, k0, T(beta), T(beta_sq), T(orbital_tolerance), T(z_precision)
    )
    
    # Wait for completion and copy back
    CUDA.synchronize()
    result_chunk_T = Array(pi4_chunk_gpu) .* T(4)
    
    # Convert to Float64 for pipeline compatibility
    result_chunk = config.gpu.precision_double ? result_chunk_T : Float64.(result_chunk_T)
    
    # Free GPU memory
    CUDA.unsafe_free!(pi4_chunk_gpu)
    CUDA.unsafe_free!(ra_gpu)
    CUDA.unsafe_free!(pha_gpu)
    CUDA.unsafe_free!(w1_gpu)
    CUDA.unsafe_free!(L_m_gpu)
    CUDA.unsafe_free!(SGNL_gpu)
    CUDA.unsafe_free!(Omega_1_gpu)
    CUDA.unsafe_free!(Omega_2_gpu)
    CUDA.unsafe_free!(psi_z_gpu)
    CUDA.unsafe_free!(z_gpu)
    
    return result_chunk
end


"""
Accumulate sharp_df terms from a Pi4 chunk to the K matrix.
pi4_chunk has shape (chunk_rows, N, KRES, KRES) and corresponds to
rows chunk_start:(chunk_start + chunk_rows - 1) of the full pi4 array.

Sharp DF formula (from MatrixCalculator.jl):
K[i_row, i_col] += selfgravity * m * pi4[j, js, ires, jres] * F0[ir_prime, 1] * weight_J[ir_prime]
where js = (ir_prime-1)*Nv + 1 (only first velocity point for each radial point)
"""
function accumulate_sharp_df_to_k_matrix!(K, pi4_chunk, chunk_start::Int, F0, weight_J, selfgravity, m, N, KRES, NR, Nv)
    chunk_rows = size(pi4_chunk, 1)
    
    # For sharp_df, we iterate over all rows j in this chunk, but only specific columns js
    # js = (ir_prime-1)*Nv + 1 corresponds to iv=1 (first velocity point)
    
    @inbounds for jres in 1:KRES
        for ires in 1:KRES
            for ir_prime in 1:NR
                js = (ir_prime - 1) * Nv + 1  # Column index for iv=1
                coeff = selfgravity * m * F0[ir_prime, 1] * weight_J[ir_prime]
                col_idx = (jres - 1) * N + js
                
                for chunk_row in 1:chunk_rows
                    j = chunk_start + chunk_row - 1
                    row_idx = (ires - 1) * N + j
                    K[row_idx, col_idx] += pi4_chunk[chunk_row, js, ires, jres] * coeff
                end
            end
        end
    end
end

end # module PiElementsChunkedCUDA
