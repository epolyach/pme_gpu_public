"""
Multi-GPU Pi Elements calculation for PME with multi-backend support.
Splits computation across multiple GPUs by distributing rows.
Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs via automatic detection.
Includes chunked computation mode for large matrices that exceed GPU memory.
"""

module PiElementsMultiGPU

using LinearAlgebra
using Printf

# Include GPUBackend for detection
include("../utils/GPUBackend.jl")
using .GPUBackend

# Detect GPU type at module load time
const _MODULE_GPU_TYPE = GPUBackend.detect_gpu_type()

# Conditionally load the appropriate GPU package and backend module
if _MODULE_GPU_TYPE == GPUBackend.NVIDIA
    using CUDA
    include("PiElementsMultiGPUCUDA.jl")
    using .PiElementsMultiGPUCUDA
    include("PiElementsChunkedCUDA.jl")
    using .PiElementsChunkedCUDA
    include("PiElementsBlockCUDA.jl")
    using .PiElementsBlockCUDA
elseif _MODULE_GPU_TYPE == GPUBackend.AMD
    using AMDGPU
    include("PiElementsMultiGPUAMD.jl")
    using .PiElementsMultiGPUAMD
    include("PiElementsBlockAMD.jl")
    using .PiElementsBlockAMD
end

export calculate_pi_elements_multi_gpu, calculate_k_matrix_chunked_gpu, estimate_memory_required, compute_pi_block_gpu



"""
Main multi-GPU Pi elements calculation function with automatic backend detection.
"""
function calculate_pi_elements_multi_gpu(config, orbit_data, z::Vector{Float64}, psi_z::Vector{Float64}, gpu_devices::Vector{Int})
    
    gpu_type = _MODULE_GPU_TYPE
    
    if gpu_type == GPUBackend.NVIDIA
        return PiElementsMultiGPUCUDA.calculate_pi_elements_multi_cuda(config, orbit_data, z, psi_z, gpu_devices)
    elseif gpu_type == GPUBackend.AMD
        return PiElementsMultiGPUAMD.calculate_pi_elements_multi_amd(config, orbit_data, z, psi_z, gpu_devices)
    else
        error("❌ No GPU detected. Cannot perform multi-GPU Pi elements calculation.")
    end
end


"""
Estimate the GPU memory required for the standard (non-chunked) Pi elements calculation.
Returns (required_bytes, available_bytes, fits_in_memory).
"""
function estimate_memory_required(config, orbit_data, gpu_devices::Vector{Int})
    NR, Nv, KRES = config.grid.NR, config.grid.Nv, config.core.kres
    N = NR * Nv
    NW = size(orbit_data.ra, 1)
    nz = config.elliptic.NZ
    
    T = config.gpu.precision_double ? Float64 : Float32
    sizeof_T = sizeof(T)
    
    gpu_type = _MODULE_GPU_TYPE
    
    # Find minimum available memory across all GPUs
    min_memory = typemax(Int)
    if gpu_type == GPUBackend.NVIDIA
        for device_id in gpu_devices
            CUDA.device!(device_id)
            mem = CUDA.available_memory()
            min_memory = min(min_memory, mem)
        end
    elseif gpu_type == GPUBackend.AMD
        for device_id in gpu_devices
            device = AMDGPU.devices()[device_id + 1]
            AMDGPU.device!(device)
            # AMDGPU memory query - approximate
            min_memory = min(min_memory, 8 * 1024^3)  # Assume 8GB for now
        end
    else
        min_memory = 0
    end
    
    # Calculate memory required for standard computation
    # Split rows evenly across GPUs
    num_gpus = length(gpu_devices)
    rows_per_gpu = div(N, num_gpus) + (N % num_gpus > 0 ? 1 : 0)
    
    # Fixed arrays: ra, pha (NW × N), w1, L_m, SGNL, Omega_1, Omega_2 (N), psi_z, z (nz)
    fixed_bytes = (2 * NW * N + 5 * N + 2 * nz) * sizeof_T
    
    # Pi4 chunk: rows_per_gpu × N × KRES × KRES
    pi4_bytes = rows_per_gpu * N * KRES * KRES * sizeof_T
    
    required_bytes = fixed_bytes + pi4_bytes
    
    fits = required_bytes <= min_memory * 0.90  # 90% threshold
    
    return required_bytes, min_memory, fits
end


"""
Calculate K matrix using chunked GPU computation.
This function computes Pi elements in memory-efficient chunks and
accumulates directly to K matrix, allowing processing of grids
larger than GPU memory can hold at once.

This is the recommended function for large grids or when memory is limited.
"""
function calculate_k_matrix_chunked_gpu(config, orbit_data, model, psi_z, z, gpu_devices::Vector{Int})
    gpu_type = _MODULE_GPU_TYPE
    if gpu_type == GPUBackend.NVIDIA
        return _calculate_k_matrix_chunked_generic(config, orbit_data, model, psi_z, z, gpu_devices, :cuda)
    elseif gpu_type == GPUBackend.AMD
        return _calculate_k_matrix_chunked_generic(config, orbit_data, model, psi_z, z, gpu_devices, :amd)
    else
        error("❌ No GPU detected. Cannot perform chunked GPU computation.")
    end
end



"""
Compute Pi elements for a single (l_i, l_j) resonance pair on GPU.
Returns an N×N matrix of Pi elements.
"""
function compute_pi_block_gpu(config, orbit_data, z::Vector{Float64}, psi_z::Vector{Float64}, 
                              l_i::Int, l_j::Int; device_id::Int=0)
    gpu_type = _MODULE_GPU_TYPE
    
    if gpu_type == GPUBackend.NVIDIA
        return PiElementsBlockCUDA.compute_pi_block_cuda(config, orbit_data, z, psi_z, l_i, l_j; device_id=device_id)
    elseif gpu_type == GPUBackend.AMD
        return PiElementsBlockAMD.compute_pi_block_amd(config, orbit_data, z, psi_z, l_i, l_j; device_id=device_id)
    else
        error("❌ No GPU detected.")
    end
end



# --- Unified chunked K builder (CUDA & AMD) ---

function _estimate_chunk_rows_generic(config, orbit_data, gpu_devices; T=Float64)
    NR, Nv = config.grid.NR, config.grid.Nv
    N  = NR * Nv
    KRES = config.core.kres
    min_free = typemax(Int)
    for dev in gpu_devices
        GPUBackend.gpu_device!(dev)
        min_free = min(min_free, GPUBackend.gpu_available_memory())
    end
    bytes_per_row = N * KRES * KRES * sizeof(T)
    max_rows = max(1, div(min_free, bytes_per_row))
    return max_rows
end

function _build_fs_mu1(config, orbit_data)
    NR, Nv = config.grid.NR, config.grid.Nv
    N  = NR * Nv
    KRES = config.core.kres
    k0 = -(KRES - 1) ÷ 2
    m  = config.core.m
    fs = zeros(Float64, KRES, N)
    for ires = 1:KRES
        l = ires + k0 - 1
        for iR in 1:NR, iv in 1:Nv
            j = (iR-1)*Nv + iv
            fs[ires, j] = (l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]) * orbit_data.FE[iR, iv] + m * orbit_data.FL[iR, iv]
        end
    end
    jac = orbit_data.jacobian
    if hasfield(typeof(orbit_data), :grid_weights)
        gw = orbit_data.grid_weights
    else
        rw = orbit_data.grids.radial.weights
        ew = orbit_data.grids.eccentricity.weights
        gw = [rw[iR] * ew[iR, iv] for iR in 1:NR, iv in 1:Nv]
    end
    mu1 = reshape((gw .* jac)', 1, N)
    return fs, mu1
end

function _accumulate_to_k_matrix!(K, pi4_chunk, chunk_start::Int, fs, mu1, selfgravity, N, KRES)
    chunk_rows = size(pi4_chunk, 1)
    @inbounds for ires in 1:KRES, jres in 1:KRES, chunk_row in 1:chunk_rows
        j = chunk_start + chunk_row - 1
        for js in 1:N
            K[(ires-1)*N + j, (jres-1)*N + js] += pi4_chunk[chunk_row, js, ires, jres] * selfgravity * fs[jres, js] * mu1[js]
        end
    end
end

# Helper function for sharp_df accumulation
function _accumulate_sharp_df_to_k!(K, pi4_chunk, chunk_start::Int, selfgravity, m, F0, weight_J, N, NR, Nv, KRES)
    chunk_rows = size(pi4_chunk, 1)
    @inbounds for ires in 1:KRES
        for chunk_row in 1:chunk_rows
            j = chunk_start + chunk_row - 1
            i_row = (ires - 1) * N + j
            
            for jres in 1:KRES
                for ir_prime in 1:NR
                    js = (ir_prime - 1) * Nv + 1  # First velocity point (L=0)
                    i_col = (jres - 1) * N + js
                    K[i_row, i_col] += selfgravity * m * pi4_chunk[chunk_row, js, ires, jres] * F0[ir_prime, 1] * weight_J[ir_prime]
                end
            end
        end
    end
end

function _calculate_k_matrix_chunked_generic(config, orbit_data, model, psi_z, z, gpu_devices::Vector{Int}, backend::Symbol)
    NR, Nv = config.grid.NR, config.grid.Nv
    N  = NR * Nv
    KRES = config.core.kres
    k0 = -(KRES - 1) ÷ 2
    m  = config.core.m
    selfgravity = config.physics.selfgravity
    T = config.gpu.precision_double ? Float64 : Float32

    max_chunk_rows = _estimate_chunk_rows_generic(config, orbit_data, gpu_devices; T=T)
    # Ensure at least one chunk per GPU for proper multi-GPU utilization
    num_gpus = length(gpu_devices)
    max_chunk_rows = min(max_chunk_rows, cld(N, num_gpus))
    num_chunks = cld(N, max_chunk_rows)
    @printf("  Chunked GPU: T=%s, devices=%s, rows/chunk=%d, chunks=%d\n", T, gpu_devices, max_chunk_rows, num_chunks)

    fs, mu1 = _build_fs_mu1(config, orbit_data)
    K = zeros(Float64, KRES * N, KRES * N)

    # Sharp DF setup - do validation and compute weights BEFORE main loop
    sharp_df = config.phase_space.sharp_df
    weight_J = nothing
    if sharp_df
        if config.phase_space.full_space
            error("sharp_df=true requires full_space=false (unidirectional disk)")
        end
        max_v0_deviation = maximum(abs.(orbit_data.grids.eccentricity.circulation_grid[:, 1]))
        if max_v0_deviation > 1e-15
            error("sharp_df=true requires v[iR,1]=0 for all radial points. Max deviation: $max_v0_deviation")
        end
        println("✓ Sharp DF validation passed: full_space=false, v[iR,1]=0")

        # Compute integration weights for dJ' integrals
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
    end

    # Launch all GPU tasks in parallel
    tasks = Vector{Task}(undef, num_chunks)
    chunk_ranges = Vector{Tuple{Int,Int,Int}}(undef, num_chunks)  # (dev, chunk_start, chunk_end)

    ng = length(gpu_devices)
    for chunk_idx in 1:num_chunks
        chunk_start = (chunk_idx - 1) * max_chunk_rows + 1
        chunk_end   = min(chunk_idx * max_chunk_rows, N)
        dev = gpu_devices[(chunk_idx - 1) % ng + 1]
        chunk_ranges[chunk_idx] = (dev, chunk_start, chunk_end)
        @printf("  Launching Chunk %d/%d on GPU %d (rows %d-%d)\n", chunk_idx, num_chunks, dev, chunk_start, chunk_end)

        tasks[chunk_idx] = Threads.@spawn backend == :cuda ?
            PiElementsMultiGPUCUDA.calculate_pi_rows_cuda(config, orbit_data, z, psi_z, dev, chunk_start, chunk_end) :
            PiElementsMultiGPUAMD.calculate_pi_rows_amd(config, orbit_data, z, psi_z, dev, chunk_start, chunk_end)
    end

    # Wait for all tasks and accumulate results
    for chunk_idx in 1:num_chunks
        dev, chunk_start, chunk_end = chunk_ranges[chunk_idx]
        @printf("  Collecting Chunk %d/%d from GPU %d... ", chunk_idx, num_chunks, dev)
        pi4_chunk = fetch(tasks[chunk_idx])
        _accumulate_to_k_matrix!(K, pi4_chunk, chunk_start, fs, mu1, selfgravity, N, KRES)

        if sharp_df
            _accumulate_sharp_df_to_k!(K, pi4_chunk, chunk_start, selfgravity, m, orbit_data.F0, weight_J, N, NR, Nv, KRES)
        end
        println("done")
    end
    GC.gc()

    # Diagonal
    @inbounds for ires in 1:KRES
        l = ires + k0 - 1
        for iR in 1:NR, iv in 1:Nv
            is = (ires - 1) * N + (iR - 1) * Nv + iv
            K[is, is] += l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]
        end
    end

    if sharp_df
        println("✓ Sharp DF term added")
    end

    return K
end

end # module PiElementsMultiGPU
