"""
Multi-GPU Pi Elements calculation for PME.
Splits computation across multiple GPUs by distributing rows.
Now supports configurable precision (Float64 or Float32).
"""

module PiElementsMultiGPUCUDA

using CUDA
using LinearAlgebra
using Printf

# Import get_k0 from Configuration (3 levels up: PiElementsMultiGPUCUDA -> PiElementsMultiGPU -> PME -> Configuration)
const get_k0 = (kres::Int) -> -(kres - 1) ÷ 2  # Local definition to avoid nested import issues

export calculate_pi_elements_multi_cuda, pi_element_chunk_kernel!, calculate_single_pi_element_gpu
"""
Main multi-GPU Pi elements calculation function.
Splits the Pi matrix calculation across multiple GPUs by row distribution.
"""
function calculate_pi_elements_multi_cuda(config, 
                                       orbit_data, 
                                       z::Vector{Float64}, 
                                       psi_z::Vector{Float64}, 
                                       gpu_devices::Vector{Int})
    
    
    NR, Nv, KRES = config.grid.NR, config.grid.Nv, config.core.kres
    N = NR * Nv
    total_elements = N * N * KRES * KRES
    
    
    # Initialize result array (always Float64 for pipeline compatibility)
    pi4 = zeros(Float64, N, N, KRES, KRES)
    
    # Calculate row distribution once and store it
    num_gpus = length(gpu_devices)
    rows_per_gpu = div(N, num_gpus)
    remainder = N % num_gpus
    
    # Store the row ranges for each GPU
    gpu_row_ranges = Vector{Tuple{Int,Int}}()
    start_row = 1
    
    for i in 1:num_gpus
        end_row = start_row + rows_per_gpu - 1
        if i <= remainder
            end_row += 1
        end
        push!(gpu_row_ranges, (start_row, end_row))
        start_row = end_row + 1
    end
    
    
    for (i, device) in enumerate(gpu_devices)
        start_row, end_row = gpu_row_ranges[i]
        elements_this_gpu = (end_row - start_row + 1) * N * KRES * KRES
        percentage = elements_this_gpu / total_elements * 100
        
    end
    
    # Create tasks with explicit row ranges to avoid variable capture issues
    tasks = Task[]
    for (i, device) in enumerate(gpu_devices)
        start_row_task, end_row_task = gpu_row_ranges[i]
        task = Threads.@spawn calculate_pi_rows_cuda(config, orbit_data, z, psi_z, device, start_row_task, end_row_task)
        push!(tasks, task)
    end
    
    
    # Collect results and assemble final matrix using stored ranges
    for (i, task) in enumerate(tasks)
        result_chunk = fetch(task)
        start_row, end_row = gpu_row_ranges[i]
        
        # Copy results back to main array
        pi4[start_row:end_row, :, :, :] = result_chunk
    end
    
    return pi4
end

"""
Calculate Pi elements for a specific range of rows on a single GPU.
"""
function calculate_pi_rows_cuda(config, 
                                orbit_data, 
                                z::Vector{Float64}, 
                                psi_z::Vector{Float64}, 
                                device_id::Int,
                                start_row::Int, 
                                end_row::Int)
    
    
    # Set GPU device
    CUDA.device!(device_id)
    
    # Get dimensions
    NR, Nv, KRES = config.grid.NR, config.grid.Nv, config.core.kres
    N = NR * Nv
    NW = size(orbit_data.ra, 1)
    nz = length(z)
    
    # Calculate chunk size
    chunk_rows = end_row - start_row + 1
    chunk_elements = chunk_rows * N * KRES * KRES
    
    # Select precision type
    T = config.gpu.precision_double ? Float64 : Float32
    memory_gb = chunk_elements * sizeof(T) / (1024^3)
    
    total_memory = CUDA.total_memory() / (1024^3)
    
    # Skip if no rows to compute
    if chunk_rows <= 0
        return zeros(Float64, 0, N, KRES, KRES)
    end
    
    # Initialize result chunk in chosen precision
    pi4_chunk = zeros(T, chunk_rows, N, KRES, KRES)
    
    # Transfer data to GPU in chosen precision
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
    
    # Use the same block size as single GPU
    threads_per_block = 256
    blocks_per_grid = cld(chunk_elements, threads_per_block)
    
    
    @cuda threads=threads_per_block blocks=blocks_per_grid pi_element_chunk_kernel!(
        pi4_chunk_gpu, ra_gpu, pha_gpu, w1_gpu, L_m_gpu, SGNL_gpu, Omega_1_gpu, Omega_2_gpu,
        psi_z_gpu, z_gpu, chunk_rows, N, NR, Nv, NW, KRES, nz, start_row,
        m, k0, T(beta), T(beta_sq), T(orbital_tolerance), T(z_precision)
    )
    
    # Wait for completion and copy back
    CUDA.synchronize()
    result_chunk_T = Array(pi4_chunk_gpu) .* T(4)  # Apply the 4x factor
    
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
GPU kernel for Pi elements calculation with row chunking.
This kernel computes a subset of rows of the full Pi matrix.
Generic over precision type T.
"""
function pi_element_chunk_kernel!(pi4_chunk, ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
                                psi_z_vals, z_vals, chunk_rows, N, NR, Nv, NW, KRES, nz, start_row,
                                m, k0, beta, beta_sq, orbital_tolerance, z_precision)
    
    T = eltype(pi4_chunk)
    
    # Global thread index
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Total elements in this chunk
    total_elements = chunk_rows * N * KRES * KRES
    
    if tid > total_elements
        return
    end
    
    # Decode 4D indices from linear index (chunk_row, j, ires, jres)
    tid_temp = tid - 1  # Convert to 0-based
    
    jres = tid_temp % KRES + 1
    tid_temp ÷= KRES
    
    ires = tid_temp % KRES + 1
    tid_temp ÷= KRES
    
    j = tid_temp % N + 1
    tid_temp ÷= N
    
    chunk_row = tid_temp % chunk_rows + 1
    
    # Convert chunk-local row index to global row index
    i = chunk_row + start_row - 1
    
    # Decode grid indices from linear indices
    iR = ((i - 1) ÷ Nv) + 1
    iv = ((i - 1) % Nv) + 1
    jL = ((j - 1) ÷ Nv) + 1
    jI = ((j - 1) % Nv) + 1
    
    # Skip invalid grid points
    if abs(ra[NW, iR, iv]) < orbital_tolerance || 
       abs(ra[NW, jL, jI]) < orbital_tolerance
        return
    end
    
    # Calculate single Pi element using the exact CPU algorithm
    pi_val = calculate_single_pi_element_gpu(
        iR, iv, jL, jI, ires, jres,
        ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
        psi_z_vals, z_vals, nz, NW,
        m, k0, beta, beta_sq, z_precision
    )
    
    # Store result in chunk array
    pi4_chunk[chunk_row, j, ires, jres] = pi_val
    
    return nothing
end

"""
GPU device function that calculates a single Pi matrix element.
This matches the exact CPU algorithm from the working single GPU implementation.
Generic over precision type.
"""
@inline function calculate_single_pi_element_gpu(
    iR, iv, jL, jI, ires, jres,
    ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
    psi_z_vals, z_vals, nz, NW,
    m, k0, beta, beta_sq, z_precision)
    
    T = eltype(ra)
    
    # Source grid point parameters (exact match to CPU)
    om_pr = Omega_2[iR, iv] 
    
    # Target grid point parameters (exact match to CPU)  
    om_pr_s = Omega_2[jL, jI]
    omega_1_inv = one(T) / Omega_1[jL, jI]
    
    result = zero(T)
    
    # Main computation loop (exact match to CPU algorithm)
    for iw1 in 1:NW
        r_1 = ra[iw1, iR, iv]
        
        # Exact trapezoidal weight for source (matching CPU)
        sw1 = if iw1 == 1
            (w1[2, iR, iv] - w1[1, iR, iv]) / T(2)
        elseif iw1 == NW
            (w1[NW, iR, iv] - w1[NW-1, iR, iv]) / T(2)
        else
            (w1[iw1+1, iR, iv] - w1[iw1-1, iR, iv]) / T(2)
        end
        
        # Calculate s2 for this iw1 (exact match to CPU)
        s2_val = zero(T)
        
        for iw1s in 1:NW
            r_1s = ra[iw1s, jL, jI]
            
            # Gravitational kernel (exact match to CPU)
            r_min = min(r_1, r_1s)
            r_max = max(r_1, r_1s)
            tmp1 = r_min^2 + r_max^2 + beta_sq
            zi = T(2) * r_min * r_max / tmp1
            
            # Interpolate psi_z (linear interpolation)
            psi_val = gpu_interpolate_linear(zi, z_vals, psi_z_vals, nz)
            
            bar_zi = one(T) - zi
            if bar_zi > z_precision
                s3_val = psi_val / sqrt(tmp1)
            else
                # Use numerically stable expression for 1 - z near z -> 1
                # bar_zi = ((r_max - r_min)^2 + beta^2) / (r_min^2 + r_max^2 + beta^2)
                bar_zi = ((r_max - r_min)^2 + beta_sq) / tmp1
                s3_val = sqrt(T(2)) * (log(T(32) / bar_zi) - (T(16)/T(3))) / (T(2)*T(π)) / sqrt(tmp1)
            end
            
            # Exact trapezoidal weight for target (matching CPU)
            sw1s = if iw1s == 1
                (w1[2, jL, jI] - w1[1, jL, jI]) / T(2)
            elseif iw1s == NW
                (w1[NW, jL, jI] - w1[NW-1, jL, jI]) / T(2)
            else
                (w1[iw1s+1, jL, jI] - w1[iw1s-1, jL, jI]) / T(2)
            end
            
            # Phase calculation (exact match to CPU)
            phi_as = w1[iw1s, jL, jI] * om_pr_s * omega_1_inv - pha[iw1s, jL, jI]
            phase_s = (jres + k0 - 1) * w1[iw1s, jL, jI] + m * phi_as
            
            s2_val += sw1s * s3_val * cos(phase_s)
        end
        
        # Final Fourier transform (exact match to CPU)
        phi_a = w1[iw1, iR, iv] * om_pr / Omega_1[iR, iv] - pha[iw1, iR, iv]
        phase_source = (ires + k0 - 1) * w1[iw1, iR, iv] + m * phi_a
        
        result += sw1 * s2_val * cos(phase_source)
    end
    
    return result
end

"""
Linear interpolation for GPU that matches CPU behavior.
Generic over precision type.
"""
@inline function gpu_interpolate_linear(x, x_vals, y_vals, n)
    if x <= x_vals[1]
        return y_vals[1]
    elseif x >= x_vals[n]
        return y_vals[n]
    end
    
    # Binary search for efficiency
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
    
    # Linear interpolation
    t = (x - x_vals[left]) / (x_vals[right] - x_vals[left])
    return y_vals[left] + t * (y_vals[right] - y_vals[left])
end

end # module

