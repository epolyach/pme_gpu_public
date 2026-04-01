module DistributedPiElementsGPU

using CUDA
using LinearAlgebra
using Printf

export calculate_pi_rows_on_gpu

"""
Calculate a subset of rows of the Pi matrix on a GPU for distributed computation.
This is based on PiElementsMultiGPU.jl but adapted for the distributed framework.

Parameters:
- `config`: Configuration data
- `orbit_data`: Orbital data  
- `z`: Z-values
- `psi_z`: Psi Z-values
- `gpu_device`: GPU device ID
- `start_row`: Starting row index (1-based)
- `end_row`: Ending row index (1-based)

Returns:
- Subsection of the Pi matrix with dimensions (num_rows, N, KRES, KRES)
"""
function calculate_pi_rows_on_gpu(config, orbit_data, z, psi_z, gpu_device, start_row, end_row)
    println("  🔥 GPU $gpu_device: Computing Pi[$start_row:$end_row, :, :, :] ($(end_row-start_row+1) rows)")
    
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
    chunk_elements = chunk_rows * N * KRES * KRES
    memory_gb = chunk_elements * 8 / (1024^3)  # Float64 = 8 bytes
    
    total_memory = CUDA.total_memory() / (1024^3)
    println("    📊 Chunk memory: $(round(memory_gb, digits=3)) GB of $(round(total_memory, digits=2)) GB available")
    
    # Skip if no rows to compute
    if chunk_rows <= 0
        println("    ⚠️  No rows to compute, returning empty chunk")
        return zeros(Float64, 0, N, KRES, KRES)
    end
    
    # Initialize result chunk
    pi4_chunk = zeros(Float64, chunk_rows, N, KRES, KRES)
    
    # Transfer data to GPU
    println("    🔄 Transferring data to GPU...")
    ra_gpu = CuArray(orbit_data.ra)
    pha_gpu = CuArray(orbit_data.pha)
    w1_gpu = CuArray(orbit_data.w1)
    L_m_gpu = CuArray(orbit_data.grids.L_m)
    SGNL_gpu = CuArray(orbit_data.grids.SGNL)
    Omega_1_gpu = CuArray(orbit_data.Omega_1)
    Omega_2_gpu = CuArray(orbit_data.Omega_2)
    psi_z_gpu = CuArray(psi_z)
    z_gpu = CuArray(z)
    pi4_chunk_gpu = CuArray(pi4_chunk)
    
    # Get parameters
    m = config.core.m
    k0 = get_k0(config.core.kres)
    beta = config.physics.beta
    beta_sq = beta * beta
    orbital_tolerance = config.tolerances.orbital_tolerance
    z_precision = config.elliptic.z_precision
    
    # Launch kernel
    println("    🚀 Launching CUDA kernel...")
    
    # Use the same block size as single GPU
    threads_per_block = 256
    blocks_per_grid = cld(chunk_elements, threads_per_block)
    
    println("      Chunk elements: $chunk_elements")
    println("      Blocks: $blocks_per_grid")
    println("      Threads per block: $threads_per_block")
    
    @cuda threads=threads_per_block blocks=blocks_per_grid pi_element_chunk_kernel!(
        pi4_chunk_gpu, ra_gpu, pha_gpu, w1_gpu, L_m_gpu, SGNL_gpu, Omega_1_gpu, Omega_2_gpu,
        psi_z_gpu, z_gpu, chunk_rows, N, NR, Nv, NW, KRES, nz, start_row,
        m, k0, beta, beta_sq, orbital_tolerance, z_precision
    )
    
    # Wait for completion and copy back
    CUDA.synchronize()
    result_chunk = Array(pi4_chunk_gpu) .* 4.0  # Apply the 4x factor
    
    # Free GPU memory
    println("    🔄 Freeing GPU memory...")
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
    
    println("    ✅ GPU $gpu_device chunk computation complete")
    
    return result_chunk
end

"""
GPU kernel for Pi elements calculation with row chunking.
This kernel computes a subset of rows of the full Pi matrix.
Based on the implementation in PiElementsGPU.jl but adapted for chunked computation.
"""
function pi_element_chunk_kernel!(pi4_chunk, ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
                                psi_z_vals, z_vals, chunk_rows, N, NR, Nv, NW, KRES, nz, start_row,
                                m, k0, beta, beta_sq, orbital_tolerance, z_precision)
    
    # Global thread index
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Total elements in this chunk
    total_elements = chunk_rows * N * KRES * KRES
    
    if tid > total_elements
        return
    end
    
    # Convert linear index to 4D indices for the chunk
    ires = ((tid - 1) % KRES) + 1
    jres = (((tid - 1) ÷ KRES) % KRES) + 1
    j = (((tid - 1) ÷ (KRES * KRES)) % N) + 1
    i_chunk = ((tid - 1) ÷ (KRES * KRES * N)) + 1  # Row index within chunk
    
    # Convert chunk row index to global row index
    i = i_chunk + start_row - 1
    
    # Convert global i,j to grid coordinates
    iR = ((i - 1) ÷ Nv) + 1
    iv = ((i - 1) % Nv) + 1
    jR = ((j - 1) ÷ Nv) + 1
    jv = ((j - 1) % Nv) + 1
    
    # Skip invalid grid points
    if abs(ra[NW, iR, iv]) < orbital_tolerance || 
       abs(ra[NW, jR, jv]) < orbital_tolerance
        return
    end
    
    # Calculate single Pi element exactly matching CPU
    pi_val = calculate_single_pi_gpu_exact(
        iR, iv, jR, jv, ires, jres,
        ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
        psi_z_vals, z_vals, nz, NW,
        m, k0, beta, beta_sq, z_precision
    )
    
    # Store in chunk array (note: using chunk indices)
    pi4_chunk[i_chunk, j, ires, jres] = pi_val
    
    return nothing
end

"""
GPU device function with EXACT CPU algorithm matching.
Based on PiElementsGPU.jl implementation.
"""
@inline function calculate_single_pi_gpu_exact(
    iR, iv, jR, jv, ires, jres,
    ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
    psi_z_vals, z_vals, nz, NW,
    m, k0, beta, beta_sq, z_precision)
    
    # Source grid point parameters (exact match to CPU)
    # sgnl = SGNL[iR, iv]
    om_pr = Omega_2[iR, iv]
    
    # Target grid point parameters (exact match to CPU)  
    # sgnl_s = SGNL[jL, jI]
    om_pr_s = Omega_2[jR, jv]
    omega_1_inv = 1.0 / Omega_1[jR, jv]
    
    result = 0.0
    
    # Main computation loop (exact match to CPU algorithm)
    for iw1 in 1:NW
        r_1 = ra[iw1, iR, iv]
        
        # Exact trapezoidal weight for source
        sw1 = if iw1 == 1
            (w1[2, iR, iv] - w1[1, iR, iv]) / 2.0
        elseif iw1 == NW
            (w1[NW, iR, iv] - w1[NW-1, iR, iv]) / 2.0
        else
            (w1[iw1+1, iR, iv] - w1[iw1-1, iR, iv]) / 2.0
        end
        
        # Calculate s2 for this iw1
        s2_val = 0.0
        
        for iw1s in 1:NW
            r_1s = ra[iw1s, jR, jv]
            
            # Gravitational kernel
            r_min = min(r_1, r_1s)
            r_max = max(r_1, r_1s)
            tmp1 = r_min^2 + r_max^2 + beta_sq
            zi = 2.0 * r_min * r_max / tmp1
            
            # Interpolate psi_z
            psi_val = gpu_interpolate_linear(zi, z_vals, psi_z_vals, nz)
            
            bar_zi = 1.0 - zi
            if bar_zi > z_precision
                s3_val = psi_val / sqrt(tmp1)
            else
                s3_val = sqrt(2.0) * (log(32.0 / bar_zi) - (16.0/3.0)) / (2*π) / sqrt(tmp1)
            end
            
            # Exact trapezoidal weight for target
            sw1s = if iw1s == 1
                (w1[2, jR, jv] - w1[1, jR, jv]) / 2.0
            elseif iw1s == NW
                (w1[NW, jR, jv] - w1[NW-1, jR, jv]) / 2.0
            else
                (w1[iw1s+1, jR, jv] - w1[iw1s-1, jR, jv]) / 2.0
            end
            
            # Phase calculation
            phi_as = w1[iw1s, jR, jv] * om_pr_s * omega_1_inv - pha[iw1s, jR, jv]
            phase_s = (jres + k0 - 1) * w1[iw1s, jR, jv] + m * phi_as
            
            s2_val += sw1s * s3_val * cos(phase_s)
        end
        
        # Final Fourier transform
        phi_a = w1[iw1, iR, iv] * om_pr / Omega_1[iR, iv] - pha[iw1, iR, iv]
        phase_source = (ires + k0 - 1) * w1[iw1, iR, iv] + m * phi_a
        
        result += sw1 * s2_val * cos(phase_source)
    end
    
    return result
end

"""
Improved linear interpolation for GPU that matches CPU spline more closely.
"""
@inline function gpu_interpolate_linear(x, x_vals, y_vals, n)
    if x <= x_vals[1]
        return y_vals[1]
    elseif x >= x_vals[n]
        return y_vals[n]
    end
    
    # Binary search for better performance
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
    i = left
    t = (x - x_vals[i]) / (x_vals[i+1] - x_vals[i])
    return y_vals[i] * (1.0 - t) + y_vals[i+1] * t
end

end # module DistributedPiElementsGPU

