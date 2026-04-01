"""
Compute Pi elements for a single (l_i, l_j) resonance pair on AMD GPU.
This module enables progressive KRES expansion by computing only the
needed resonance blocks without computing the full Pi4 array.
"""
module PiElementsBlockAMD

using AMDGPU
using LinearAlgebra
using Printf

# Local get_k0 function
const get_k0 = (kres::Int) -> -(kres - 1) ÷ 2

export compute_pi_block_amd


"""
GPU kernel for computing Pi elements for a single resonance pair (l_i, l_j).
This computes the N×N matrix pi4[:, :, ires, jres] for fixed l_i, l_j.
"""
function pi_block_kernel!(pi4_block, ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
                          psi_z_vals, z_vals, N, NR, Nv, NW, nz,
                          l_i, l_j, m, beta, beta_sq, orbital_tolerance, z_precision)
    
    T = eltype(pi4_block)
    
    # Global thread index (AMD uses workgroupIdx/workitemIdx)
    tid = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    
    # Total elements: N × N
    total_elements = N * N
    
    if tid > total_elements
        return
    end
    
    # Decode 2D indices (i, j) from linear index
    tid_temp = tid - 1
    j = tid_temp % N + 1
    i = tid_temp ÷ N + 1
    
    # Decode grid indices
    iR = ((i - 1) ÷ Nv) + 1
    iv = ((i - 1) % Nv) + 1
    jL = ((j - 1) ÷ Nv) + 1
    jI = ((j - 1) % Nv) + 1
    
    # Skip invalid grid points
    if abs(ra[NW, iR, iv]) < orbital_tolerance || 
       abs(ra[NW, jL, jI]) < orbital_tolerance
        return
    end
    
    # Calculate Pi element for this (i, j) pair at resonance (l_i, l_j)
    pi_val = calculate_single_pi_element_block(
        iR, iv, jL, jI, l_i, l_j,
        ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
        psi_z_vals, z_vals, nz, NW,
        m, beta, beta_sq, z_precision
    )
    
    pi4_block[i, j] = pi_val
    
    return nothing
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
    return y_vals[left] * (one(eltype(y_vals)) - t) + y_vals[right] * t
end


"""
GPU device function that calculates a single Pi matrix element for specific l_i, l_j.
This matches the exact CPU algorithm from the working single GPU implementation.
"""
@inline function calculate_single_pi_element_block(
    iR, iv, jL, jI, l_i, l_j,
    ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
    psi_z_vals, z_vals, nz, NW,
    m, beta, beta_sq, z_precision)
    
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
            
            # Phase calculation using l_j directly (instead of jres + k0 - 1)
            phi_as = w1[iw1s, jL, jI] * om_pr_s * omega_1_inv - pha[iw1s, jL, jI]
            phase_s = l_j * w1[iw1s, jL, jI] + m * phi_as
            
            s2_val += sw1s * s3_val * cos(phase_s)
        end
        
        # Final Fourier transform using l_i directly (instead of ires + k0 - 1)
        phi_a = w1[iw1, iR, iv] * om_pr / Omega_1[iR, iv] - pha[iw1, iR, iv]
        phase_source = l_i * w1[iw1, iR, iv] + m * phi_a
        
        result += sw1 * s2_val * cos(phase_source)
    end
    
    return result
end


"""
Compute Pi elements for a single (l_i, l_j) resonance pair on AMD GPU.
Returns an N×N matrix of Pi elements.

Parameters:
- config: PME configuration
- orbit_data: Orbital trajectory data
- z, psi_z: Elliptic function lookup table
- l_i, l_j: Resonance numbers for source and target
- device_id: GPU device to use (default 0)
"""
function compute_pi_block_amd(config, orbit_data, z::Vector{Float64}, psi_z::Vector{Float64}, 
                              l_i::Int, l_j::Int; device_id::Int=0)
    
    # Set GPU device (AMDGPU requires HIPDevice object, not integer)
    AMDGPU.device!(AMDGPU.devices()[device_id + 1])
    
    NR, Nv = config.grid.NR, config.grid.Nv
    N = NR * Nv
    NW = size(orbit_data.ra, 1)
    nz = length(z)
    
    # Select precision type
    T = config.gpu.precision_double ? Float64 : Float32
    
    # Initialize result block
    pi4_block = zeros(T, N, N)
    
    # Transfer data to GPU
    ra_gpu = ROCArray(T.(orbit_data.ra))
    pha_gpu = ROCArray(T.(orbit_data.pha))
    w1_gpu = ROCArray(T.(orbit_data.w1))
    L_m_gpu = ROCArray(T.(orbit_data.grids.L_m))
    SGNL_gpu = ROCArray(T.(orbit_data.grids.SGNL))
    Omega_1_gpu = ROCArray(T.(orbit_data.Omega_1))
    Omega_2_gpu = ROCArray(T.(orbit_data.Omega_2))
    psi_z_gpu = ROCArray(T.(psi_z))
    z_gpu = ROCArray(T.(z))
    pi4_block_gpu = ROCArray(pi4_block)
    
    # Get parameters
    m = config.core.m
    beta = config.physics.beta
    beta_sq = beta * beta
    orbital_tolerance = config.tolerances.orbital_tolerance
    z_precision = config.elliptic.z_precision
    
    # Launch kernel
    total_elements = N * N
    threads_per_block = 256
    blocks_per_grid = cld(total_elements, threads_per_block)
    
    @roc groupsize=threads_per_block gridsize=blocks_per_grid pi_block_kernel!(
        pi4_block_gpu, ra_gpu, pha_gpu, w1_gpu, L_m_gpu, SGNL_gpu, Omega_1_gpu, Omega_2_gpu,
        psi_z_gpu, z_gpu, N, NR, Nv, NW, nz,
        l_i, l_j, m, T(beta), T(beta_sq), T(orbital_tolerance), T(z_precision)
    )
    
    # Wait and copy back
    AMDGPU.synchronize()
    result_T = Array(pi4_block_gpu) .* T(4)  # Apply 4x factor
    
    # Return native precision (Float32 or Float64 based on config)
    # K matrix pipeline will handle the precision consistently
    result = result_T
    
    # Free GPU memory
    AMDGPU.unsafe_free!(pi4_block_gpu)
    AMDGPU.unsafe_free!(ra_gpu)
    AMDGPU.unsafe_free!(pha_gpu)
    AMDGPU.unsafe_free!(w1_gpu)
    AMDGPU.unsafe_free!(L_m_gpu)
    AMDGPU.unsafe_free!(SGNL_gpu)
    AMDGPU.unsafe_free!(Omega_1_gpu)
    AMDGPU.unsafe_free!(Omega_2_gpu)
    AMDGPU.unsafe_free!(psi_z_gpu)
    AMDGPU.unsafe_free!(z_gpu)
    
    return result
end


end # module PiElementsBlockAMD
