module PiElementsAMD

"""
AMD GPU (ROCm) accelerated Pi elements calculation.
Port of PiElementsGPU.jl for AMD GPUs using AMDGPU.jl.

Translation from CUDA to ROCm:
- @cuda → @roc
- CuArray → ROCArray
- blockIdx().x → workgroupIdx().x
- threadIdx().x → workitemIdx().x
- blockDim().x → workgroupDim().x
- CUDA.synchronize() → AMDGPU.synchronize()
"""

using AMDGPU
using LinearAlgebra
using Printf

# Import get_k0
const get_k0 = (kres::Int) -> -(kres - 1) ÷ 2
using Interpolations
using ..GridConstruction

export calculate_pi_elements_amd
export calculate_pi_elements_multi_amd

"""
AMD GPU kernel for Pi elements computation.
Each work item computes one full Pi element pi4[i,j,ires,jres].
"""
function pi_element_kernel_amd!(pi4, ra, pha, w1, L_m, SGNL, Omega_1, Omega_2, 
                                psi_z_vals, z_vals, nz, N, NR, Nv, NW, KRES,
                                m, k0, beta, beta_sq, orbital_tolerance, z_precision)
    
    # Get work item index - AMD uses workgroupIdx and workitemIdx
    tid = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    
    T = eltype(pi4)
    total_elements = N * N * KRES * KRES
    if tid <= total_elements
        # Convert linear index to 4D indices
        ires = ((tid - 1) % KRES) + 1
        jres = (((tid - 1) ÷ KRES) % KRES) + 1
        j = (((tid - 1) ÷ (KRES * KRES)) % N) + 1
        i = ((tid - 1) ÷ (KRES * KRES * N)) + 1
        
        # Convert i,j to grid coordinates
        iR = ((i - 1) ÷ Nv) + 1
        iv = ((i - 1) % Nv) + 1
        jL = ((j - 1) ÷ Nv) + 1
        jI = ((j - 1) % Nv) + 1
        
        # Skip invalid grid points
        if abs(ra[NW, iR, iv]) < orbital_tolerance || 
           abs(ra[NW, jL, jI]) < orbital_tolerance
            return
        end
        
        # Calculate single Pi element exactly matching CPU
        pi_val = calculate_single_pi_amd_exact(
            iR, iv, jL, jI, ires, jres,
            ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
            psi_z_vals, z_vals, nz, NW,
            m, k0, beta, beta_sq, z_precision
        )
        
        pi4[i, j, ires, jres] = T(4) * pi_val
    end
    
    return nothing
end

"""
AMD GPU device function with exact CPU algorithm matching.
"""
@inline function calculate_single_pi_amd_exact(
    iR, iv, jL, jI, ires, jres,
    ra, pha, w1, L_m, SGNL, Omega_1, Omega_2,
    psi_z_vals, z_vals, nz, NW,
    m, k0, beta, beta_sq, z_precision)
    
    T = eltype(ra)
    
    # Source grid point parameters
    om_pr = Omega_2[iR, iv]
    
    # Target grid point parameters
    om_pr_s = Omega_2[jL, jI]
    omega_1_inv = one(T) / Omega_1[jL, jI]
    
    result = zero(T)
    
    # Main computation loop
    for iw1 in 1:NW
        r_1 = ra[iw1, iR, iv]
        
        # Exact trapezoidal weight for source
        sw1 = if iw1 == 1
            (w1[2, iR, iv] - w1[1, iR, iv]) / T(2)
        elseif iw1 == NW
            (w1[NW, iR, iv] - w1[NW-1, iR, iv]) / T(2)
        else
            (w1[iw1+1, iR, iv] - w1[iw1-1, iR, iv]) / T(2)
        end
        
        # Calculate s2 for this iw1
        s2_val = zero(T)
        
        for iw1s in 1:NW
            r_1s = ra[iw1s, jL, jI]
            
            # Gravitational kernel
            r_min = min(r_1, r_1s)
            r_max = max(r_1, r_1s)
            tmp1 = r_min^2 + r_max^2 + beta_sq
            zi = T(2) * r_min * r_max / tmp1
            
            # Interpolate psi_z
            psi_val = amd_interpolate_linear(zi, z_vals, psi_z_vals, nz)
            
            bar_zi = one(T) - zi
            if bar_zi > z_precision
                s3_val = psi_val / sqrt(tmp1)
            else
                # Use numerically stable expression for z near 1
                bar_zi = ((r_max - r_min)^2 + beta_sq) / tmp1
                s3_val = sqrt(T(2)) * (log(T(32) / bar_zi) - (T(16)/T(3))) / (T(2)*T(π)) / sqrt(tmp1)
            end
            
            # Exact trapezoidal weight for target
            sw1s = if iw1s == 1
                (w1[2, jL, jI] - w1[1, jL, jI]) / T(2)
            elseif iw1s == NW
                (w1[NW, jL, jI] - w1[NW-1, jL, jI]) / T(2)
            else
                (w1[iw1s+1, jL, jI] - w1[iw1s-1, jL, jI]) / T(2)
            end
            
            # Phase calculation
            phi_as = w1[iw1s, jL, jI] * om_pr_s * omega_1_inv - pha[iw1s, jL, jI]
            phase_s = (jres + k0 - 1) * w1[iw1s, jL, jI] + m * phi_as
            
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
Linear interpolation for AMD GPU.
"""
@inline function amd_interpolate_linear(x, x_vals, y_vals, n)
    if x <= x_vals[1]
        return y_vals[1]
    elseif x >= x_vals[n]
        return y_vals[n]
    end
    
    # Binary search
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
    return y_vals[i] * (one(eltype(y_vals)) - t) + y_vals[i+1] * t
end

"""
Main AMD GPU calculation function with configurable precision.
"""
function calculate_pi_elements_amd(config, orbit_data, z, psi_z)
    # Check AMDGPU availability
    if !AMDGPU.functional()
        error("❌ AMDGPU/ROCm is not functional. GPU computation cannot proceed.")
    end
    
    # Select precision type based on config
    T = config.gpu.precision_double ? Float64 : Float32
    
    # Extract dimensions
    NR, Nv, NW = config.grid.NR, config.grid.Nv, config.grid.NW
    N = NR * Nv
    KRES = config.core.kres
    
    # Extract physics parameters
    m = config.core.m
    k0 = get_k0(config.core.kres)
    beta = config.physics.beta
    beta_sq = beta^2
    orbital_tolerance = config.tolerances.orbital_tolerance
    z_precision = config.elliptic.z_precision
    
    total_elements = N * N * KRES * KRES
    
    # Memory requirement check
    memory_requirement = N^2 * KRES^2 * sizeof(T) / 1e9  # GB
    
    # Get AMD GPU memory info
    gpu_memory = try
        info = AMDGPU.Runtime.Mem.info()
        info[1] / 1e9  # Free memory in GB
    catch
        16.0  # Default assumption if query fails
    end
    
    if memory_requirement > gpu_memory * 0.8
        error("❌ Insufficient GPU memory. Required: $(round(memory_requirement, digits=2)) GB, Available: $(round(gpu_memory, digits=2)) GB")
    end
    
    # Transfer all orbital data to GPU in chosen precision using ROCArray
    ra_gpu = ROCArray(T.(orbit_data.ra))
    pha_gpu = ROCArray(T.(orbit_data.pha))  
    w1_gpu = ROCArray(T.(orbit_data.w1))
    L_m_gpu = ROCArray(T.(orbit_data.grids.L_m))
    SGNL_gpu = ROCArray(T.(orbit_data.grids.SGNL))
    Omega_1_gpu = ROCArray(T.(orbit_data.Omega_1))
    Omega_2_gpu = ROCArray(T.(orbit_data.Omega_2))
    z_gpu = ROCArray(T.(z))
    psi_z_gpu = ROCArray(T.(psi_z))
    nz = length(z)
    
    # Allocate result on GPU in chosen precision
    pi4_gpu = ROCArray(zeros(T, N, N, KRES, KRES))
    
    # Configure kernel launch parameters
    # AMD uses groupsize (threads per block) and gridsize (total blocks)
    groupsize = 256
    gridsize = (total_elements + groupsize - 1) ÷ groupsize
    
    # Launch kernel using @roc macro
    @roc groupsize=groupsize gridsize=gridsize pi_element_kernel_amd!(
        pi4_gpu, ra_gpu, pha_gpu, w1_gpu, L_m_gpu, SGNL_gpu, Omega_1_gpu, Omega_2_gpu,
        psi_z_gpu, z_gpu, nz, N, NR, Nv, NW, KRES,
        m, k0, T(beta), T(beta_sq), T(orbital_tolerance), T(z_precision)
    )
    
    # Wait for kernel to complete
    AMDGPU.synchronize()
    
    # Transfer results back to CPU
    pi4_T = Array(pi4_gpu)
    
    # Convert to Float64 for compatibility with rest of pipeline
    pi4 = config.gpu.precision_double ? pi4_T : Float64.(pi4_T)
    
    # Validate matrix
    if any(!isfinite, pi4)
        @warn "Pi matrix contains NaN or Inf values!"
    end
    
    # Note: AMDGPU manages memory differently; explicit free not typically needed
    # The GC will handle ROCArray cleanup
    
    return pi4
end

"""
Multi-GPU Pi elements calculation for AMD GPUs.
"""
function calculate_pi_elements_multi_amd(config, orbit_data, z, psi_z, gpu_devices)
    if length(gpu_devices) == 1
        # Single GPU - use existing function
        dev = AMDGPU.devices()[gpu_devices[1] + 1]  # 0-indexed to 1-indexed
        AMDGPU.device!(dev)
        return calculate_pi_elements_amd(config, orbit_data, z, psi_z)
    end
    
    # Extract dimensions
    N = config.grid.NR * config.grid.Nv
    KRES = config.core.kres
    
    # Split work across GPUs
    num_gpus = length(gpu_devices)
    elements_per_gpu = div(N, num_gpus)
    remainder = N % num_gpus
    
    # Initialize result matrix
    pi4_result = zeros(Float64, N, N, KRES, KRES)
    
    # Create tasks for each GPU
    tasks = []
    start_idx = 1
    
    for (gpu_idx, device_id) in enumerate(gpu_devices)
        # Calculate range for this GPU
        chunk_size = elements_per_gpu + (gpu_idx <= remainder ? 1 : 0)
        end_idx = start_idx + chunk_size - 1
        
        if start_idx > N
            break
        end
        
        # Create task for this GPU
        task = Threads.@spawn begin
            calculate_pi_chunk_amd(
                config, orbit_data, z, psi_z,
                device_id, start_idx, end_idx
            )
        end
        push!(tasks, (task, start_idx, end_idx))
        
        start_idx = end_idx + 1
    end
    
    # Wait for all tasks and collect results
    for (task, start_idx, end_idx) in tasks
        chunk_result = fetch(task)
        pi4_result[start_idx:end_idx, :, :, :] = chunk_result
    end
    
    return pi4_result
end

"""
Calculate Pi elements for a specific row range on a single AMD GPU.
"""
function calculate_pi_chunk_amd(config, orbit_data, z, psi_z, device_id, start_idx, end_idx)
    # Set AMD GPU device (0-indexed input, need 1-indexed for AMDGPU)
    devs = AMDGPU.devices()
    if device_id + 1 > length(devs)
        error("AMD GPU device $device_id not found")
    end
    AMDGPU.device!(devs[device_id + 1])
    
    # Calculate full matrix on this GPU
    chunk_config = deepcopy(config)
    pi4_full = calculate_pi_elements_amd(chunk_config, orbit_data, z, psi_z)
    
    # Return only the requested rows
    return pi4_full[start_idx:end_idx, :, :, :]
end

end # module PiElementsAMD
