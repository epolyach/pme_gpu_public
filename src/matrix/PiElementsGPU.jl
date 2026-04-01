module PiElementsGPU

"""
GPU-accelerated Pi elements calculation with multi-backend support.
Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs via automatic detection.

The module dispatches to the appropriate backend:
- NVIDIA GPUs: Uses CUDA.jl with @cuda kernels (via PiElementsCUDA.jl)
- AMD GPUs: Uses AMDGPU.jl with @roc kernels (via PiElementsAMD.jl)
"""

using LinearAlgebra
using Printf
using Interpolations
using ..GridConstruction

# Include GPUBackend for detection
include("../utils/GPUBackend.jl")
using .GPUBackend

export calculate_pi_elements_gpu
export calculate_pi_elements_multi_gpu

# Detect GPU type at module load time
const _MODULE_GPU_TYPE = GPUBackend.detect_gpu_type()

# Conditionally load the appropriate GPU package and backend module
if _MODULE_GPU_TYPE == GPUBackend.NVIDIA
    using CUDA
    include("PiElementsCUDA.jl")
    using .PiElementsCUDA
elseif _MODULE_GPU_TYPE == GPUBackend.AMD
    using AMDGPU
    include("PiElementsAMD.jl")
    using .PiElementsAMD
end

"""
    calculate_pi_elements_gpu(config, orbit_data, z, psi_z)

Main GPU calculation function with automatic backend detection.
Dispatches to CUDA or AMD implementation based on detected GPU type.
"""
function calculate_pi_elements_gpu(config, orbit_data, z, psi_z)
    gpu_type = _MODULE_GPU_TYPE
    
    if gpu_type == GPUBackend.NVIDIA
        return PiElementsCUDA.calculate_pi_elements_cuda(config, orbit_data, z, psi_z)
    elseif gpu_type == GPUBackend.AMD
        return PiElementsAMD.calculate_pi_elements_amd(config, orbit_data, z, psi_z)
    else
        error("❌ No GPU detected. Cannot perform GPU Pi elements calculation.")
    end
end

"""
    calculate_pi_elements_multi_gpu(config, orbit_data, z, psi_z, gpu_devices)

Multi-GPU Pi elements calculation with automatic backend detection.
"""
function calculate_pi_elements_multi_gpu(config, orbit_data, z, psi_z, gpu_devices)
    gpu_type = _MODULE_GPU_TYPE
    
    if gpu_type == GPUBackend.NVIDIA
        return PiElementsCUDA.calculate_pi_elements_multi_cuda(config, orbit_data, z, psi_z, gpu_devices)
    elseif gpu_type == GPUBackend.AMD
        return PiElementsAMD.calculate_pi_elements_multi_amd(config, orbit_data, z, psi_z, gpu_devices)
    else
        error("❌ No GPU detected. Cannot perform multi-GPU Pi elements calculation.")
    end
end

end # module PiElementsGPU
