# src/utils/GPUBackend.jl
"""
GPU Backend Detection and Abstraction Layer

Provides automatic detection of GPU type (NVIDIA/AMD) and a unified interface
for GPU operations across different backends (CUDA.jl / AMDGPU.jl).

Usage:
    using .GPUBackend
    
    gpu_type = detect_gpu_type()  # Returns NVIDIA, AMD, or NONE
    if gpu_functional()
        gpu_device!(0)
        # ... GPU operations ...
    end
"""
module GPUBackend

export GPUType, NVIDIA, AMD, NONE
export detect_gpu_type, get_gpu_type, gpu_functional
export gpu_device!, gpu_devices, gpu_device_count, gpu_synchronize
export gpu_available_memory, gpu_device_name, gpu_array_type

"""
GPU vendor type enumeration
"""
@enum GPUType begin
    NVIDIA  # CUDA-compatible GPU
    AMD     # ROCm-compatible GPU
    NONE    # No GPU detected
end

# Cached GPU type to avoid repeated detection
const _cached_gpu_type = Ref{Union{GPUType, Nothing}}(nothing)

# Flag to track if GPU packages are loaded
const _cuda_loaded = Ref{Bool}(false)
const _amdgpu_loaded = Ref{Bool}(false)

"""
    detect_gpu_type()::GPUType

Detect available GPU hardware by checking for vendor-specific tools.
- Checks for `nvidia-smi` → NVIDIA GPU detected
- Checks for `rocm-smi` → AMD GPU detected
- Neither found → NONE

Returns: GPUType enum value
"""
function detect_gpu_type()::GPUType
    # Try nvidia-smi first (most common)
    try
        result = run(pipeline(`nvidia-smi -L`, stdout=devnull, stderr=devnull), wait=true)
        if result.exitcode == 0
            return NVIDIA
        end
    catch
        # nvidia-smi not found or failed
    end
    
    # Try rocm-smi for AMD GPUs
    try
        result = run(pipeline(`rocm-smi -i`, stdout=devnull, stderr=devnull), wait=true)
        if result.exitcode == 0
            return AMD
        end
    catch
        # rocm-smi not found or failed
    end
    
    return NONE
end

"""
    get_gpu_type()::GPUType

Get the detected GPU type, using cached value if available.
Call `reset_gpu_cache!()` to force re-detection.
"""
function get_gpu_type()::GPUType
    if _cached_gpu_type[] === nothing
        _cached_gpu_type[] = detect_gpu_type()
    end
    return _cached_gpu_type[]
end

"""
    reset_gpu_cache!()

Reset the cached GPU type, forcing re-detection on next call.
"""
function reset_gpu_cache!()
    _cached_gpu_type[] = nothing
end

"""
    ensure_gpu_package_loaded()

Ensure the appropriate GPU package is loaded based on detected GPU type.
"""
function ensure_gpu_package_loaded()
    gpu_type = get_gpu_type()
    
    if gpu_type == NVIDIA && !_cuda_loaded[]
        @eval Main using CUDA
        _cuda_loaded[] = true
    elseif gpu_type == AMD && !_amdgpu_loaded[]
        @eval Main using AMDGPU
        _amdgpu_loaded[] = true
    end
end

"""
    gpu_functional()::Bool

Check if a functional GPU is available and the appropriate package is working.
"""
function gpu_functional()::Bool
    gpu_type = get_gpu_type()
    
    if gpu_type == NVIDIA
        try
            ensure_gpu_package_loaded()
            return Main.CUDA.functional()
        catch e
            @warn "CUDA.jl check failed: $e"
            return false
        end
    elseif gpu_type == AMD
        try
            ensure_gpu_package_loaded()
            return Main.AMDGPU.functional()
        catch e
            @warn "AMDGPU.jl check failed: $e"
            return false
        end
    end
    
    return false
end

"""
    gpu_device!(id::Int)

Set the active GPU device by index (0-based for consistency with CUDA convention).
"""
function gpu_device!(id::Int)
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        Main.CUDA.device!(id)
    elseif gpu_type == AMD
        # AMDGPU uses 1-based indexing internally
        devs = Main.AMDGPU.devices()
        if id + 1 > length(devs)
            error("GPU device $id not found. Available devices: 0-$(length(devs)-1)")
        end
        Main.AMDGPU.device!(devs[id + 1])
    else
        error("No GPU available")
    end
end

"""
    gpu_devices()

Get the list of available GPU devices.
"""
function gpu_devices()
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        return Main.CUDA.devices()
    elseif gpu_type == AMD
        return Main.AMDGPU.devices()
    end
    
    return []
end

"""
    gpu_device_count()::Int

Get the number of available GPU devices.
"""
function gpu_device_count()::Int
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        return length(Main.CUDA.devices())
    elseif gpu_type == AMD
        return length(Main.AMDGPU.devices())
    end
    
    return 0
end

"""
    gpu_synchronize()

Synchronize GPU execution (wait for all GPU operations to complete).
"""
function gpu_synchronize()
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        Main.CUDA.synchronize()
    elseif gpu_type == AMD
        Main.AMDGPU.synchronize()
    end
end

"""
    gpu_available_memory()::Int

Get available GPU memory in bytes for the current device.
"""
function gpu_available_memory()::Int
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        return Main.CUDA.available_memory()
    elseif gpu_type == AMD
        # AMD: Use HIP properties for total memory
        # Note: AMDGPU.Runtime.Mem.info()[1] reports incorrect values
        try
            dev = Main.AMDGPU.device()
            props = Main.AMDGPU.HIP.properties(dev)
            return props.totalGlobalMem
        catch
            # Fallback: 24GB default for RX 7900 XTX
            return 24 * 1024^3
        end
    end
    
    return 0
end

"""
    gpu_device_name(id::Int=0)::String

Get the name of the specified GPU device.
"""
function gpu_device_name(id::Int=0)::String
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        dev = Main.CUDA.device(id)
        return Main.CUDA.name(dev)
    elseif gpu_type == AMD
        devs = Main.AMDGPU.devices()
        if id + 1 <= length(devs)
            return string(devs[id + 1])
        end
        return "Unknown AMD GPU"
    end
    
    return "No GPU"
end

"""
    gpu_array_type()

Get the appropriate GPU array type for the detected backend.
Returns CuArray for NVIDIA, ROCArray for AMD.
"""
function gpu_array_type()
    gpu_type = get_gpu_type()
    ensure_gpu_package_loaded()
    
    if gpu_type == NVIDIA
        return Main.CUDA.CuArray
    elseif gpu_type == AMD
        return Main.AMDGPU.ROCArray
    end
    
    error("No GPU available - cannot determine array type")
end

"""
    print_gpu_info()

Print information about detected GPU hardware.
"""
function print_gpu_info()
    gpu_type = get_gpu_type()
    
    if gpu_type == NONE
        println("❌ No GPU detected")
        println("   Install nvidia-smi (NVIDIA) or rocm-smi (AMD) for GPU support")
        return
    end
    
    vendor = gpu_type == NVIDIA ? "NVIDIA (CUDA)" : "AMD (ROCm)"
    println("🎮 GPU Backend: $vendor")
    
    if gpu_functional()
        n_devices = gpu_device_count()
        println("✅ GPU functional with $n_devices device(s)")
        
        for i in 0:(n_devices-1)
            name = gpu_device_name(i)
            println("   Device $i: $name")
        end
        
        mem_gb = round(gpu_available_memory() / 1e9, digits=2)
        println("   Available memory: $mem_gb GB")
    else
        println("⚠️  GPU detected but not functional")
        println("   Check driver installation")
    end
end

end # module GPUBackend
