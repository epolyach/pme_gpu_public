# src/utils/CUDAUtils.jl
"""
CUDA utility functions for checking CUDA availability and providing fallbacks.
"""
module CUDAUtils

export pme_has_cuda, cuda_functional, gpu_info, fallback_to_cpu

"""
    pme_has_cuda() -> Bool

Check if CUDA.jl is available (installed).
"""
function pme_has_cuda()
    try
        @eval using CUDA
        return true
    catch
        return false
    end
end

"""
    cuda_functional() -> Bool

Check if CUDA is both available and functional (can detect GPU hardware).
"""
function cuda_functional()
    if !pme_has_cuda()
        return false
    end
    
    try
        @eval using CUDA
        return CUDA.functional()
    catch
        return false
    end
end

"""
    gpu_info()

Print information about GPU availability and status.
"""
function gpu_info()
#     println("🔍 GPU Status Check:")
    
    if !pme_has_cuda()
#         println("  ❌ CUDA.jl not installed")
#         println("  💡 To enable GPU support, run: using Pkg; Pkg.add(\"CUDA\")")
        return
    end
    
    @eval using CUDA
    
    if !CUDA.functional()
#         println("  ⚠️  CUDA.jl installed but not functional")
#         println("  📝 Reason: No CUDA-capable GPU detected or CUDA drivers not installed")
#         println("  💡 Running in CPU-only mode")
        return
    end
    
#     println("  ✅ CUDA.jl functional")
#     println("  🎯 GPU Device: $(CUDA.device())")
#     println("  💾 GPU Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")
#     println("  🚀 GPU acceleration enabled")
end

"""
    fallback_to_cpu(func_name::String)

Print a message about falling back to CPU for a specific function.
"""
function fallback_to_cpu(func_name::String)
#     println("⚠️  GPU not available for $func_name - using CPU implementation")
end

end # module CUDAUtils
