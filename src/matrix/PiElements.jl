# src/matrix/PiElements.jl
"""
Calculation of Pi matrix elements with high-performance parallelization.
Optimized for 64-core AMD systems with 256GB RAM.
This is an optimized Julia port of the C code in `schwarz_eRc_pi_full.c`.
"""

module PiElements

# get_k0 is computed from kres
const get_k0 = (config) -> -(config.core.kres - 1) ÷ 2

using Interpolations
using LinearAlgebra
using ..GridConstruction
using Base.Threads
using Printf
using ProgressMeter
using SharedArrays
using LoopVectorization
using ..ProgressUtils: print_progress_bar

export calculate_pi_elements, calculate_single_pi_element_exact_c_match!

"""
    calculate_pi_elements(config, orbits, z::Vector{Float64}, psi_z::Vector{Float64})

Calculate the Pi matrix elements with high-performance parallel processing optimized for 64-core systems.
Uses block-based computation, NUMA-aware memory allocation, and advanced progress monitoring.
"""
function calculate_pi_elements(config, orbits, z::Vector{Float64}, psi_z::Vector{Float64})
    
    # Extract configuration parameters with proper access patterns
    NR, Nv, NW = config.grid.NR, config.grid.Nv, config.grid.NW
    N = NR * Nv
    KRES = config.core.kres
    
    # Optimize thread usage for 64-core systems
    nthreads = Threads.nthreads()
    
    # Note: Thread pinning could be added here for NUMA optimization on large systems
    if nthreads > 32
#         println("  ⚠ Large system detected - consider manual thread pinning for optimal NUMA performance")
    end
    
#     println("  High-performance matrix calculation using $nthreads threads")
#     println("  Total matrix elements: $(N^2) x $(KRES^2) = $(N^2 * KRES^2)")
#     println("  Memory requirement: ~$(round((N^2 * KRES^2 * 8) / 1e9, digits=2)) GB")
    
    # Setup thread-safe interpolation for psi_z with safe flat extrapolation at boundaries
    base_itp = interpolate((z,), psi_z, Gridded(Linear()))
    spline_psiz = extrapolate(base_itp, Flat())
    
    # Determine precision based on config
    T = config.cpu.precision_double ? Float64 : Float32
    
    # Use SharedArray for large matrices to optimize memory access on NUMA systems
    if N^2 * KRES^2 > 1e8  # Use SharedArray for large problems
    
#         println("  Using SharedArray for NUMA-optimized memory allocation")
        pi4 = SharedArray{T}(N, N, KRES, KRES)
        fill!(pi4, zero(T))
    else
        pi4 = zeros(T, N, N, KRES, KRES)
    end
    
    # Create work distribution optimized for cache efficiency  
    # Match MATLAB convention: iR (radial) outer, iv (circulation) inner
    grid_pairs = [(iR, iv) for iR in 1:NR for iv in 1:Nv]
    
    # Filter out invalid grid points and sort by computational complexity
    orbital_tolerance = config.tolerances.orbital_tolerance
    valid_pairs = filter(grid_pairs) do (iR, iv)
        abs(orbits.ra[NW, iR, iv]) >= orbital_tolerance
    end
    
    # Sort pairs by radial index for better cache locality
    sort!(valid_pairs, by = x -> x[1])
    
#     println("  Valid grid points: $(length(valid_pairs)) / $(length(grid_pairs))")
    
    # Advanced progress monitoring with time estimation
    progress = Progress(length(valid_pairs), dt=1.0, 
                       desc="Computing Pi elements: ",
                       barglyphs=BarGlyphs("[=> ]"),
                       barlen=50)
    
    # Block-based parallel computation for better load balancing
    block_size = max(1, length(valid_pairs) ÷ (nthreads * 4))  # 4 blocks per thread
    blocks = [valid_pairs[i:min(i+block_size-1, end)] for i in 1:block_size:length(valid_pairs)]
    
#     println("  Using $(length(blocks)) computation blocks for optimal load balancing")
    
    # High-performance parallel computation with optimized memory access
    # Use Threads.@threads for reliable parallelization
    Threads.@threads for block in blocks
        # Thread-local pre-allocated arrays (reused across iterations)
        local_s1 = Matrix{T}(undef, KRES, KRES)
        
        for (iR, iv) in block
            i_ri = (iR - 1) * Nv + iv
            
            # Skip invalid grid points early
            orbital_tolerance = config.tolerances.orbital_tolerance
            if abs(orbits.ra[NW, iR, iv]) < orbital_tolerance
                next!(progress)
                continue
            end
            
            
            # Calculate matrix elements for all target grid points with vectorized operations
            # Match C code order: for(iRs=0; iRs<NR; iRs++) for(ivs=0; ivs<Nv; ivs++)
            @inbounds for iRs in 1:NR, ivs in 1:Nv
                i_ris = (iRs - 1) * Nv + ivs
                
                # Skip invalid target points
                orbital_tolerance = config.tolerances.orbital_tolerance
                if abs(orbits.ra[NW, iRs, ivs]) < orbital_tolerance
                    continue
                end
                
                # Calculate Pi elements for this pair of grid points (ultra-optimized)
                calculate_single_pi_element_ultra_optimized!(local_s1,
                                                           iR, iv, iRs, ivs, orbits, spline_psiz, 
                                                           config)
                
                # Store results with cache-friendly memory access pattern
                # Use simple loops for compatibility (avoid @turbo hangs with large matrices)
                @inbounds for ires in 1:KRES, jres in 1:KRES
                    pi4[i_ri, i_ris, ires, jres] = T(4.0) * local_s1[ires, jres]
                end
            end
            
            # Update progress (thread-safe)
            next!(progress)
        end
    end
    
    finish!(progress)
#     println("  ✓ Matrix elements calculated successfully")
    
    # Validate matrix and print unified range info
    min_val = minimum(pi4)
    max_val = maximum(pi4)
    if any(!isfinite, pi4)
#         println("  ⚠ WARNING: Pi matrix contains NaN or Inf values!")
#         println("  Pi element range: $(round(min_val, digits=4)) - $(round(max_val, digits=4))")
    else
#         println("  ✓ Pi matrix validated: no NaN or Inf values found")
#         println("  Pi element range: $(round(min_val, digits=4)) - $(round(max_val, digits=4))")
    end
    
    # Debug output: save Pi elements if debug mode is enabled
    if config.io.debug
#         println("  Debug mode: saving Pi matrix elements...")
        
        # Setup data directory
        data_dir = config.io.data_path
        mkpath(data_dir)
        binary_dir = joinpath(data_dir, "binary")
        mkpath(binary_dir)
        
        # Save Pi elements with configurable precision
        pi_filename = joinpath(binary_dir, "pi_elements.bin")
        try
            if config.io.single_precision
                # Write as Float32
                open(pi_filename, "w") do io
                    pi4_f32 = Float32.(pi4)
                    write(io, pi4_f32)
                end
#                 println("  ✓ Pi matrix elements saved to $pi_filename (Float32 precision)")
            else
                # Write as Float64
                open(pi_filename, "w") do io
                    pi4_f64 = Float64.(pi4)
                    write(io, pi4_f64)
                end
#                 println("  ✓ Pi matrix elements saved to $pi_filename (Float64 precision)")
            end
#             println("  ✓ Pi matrix dimensions: $(size(pi4))")
        catch e
#             println("  ⚠ WARNING: Failed to save Pi elements: $e")
        end
    end
    
    return pi4
end

"""
High-performance optimized calculation of Pi matrix elements for a single pair of grid points.
Uses pre-allocated arrays and vectorized operations for maximum performance.
"""
function calculate_single_pi_element_ultra_optimized!(s1, iR, iv, iLs, iIs, orbits, spline_psiz, config)
    
    NW = config.grid.NW
    KRES = config.core.kres
    k0 = get_k0(config)
    m = config.core.m
    beta = config.physics.beta
    beta_sq = beta^2
    
    # Local working arrays
    s2 = Matrix{Float64}(undef, NW, KRES)
    s3 = Vector{Float64}(undef, NW)
    
    # Source grid point parameters
    # sgnl = orbits.grids.SGNL[iR, iv]
    om_pr = orbits.Omega_2[iR, iv]
    w1_1_ = @view orbits.w1[:, iR, iv]
    Sw1 = GridConstruction.trapezoidal_coef(w1_1_)
    
    # Target grid point parameters
    # sgnl_s = orbits.grids.SGNL[iLs, iIs]
    om_pr_s = orbits.Omega_2[iLs, iIs]
    omega_1_inv = 1.0 / orbits.Omega_1[iLs, iIs]
    
    # Pre-compute phase arrays for vectorization
    w1s_1_ = @view orbits.w1[:, iLs, iIs]
    pha_s = @view orbits.pha[:, iLs, iIs]
    ra_s = @view orbits.ra[:, iLs, iIs]
    
    # Pre-compute Fourier coefficients for source
    pha_a = @view orbits.pha[:, iR, iv]
    ra = @view orbits.ra[:, iR, iv]
    
    Sw1s = GridConstruction.trapezoidal_coef(w1s_1_)
    
    # CRITICAL FIX: Pre-compute phi_as array OUTSIDE the iw1 loop (matching C code lines 290-295)
    phi_as = Vector{Float64}(undef, NW)
    @inbounds for iw1s in 1:NW
        phi_as[iw1s] = w1s_1_[iw1s] * om_pr_s * omega_1_inv - pha_s[iw1s]  # C code line 295, theta' in (2.12)
    end
    
    # Clear pre-allocated arrays
    fill!(s1, 0.0)
    fill!(s2, 0.0)
    
    # Ultra-optimized double loop with maximum vectorization
    @inbounds for iw1 in 1:NW
        r_1 = ra[iw1]
        
        # Calculate gravitational kernel - moved conditional outside @turbo
        @inbounds for iw1s in 1:NW
            r_1s = ra_s[iw1s]
            
            r_min = min(r_1, r_1s)
            r_max = max(r_1, r_1s)
            
            tmp1 = r_min^2 + r_max^2 + beta_sq
            zi = 2.0 * r_min * r_max / tmp1
            
            # Conditional evaluation outside @turbo - match C code exactly
            z_precision = config.elliptic.z_precision
            bar_zi = 1.0 - zi
            if bar_zi > z_precision
                s3[iw1s] = spline_psiz(zi) / sqrt(tmp1)
            else
                # Use numerically stable expression for 1 - z near z -> 1
                # bar_zi = ((r_max - r_min)^2 + beta^2) / (r_min^2 + r_max^2 + beta^2)
                bar_zi = ((r_max - r_min)^2 + beta_sq) / tmp1
                s3[iw1s] = sqrt(2.0) * (log(32.0 / bar_zi) - (16.0/3.0)) / (2*π) / sqrt(tmp1)  # m=2 hardcoded
            end
        end
        
        # Use pre-computed phi_as (computed once outside iw1 loop)
        @inbounds for iw1s in 1:NW
            # Vectorized accumulation over w_s
            for jres in 1:KRES
                phase_s = (jres + k0 - 1) * w1s_1_[iw1s] + m * phi_as[iw1s]
                s2[iw1, jres] += Sw1s[iw1s] * s3[iw1s] * cos(phase_s)
            end
        end
    end
    
    # Final Fourier transform - fully vectorized with optimal memory access pattern
    @inbounds for ires in 1:KRES
        for jres in 1:KRES
            # Vectorized accumulation over w
            for iw1 in 1:NW
                phi_a = w1_1_[iw1] * om_pr / orbits.Omega_1[iR, iv] - pha_a[iw1] # theta in (2.12)
                phase_source = (ires + k0 - 1) * w1_1_[iw1] + m * phi_a
                s1[ires, jres] += Sw1[iw1] * s2[iw1, jres] * cos(phase_source)
            end
        end
    end
    
return nothing
end

# Legacy function removed - use calculate_single_pi_element_ultra_optimized! for best performance

"""
Exact C code structure match for Pi calculation - follows C code line by line
"""
function calculate_single_pi_element_exact_c_match!(s1, s2, s3, iR, iv, iLs, iIs, orbits, spline_psiz, config)
    
    NW = config.grid.NW
    KRES = config.core.kres
    k0 = get_k0(config)
    m = config.core.m
    beta = config.physics.beta
    
    # Source grid point parameters - exactly matching C code
    # sgnl = orbits.grids.SGNL[iR, iv]
 
    om_pr = orbits.Omega_2[iR, iv]
    w1_1_ = @view orbits.w1[:, iR, iv]
    Sw1 = GridConstruction.trapezoidal_coef(w1_1_)
    
    # Target grid point parameters - exactly matching C code
    # sgnl_s = orbits.grids.SGNL[iLs, iIs]
    om_pr_s = orbits.Omega_2[iLs, iIs]
    
    # Pre-compute w1s_1_ and phi_as_ arrays (matching C code lines 290-295)
    w1s_1_ = zeros(Float64, NW)
    phi_as_ = zeros(Float64, NW)
    
    for iw1s in 1:NW
        w1s_1_[iw1s] = orbits.w1[iw1s, iLs, iIs]
        phi_as_[iw1s] = w1s_1_[iw1s] * om_pr_s / orbits.Omega_1[iLs, iIs] - orbits.pha[iw1s, iLs, iIs]
    end
    
    # Compute trapezoidal coefficients for w1s (matching C code lines 298-303)
    Swt = zeros(Float64, NW-1)
    Sw1s = zeros(Float64, NW)
    
    for iw1s in 1:(NW-1)
        Swt[iw1s] = (w1s_1_[iw1s+1] - w1s_1_[iw1s]) / 2.0
    end
    
    for iw1s in 2:(NW-1)
        Sw1s[iw1s] = Swt[iw1s] + Swt[iw1s-1]
    end
    Sw1s[1] = Swt[1]
    Sw1s[NW] = Swt[NW-1]
    
    # Initialize s2 array - 1D array matching C code storage pattern
    s2_1d = zeros(Float64, NW * KRES)
    
    # Create s2_ array for intermediate storage (matching C code)
    s2_ = zeros(Float64, NW * KRES * KRES)
    
    # Main calculation loop - exactly matching C code structure
    for iw1 in 1:NW
        r_1 = orbits.ra[iw1, iR, iv]
        w1_1 = w1_1_[iw1]
        phi_a = w1_1 * om_pr / orbits.Omega_1[iR, iv] - orbits.pha[iw1, iR, iv]
        
        # Calculate gravitational kernel s3 (matching C code lines 318-340)
        for iw1s in 1:NW
            r_1s = orbits.ra[iw1s, iLs, iIs]
            
            r_min = min(r_1, r_1s)
            r_max = max(r_1, r_1s)
            
            tmp1 = r_min*r_min + r_max*r_max + beta*beta
            zi = 2.0 * r_min * r_max / tmp1
            
            if (1.0 - zi) > 1e-8 && zi > 0.0
                tmp = spline_psiz(zi)
                s3[iw1s] = tmp / sqrt(tmp1)
            else
                if zi <= 0.0
                    s3[iw1s] = 0.0
                else
                    s3[iw1s] = 4.5 / sqrt(tmp1)
                end
            end
        end
        
        # First accumulation into s2 (matching C code lines 343-345)
        for jres in 1:KRES
            for iw1s in 1:NW
                # C indexing: s2[iw1 + jres*NW] with jres from 0 to KRES-1
                # Julia indexing: s2_1d[(jres-1)*NW + iw1] with jres from 1 to KRES
                idx = (jres-1)*NW + iw1
                s2_1d[idx] += Sw1s[iw1s] * s3[iw1s] * cos((jres + k0 - 1) * w1s_1_[iw1s] + m * phi_as_[iw1s])
            end
        end
        
        # Compute s2_ array (matching C code lines 347-351)
        for ires in 1:KRES
            for jres in 1:KRES
                # C indexing: s2_[iw1 + NW*(ires + KRES*jres)]
                idx_s2 = (jres-1)*NW + iw1  # corresponds to s2[iw1 + jres*NW]
                idx_s2_ = iw1 + NW*((ires-1) + KRES*(jres-1))
                s2_[idx_s2_] = s2_1d[idx_s2] * cos((ires + k0 - 1) * w1_1 + m * phi_a)
            end
        end
    end
    
    # Final integration (matching C code lines 354-361)
    fill!(s1, 0.0)
    
    for ires in 1:KRES
        for jres in 1:KRES
            for iw1 in 1:NW
                idx_s2_ = iw1 + NW*((ires-1) + KRES*(jres-1))
                s1[ires, jres] += Sw1[iw1] * s2_[idx_s2_]
            end
        end
    end
    
    return nothing
end


end # module PiElements