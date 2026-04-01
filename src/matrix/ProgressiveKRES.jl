"""
Progressive KRES expansion for iterative eigenvalue refinement.
Implements Option 2: Incremental K matrix expansion.

This module allows computing eigenvalues starting with a small KRES
and progressively increasing it, reusing previously computed K matrix blocks.
"""
module ProgressiveKRES

using LinearAlgebra
using Printf
using Arpack
using Dates

# Local get_k0 function
const get_k0 = (kres::Int) -> -(kres - 1) ÷ 2
"""
Generate log filename with timestamp and parameters.
Format: progressive_NRxNvxNW_beta_gridRc_gridv_YYYYMMDD_HHMMSS.log
"""
function generate_log_filename(config)
    NR = config.grid.NR
    Nv = config.grid.Nv
    NW = config.grid.NW
    beta = config.physics.beta
    grid_Rc = config.grid.radial_grid_type
    grid_v = config.grid.circulation_grid_type
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    return @sprintf("progressive_%dx%dx%d_beta%s_gr%d_gv%d_%s.log", 
                    NR, Nv, NW, beta, grid_Rc, grid_v, timestamp)
end

# Global log file reference for tee output
const _log_file = Ref{Union{IOStream, Nothing}}(nothing)

function tee_print(args...)
    print(stdout, args...)
    if _log_file[] !== nothing
        print(_log_file[], args...)
        flush(_log_file[])
    end
    if isdefined(Main, :LOG_FILE) && Main.LOG_FILE[] !== nothing
        print(Main.LOG_FILE[], args...)
        flush(Main.LOG_FILE[])
    end
end

function tee_println(args...)
    println(stdout, args...)
    # Write to module-local log file if set
    if _log_file[] !== nothing
        println(_log_file[], args...)
        flush(_log_file[])
    end
    # Also write to global log file from Main if available
    if isdefined(Main, :LOG_FILE) && Main.LOG_FILE[] !== nothing
        println(Main.LOG_FILE[], args...)
        flush(Main.LOG_FILE[])
    end
end

function start_logging(config)
    log_dir = config.io.data_path
    mkpath(log_dir)
    log_filename = generate_log_filename(config)
    log_path = joinpath(log_dir, log_filename)
    _log_file[] = open(log_path, "w")
    
    # Print header
    tee_println("# Progressive KRES Log")
    tee_println("# Generated: $(Dates.now())")
    tee_println("# Config: NR=$(config.grid.NR), Nv=$(config.grid.Nv), NW=$(config.grid.NW)")
    tee_println("# Beta: $(config.physics.beta)")
    tee_println("# Radial grid: type=$(config.grid.radial_grid_type), R_min=$(config.grid.R_min), R_max=$(config.grid.R_max)")
    tee_println("# Circulation grid type: $(config.grid.circulation_grid_type)")
    tee_println("#")
    
    return log_path
end

function stop_logging(log_path)
    if _log_file[] !== nothing
        flush(_log_file[])
        close(_log_file[])
        _log_file[] = nothing
        # Log path will be printed at the end of main workflow
    end
end


export progressive_eigenvalue_solve, expand_k_matrix!, kres_sequence, start_logging, stop_logging, tee_println,
       resonance_numbers, new_resonance_pairs


"""
Generate the sequence of KRES values for progressive expansion.
All values are odd.
"""
function kres_sequence(kres_start::Int, kres_max::Int, kres_step::Int)
    @assert isodd(kres_start) "kres_start must be odd"
    @assert isodd(kres_max) "kres_max must be odd"
    @assert iseven(kres_step) "kres_step must be even to keep KRES odd"
    
    return collect(kres_start:kres_step:kres_max)
end


"""
Get the resonance numbers (l values) for a given KRES.
Returns a vector of integers from -k to +k where k = (KRES-1)/2.
"""
function resonance_numbers(kres::Int)
    k = (kres - 1) ÷ 2
    return collect(-k:k)
end


"""
Get the new resonance numbers when expanding from kres_old to kres_new.
Returns the l values that need to be computed (the outer ring).
"""
function new_resonance_numbers(kres_old::Int, kres_new::Int)
    l_old = resonance_numbers(kres_old)
    l_new = resonance_numbers(kres_new)
    return setdiff(l_new, l_old)
end


"""
Get the new resonance pairs (l_i, l_j) when expanding from kres_old to kres_new.
These are the pairs where at least one of l_i or l_j is new.
"""
function new_resonance_pairs(kres_old::Int, kres_new::Int)
    l_old = resonance_numbers(kres_old)
    l_new_only = new_resonance_numbers(kres_old, kres_new)
    l_all = resonance_numbers(kres_new)
    
    pairs = Tuple{Int,Int}[]
    
    # New rows × all columns
    for l_i in l_new_only
        for l_j in l_all
            push!(pairs, (l_i, l_j))
        end
    end
    
    # Old rows × new columns (excluding corners already counted)
    for l_i in l_old
        for l_j in l_new_only
            push!(pairs, (l_i, l_j))
        end
    end
    
    return pairs
end


"""
Map resonance number l to array index given k0.
"""
function l_to_index(l::Int, k0::Int)
    return l - k0 + 1
end


"""
Map array index to resonance number given k0.
"""
function index_to_l(idx::Int, k0::Int)
    return idx + k0 - 1
end


"""
Copy existing K matrix blocks from old K to new K with proper index mapping.
"""
function copy_existing_blocks!(K_new::Matrix{T}, K_old::Matrix{T}, 
                                kres_old::Int, kres_new::Int, N::Int) where T
    k0_old = get_k0(kres_old)
    k0_new = get_k0(kres_new)
    
    l_old = resonance_numbers(kres_old)
    
    for l_i in l_old
        for l_j in l_old
            # Old indices
            ires_old = l_to_index(l_i, k0_old)
            jres_old = l_to_index(l_j, k0_old)
            
            # New indices
            ires_new = l_to_index(l_i, k0_new)
            jres_new = l_to_index(l_j, k0_new)
            
            # Block ranges
            row_old = ((ires_old - 1) * N + 1):(ires_old * N)
            col_old = ((jres_old - 1) * N + 1):(jres_old * N)
            row_new = ((ires_new - 1) * N + 1):(ires_new * N)
            col_new = ((jres_new - 1) * N + 1):(jres_new * N)
            
            # Copy block
            K_new[row_new, col_new] = K_old[row_old, col_old]
        end
    end
end


"""
Add a single Pi4 block's contribution to the K matrix.
pi4_block: N×N matrix of Pi elements for resonance pair (l_i, l_j)
"""
function add_pi_block_to_k!(K::Matrix{<:Real}, pi4_block::Matrix{<:Real},
                            l_i::Int, l_j::Int, k0::Int, N::Int,
                            fs::Matrix{<:Real}, mu1::AbstractMatrix{<:Real}, selfgravity::Real)
    ires = l_to_index(l_i, k0)
    jres = l_to_index(l_j, k0)
    
    # K[(ires-1)*N + j, (jres-1)*N + js] = selfgravity * pi4[j, js, ires, jres] * fs[jres, js] * mu1[js]
    @inbounds for js in 1:N
        coeff = selfgravity * fs[jres, js] * mu1[1, js]
        col_idx = (jres - 1) * N + js
        
        for j in 1:N
            row_idx = (ires - 1) * N + j
            K[row_idx, col_idx] += pi4_block[j, js] * coeff
        end
    end
end


"""
Accumulate sharp_df terms for a single (l_i, l_j) Pi block.
Matches the sharp DF extension used in MatrixCalculator.jl.2.
"""
function add_sharp_df_block_to_k!(K::Matrix{<:Real}, pi4_block::Matrix{<:Real},
                                  l_i::Int, l_j::Int, k0::Int, N::Int,
                                  selfgravity::Real, m::Int,
                                  F0::AbstractMatrix{<:Real}, weight_J::AbstractVector{<:Real},
                                  NR::Int, Nv::Int)
    ires = l_to_index(l_i, k0)
    jres = l_to_index(l_j, k0)

    # Sharp DF formula (from MatrixCalculator.jl.2):
    # K[i_row, i_col] += selfgravity * m * pi4[j, js, ires, jres] * F0[ir_prime, 1] * weight_J[ir_prime]
    # where js = (ir_prime-1)*Nv + 1 (first velocity point for each radial point)
    @inbounds for ir_prime in 1:NR
        js = (ir_prime - 1) * Nv + 1
        coeff = selfgravity * m * F0[ir_prime, 1] * weight_J[ir_prime]
        col_idx = (jres - 1) * N + js

        for j in 1:N
            row_idx = (ires - 1) * N + j
            K[row_idx, col_idx] += pi4_block[j, js] * coeff
        end
    end
end




"""
Compute fs coefficients for K matrix construction.
"""
function compute_fs(::Type{T}, orbit_data, kres::Int, m::Int, NR::Int, Nv::Int) where T
    N = NR * Nv
    k0 = get_k0(kres)
    
    fs = zeros(T, kres, N)
    for ires = 1:kres
        l = ires + k0 - 1
        for iR in 1:NR
            for iv in 1:Nv
                i_ri = (iR - 1) * Nv + iv
                fs[ires, i_ri] = (l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]) * orbit_data.FE[iR, iv] +
                                 m * orbit_data.FL[iR, iv]
            end
        end
    end
    return fs
end


"""
Compute mu1 weights for K matrix construction.
"""
function compute_mu1(::Type{T}, orbit_data, NR::Int, Nv::Int) where T
    N = NR * Nv
    jac = orbit_data.jacobian
    
    radial_weights = orbit_data.grids.radial.weights
    eccentricity_weights = orbit_data.grids.eccentricity.weights
    
    grid_weights = zeros(T, NR, Nv)
    for iR in 1:NR
        for iv in 1:Nv
            grid_weights[iR, iv] = radial_weights[iR] * eccentricity_weights[iR, iv]
        end
    end
    
    SS = grid_weights .* jac
    mu1 = reshape(SS', 1, N)
    return mu1
end


"""
Expand K matrix from kres_old to kres_new.
Returns a new K matrix with the expanded size.
"""
function expand_k_matrix!(K_old::Matrix{T}, kres_old::Int, kres_new::Int, N::Int,
                          compute_pi_block::Function, fs_new::AbstractMatrix{<:Real}, mu1::AbstractMatrix{<:Real},
                          selfgravity::Float64, orbit_data, m::Int; sharp_df::Bool=false, F0=nothing, weight_J=nothing) where T
    
    k0_new = get_k0(kres_new)
    
    # Allocate new K matrix
    K_size_new = kres_new * N
    K_new = zeros(T, K_size_new, K_size_new)
    
    # Copy existing blocks
    tee_println("    Copying existing $(kres_old)×$(kres_old) blocks...")
    copy_existing_blocks!(K_new, K_old, kres_old, kres_new, N)
    
    # Get new resonance pairs to compute
    new_pairs = new_resonance_pairs(kres_old, kres_new)
    tee_println("    Computing $(length(new_pairs)) new resonance pair blocks...")
    
    # Compute and add new blocks
    for (i, (l_i, l_j)) in enumerate(new_pairs)
        pi4_block = compute_pi_block(l_i, l_j)
        add_pi_block_to_k!(K_new, pi4_block, l_i, l_j, k0_new, N, fs_new, mu1, selfgravity)
        
        if sharp_df && F0 !== nothing && weight_J !== nothing
            NR_local = size(orbit_data.Omega_1, 1)
            Nv_local = size(orbit_data.Omega_1, 2)
            add_sharp_df_block_to_k!(K_new, pi4_block, l_i, l_j, k0_new, N,
                                     selfgravity, m, F0, weight_J, NR_local, Nv_local)
        end
        
        if i % 5 == 0 || i == length(new_pairs)
            print("      Block $(i)/$(length(new_pairs))\r")
        end
    end
    tee_println()
    
    # Add diagonal frequency terms for new resonances
    NR = size(orbit_data.Omega_1, 1)
    Nv = size(orbit_data.Omega_1, 2)
    new_l = new_resonance_numbers(kres_old, kres_new)
    
    for l in new_l
        ires = l_to_index(l, k0_new)
        for iR in 1:NR
            for iv in 1:Nv
                i_ri = (iR - 1) * Nv + iv
                is = (ires - 1) * N + i_ri
                K_new[is, is] += l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]
            end
        end
    end
    
    return K_new
end


"""
Compute initial K matrix for the starting KRES.
Uses all resonance pairs from scratch.
"""
function compute_initial_k_matrix(kres::Int, N::Int, compute_pi_block::Function, 
                                   fs::Matrix{T}, mu1::AbstractMatrix{<:Real},
                                   selfgravity::Float64, orbit_data, m::Int; sharp_df::Bool=false, F0=nothing, weight_J=nothing) where T
    
    k0 = get_k0(kres)
    K_size = kres * N
    K = zeros(Float64, K_size, K_size)
    
    l_all = resonance_numbers(kres)
    total_pairs = length(l_all)^2
    
    tee_println("    Computing $(total_pairs) resonance pair blocks...")
    
    count = 0
    for l_i in l_all
        for l_j in l_all
            count += 1
            pi4_block = compute_pi_block(l_i, l_j)
            add_pi_block_to_k!(K, pi4_block, l_i, l_j, k0, N, fs, mu1, selfgravity)
            
            if sharp_df && F0 !== nothing && weight_J !== nothing
                NR_local = size(orbit_data.Omega_1, 1)
                Nv_local = size(orbit_data.Omega_1, 2)
                add_sharp_df_block_to_k!(K, pi4_block, l_i, l_j, k0, N,
                                         selfgravity, m, F0, weight_J, NR_local, Nv_local)
            end
            
            if count % 10 == 0 || count == total_pairs
                print("      Block $(count)/$(total_pairs)\r")
            end
        end
    end
    tee_println()
    
    # Add diagonal frequency terms
    NR = size(orbit_data.Omega_1, 1)
    Nv = size(orbit_data.Omega_1, 2)
    
    for l in l_all
        ires = l_to_index(l, k0)
        for iR in 1:NR
            for iv in 1:Nv
                i_ri = (iR - 1) * Nv + iv
                is = (ires - 1) * N + i_ri
                K[is, is] += l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]
            end
        end
    end
    
    return K
end


"""
Solve eigenvalues using full LAPACK decomposition.
Returns eigenvalues sorted by growth rate (descending).
"""
function solve_eigenvalues_full(K::Matrix{<:Real}, m::Int, num_output::Int)
    tee_println("  Computing full eigenvalue decomposition (LAPACK)...")
    
    # Convert to Float64 for eigenvalue computation stability
    K_f64 = Float64.(K)
    eigenvalues = eigvals(K_f64)
    
    # Sort by growth rate (imaginary part) descending
    sort_idx = sortperm(imag.(eigenvalues), rev=true)
    eigenvalues_sorted = eigenvalues[sort_idx]
    
    # Return top num_output eigenvalues
    n_return = min(num_output, length(eigenvalues_sorted))
    return Vector{ComplexF64}(eigenvalues_sorted[1:n_return]), nothing
end


"""
Solve eigenvalues using iterative solver with shift-invert.
"""
function solve_eigenvalues_iterative(K::Matrix{<:Real}, shift_Omega_p::Vector{Float64},
                                      shift_gamma::Vector{Float64}, m::Int;
                                      krylov_dim::Int=50, tol::Float64=1e-8)
    n = size(K, 1)
    K_complex = Complex{Float64}.(K)
    
    all_vals = ComplexF64[]
    all_vecs = Vector{ComplexF64}[]
    
    num_shifts = length(shift_Omega_p)
    ncv = min(max(krylov_dim, 3), n - 1)
    
    for i in 1:num_shifts
        sigma = m * shift_Omega_p[i] + shift_gamma[i] * im
        
        try
            vals, vecs, nconv, niter, nmult, resid = eigs(
                K_complex, 
                nev=1,
                which=:LM,
                sigma=sigma,
                ncv=ncv,
                tol=tol,
                maxiter=300
            )
            
            if nconv > 0
                val = vals[1]
                Omega_p = real(val) / m
                gamma = imag(val)
                print("      Shift $i: Ωₚ = $(round(Omega_p, digits=4)), γ = $(round(gamma, digits=4))")
                
                # Check for duplicates
                is_duplicate = false
                for existing_val in all_vals
                    if abs(val - existing_val) < 1e-6 * abs(existing_val)
                        is_duplicate = true
                        break
                    end
                end
                
                if is_duplicate
                    tee_println(" (duplicate)")
                else
                    tee_println()
                    push!(all_vals, val)
                    push!(all_vecs, vecs[:, 1])
                end
            end
        catch e
            tee_println("      Shift $i failed: $(typeof(e))")
        end
    end
    
    if isempty(all_vals)
        error("No eigenvalues converged")
    end
    
    # Sort by growth rate
    perm = sortperm(imag.(all_vals), rev=true)
    vals_sorted = all_vals[perm]
    vecs_sorted = all_vecs[perm]
    
    # Convert to matrix
    eigenvectors = zeros(ComplexF64, n, length(vals_sorted))
    for i in 1:length(vals_sorted)
        eigenvectors[:, i] = vecs_sorted[i]
    end
    
    return Vector{ComplexF64}(vals_sorted), eigenvectors
end


"""
Check if eigenvalues have converged between iterations.
"""
function check_convergence(eigenvalues_new::Vector{ComplexF64}, 
                           eigenvalues_old::Vector{ComplexF64},
                           tol::Float64)
    n = min(length(eigenvalues_new), length(eigenvalues_old))
    if n == 0
        return false, Inf
    end
    
    # Sort both by growth rate for comparison
    sort_new = sortperm(imag.(eigenvalues_new), rev=true)
    sort_old = sortperm(imag.(eigenvalues_old), rev=true)
    
    max_diff = 0.0
    for i in 1:n
        diff = abs(eigenvalues_new[sort_new[i]] - eigenvalues_old[sort_old[i]])
        max_diff = max(max_diff, diff)
    end
    
    return max_diff < tol, max_diff
end


"""
Print eigenvalue comparison table between two KRES iterations.
Modes are sorted by Ωₚ (decreasing) for stable tracking across iterations.
"""
function print_eigenvalue_comparison(eigenvalues_old::Vector{ComplexF64},
                                     eigenvalues_new::Vector{ComplexF64},
                                     kres_old::Int, kres_new::Int, m::Int)
    # Sort by Ωₚ (decreasing) for comparison - this gives stable mode tracking
    sort_new = sortperm(imag.(eigenvalues_new), rev=true)
    sort_old = sortperm(imag.(eigenvalues_old), rev=true)
    
    tee_println("\n  ┌───────────────────────────────────────────────────────────────────────────┐")
    tee_println("  │ Eigenvalue comparison: KRES=$(kres_old) → $(kres_new)  (sorted by γ)")
    tee_println("  ├───────┬────────────┬────────────┬───────────┬───────────┬─────────┬─────────┤")
    tee_println("  │ Rank  │  Ωₚ (old)  │  Ωₚ (new)  │  γ (old)  │  γ (new)  │   ΔΩₚ   │    Δγ   │")
    tee_println("  ├───────┼────────────┼────────────┼───────────┼───────────┼─────────┼─────────┤")
    
    n = min(length(eigenvalues_old), length(eigenvalues_new), 10)
    for i in 1:n
        v_old = eigenvalues_old[sort_old[i]]
        v_new = eigenvalues_new[sort_new[i]]
        Ωp_old_i = real(v_old) / m
        Ωp_new_i = real(v_new) / m
        γ_old = imag(v_old)
        γ_new = imag(v_new)
        dΩp = abs(Ωp_new_i - Ωp_old_i)
        dγ = abs(γ_new - γ_old)
        
        tee_println(@sprintf("  │ %4d  │ %10.4f │ %10.4f │ %9.4f │ %9.4f │ %7.4f │ %7.4f │", 
                         i, Ωp_old_i, Ωp_new_i, γ_old, γ_new, dΩp, dγ))
    end
    tee_println("  └───────┴────────────┴────────────┴───────────┴───────────┴─────────┴─────────┘")
end


"""
Print final eigenvalue table.
Eigenvalues are sorted by γ (descending).
"""
function print_eigenvalue_table(eigenvalues::Vector{ComplexF64}, kres::Int, m::Int)
    # Sort by Ωₚ (decreasing) for consistent ordering
    Ωp_vals = real.(eigenvalues) ./ m
    sort_idx = sortperm(Ωp_vals, rev=true)
    
    # Find which one has max growth rate
    γ_vals = imag.(eigenvalues)
    max_γ_idx = argmax(γ_vals)
    
    tee_println("\n  Eigenvalues (KRES=$kres, sorted by Ωₚ):")
    tee_println("  ──────────────────────────────────────────────────")
    tee_println("  Rank │           Ωₚ │            γ")
    tee_println("  ──────────────────────────────────────────────────")
    
    n = length(eigenvalues)
    for i in 1:n
        idx = sort_idx[i]
        v = eigenvalues[idx]
        Ωp = real(v) / m
        γ = imag(v)
        suffix = (idx == max_γ_idx) ? "  ← max γ" : ""
        tee_println(@sprintf("  %4d │ %12.4f │ %12.4f%s", i, Ωp, γ, suffix))
    end
    tee_println("  ──────────────────────────────────────────────────")
end

"""
Print summary table of dominant mode for each KRES.
"""
function print_kres_summary(kres_history::Vector{Int}, eigenvalues_history::Vector{Vector{ComplexF64}}, m::Int)
    tee_println("\n" * "="^70)
    tee_println("PROGRESSIVE KRES SUMMARY - Dominant Mode")
    tee_println("="^70)
    tee_println("  KRES │           Ωₚ │            γ │    K size")
    tee_println("  ─────┼──────────────┼──────────────┼───────────")
    
    for (i, kres) in enumerate(kres_history)
        eigenvalues = eigenvalues_history[i]
        sort_idx = sortperm(imag.(eigenvalues), rev=true)
        dominant = eigenvalues[sort_idx[1]]
        Ωp = real(dominant) / m
        γ = imag(dominant)
        K_size = kres * 451  # N = NR * Nv, but we don't have it here, approximate
        tee_println(@sprintf("  %4d │ %12.4f │ %12.4f │ %9d", kres, Ωp, γ, kres))
    end
    tee_println("  ─────┴──────────────┴──────────────┴───────────")
end


"""
Save eigenvalue history to file.
"""
function save_eigenvalue_history(filename::String, kres_history::Vector{Int}, 
                                  eigenvalues_history::Vector{Vector{ComplexF64}}, 
                                  m::Int, config)
    open(filename, "w") do f
        # Header
        println(f, "# Progressive KRES Eigenvalue Results")
        println(f, "# Generated: $(Dates.now())")
        println(f, "# Grid: $(config.grid.NR)×$(config.grid.Nv)")
        println(f, "# m = $(m)")
        println(f, "#")
        
        for (i, kres) in enumerate(kres_history)
            eigenvalues = eigenvalues_history[i]
            sort_idx = sortperm(imag.(eigenvalues), rev=true)
            
            println(f, "# KRES = $kres")
            println(f, "# Rank, Omega_p, gamma")
            
            for (rank, idx) in enumerate(sort_idx)
                v = eigenvalues[idx]
                Ωp = real(v) / m
                γ = imag(v)
                println(f, @sprintf("%d, %.8f, %.8f", rank, Ωp, γ))
            end
            println(f, "#")
        end
    end
end


"""
Main progressive eigenvalue solve function.

Progressively expands KRES from kres_start to kres_max, reusing
previously computed K matrix blocks and using eigenvalues from
previous iterations as initial guesses.

Parameters:
- config: PME configuration with eigenvector settings
- orbit_data: Orbital trajectory data
- model: Galaxy model results
- psi_z, z: Elliptic function lookup table
- compute_pi_block: Function (l_i, l_j) -> Matrix{Float64} that computes Pi for resonance pair
- K_precomputed: Optional pre-computed K matrix (used when progressive=false to avoid recomputation)

Returns: (eigenvalues, eigenvectors, final_kres, log_path)
"""
function progressive_eigenvalue_solve(config, orbit_data, model, psi_z, z, compute_pi_block::Function; K_precomputed::Union{Matrix, Nothing}=nothing)
    
    # Logging header (main log file already started by run_pme_gpu.jl)
    log_path = "managed_by_run_pme_gpu"

    NR, Nv = config.grid.NR, config.grid.Nv
    N = NR * Nv
    m = config.core.m
    selfgravity = config.physics.selfgravity
    
    # Get progressive settings
    kres_start = config.eigenvectors.kres_start
    kres_max = config.eigenvectors.kres_max
    kres_step = config.eigenvectors.kres_step
    conv_tol = config.eigenvectors.convergence_tol
    use_iterative = config.eigenvectors.iterative
    # Compute enough eigenvalues for both output and display
    num_output = max(config.eigenvectors.num_output, config.io.max_display_modes)
    
    # Generate KRES sequence
    kres_seq = kres_sequence(kres_start, kres_max, kres_step)
    
    tee_println("\n" * "="^70)
    tee_println("PROGRESSIVE KRES EIGENVALUE SOLVE")
    tee_println("="^70)
    tee_println("  KRES sequence: $(kres_seq)")
    tee_println("  Convergence tolerance: $conv_tol")
    tee_println("  Grid: $(NR)×$(Nv), N=$(N)")
    tee_println("  Solver: $(use_iterative ? "iterative (Arpack)" : "full (LAPACK)")")
    tee_println("  Collecting top $num_output modes per KRES")
    
    # Pre-compute mu1 (same for all KRES)
    # Determine precision type from GPU config (default Float64 for CPU-only)
    T = hasfield(typeof(config), :gpu) && !config.gpu.precision_double ? Float32 : Float64
    mu1 = compute_mu1(T, orbit_data, NR, Nv)
    

    # Sharp DF setup (if enabled)
    sharp_df = hasfield(typeof(config), :phase_space) ? config.phase_space.sharp_df : false
    weight_J = nothing
    if sharp_df
        if config.phase_space.full_space
            error("sharp_df=true requires full_space=false (unidirectional disk)")
        end

        max_v0_deviation = maximum(abs.(orbit_data.grids.eccentricity.circulation_grid[:, 1]))
        if max_v0_deviation > 1e-15
            error("sharp_df=true requires v[iR,1]=0 for all radial points. Max deviation: $(max_v0_deviation)")
        end

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

        tee_println("  Sharp DF mode: enabled")
    end
    # Initial shifts from config (for iterative mode)
    shift_Omega_p = copy(config.eigenvectors.shift_Omega_p)
    shift_gamma = copy(config.eigenvectors.shift_gamma)
    
    K = nothing
    eigenvalues = nothing
    eigenvalues_prev = nothing
    eigenvectors = nothing
    kres_old = 0
    
    # History arrays for collecting results
    kres_history = Int[]
    eigenvalues_history = Vector{ComplexF64}[]
    
    for (iter, kres) in enumerate(kres_seq)
        tee_println("\n" * "-"^70)
        tee_println("  Iteration $iter: KRES = $kres (l ∈ {$(get_k0(kres)), ..., $(get_k0(kres) + kres - 1)})")
        tee_println("-"^70)
        
        # Compute fs for this KRES
        fs = compute_fs(T, orbit_data, kres, m, NR, Nv)
        
        if iter == 1 && K_precomputed !== nothing
            # Use pre-computed K matrix (single KRES mode, no recomputation needed)
            tee_println("  Using pre-computed K matrix...")
            K = K_precomputed
        elseif iter == 1
            # First iteration: compute full K matrix
            tee_println("  Computing initial K matrix...")
            K = compute_initial_k_matrix(kres, N, compute_pi_block, fs, mu1, selfgravity, orbit_data, m; sharp_df=sharp_df, F0=orbit_data.F0, weight_J=weight_J)
        else
            # Expand K matrix from previous iteration
            tee_println("  Expanding K matrix from KRES=$(kres_old) to KRES=$(kres)...")
            K = expand_k_matrix!(K, kres_old, kres, N, compute_pi_block, fs, mu1, selfgravity, orbit_data, m; sharp_df=sharp_df, F0=orbit_data.F0, weight_J=weight_J)
        end
        
        K_size = size(K, 1)
        K_mem = K_size * K_size * sizeof(eltype(K)) / 1e9
        tee_println("    K matrix size: $(K_size) × $(K_size) ($(round(K_mem, digits=2)) GB)")
        
        # Solve eigenvalues
        if use_iterative
            tee_println("  Solving eigenvalues with $(length(shift_Omega_p)) shift points...")
            eigenvalues, eigenvectors = solve_eigenvalues_iterative(
                K, shift_Omega_p, shift_gamma, m;
                krylov_dim=config.eigenvectors.krylov_dim,
                tol=config.eigenvectors.tol
            )
        else
            eigenvalues, eigenvectors = solve_eigenvalues_full(K, m, num_output)
        end
        tee_println("    Found $(length(eigenvalues)) eigenvalues")
        
        # Store results for this KRES
        
        # Print eigenvalue table for first iteration
        if iter == 1
            print_eigenvalue_table(eigenvalues, kres, m)
        end
        push!(kres_history, kres)
        push!(eigenvalues_history, copy(eigenvalues))
        
        # Print DOMINANT MODE vs KRES table (cumulative, per iteration)
        print_kres_summary_table(kres_history, eigenvalues_history, m, N)
        
        if eigenvalues_prev !== nothing
            print_eigenvalue_comparison(eigenvalues_prev, eigenvalues, kres_old, kres, m)
        end
        
        
        # Check convergence (if not first iteration)
        if eigenvalues_prev !== nothing
            converged, max_diff = check_convergence(eigenvalues, eigenvalues_prev, conv_tol)
            tee_println("    Max eigenvalue change: $(round(max_diff, sigdigits=3))")
            
            if converged
                tee_println("\n  ✓ Converged! Max change $(round(max_diff, sigdigits=3)) < tolerance $conv_tol")
                break
            end
        end
        
        # Update shifts for next iteration using current eigenvalues (for iterative mode)
        if use_iterative
            n_shifts = min(length(eigenvalues), length(shift_Omega_p))
            sort_idx = sortperm(imag.(eigenvalues), rev=true)
            for i in 1:n_shifts
                shift_Omega_p[i] = real(eigenvalues[sort_idx[i]]) / m
                shift_gamma[i] = imag(eigenvalues[sort_idx[i]])
            end
        end
        
        eigenvalues_prev = copy(eigenvalues)
        kres_old = kres
    end
    
    # Save eigenvalue history to file
    output_dir = config.io.data_path
    mkpath(output_dir)
    history_file = joinpath(output_dir, "progressive_kres_eigenvalues.csv")
    save_eigenvalue_history(history_file, kres_history, eigenvalues_history, m, config)
    tee_println("\n  Eigenvalue history saved to: $history_file")
    
    # Print summary table
    print_kres_summary_table(kres_history, eigenvalues_history, m, N)
    
    # Print final eigenvalue table
    print_eigenvalue_table(eigenvalues, kres_history[end], m)
    
    # Logging handled by run_pme_gpu.jl

    return eigenvalues, eigenvectors, kres_history[end], log_path
end


"""
Print summary table of dominant mode for each KRES (improved version).
"""
function print_kres_summary_table(kres_history::Vector{Int}, eigenvalues_history::Vector{Vector{ComplexF64}}, m::Int, N::Int)
    tee_println("\n" * "="^70)
    tee_println("DOMINANT MODE vs KRES")
    tee_println("="^70)
    tee_println("  KRES │           Ωₚ │            γ │    K matrix size")
    tee_println("  ─────┼──────────────┼──────────────┼──────────────────")
    
    for (i, kres) in enumerate(kres_history)
        eigenvalues = eigenvalues_history[i]
        sort_idx = sortperm(imag.(eigenvalues), rev=true)
        dominant = eigenvalues[sort_idx[1]]
        Ωp = real(dominant) / m
        γ = imag(dominant)
        K_size = kres * N
        tee_println(@sprintf("  %4d │ %12.4f │ %12.4f │ %8d × %d", kres, Ωp, γ, K_size, K_size))
    end
    tee_println("  ─────┴──────────────┴──────────────┴──────────────────")
end


end # module ProgressiveKRES
