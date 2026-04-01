# src/matrix/KEigenvalues.jl
"""
Calculates the eigenvalues of the K matrix.
Supports both full eigenvalue decomposition (LAPACK) and iterative methods (KrylovKit).
"""
module KEigenvalues

using LinearAlgebra
using CSV
using DataFrames
using Printf
using Dates
using KrylovKit
using Arpack
using ..Configuration: PMEConfig, EigenvectorConfig
# get_k0 is computed from kres
const get_k0 = (config) -> -(config.core.kres - 1) ÷ 2

export calculate_eigenvalues, save_eigenvectors_and_grid

"""
    format_complex_array(arr::AbstractArray{<:Complex}; digits=4)

Format an array of complex numbers using mathematical 'i' notation instead of Julia's 'im'.
"""
function format_complex_array(arr::AbstractArray{<:Complex}; digits=4)
    formatted = String[]
    for z in arr
        real_part = real(z)
        imag_part = imag(z)
        
        if imag_part == 0
            push!(formatted, string(real_part))
        elseif real_part == 0
            if imag_part == 1
                push!(formatted, "i")
            elseif imag_part == -1
                push!(formatted, "-i")
            else
                push!(formatted, "$(imag_part)i")
            end
        else
            sign_str = imag_part >= 0 ? " + " : " - "
            abs_imag = abs(imag_part)
            if abs_imag == 1
                push!(formatted, "$(real_part)$(sign_str)i")
            else
                push!(formatted, "$(real_part)$(sign_str)$(abs_imag)i")
            end
        end
    end
    return formatted
end

"""
    save_eigenvectors_and_grid(eigenvalues, eigenvectors, orbit_data, config::PMEConfig, model_type::String)

Save eigenvectors and grid data to CSV files in the specified format.

Eigenvectors are saved as separate files for each mode with format:
modeltype_grid_beta_vectornum_omegap_gamma.csv

Grid data is saved as a single file with format:
modeltype_grid.csv
"""
function save_eigenvectors_and_grid(eigenvalues, eigenvectors, orbit_data, config::PMEConfig, model_type::String, timestamp::String)
    if eigenvectors === nothing
        @warn "Eigenvectors not computed, cannot save to files"
        return
    end
    
    # Create output directory
    eigenvector_dir = joinpath(config.io.data_path, "results", "eigenvectors")
    mkpath(eigenvector_dir)
    
    
    # Grid dimensions
    NR = config.grid.NR
    Nv = config.grid.Nv
    KRES = config.core.kres
    k0 = get_k0(config)
    m = config.core.m
    beta = config.physics.beta
    
    # Create column names for resonance terms: k0, k0+1, ..., k0+KRES-1
    resonance_columns = [string(k0 + i) for i in 0:(KRES-1)]
    
    # Sort eigenvalues by growth rate (descending) to get most unstable first
    sorted_indices = sortperm(imag(eigenvalues), rev=true)
    
    # Save each eigenvector as a separate file
    for (vector_num, idx) in enumerate(sorted_indices[1:min(config.eigenvectors.num_output, length(eigenvalues))])
        eigenvalue = eigenvalues[idx]
        eigenvector = eigenvectors[:, idx]
        
        # Calculate pattern speed and growth rate
        Ωₚ = real(eigenvalue) / m
        γ = imag(eigenvalue)
        
        # Generate common filename components
        grid_str = "$(NR)x$(Nv)x$(KRES)"
        beta_str = "beta$(beta)"
        omegap_str = string(round(Ωₚ, digits=4))
        gamma_str = string(round(γ, digits=4))
        
        # Check if sharp_df mode is enabled
        sharp_df = config.phase_space.sharp_df        
        
        # Standard mode: Reshape eigenvector from (NR*Nv*KRES,) to (NR*Nv, KRES)
        eigenvector_matrix = reshape(eigenvector, (NR*Nv, KRES))
        
        # Create DataFrame
        df_eigenvector = DataFrame(eigenvector_matrix, resonance_columns)
        
        # Save eigenvector
        filename = "$(model_type)_$(grid_str)_$(timestamp)_$(beta_str)_$(vector_num)_$(omegap_str)_$(gamma_str).csv"
        filepath = joinpath(eigenvector_dir, filename)
        CSV.write(filepath, df_eigenvector)
    end
    
    # Save grid data
    # Extract grid data - flatten in MATLAB order (iR outer, iv inner)
    Rc_flat = Float64[]
    v_iR_flat = Float64[]
    E_flat = Float64[]
    L_m_flat = Float64[]
    Ir_flat = Float64[]
    Omega_1_flat = Float64[]
    Omega_2_flat = Float64[]
    R_1_flat = Float64[]
    R_2_flat = Float64[]
    F0_flat = Float64[]
    FE_flat = Float64[]
    FL_flat = Float64[]
    
    for iR in 1:NR
        for iv in 1:Nv
            push!(Rc_flat, orbit_data.grids.Rc[iR, iv])
            push!(v_iR_flat, orbit_data.grids.eccentricity.circulation_grid[iR, iv])
            push!(E_flat, orbit_data.grids.E[iR, iv])
            push!(L_m_flat, orbit_data.grids.L_m[iR, iv])
            push!(Ir_flat, orbit_data.Ir[iR, iv])
            push!(Omega_1_flat, orbit_data.Omega_1[iR, iv])
            push!(Omega_2_flat, orbit_data.Omega_2[iR, iv])
            push!(R_1_flat, orbit_data.grids.R1[iR, iv])
            push!(R_2_flat, orbit_data.grids.R2[iR, iv])
            push!(F0_flat, orbit_data.F0[iR, iv])
            push!(FE_flat, orbit_data.FE[iR, iv])
            push!(FL_flat, orbit_data.FL[iR, iv])
        end
    end
    
    # Create DataFrame for grid data
    df_grid = DataFrame(
        Rc = Rc_flat,
        v_iR = v_iR_flat,
        E = E_flat,
        L_m = L_m_flat,
        Ir = Ir_flat,
        Omega_1 = Omega_1_flat,
        Omega_2 = Omega_2_flat,
        R_1 = R_1_flat,
        R_2 = R_2_flat,
        F0 = F0_flat,
        FE = FE_flat,
        FL = FL_flat
    )
    
    # Generate grid filename
    grid_str = "$(NR)x$(Nv)"
    grid_filename = "$(model_type)_$(grid_str)_$(timestamp).csv"
    grid_filepath = joinpath(eigenvector_dir, grid_filename)
    
    # Save grid data to CSV
    CSV.write(grid_filepath, df_grid)
end

"""
    calculate_eigenvalues_iterative(K::Matrix{Float64}, config::EigenvectorConfig)

Use KrylovKit's Arnoldi iteration to find eigenvalues with largest imaginary part.
Much faster than full eigenvalue decomposition for large matrices when only a few
eigenvalues are needed.
"""
function calculate_eigenvalues_iterative(K::Matrix{<:Real}, config::EigenvectorConfig, m::Int=2)
    n = size(K, 1)
    num_shifts = length(config.shift_Omega_p)
    
    # Convert K to complex for shift-invert with complex shift
    K_complex = Complex{Float64}.(K)
    
    # Collect all eigenvalues and eigenvectors from all shift points
    all_vals = ComplexF64[]
    all_vecs = Vector{ComplexF64}[]
    
    ncv = min(max(config.krylov_dim, 3), n - 1)
    
    println("  Searching for $num_shifts eigenvalues...")
    
    for i in 1:num_shifts
        # σ = m * Ωₚ + i * γ
        sigma = m * config.shift_Omega_p[i] + config.shift_gamma[i] * im
        
        try
            vals, vecs, nconv, niter, nmult, resid = eigs(
                K_complex, 
                nev=1,  # Find 1 eigenvalue per shift
                which=:LM,
                sigma=sigma,
                ncv=ncv,
                tol=config.tol,
                maxiter=300
            )
            
            if nconv > 0
                val = vals[1]
                Omega_p = real(val) / m
                gamma = imag(val)
                # Print as found
                print("    $i: Ωₚ = $(round(Omega_p, digits=4)), γ = $(round(gamma, digits=4))")
                
                # Check for duplicates (within tolerance)
                is_duplicate = false
                for existing_val in all_vals
                    if abs(val - existing_val) < 1e-6 * abs(existing_val)
                        is_duplicate = true
                        break
                    end
                end
                
                if is_duplicate
                    println(" (duplicate, skipped)")
                else
                    println()
                    push!(all_vals, val)
                    push!(all_vecs, vecs[:, 1])
                end
            end
        catch e
            @warn "Shift $i failed: $e"
        end
    end
    
    if isempty(all_vals)
        error("No eigenvalues converged for any shift point")
    end
    
    # Sort by imaginary part (growth rate) descending
    perm = sortperm(imag.(all_vals), rev=true)
    vals_sorted = all_vals[perm]
    vecs_sorted = all_vecs[perm]
    
    # Convert to matrix format
    num_found = length(vals_sorted)
    eigenvectors = zeros(ComplexF64, n, num_found)
    for i in 1:num_found
        eigenvectors[:, i] = vecs_sorted[i]
    end
    
    println("  Found $num_found unique eigenvalues")
    
    return Vector{ComplexF64}(vals_sorted), eigenvectors
end

"""
    calculate_eigenvalues(K::Matrix{Float64}, config::EigenvectorConfig; blas_threads::Union{Int,String,Nothing}=nothing)

Calculate the eigenvalues and (optionally) eigenvectors of the real K matrix.
Returns complex eigenvalues and selected number of eigenvectors.

Uses iterative solver (KrylovKit) by default for efficiency. Set config.eigenvectors.iterative=false
to use full LAPACK eigenvalue decomposition.

Optional arguments:
- blas_threads: Number of BLAS threads (Int), "auto" for all available, or nothing for default
"""
function calculate_eigenvalues(K::Matrix{<:Real}, config::EigenvectorConfig, m::Int=2; blas_threads::Union{Int,String,Nothing}=nothing)

    # Set BLAS threads if specified
    if blas_threads !== nothing
        BLAS.set_num_threads(blas_threads)
    end

    # Use iterative solver only when explicitly requested via config
    if config.iterative
        # Iterative solver (KrylovKit) - recommended for matrices > 20000x20000
        eigenvalues, eigenvectors = calculate_eigenvalues_iterative(K, config, m)
        
        if config.compute
            return eigenvalues, eigenvectors
        else
            return eigenvalues, nothing
        end
    else
        # Full eigenvalue decomposition (LAPACK) - default
        if config.compute
            eigen_decomp = eigen(K)
            eigenvalues = eigen_decomp.values
            eigenvectors = eigen_decomp.vectors
            return eigenvalues, eigenvectors
        else
            eigenvalues = eigvals(K)
            return eigenvalues, nothing
        end
    end
end

end # module KEigenvalues
