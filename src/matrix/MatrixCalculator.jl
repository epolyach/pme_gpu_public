# src/matrix/MatrixCalculator.jl

"""
High-performance K matrix construction optimized for 64-core AMD systems.
Constructs the K matrix, a core component for determining galactic normal modes.
This is an optimized Julia port of the logic from `JJ_eRc_modes.m`.
"""
module MatrixCalculator

using Base.Threads
using Printf
using ..AbstractModel
# get_k0 is computed from kres
const get_k0 = (config) -> -(config.core.kres - 1) ÷ 2
using LinearAlgebra
using ProgressMeter
using SharedArrays
using LoopVectorization
using ..ProgressUtils: print_progress_bar

export construct_k_matrix


function construct_k_matrix(config, model::ModelResults, orbit_data, pi4::AbstractArray{<:Real, 4})
    NR, Nv = config.grid.NR, config.grid.Nv
    N = NR * Nv
    KRES = config.core.kres
    k0 = get_k0(config)
    m = config.core.m
    sharp_df = config.phase_space.sharp_df
    
# Get selfgravity parameter from config.physics (loaded from model file's [physics] section)
    selfgravity = config.physics.selfgravity

    # Determine precision based on config
    T = config.cpu.precision_double ? Float64 : Float32
    
    # Calculate fs coefficients
    fs = zeros(T, KRES, N)
    for ires = 1:KRES
        l = ires + k0 - 1
        # Match MATLAB loop order: iR (radial) outer, iv (circulation) inner
        for iR in 1:NR  # iR in MATLAB
            for iv in 1:Nv  # ie in MATLAB
                i_ri = (iR - 1) * Nv + iv  # (iR-1)*Ne + ie in MATLAB
                # * orbit_data.grids.SGNL[iR, iv]
                fs[ires, i_ri] = (l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]) * orbit_data.FE[iR, iv] +
                                 m * orbit_data.FL[iR, iv]
            end
        end
    end

    # Use Jacobian from orbit_data (calculated analytically)
    jac = orbit_data.jacobian
    
    # Calculate mu1 following MATLAB: SS = repmat(S_RC',1,Ne) .* S_e .*Jac; mu1 = reshape(SS', 1, n)
    if hasfield(typeof(orbit_data), :grid_weights)  # Distributed mode with precomputed grid_weights
        grid_weights = orbit_data.grid_weights
    else  # Local mode with OrbitData struct - compute grid_weights from grids
        # Compute grid weights product: S_RC' * S_e (radial weights * eccentricity weights)
        radial_weights = orbit_data.grids.radial.weights
        eccentricity_weights = orbit_data.grids.eccentricity.weights
        
        # Create 2D grid_weights following MATLAB logic: repmat(S_RC',1,Ne) .* S_e
        grid_weights = zeros(T, NR, Nv)
        for iR in 1:NR
            for iv in 1:Nv
                grid_weights[iR, iv] = radial_weights[iR] * eccentricity_weights[iR, iv]
            end
        end
    end
    
    # Create SS matrix using the precomputed product and jacobian
    SS = grid_weights .* jac
    mu1 = reshape(SS', 1, N)

    # Output mu1 in binary format to data directory
    data_dir = config.io.data_path
    mkpath(joinpath(data_dir, "binary"))
    mu1_file = joinpath(data_dir, "binary", "mu1.bin")
    open(mu1_file, "w") do file
        write(file, mu1)
    end
    
    # Validation for sharp_df mode
    if sharp_df
        if config.phase_space.full_space
            error("sharp_df=true requires full_space=false (unidirectional disk)")
        end
        
        # Check that first velocity grid point is at v=0
        max_v0_deviation = maximum(abs.(orbit_data.grids.eccentricity.circulation_grid[:, 1]))
        if max_v0_deviation > 1e-15
            error("sharp_df=true requires v[iR,1]=0 for all radial points. Max deviation: $max_v0_deviation")
        end
        
        println("✓ Sharp DF validation passed: full_space=false, v[iR,1]=0")
    end
    
    # Construct K matrix
    n_res = KRES
    K_size = n_res * N
    println("K matrix size = $(K_size) x $(K_size)")
    K = zeros(T, K_size, K_size)

    # Apply selfgravity scaling to the DF derivative combination (off-diagonal terms)
    # K_ff block - existing code
    for ires_ in 1:KRES
        for jres_ in 1:KRES
            for j in 1:N
                for js in 1:N
                    K[(ires_ - 1) * N + j, (jres_ - 1) * N + js] = selfgravity * pi4[j, js, ires_, jres_] * fs[jres_, js] * mu1[js]
                end
            end
        end
    end

    # Add diagonal frequency terms (NOT scaled by selfgravity)
    # K_ff diagonal
    for ires_ in 1:KRES
        l = ires_ + k0 - 1
        # Match MATLAB loop order: iR (radial) outer, iv (circulation) inner
        for iR in 1:NR  # iR in MATLAB
            for iv in 1:Nv  # ie in MATLAB
                i_ri = (iR - 1) * Nv + iv  # (iR-1)*Ne + ie in MATLAB
                is = (ires_ - 1) * N + i_ri
                # * orbit_data.grids.SGNL[iR, iv]
                K[is, is] += l * orbit_data.Omega_1[iR, iv] + m * orbit_data.Omega_2[iR, iv]
            end
        end
    end

    # Sharp DF extensions
    if sharp_df
        # Compute integration weights for dJ' integrals
        weight_J = zeros(T, NR)
        for ir in 1:NR
            if ir == 1
                weight_J[ir] = (orbit_data.Ir[2, 1] - orbit_data.Ir[1, 1]) / 2
            elseif ir == NR
                weight_J[ir] = (orbit_data.Ir[NR, 1] - orbit_data.Ir[NR-1, 1]) / 2
            else
                weight_J[ir] = (orbit_data.Ir[ir+1, 1] - orbit_data.Ir[ir-1, 1]) / 2
            end
        end

        for ires in 1:KRES
            for ir in 1:NR
                for iv in 1:Nv  
                    j = (ir-1)*Nv + iv  
                    i_row = (ires-1)*N + j
                        
                    for jres in 1:KRES
                        for ir_prime in 1:NR
                            js = (ir_prime-1)*Nv + 1
                            i_col = (jres-1)*N + js                        
                            K[i_row, i_col] +=  selfgravity * m * pi4[j, js, ires, jres] * orbit_data.F0[ir_prime, 1] * weight_J[ir_prime]
                        end
                    end
                end
            end
        end

        println("✓ Sharp DF term added")
    end

    return K
end


end # module MatrixCalculator
