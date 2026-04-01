# src/grids/GridConstruction.jl
module GridConstruction

using LinearAlgebra

export simpson_coef, trapezoidal_coef
export RadialGrid, EccentricityGrid, PMEGrids
export create_radial_grid, create_eccentricity_grid, create_pme_grids

"""
Numerical integration coefficients
"""
function simpson_coef(x::AbstractVector{T}) where T<:Real
    n = length(x)
    if n < 3
        error("Simpson's rule requires at least 3 points")
    end
    
    # Check if spacing is uniform
    dx = diff(x)
    if !all(abs.(dx .- dx[1]) .< 1e-12)
        # Non-uniform spacing - use composite rule
        return trapezoidal_coef(x)
    end
    
    # Uniform spacing Simpson's rule
    h = dx[1]
    coef = zeros(T, n)
    
    if isodd(n)
        # Standard Simpson's rule for odd number of points
        coef[1] = h/3
        coef[end] = h/3
        
        for i in 2:2:n-1
            coef[i] = 4*h/3  # Even indices (middle points)
        end
        for i in 3:2:n-2  
            coef[i] = 2*h/3  # Odd indices
        end
    else
        # Even number of points - use Simpson's 3/8 rule for last interval
        for i in 1:n-3
            if i == 1
                coef[i] = h/3
            elseif isodd(i)
                coef[i] = 2*h/3
            else
                coef[i] = 4*h/3
            end
        end
        
        # 3/8 rule for last 4 points
        h38 = 3*h/8
        coef[n-3] += h38
        coef[n-2] = 3*h38
        coef[n-1] = 3*h38  
        coef[n] = h38
    end
    
    return coef
end

function trapezoidal_coef(x::AbstractVector{T}) where T<:Real
    n = length(x)
    coef = zeros(T, n)
    
    if n >= 2
        coef[1] = (x[2] - x[1]) / 2
        coef[end] = (x[end] - x[end-1]) / 2
        
        for i in 2:n-1
            coef[i] = (x[i+1] - x[i-1]) / 2
        end
    end
    
    return coef
end

"""
Radial grid construction with multiple coordinate mappings
"""
struct RadialGrid{T<:Real}
    points::Vector{T}           # Rc_
    weights::Vector{T}          # S_RC  
    grid_type::Int
    parameters::Dict{String,Any}
end

function create_radial_grid(grid_type::Int, NR::Int, grid_params::Dict{String,Any})::RadialGrid{Float64}
# Extract parameters from config
R_min = get(grid_params, "R_min", 0.1)
R_max = get(grid_params, "R_max", 16.0)
alpha = get(grid_params, "alpha", 3.0)

    # Normalized coordinates
    ne = 1:NR
    ue_n = 1 .- 0.5/NR .- (ne .- 1)/NR
    due_n = ue_n[1] - ue_n[2]
    
    if grid_type == 0
        # Exponential grid: Rc = -α log(u)
        Rc_ = -alpha * log.(ue_n)
        S_RC = alpha * due_n ./ ue_n
        
    elseif grid_type == 1
        # Rational grid: Rc = α(1-u)/u  
        Rc_ = alpha * (1 .- ue_n) ./ ue_n
        S_RC = alpha * due_n ./ ue_n.^2
        
    elseif grid_type == 2
        # Linear grid
        Rc_ = range(R_min, R_max, length=NR)
        S_RC = simpson_coef(collect(Rc_))
        
    elseif grid_type == 3
        # Logarithmic grid
        Rc_ = 10 .^ range(log10(R_min), log10(R_max), length=NR)
        uRc_ = log.(Rc_)
        S_RC = trapezoidal_coef(collect(uRc_)) .* Rc_  

    elseif grid_type == 4
        # Logarithmic grid
        Rc_ = 10 .^ range(log10(R_min), log10(R_max), length=NR)
        uRc_ = log.(Rc_)
        S_RC = simpson_coef(collect(uRc_)) .* Rc_
        # S_RC = (uRc_[2] - uRc_[1]) .* Rc_
        
    else
        error("Unknown radial grid type: $grid_type. Supported types: 0 (exponential), 1 (rational), 2 (linear), 3 (logarithmic)")
    end
    
    params = Dict{String,Any}(
        "R_min" => R_min,
        "R_max" => R_max,
        "alpha" => alpha
    )
    
    return RadialGrid{Float64}(collect(Rc_), collect(S_RC), grid_type, params)
end

"""
Eccentricity grid construction with optimization
"""
struct EccentricityGrid{T<:Real}
    points::Matrix{T}           # e(iR,ie) - NR × Nv
    weights::Matrix{T}          # S_e(iR,ie)
    circulation_grid::Matrix{T} # v_iR(iR,ie) - circulation connected to eccentricity
    grid_type::Int
end

function create_eccentricity_grid(grid_type::Int, Nv::Int, radial_grid::RadialGrid{T},
                                 model; 
                                 optimization_on::Bool=true,
                                 full_space::Bool=true,
                                 delta_F0::Real,
                                 alpha_v::Real=3.0,
                                 v_jump_scale::Real=0.05,
                                 Nvt::Union{Int, Nothing}=nothing) where T<:Real
    
    NR = length(radial_grid.points)
    Rc_ = radial_grid.points
    
    l_delta = log(delta_F0)
    S_e = zeros(T, NR, Nv)
    v_iR = zeros(T, NR, Nv) 
    e = zeros(T, NR, Nv)

    for iR in 1:NR
        R_i = Rc_[iR]
        
        # Determine circulation range with optimization
        if optimization_on
            # Get velocity dispersion at this radius
            sigma_r_val = if model.velocity_dispersion !== nothing
                model.velocity_dispersion(R_i)
            else                
                0.1  # Default fallback
            end
            
            Omega_val = model.rotation_frequency(R_i)
            
            if full_space
                v_min = max(-1, 1 - sigma_r_val / (Omega_val * R_i) * sqrt(-2*l_delta))
            else
                v_min = max(0, 1 - sigma_r_val / (Omega_val * R_i) * sqrt(-2*l_delta))
            end
        else
            v_min = full_space ? -1.0 : 0.0
        end
        
        if grid_type == 1
            # Linear circulation grid
            v = range(v_min, 1, length=Nv+1)
            dv = step(v)
            v = v[2:end] .- dv/2  # Midpoint rule
            S_e[iR, :] = fill(dv, Nv)
            
        elseif grid_type == 2
            # Trapezoidal circulation grid
            v = range(v_min, 1, length=Nv)
            S_e[iR, :] = trapezoidal_coef(collect(v))
            
        elseif grid_type == 3
            # Simpson's rule circulation grid
            v = range(v_min, 1, length=Nv)
            S_e[iR, :] = simpson_coef(collect(v))
            
        elseif grid_type == 4
            # Power-stretched circulation grid with clustering near v ~ 0
            # Uses exponent alpha_v (>1) to concentrate points toward the lower end
            # Mapping: v = sign(w) * |w|^alpha_v
            # Choose w-range to honor v_min and domain
            if full_space && v_min < 0
                w_low = -abs(v_min)^(1.0/alpha_v)
            else
                # half-space or v_min >= 0
                w_low = (v_min)^(1.0/alpha_v)
            end
            w = range(w_low, 1.0, length=Nv)
            v = sign.(w) .* abs.(w).^(alpha_v)
            v[end]= 1.0  # Ensure last point is exactly 1.0
            # |dv/dw| = alpha_v * |w|^(alpha_v-1)
            S_e[iR, :] = simpson_coef(collect(w)) .* (alpha_v .* abs.(w).^(alpha_v .- 1.0))

        elseif grid_type == 5
            # Arctan-inverse mapped grid: s = atan(v / v0), v(s) = v0 * tan(s)
            # Here, we use v0 = v_jump_scale from config (typical small value, e.g., 0.05)
            v0 = v_jump_scale
            # Choose s-range to honor v_min and v=1 upper bound
            s_min = atan(v_min / v0)
            s_max = atan(1.0 / v0)
            s = range(s_min, s_max, length=Nv)
            v = v0 .* tan.(s)
            v[end]= 1.0  # Ensure last point is exactly 1.0
            # dv/ds = v0 * sec^2(s) = v0 / cos^2(s)
            dvds = v0 ./(cos.(s).^2)
            S_e[iR, :] = simpson_coef(collect(s)) .* dvds

            if get(ENV, "PME_VERBOSE", "false") == "true"
                if iR <= 5
                    @info "Circulation grid type 5 (atan map) at iR=$iR" v_min=v_min v0=v0 s_min=s_min s_max=s_max v_first=v[1] v_last=v[end]
                end
            end

        elseif grid_type == 6
            # Special grid for MiyamotoTaperTanh: half points from -v_0 to v_0, half from v_0 to 1
            # Trapezoidal weights
            # Get v_0 from model params or helper_functions (model-independent)
            # Get v_0 based on model type
            if model.model_type == "KuzminTaperPoly3L"
                # For KuzminTaperPoly3L, use a default v_0 value since it does not have this parameter
                v_0 = 0.5  # Default value for KuzminTaperPoly3L
            elseif haskey(model.parameters, "v_0")
                v_0 = model.parameters["v_0"]
            elseif hasfield(typeof(model.helper_functions), :v_0)
                v_0 = model.helper_functions.v_0
            else
                error("Grid type 6 requires v_0 parameter")
            end
            if Nvt === nothing
                error("Grid type 6 requires Nvt parameter to be set in configuration")
            end
            
            # Calculate number of points in each half (with overlap at v_0)
            # Number of points: Nvt in taper region, Nv-Nvt+1 in outer region
            
            # Create first segment: -v_0 to v_0
            v_first = collect(range(-v_0, v_0, length=Nvt))
            
            # Create second segment: v_0 to 1.0 (skip first point to avoid doubling v_0)
            v_second = collect(range(v_0, 1.0, length=Nv-Nvt+1))[2:end]
            
            # Combine segments
            v = vcat(v_first, v_second)
            
            # Verify we have exactly Nv points
            if length(v) != Nv
                error("Grid type 6: Expected $Nv points, got $(length(v))")
            end
            
            # Apply trapezoidal weights
            S_e[iR, :] = trapezoidal_coef(v)
            

        elseif grid_type == 7
            # Special grid for MiyamotoTaperTanh/KuzminTaperPoly3: half points from -v_0 to v_0, half from v_0 to 1
            # Simpson's rule weights - calculated separately for each segment then combined
            # Get v_0 based on model type
            if model.model_type == "KuzminTaperPoly3L"
                # For KuzminTaperPoly3L, use a default v_0 value since it does not have this parameter
                v_0 = 0.5  # Default value for KuzminTaperPoly3L
            elseif haskey(model.parameters, "v_0")
                v_0 = model.parameters["v_0"]
            elseif hasfield(typeof(model.helper_functions), :v_0)
                v_0 = model.helper_functions.v_0
            else
                error("Grid type 7 requires v_0 parameter")
            end
            if Nvt === nothing
                error("Grid type 7 requires Nvt parameter to be set in configuration")
            end
            
            # Calculate number of points in each half (with overlap at v_0)
            # Number of points: Nvt in taper region, Nv-Nvt+1 in outer region
            
            # Create first segment: -v_0 to v_0
            v_first = collect(range(-v_0, v_0, length=Nvt))
            
            # Create second segment: v_0 to 1.0
            v_second = collect(range(v_0, 1.0, length=Nv-Nvt+1))
            
            # Calculate Simpson weights for each segment separately
            S_first = simpson_coef(v_first)
            S_second = simpson_coef(v_second)
            
            # Combine: first segment (excluding last point) + second segment
            # The common point at v_0 gets the sum of weights from both segments
            S_combined = zeros(T, Nv)
            S_combined[1:Nvt-1] = S_first[1:end-1]  # First segment without last point
            S_combined[Nvt] = S_first[end] + S_second[1]  # Common point at v_0: add both weights
            S_combined[Nvt+1:end] = S_second[2:end]  # Second segment without first point
            
            # Combine grid points
            v = vcat(v_first, v_second[2:end])
            
            # Verify we have exactly Nv points
            if length(v) != Nv
                error("Grid type 7: Expected $Nv points, got $(length(v))")
            end
            
            S_e[iR, :] = S_combined

        elseif grid_type == 8
            # Special grid for MiyamotoTaperTanh/KuzminTaperPoly3: wide central region [-4v_0, 4v_0], then 4v_0 to 1
            # Trapezoidal weights
            # Get v_0 based on model type
            if model.model_type == "KuzminTaperPoly3L"
                v_0 = 0.5  # Default value for KuzminTaperPoly3L
            elseif haskey(model.parameters, "v_0")
                v_0 = model.parameters["v_0"]
            elseif hasfield(typeof(model.helper_functions), :v_0)
                v_0 = model.helper_functions.v_0
            else
                error("Grid type 8 requires v_0 parameter")
            end
            if 4.0 * v_0 >= 1.0
                error("Grid type 8 requires 4*v_0 < 1.0, got v_0=$(v_0)")
            end
            if Nvt === nothing
                error("Grid type 8 requires Nvt parameter to be set in configuration")
            end

            # Calculate number of points in each half (with overlap at 4v_0)
            # Number of points: Nvt in taper region, Nv-Nvt+1 in outer region

            # Create first segment: -4v_0 to 4v_0
            v_first = collect(range(-4.0*v_0, 4.0*v_0, length=Nvt))

            # Create second segment: 4v_0 to 1.0 (skip first point to avoid doubling 4v_0)
            v_second = collect(range(4.0*v_0, 1.0, length=Nv-Nvt+1))[2:end]

            # Combine segments
            v = vcat(v_first, v_second)

            # Verify we have exactly Nv points
            if length(v) != Nv
                error("Grid type 8: Expected $Nv points, got $(length(v))")
            end

            # Apply trapezoidal weights
            S_e[iR, :] = trapezoidal_coef(v)

        else
            error("Unknown eccentricity grid type: $grid_type. Supported types: 1 (linear), 2 (trapezoidal), 3 (Simpson), 4 (power-stretched), 5 (arctan), 6 (MiyamotoTaperTanh-trapezoidal), 7 (MiyamotoTaperTanh-Simpson), 8 (MiyamotoTaperTanh-wide trapezoidal)")
        end
        
        v_iR[iR, :] = v
        e[iR, :] = 1 .- abs.(v)
    end
    
    return EccentricityGrid{T}(e, S_e, v_iR, grid_type)
end

"""
Combined grid structure
"""
struct PMEGrids{T<:Real}
    radial::RadialGrid{T}
    eccentricity::EccentricityGrid{T}
    
    # Derived quantities
    Rc::Matrix{T}      # Rc values on 2D grid
    R1::Matrix{T}      # Periapsis radii  
    R2::Matrix{T}      # Apoapsis radii
    E::Matrix{T}       # Energies
    L_m::Matrix{T}     # Angular momenta
    L2_m::Matrix{T}    # L² values
    SGNL::Matrix{T}    # Sign of angular momentum
    v::Matrix{T}       # Circulation parameter v = 1 - |e|
end
function create_pme_grids(config, model)::PMEGrids{Float64}
    
    # Create radial grid
    radial_grid_type = config.grid.radial_grid_type
    # Extract grid parameters from config struct
    grid_params = Dict{String,Any}(
        "R_min" => config.grid.R_min,
        "R_max" => config.grid.R_max,
        "alpha" => config.grid.alpha
    )

    radial_grid = create_radial_grid(radial_grid_type, config.grid.NR, grid_params)
    
    # Create eccentricity grid  
    ecc_grid_type = config.grid.circulation_grid_type
    optimization_on = config.phase_space.optimization_on
    full_space = config.phase_space.full_space
    
    # Extract delta_F0 from config
    delta_F0 = config.phase_space.delta_F0
    
    ecc_grid = create_eccentricity_grid(ecc_grid_type, config.grid.Nv, radial_grid, model;
                                       optimization_on=optimization_on, 
                                       full_space=full_space,
                                       delta_F0=delta_F0,
                                        alpha_v=config.grid.alpha_v,
                                        Nvt=config.grid.Nvt,
                                       v_jump_scale=config.grid.v_jump_scale)
    
    NR, Nv = config.grid.NR, config.grid.Nv
    
    # Create 2D coordinate arrays
    Rc = repeat(radial_grid.points, 1, Nv)
    e = ecc_grid.points
    v_iR = ecc_grid.circulation_grid
    
    
    # Calculate circulation parameter v = 1 - |e|
    v = v_iR
    R1 = Rc .* (1.0 .- e)
    R2 = Rc .* (1.0 .+ e)
    
    # Calculate energies using the model potential
    E = zeros(Float64, NR, Nv)
    for iR in 1:NR, ie in 1:Nv

        r1, r2 = R1[iR, ie], R2[iR, ie]
        V1, V2 = model.potential(r1), model.potential(r2)
        E[iR, ie] = (V2 * r2^2 - V1 * r1^2) / (r2^2 - r1^2)

        """
        if !isfinite(E[iR, ie])
            if e[iR, ie] < 1.0
                # For e < 1, use E = Rc*dV(Rc)/2 + V(Rc)
                Rc_val = Rc[iR, ie]
                E[iR, ie] = Rc_val * model.potential_derivative(Rc_val) / 2 + model.potential(Rc_val)
            else
                # Default fallback
                E[iR, ie] = model.potential(Rc[iR, ie])
            end
        end 
        """
        
        # For near-circular orbits, use analytic circular-limit energy to avoid cancellation
        if e[iR, ie] < 1e-10
            Rc_val = Rc[iR, ie]
            E[iR, ie] = Rc_val * model.potential_derivative(Rc_val) / 2 + model.potential(Rc_val)
        end

        # For near-radial orbits (v ≈ 0), use E = V(r2) to avoid V(0)=-Inf issues
        if abs(v[iR, ie]) < 1e-10
            E[iR, ie] = V2
        end

    end


    # Calculate angular momenta
    L2_m = zeros(Float64, NR, Nv)
    for iR in 1:NR, ie in 1:Nv
        r2 = R2[iR, ie]
        E_val = E[iR, ie]
        V2_val = model.potential(r2)
        
        L2_val = 2 * (E_val - V2_val) * r2^2

        # Handle NaN and small negative L2_m values (MATLAB: L2_m(find(L2_m<1e-14))=0;)
        if isnan(L2_val)
            L2_m[iR, ie] = 0.0
        elseif L2_val < 1e-14
            # MATLAB approach: clamp small negative/zero values to exactly zero
            L2_m[iR, ie] = 0.0
        else
            L2_m[iR, ie] = L2_val
        end
        # For near-circular orbits, enforce Lc^2 = Rc^3 * dV(Rc) for numerical stability
        if e[iR, ie] < 1e-10
            Rc_val = Rc[iR, ie]
            tmp = Rc_val^3 * model.potential_derivative(Rc_val)
            if isnan(tmp) || tmp < 0
                error("NaN or negative L2_m for near-circular orbit at iR=$(iR), ie=$(ie). This indicates numerical issues in grid construction.")
                L2_m[iR, ie] = 0
            end
            L2_m[iR, ie] = max(0.0, tmp)
        end
    end

    # Apply sign from velocity grid
    # For radial orbits (v_iR ≈ 0), SGNL should be exactly 0, not forced to +1
    SGNL = zeros(Float64, size(v_iR))
    for i in eachindex(v_iR)
        if abs(v_iR[i]) < 1e-12
            SGNL[i] = 0.0  # Exactly zero for radial orbits
        else
            SGNL[i] = sign(v_iR[i])
        end
    end

    L_m = sqrt.(L2_m) .* SGNL


    # Debug: inspect first circulation column values at a few radii
    if get(ENV, "PME_VERBOSE", "false") == "true"
        for iR in 1:min(NR, 5)
            ie = 1
            Rc_val = Rc[iR, ie]
            e_val = e[iR, ie]
            v_val = v_iR[iR, ie]
            r1_val = R1[iR, ie]
            r2_val = R2[iR, ie]
            E_val = E[iR, ie]
            V2_val = model.potential(r2_val)
            dV_val = model.potential_derivative(Rc_val)
            L2_val = L2_m[iR, ie]
            L_val = L_m[iR, ie]
            sgnl_val = SGNL[iR, ie]
            @info "Grid debug at iR=$(iR), ie=1" v=v_val e=e_val Rc=Rc_val R1=r1_val R2=r2_val E=E_val V_R2=V2_val dV_Rc=dV_val L2=L2_val SGNL=sgnl_val L=L_val
        end
    end

    return PMEGrids{Float64}(
        radial_grid, ecc_grid,
        Rc, R1, R2, E, L_m, L2_m, SGNL, v
    )
end

end # module GridConstruction
