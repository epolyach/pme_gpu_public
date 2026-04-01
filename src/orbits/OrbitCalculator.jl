# src/orbits/OrbitCalculator.jl
module OrbitCalculator

using LinearAlgebra
using Printf
using ..AbstractModel: ModelResults
using ..Configuration
using ..ProgressUtils: print_progress_bar

export OrbitData, calculate_orbits, evaluate_distribution_function

"""
    call_with_grids(func, E, L, grids, iR, ie)

Helper function to call exponential taper distribution functions with grid data.
"""
function call_with_grids(func, E_val, L_val, grids, iR, ie)
    # Import the required modules to access the internal closure functions
    try
        # Try to access internal closure functions that need grids
        if hasfield(typeof(func), :env) && hasfield(typeof(func.env), :DF_closure)
            # Access the closure function that needs grids
            return func.env.DF_closure(E_val, L_val, iR, ie, grids)
        elseif hasfield(typeof(func), :env) && hasfield(typeof(func.env), :DF_dE_closure)
            return func.env.DF_dE_closure(E_val, L_val, iR, ie, grids)
        elseif hasfield(typeof(func), :env) && hasfield(typeof(func.env), :DF_dL_closure)
            return func.env.DF_dL_closure(E_val, L_val, iR, ie, grids)
        else
            # Fallback: try to call the function with the additional parameters
            # This assumes the function can handle additional arguments gracefully
            return func(E_val, L_val, iR, ie, grids)
        end
    catch
        # If all else fails, try a simpler approach with just the required params
        try 
            return func(E_val, L_val, iR, ie)
        catch
            error("Unable to call exponential taper distribution function with grid data")
        end
    end
end

"""
    OrbitData

A mutable struct to hold all data related to calculated orbits.
Making it mutable allows us to add the distribution function values later.
"""
mutable struct OrbitData{T<:Real}
    # Grid dimensions
    NR::Int
    Ne::Int  
    nwa::Int
    
    # Frequencies
    Omega_1::Matrix{T}    # Radial frequency (NR × Ne)
    Omega_2::Matrix{T}    # Azimuthal frequency (NR × Ne)
    
    # Action variables
    Ir::Matrix{T}         # Radial action (NR × Ne)
    
    # Orbital trajectories (nwa × NR × Ne)
    w1::Array{T,3}        # Radial phase angles
    ra::Array{T,3}        # Radial positions  
    pha::Array{T,3}       # Azimuthal phases
    
    # Grid data
    grids

    # Distribution function values - to be added later
    F0::Matrix{T}
    FE::Matrix{T}
    FL::Matrix{T}
    t1::Matrix{T}         # Denominator for dr1/dE and dr1/dL
    t2::Matrix{T}         # Denominator for dr2/dE and dr2/dL
    
    # Jacobian for action-angle transformation
    jacobian::Matrix{T}

    # Constructor for initialization
    function OrbitData(NR, Ne, nwa, Omega_1, Omega_2, Ir, w1, ra, pha, grids)
        T = eltype(Omega_1)
        new{T}(NR, Ne, nwa, Omega_1, Omega_2, Ir, w1, ra, pha, grids, 
               zeros(T, NR, Ne), zeros(T, NR, Ne), zeros(T, NR, Ne), zeros(T, NR, Ne),
               zeros(T, NR, Ne), zeros(T, NR, Ne))
    end
end

"""
    calculate_orbits(grids, model, config) -> OrbitData

Calculates orbital properties (frequencies, actions, trajectories) for all points
on the computational grid using a multi-threaded approach.
"""
function calculate_orbits(grids, model, config::PMEConfig)
    
    NR, Nv = config.grid.NR, config.grid.Nv
    nwa = config.grid.NW
    full_space = config.phase_space.full_space
    
    # Initialize arrays
    T = Float64
    Omega_1 = zeros(T, NR, Nv)
    Omega_2 = zeros(T, NR, Nv) 
    Ir = zeros(T, NR, Nv)
    w1 = zeros(T, nwa, NR, Nv)
    ra = zeros(T, nwa, NR, Nv)
    pha = zeros(T, nwa, NR, Nv)
    
    # Setup angular grid for orbit integration
    # Use high-resolution grid for accurate orbit integration
    nw = config.grid.NW_orbit  # Configurable orbit integration points
    nwai = (nw - 1) ÷ (nwa - 1)
    Ia = 1:nwai:nw
    
    eps_w = 1e-10
    w = range(eps_w, π - eps_w, length=nw)
    dw = step(w)
    cosw = cos.(w)
    sinw = sin.(w)
    wa = w[Ia]
    
    # Calculating orbital trajectories
    
    # Progress tracking for multi-threading
    completed_rows = Threads.Atomic{Int}(0)
    total_rows = NR
    eccentric_warning_printed = Threads.Atomic{Bool}(false)
    # print_progress_bar(0, total_rows, "Orbit calculation")

    Threads.@threads for iR in 1:NR
        
        R_i = grids.Rc[iR, 1]
        kappa_iR = model.epicyclic_frequency(R_i)
        Omega_iR = model.rotation_frequency(R_i)
        
        for iv in 1:Nv
            E_i = grids.E[iR, iv]
            L_j = grids.L_m[iR, iv]
            L2_j = grids.L2_m[iR, iv]
            e_val = grids.eccentricity.points[iR, iv]
            
            if abs(e_val) < 1e-10
                calculate_circular_orbit!(iR, iv, R_i, kappa_iR, Omega_iR, grids.SGNL[iR, iv], wa, Omega_1, Omega_2, Ir, w1, ra, pha)
            elseif e_val < 0.01
                calculate_epicyclic_orbit!(iR, iv, R_i, kappa_iR, Omega_iR, grids.R1[iR, iv], grids.R2[iR, iv], grids.SGNL[iR, iv], wa, cosw, sinw, Ia, Omega_1, Omega_2, Ir, w1, ra, pha)
            else
                calculate_general_orbit!(iR, iv, E_i, L_j, L2_j, e_val, grids.R1[iR, iv], grids.R2[iR, iv], grids.SGNL[iR, iv], full_space, model, w, dw, cosw, sinw, Ia, wa, Omega_1, Omega_2, Ir, w1, ra, pha)
            end
        end
        
        # Update progress
        new_val = Threads.atomic_add!(completed_rows, 1)
        # print_progress_bar(new_val, total_rows, "Orbit calculation")
    end
    
    # Orbital trajectories completed
    

    # ===========================================================================
    # CORRECT OMEGA_2 FOR RADIAL ORBITS BASED ON OMEGA_2_LIM
    # ===========================================================================
    # Omega_2_lim semantics:
    #   +1: limit L -> 0+ ⇒ Ω₂ = Ω₁/2
    #    0: enforce Ω₂(J, 0) = 0 (radial orbits)
    #   -1: limit L -> 0- ⇒ Ω₂ = -Ω₁/2
    #
    # IMPORTANT: Radial orbits are identified by L_m ≈ 0, NOT by index iv=1.
    # The index iv=1 corresponds to L=0 ONLY when sharp_df=true and v[iR,1]=0.
    # For full_space=true (bidirectional), iv=1 is NOT necessarily L=0.
    
    Omega_2_lim = config.phase_space.Omega_2_lim
    if Omega_2_lim == 1 || Omega_2_lim == 0 || Omega_2_lim == -1
        # Tolerance for identifying radial orbits
        L_tol = 1e-11
        
        # Count how many radial orbits we find and correct
        n_corrected = 0
        
        for ir in 1:NR
            for iv in 1:Nv
                # Check if this is a radial orbit (|L| ≈ 0)
                if abs(grids.L_m[ir, iv]) < L_tol
                    if Omega_2_lim == 1
                        # Limit L->0+: Ω₂ = Ω₁/2
                        Omega_2[ir, iv] = Omega_1[ir, iv] / 2.0
                    elseif Omega_2_lim == 0
                        # Enforce Ω₂(J,0) = 0
                        Omega_2[ir, iv] = 0.0
                    elseif Omega_2_lim == -1
                        # Limit L->0-: Ω₂ = -Ω₁/2
                        Omega_2[ir, iv] = -Omega_1[ir, iv] / 2.0
                    end
                    n_corrected += 1
                end
            end
        end
        
        if n_corrected > 0
            #             println("✓ Omega_2 corrected for $(n_corrected) radial orbits (|L|<$(L_tol)) using limit mode $(Omega_2_lim)")
        end
    elseif Omega_2_lim != 999  # sentinel for "no correction"
        error("Invalid Omega_2_lim value: $(Omega_2_lim). Must be +1, 0, or -1")
    end
    
    orbit_data = OrbitData(NR, Nv, nwa, Omega_1, Omega_2, Ir, w1, ra, pha, grids)
    
    # Calculate jacobian after orbit frequencies are computed
    calculate_jacobian!(orbit_data, model)
    
    return orbit_data
end

function calculate_circular_orbit!(iR::Int, ie::Int, R_i::Real, 
                                  kappa_iR::Real, Omega_iR::Real, SGNL::Real,
                                  wa::AbstractVector,
                                  Omega_1::Matrix, Omega_2::Matrix, Ir::Matrix,
                                  w1::Array, ra::Array, pha::Array)
    
    Ir[iR, ie] = 0.0
    Omega_1[iR, ie] = kappa_iR  
    Omega_2[iR, ie] = Omega_iR * SGNL
    
    # Debug info for Omega_1 calculation in circular orbits
    w1[:, iR, ie] = wa
    ra[:, iR, ie] .= R_i
    pha[:, iR, ie] = wa * Omega_2[iR, ie] / Omega_1[iR, ie]
end

function calculate_epicyclic_orbit!(iR::Int, ie::Int, R_i::Real,
                                   kappa_iR::Real, Omega_iR::Real,
                                   R1::Real, R2::Real, SGNL::Real,
                                   wa::AbstractVector, cosw::AbstractVector,
                                   sinw::AbstractVector, Ia::AbstractVector,
                                   Omega_1::Matrix, Omega_2::Matrix, Ir::Matrix,
                                   w1::Array, ra::Array, pha::Array)
    
    a_j = (R2 - R1) / 2
    Ir[iR, ie] = a_j^2 * kappa_iR / 2
    Omega_1[iR, ie] = kappa_iR
    Omega_2[iR, ie] = Omega_iR * SGNL
    
    w1[:, iR, ie] = wa
    ra[:, iR, ie] = R_i .- a_j .* cosw[Ia]
    pha[:, iR, ie] = (wa .* Omega_2[iR, ie] ./ Omega_1[iR, ie] .+ 
                      2 .* Omega_iR ./ kappa_iR .* a_j ./ R_i .* sinw[Ia] .* SGNL)
end

function calculate_general_orbit!(iR::Int, ie::Int, E_i::Real, L_j::Real, L2_j::Real,
                                 e_val::Real, R1::Real, R2::Real, SGNL::Real,
                                 full_space::Bool, model, w::AbstractVector, dw::Real,
                                 cosw::AbstractVector, sinw::AbstractVector,
                                 Ia::AbstractVector, wa::AbstractVector,
                                 Omega_1::Matrix, Omega_2::Matrix, Ir::Matrix,
                                 w1::Array, ra::Array, pha::Array)
    
    r1, r2 = R1, R2
    rs = (r1 + r2) / 2
    drs = (r2 - r1) / 2    
    xs = rs .- drs * cosw
    
    rvr = zeros(length(xs))
    if L2_j>1e-13
        for i in eachindex(xs)
            x = xs[i]
            V_x = model.potential(x)
            vr_sq = 2 * (E_i - V_x) * x^2 - L2_j
            if vr_sq < 0
                Omega_0 = model.rotation_frequency(0)
                vr_sq = (2.0*rs*Omega_0)^2 * (1 - (x/2/rs)^2) * x^2 - L2_j
            end
            rvr[i] = sqrt(vr_sq)
            # rvr[i] = sqrt(max(0.0, vr_sq))
        end   
        svr = sinw .* rvr ./ xs
        svr_time = sinw .* xs ./ rvr
        svr_time[1] = svr_time[2] * 2 - svr_time[3]
    else
        # Radial orbit case (L=0)
        for i in eachindex(xs)
            x = xs[i]
            V_x = model.potential(x)
            vr_sq = 2 * (E_i - V_x)
            if vr_sq <= 0
                Omega_0 = model.rotation_frequency(0)
                vr_sq = (2.0*rs*Omega_0)^2 * (1 - (x/2/rs)^2)
                # println("   i=$i, vr_sq=$vr_sq, E_i=$E_i, V_x=$V_x, x=$x, rs=$rs, Omega_0=$Omega_0")
            end
            rvr[i] = sqrt(vr_sq)
            # rvr[i] = sqrt(max(0.0, vr_sq))
        end        
        svr = sinw .* rvr
        svr_time = sinw ./ rvr
    end
    Ir[iR, ie] = drs * sum(svr) * dw / π
    svr_time[end] = svr_time[end-1] * 2 - svr_time[end-2] 
    
    dt1 = drs * dw * svr_time
    dt2 = zeros(length(w))
    dt2[2:end] = (dt1[1:end-1] + dt1[2:end]) / 2
    
    # Debug: Check for non-finite values with detailed diagnostics
    has_nonfinite = false
    nonfinite_info = String[]
    
    if any(.!isfinite.(svr_time))
        println(svr_time[1:4])
        has_nonfinite = true
        idx = findall(.!isfinite.(svr_time))
        push!(nonfinite_info, "svr_time non-finite at indices: $idx")
        # Show rvr values at problematic indices
        for (count, i) in enumerate(idx)
            if count > 3  # Limit to first 3 indices
                push!(nonfinite_info, "  ... (showing first 3 of $(length(idx)) non-finite values)")
                break
            end
            if i <= length(rvr)
                push!(nonfinite_info, "  idx=$i: rvr[$i]=$(rvr[i]), sinw[$i]=$(sinw[i]), xs[$i]=$(xs[i])")
            end
        end
    end
    if any(.!isfinite.(dt1))
        has_nonfinite = true
        idx = findall(.!isfinite.(dt1))
        push!(nonfinite_info, "dt1 non-finite at indices: $idx")
    end
    if any(.!isfinite.(dt2))
        has_nonfinite = true
        idx = findall(.!isfinite.(dt2))
        push!(nonfinite_info, "dt2 non-finite at indices: $idx")
    end
    
    if has_nonfinite
        # Print diagnostics directly to stdout before throwing error
        println("\n=== DIAGNOSTICS (printed before error) ===")
        println("\n  iR=$iR, ie=$ie")
        for line in nonfinite_info
            println(line)
        end
        println("=== END DIAGNOSTICS ===\n")
        # Compute Rc (circular orbit radius) for context
        Rc = (r1 + r2) / 2  # approximation for small e
        
        # Find indices where rvr is zero or very small
        zero_rvr_idx = findall(x -> x < 1e-15, rvr)
        
        error_msg = """
Non-finite values detected in orbit calculation!

=== Orbit Parameters ===
iR=$iR, ie=$ie
E_i=$E_i
L_j=$L_j, L2_j=$L2_j
e_val=$e_val
R1 (pericenter)=$r1
R2 (apocenter)=$r2
rs (semi-major axis)=$rs
drs (half-width)=$drs
Rc (approx circular radius)=$Rc

=== Diagnostics ===
$(join(nonfinite_info, "\n"))

Zero/tiny rvr at indices: $zero_rvr_idx (total $(length(zero_rvr_idx)) of $(length(rvr)))
rvr range: min=$(minimum(rvr)), max=$(maximum(rvr))
xs range: min=$(minimum(xs)), max=$(maximum(xs))
sinw range: min=$(minimum(sinw)), max=$(maximum(sinw))

=== First/Last few values ===
rvr[1:5] = $(rvr[1:min(5,length(rvr))])
rvr[end-4:end] = $(rvr[max(1,length(rvr)-4):end])
svr_time[1:5] = $(svr_time[1:min(5,length(svr_time))])
svr_time[end-4:end] = $(svr_time[max(1,length(svr_time)-4):end])
"""
        error(error_msg)
    end
    
    t = cumsum(dt2)
    T_radial = t[end]
    Omega_1[iR, ie] = π / T_radial
    
    if !isfinite(Omega_1[iR, ie])
        Rc = (r1 + r2) / 2
        error("""
Non-finite Omega_1 at grid point ($iR, $ie)!

=== Parameters ===
T_radial=$T_radial
e_val=$e_val
E_i=$E_i
L2_j=$L2_j
R1=$r1, R2=$r2
Rc (approx)=$Rc

t[end-4:end] = $(t[max(1,length(t)-4):end])
dt2[end-4:end] = $(dt2[max(1,length(dt2)-4):end])
""")
    end
    
    w1[:, iR, ie] .= t[Ia] .* Omega_1[iR, ie]
    ra[:, iR, ie] .= xs[Ia]
    
    if abs(1 - e_val) > 1e-10
        svr_phi = sinw ./ rvr ./ xs
        svr_phi[1] = svr_phi[2] * 2 - svr_phi[3] 
        svr_phi[end] = svr_phi[end-1] * 2 - svr_phi[end-2] 
        
        dt3 = drs * dw * svr_phi
        dt4 = zeros(length(w))
        dt4[2:end] = (dt3[1:end-1] + dt3[2:end]) / 2
        
        phi = cumsum(dt4)
        ph = L_j * phi[Ia]
    else
        ph = fill(π/2, length(Ia))

        ### Modified here 6/11/2025
        ph[1] = 0.0
    end
    
    Omega_2[iR, ie] = Omega_1[iR, ie] * ph[end] / π
    pha[:, iR, ie] = ph
    
end

function evaluate_distribution_function(orbit_data::OrbitData, model_results::ModelResults)
    
    NR, Ne = orbit_data.NR, orbit_data.Ne
    grids = orbit_data.grids
    
    df_type = get(model_results.parameters, "DF_type", 0)
    taper_type = get(model_results.parameters, "taper_type", "")
    
    Threads.@threads for iR in 1:NR
        for ie in 1:Ne
            local F_val, FE_val, FL_val
            if df_type == 0
                E_val = grids.E[iR, ie]
                L_val = grids.L_m[iR, ie]
                
                # Check if this is the exponential taper model that needs grid data
                if taper_type == "Exp" || taper_type == "TanhEnergy" || taper_type == "Tanh" || taper_type == "Poly3"
                    # Call distribution functions with additional grid parameters
                    F_val = model_results.distribution_function(E_val, L_val, iR, ie, grids)
                    FE_val = model_results.df_energy_derivative(E_val, L_val, iR, ie, grids)  
                    FL_val = model_results.df_angular_derivative(E_val, L_val, iR, ie, grids)
                else
                    F_val = model_results.distribution_function(E_val, L_val)
                    FE_val = model_results.df_energy_derivative(E_val, L_val)  
                    FL_val = model_results.df_angular_derivative(E_val, L_val)
                end
                
            elseif df_type == 1
                Ir_val = orbit_data.Ir[iR, ie]
                L_val = grids.L_m[iR, ie]
                
                # Check if this is the exponential taper model that needs grid indices
                if taper_type == "Exp" || taper_type == "TanhEnergy" || taper_type == "Tanh"
                    F_val = model_results.distribution_function(Ir_val, L_val, iR, ie)
                    FE_val = model_results.df_energy_derivative(Ir_val, L_val, iR, ie)
                    FL_val = model_results.df_angular_derivative(Ir_val, L_val, iR, ie)
                else
                    F_val = model_results.distribution_function(Ir_val, L_val)
                    FE_val = model_results.df_energy_derivative(Ir_val, L_val)
                    FL_val = model_results.df_angular_derivative(Ir_val, L_val)
                end
                
            else
                Rc_val = grids.Rc[iR, ie]
                e_val = grids.eccentricity.points[iR, ie]
                
                # Check if this is the exponential taper model that needs grid indices
                if taper_type == "Exp" || taper_type == "TanhEnergy" || taper_type == "Tanh"
                    F_val = model_results.distribution_function(Rc_val, e_val, iR, ie)
                    FE_val = model_results.df_energy_derivative(Rc_val, e_val, iR, ie)
                    FL_val = model_results.df_angular_derivative(Rc_val, e_val, iR, ie)
                else
                    F_val = model_results.distribution_function(Rc_val, e_val)
                    FE_val = model_results.df_energy_derivative(Rc_val, e_val)
                    FL_val = model_results.df_angular_derivative(Rc_val, e_val)
                end
            end
            
            orbit_data.F0[iR, ie] = isfinite(F_val) ? F_val : 0.0
            orbit_data.FE[iR, ie] = isfinite(FE_val) ? FE_val : 0.0
            orbit_data.FL[iR, ie] = isfinite(FL_val) ? FL_val : 0.0
            
            # Compute t1 and t2
            r1 = grids.R1[iR, ie]
            r2 = grids.R2[iR, ie]
            E_val = grids.E[iR, ie]
            V_func = model_results.potential
            dV_func = model_results.potential_derivative
            V1 = V_func(r1)
            V2 = V_func(r2)
            dV1 = dV_func(r1)
            dV2 = dV_func(r2)
            if abs(grids.v[iR, ie]) > 1e-10
                orbit_data.t1[iR, ie] = 2*(E_val - V1)*r1 - dV1*r1^2
            else
                orbit_data.t1[iR, ie] = 0
            end
            orbit_data.t2[iR, ie] = 2*(E_val - V2)*r2 - dV2*r2^2

            # Debug: trace how column 1 becomes nonzero with TanhEnergy taper (eta ≈ 0.02)
            if iR <= 5 && ie == 1 && haskey(model_results.parameters, "taper_type") && model_results.parameters["taper_type"] == "TanhEnergy"
                eta_dbg = get(model_results.parameters, "eta", nothing)
                if eta_dbg !== nothing && abs(eta_dbg - 0.02) < 1e-9
                    # Reconstruct taper diagnostics
                    E_dbg = grids.E[iR, ie]
                    L_dbg = grids.L_m[iR, ie]
                    Lc_dbg = grids.L_m[iR, end]
                    x_arg = L_dbg / (eta_dbg * Lc_dbg)
                    taper_val = 0.5 * (1 + tanh(x_arg))
                    x_arg_expected = -1.0 / eta_dbg
                    x_arg_diff = x_arg - x_arg_expected
                    @info "TanhEnergy debug (eta=0.02) at iR=$(iR), ie=1" E=E_dbg L=L_dbg Lc=Lc_dbg x_arg=x_arg x_arg_expected=x_arg_expected x_arg_diff=x_arg_diff taper=taper_val F0=orbit_data.F0[iR, ie] FE=orbit_data.FE[iR, ie] FL=orbit_data.FL[iR, ie]
                end
            end
        end
    end
    
    return orbit_data
end

"""
    calculate_jacobian!(orbit_data, model)

Calculate the Jacobian transformation matrix for action-angle variables.
This is called after orbit frequencies are computed.
"""
function calculate_jacobian!(orbit_data::OrbitData, model)
    NR, Ne = orbit_data.NR, orbit_data.Ne
    grids = orbit_data.grids
    
    # Calculating jacobian transformation
    
    # Calculate jacobian using potential
    for iR in 1:NR, ie in 1:Ne
        r1 = grids.R1[iR, ie]
        r2 = grids.R2[iR, ie]
        rc = grids.Rc[iR, ie]
        E_val = grids.E[iR, ie]
        L_val = grids.L_m[iR, ie]
        omega1 = orbit_data.Omega_1[iR, ie]
        
        orbit_data.jacobian[iR, ie] = 0.0

        # General orbit check (not circular, not radial)
        general_orbit = (abs(r2 - r1) > 1e-12) && (abs(L_val) > 1e-12)
        if general_orbit
            V1 = model.potential(r1)
            V2 = model.potential(r2)
            dV1 = model.potential_derivative(r1)
            dV2 = model.potential_derivative(r2)

            # Partial derivatives
            t1 = 2 * (E_val - V1) * r1 - dV1 * r1^2
            t2 = 2 * (E_val - V2) * r2 - dV2 * r2^2
            denominator = r2^2 - r1^2
            det_jac = abs(t1 * t2 / denominator)
            orbit_data.jacobian[iR, ie] = 2 * det_jac * rc / omega1 / abs(L_val)
        end

        # Radial orbit check 
        radial_orbit = abs(L_val) <= 1e-12
        if radial_orbit
            DelE = 2.0*(E_val - model.potential(r1))
            # Guard against negative radicand due to discretization or probing outside domain
            orbit_data.jacobian[iR, ie] = sqrt(max(DelE, 0.0)) * model.potential_derivative(r2) * 2*rc / omega1
            if !isfinite(orbit_data.jacobian[iR, ie])
                orbit_data.jacobian[iR, ie] = 0.0
            end
        end

    end
    
    # Jacobian calculation completed
end

end # module OrbitCalculator
