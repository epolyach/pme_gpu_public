
# src/models/MiyamotoTaperTanh.jl

"""
Miyamoto model with tanh taper in circulation space for Kuzmin-Toomre disk with Plummer potential
Based on Miyamoto (1971) and Hunter (1992) formulations with tanh taper T(v) = (1 + tanh(v/v_0))/2, v = sign(L)*r1/Rc
"""
module MiyamotoTaperTanh

using SpecialFunctions: gamma
using QuadGK: quadgk
using ..AbstractModel: AbstractGalacticModel, ModelResults

export MiyamotoTaperTanhModel, create_miyamoto_taper_tanh_model, setup_model

# =============================================================================
# MODEL DEFINITION
# =============================================================================

struct MiyamotoTaperTanhModel <: AbstractGalacticModel end

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

"""
    create_miyamoto_taper_tanh_model(config::PMEConfig) -> ModelResults

Create a Miyamoto model with tanh taper in circulation space from configuration parameters.
"""
function create_miyamoto_taper_tanh_model(config)
    n_M = config.model.n_M
    # Support both v_0 (tanh) and legacy L_0
    v_0 = hasproperty(config.model, :v_0) ? getfield(config.model, :v_0) : (hasproperty(config.model, :L_0) ? getfield(config.model, :L_0) : 0.2)
    unit_mass = config.model.unit_mass
    unit_length = config.model.unit_length
    selfgravity = config.physics.selfgravity

    return setup_model(MiyamotoTaperTanhModel, n_M, v_0; 
                       unit_mass=unit_mass, 
                       unit_length=unit_length,
                       selfgravity=selfgravity)
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    double_factorial_odd(n::Int) -> Int

Compute the odd double factorial (2n-1)!!
"""
function double_factorial_odd(n::Int)
    if n <= 0
        return 1
    end
    result = 1
    for k in 1:n
        result *= (2*k - 1)
    end
    return result
end

"""
    compute_coefficient(n_M::Int, s::Int) -> Float64

Compute the coefficient for term s in the Miyamoto DF sum.
From Expression (3.126):
coefficient = n_M! * Γ(2n_M + 4) / (s! * (n_M - s)! * (2s - 1)!! * Γ(2n_M - s + 3))
"""
function compute_coefficient(n_M::Int, s::Int)
    if s < 0 || s > n_M
        return 0.0
    end
    n_factorial = factorial(n_M)
    s_factorial = factorial(s)
    n_minus_s_factorial = factorial(n_M - s)
    double_fact = double_factorial_odd(s)
    gamma_num = gamma(2*n_M + 4)
    gamma_den = gamma(2*n_M - s + 3)
    return n_factorial * gamma_num * (2.0^s) / (s_factorial * n_minus_s_factorial * double_fact * gamma_den)
end

"""
    compute_radial_action(E::Real, L::Real, V::Function) -> Float64
"""
function compute_radial_action(E::Real, L::Real, V::Function)
    if E >= 0.0
        return 0.0
    end
    L2 = L^2
    V_eff(r) = V(r) + L2 / (2 * r^2)
    r_min = 1e-6; r_max = 100.0
    function find_turning_point(r_low, r_high, target_E)
        for _ in 1:50
            r_mid = (r_low + r_high)/2
            V_mid = V_eff(r_mid)
            if abs(V_mid - target_E) < 1e-10
                return r_mid
            end
            if V_mid > target_E; r_low = r_mid; else; r_high = r_mid; end
        end
        return (r_low + r_high)/2
    end
    r_test = range(r_min, r_max, length=1000)
    V_test = V_eff.(r_test)
    crossings = findall(diff(sign.(V_test .- E)) .!= 0)
    if length(crossings) < 2; return 0.0; end
    r1 = find_turning_point(r_test[crossings[1]], r_test[crossings[1]+1], E)
    r2 = find_turning_point(r_test[crossings[end]], r_test[crossings[end]+1], E)
    function integrand(r)
        vr2 = 2*(E - V(r)) - L2/r^2
        return vr2 > 0 ? sqrt(vr2) : 0.0
    end
    Jr_val, _ = quadgk(integrand, r1, r2, rtol=1e-8)
    return Jr_val/π
end

"""
    compute_frequencies(E::Real, L::Real, V::Function, dV::Function) -> (Float64, Float64)
"""
function compute_frequencies(E::Real, L::Real, V::Function, dV::Function)
    if E >= 0.0
        return (0.0, 0.0)
    end
    L2 = L^2
    V_eff(r) = V(r) + L2 / (2*r^2)
    r_min = 1e-6; r_max = 100.0
    function find_turning_point(r_low, r_high, target_E)
        for _ in 1:50
            r_mid = (r_low + r_high)/2
            V_mid = V_eff(r_mid)
            if abs(V_mid - target_E) < 1e-10
                return r_mid
            end
            if V_mid > target_E; r_low = r_mid; else; r_high = r_mid; end
        end
        return (r_low + r_high)/2
    end
    r_test = range(r_min, r_max, length=1000)
    V_test = V_eff.(r_test)
    crossings = findall(diff(sign.(V_test .- E)) .!= 0)
    if length(crossings) < 2; return (0.0, 0.0); end
    r1 = find_turning_point(r_test[crossings[1]], r_test[crossings[1]+1], E)
    r2 = find_turning_point(r_test[crossings[end]], r_test[crossings[end]+1], E)
    function time_integrand(r)
        vr2 = 2*(E - V(r)) - L2/r^2
        return vr2 > 0 ? 1.0/sqrt(vr2) : 0.0
    end
    T_radial, _ = quadgk(time_integrand, r1, r2, rtol=1e-8)
    T_radial *= 2
    Omega_1 = 2π / T_radial
    function phi_integrand(r)
        vr2 = 2*(E - V(r)) - L2/r^2
        return vr2 > 0 ? L/(r^2*sqrt(vr2)) : 0.0
    end
    delta_phi, _ = quadgk(phi_integrand, r1, r2, rtol=1e-8)
    delta_phi *= 2
    Omega_2 = Omega_1 * delta_phi / (2π)
    return (Omega_1, Omega_2)
end

# =============================================================================
# SETUP FUNCTION
# =============================================================================

"""
    setup_model(::Type{MiyamotoTaperTanhModel}, n_M::Int, v_0::Real; kwargs...) -> ModelResults
"""
function setup_model(::Type{MiyamotoTaperTanhModel}, n_M::Int, v_0::Real; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    if n_M < 0; error("n_M must be non-negative, got $n_M"); end
    if v_0 <= 0; error("v_0 must be positive, got $v_0"); end
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Miyamoto model with tanh taper (n_M=$n_M, v_0=$v_0, selfgravity=$selfgravity)")
    end

    # Plummer potential
    V(r) = -1.0 / sqrt(1.0 + r^2)
    dV(r) = r / (1.0 + r^2)^(3/2)
    Omega(r) = 1.0 / (1.0 + r^2)^(3/4)
    function kappa(r)
        r2 = r^2
        return sqrt((r2 + 4.0) / (1.0 + r2)^(5/2))
    end
    Sigma_d(r) = 1.0 / (2π * (1.0 + r^2)^(3/2))

    # Taper
    # Tanh-based implementation
    taper(v) = 0.5 * (1.0 + tanh(v / v_0))
    taper_deriv(v) = 0.5 / v_0 * (1.0 - tanh(v / v_0)^2)
    
    
    """
    # Piecewise polynomial implementation (alternative): T(x) = 1/2 + 3x/4 - x^3/4, x = v/v_0
    # # For |x| <= 1; T(x) = 0 for x < -1; T(x) = 1 for x > 1
    # function taper(v)
    #     x = v / v_0
    #     if x < -1.0
    #         return 0.0
    #     elseif x > 1.0
    #         return 1.0
    #     else
    #         return 0.5 + 0.75*x - 0.25*x^3
    #     end
    # end
    # 
    # # Derivative: dT/dv = (3/4 - 3x^2/4) / v_0 for |x| <= 1; 0 elsewhere
    # function taper_deriv(v)
    #     x = v / v_0
    #     if x < -1.0 || x > 1.0
    #         return 0.0
    #     else
    #         return (0.75 - 0.75*x^2) / v_0
    #     end
    # end
    # 

    """

    # # Precompute coefficients
    coeffs = [compute_coefficient(n_M, s) for s in 0:n_M]

    # DF with circulation taper
    function DF(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        r1 = grids.R1[iR, iv]
        Rc = grids.Rc[iR, iv]
        v = grids.v[iR, iv]
        sum_value = coeffs[1]
        L2_over_neg2E = L^2 / (-2.0*E)
        for s in 1:n_M
            sum_value += coeffs[s+1] * L2_over_neg2E^s
        end
        F_base = (2.0 / (2π)^2) * (-E)^(2*n_M + 2) * sum_value
        return F_base * taper(v)
    end

    # Energy derivative
    function DF_dE_inner(E::Real, L::Real, iR::Int, iv::Int, grids)::Real

        r1 = grids.R1[iR, iv]; r2 = grids.R2[iR, iv]; Rc = grids.Rc[iR, iv]
        sign_L = grids.SGNL[iR, iv]
        v = grids.v[iR, iv]
        if abs(L) < 1e-10
            # Proper limit as L→0
            dv_dE = 0.0
        else
            V1 = V(r1); V2 = V(r2); dV1 = dV(r1); dV2 = dV(r2)
            t1 = 2*(E - V1)*r1 - dV1*r1^2
            t2 = 2*(E - V2)*r2 - dV2*r2^2
            dr1_dE = -r1^2 / t1
            dr2_dE = -r2^2 / t2
            dRc_dE = 0.5*(dr1_dE + dr2_dE)
            dv_dE = sign_L * (dr1_dE/Rc - r1*dRc_dE/Rc^2)
        end

        sum_value = coeffs[1]; sum_deriv = 0.0
        L2_over_neg2E = L^2 / (-2.0*E)
        for s in 1:n_M
            sum_value += coeffs[s+1] * L2_over_neg2E^s
            sum_deriv += coeffs[s+1] * s * L2_over_neg2E^(s-1) * L^2 / (2.0*E^2)
        end
        factor1 = -(2*n_M + 2) * (-E)^(2*n_M + 1) * sum_value
        factor2 = (-E)^(2*n_M + 2) * sum_deriv
        F_base = (2.0 / (2π)^2) * (-E)^(2*n_M + 2) * sum_value
        dF_base_dE = (2.0 / (2π)^2) * (factor1 + factor2)
        return dF_base_dE * taper(v) + F_base * taper_deriv(v) * dv_dE
    end

    # Angular momentum derivative
    function DF_dL_inner(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        r1 = grids.R1[iR, iv]; r2 = grids.R2[iR, iv]; Rc = grids.Rc[iR, iv]
        sign_L = grids.SGNL[iR, iv]
        v = grids.v[iR, iv]

        # For L ≈ 0 (radial orbits)
        if abs(L) < 1e-10
            # Proper limit as L→0
            V0 = V(0.0)
            dv_dL = 1.0 / (Rc * sqrt(2.0 * (E - V0)))
            # Use this instead of sign_L * (dr1_dL/Rc - ...)
        else
            V1 = V(r1); V2 = V(r2); dV1 = dV(r1); dV2 = dV(r2)
            t1 = 2*(E - V1)*r1 - dV1*r1^2
            t2 = 2*(E - V2)*r2 - dV2*r2^2
            dr1_dL = L / t1
            dr2_dL = L / t2
            dRc_dL = 0.5*(dr1_dL + dr2_dL)
            dv_dL = sign_L * (dr1_dL/Rc - r1*dRc_dL/Rc^2)
        end

        sum_value = coeffs[1]; sum_deriv = 0.0
        L2_over_neg2E = L^2 / (-2.0*E)
        for s in 1:n_M
            sum_value += coeffs[s+1] * L2_over_neg2E^s
            sum_deriv += coeffs[s+1] * s * L2_over_neg2E^(s-1) * (2.0*L) / (-2.0*E)
        end
        F_base = (2.0 / (2π)^2) * (-E)^(2*n_M + 2) * sum_value
        dF_base_dL = (2.0 / (2π)^2) * (-E)^(2*n_M + 2) * sum_deriv
        return dF_base_dL * taper(v) + F_base * taper_deriv(v) * dv_dL
    end

    # Wrappers (signatures expected by PME)
    DF_dE(E::Real, L::Real, iR::Int, iv::Int, grids) = DF_dE_inner(E, L, iR, iv, grids)
    DF_dL(E::Real, L::Real, iR::Int, iv::Int, grids) = DF_dL_inner(E, L, iR, iv, grids)

    # ACTION-ANGLE FUNCTIONS
    Jr(E, L) = compute_radial_action(E, L, V)
    function Omega_1(E, L)
        omega1, _ = compute_frequencies(E, L, V, dV); return omega1
    end
    function Omega_2(E, L)
        _, omega2 = compute_frequencies(E, L, V, dV); return omega2
    end

    # CREATE MODEL RESULTS
    helpers = (
        n_M = n_M,
        v_0 = v_0,
        Jr = Jr, Omega_1 = Omega_1, Omega_2 = Omega_2,
        taper_function = v -> taper(v)
    )
    params = Dict{String,Any}(
        "n_M" => n_M,
        "v_0" => v_0,
        "unit_mass" => unit_mass,
        "unit_length" => unit_length,
        "selfgravity" => selfgravity,
        "potential" => "Plummer",
        "disk_type" => "Kuzmin-Toomre",
        "DF_type" => 0,
        "taper_type" => "Tanh"
    )

    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF,
        DF_dE, DF_dL,
        Sigma_d,
        nothing,
        helpers,
        "MiyamotoTaperTanh",
        params
    )
end

end # module MiyamotoTaperTanh
