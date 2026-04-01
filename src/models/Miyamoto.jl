# src/models/Miyamoto.jl

"""
Miyamoto model for Kuzmin-Toomre disk with Plummer potential
Based on Miyamoto (1971) and Hunter (1992) formulations
"""
module Miyamoto

# NOTE: Sharp-edge model (no taper). Requires the generalized matrix method.

using SpecialFunctions: gamma
using QuadGK: quadgk
using ..AbstractModel: AbstractGalacticModel, ModelResults

export MiyamotoModel, create_miyamoto_model, setup_model

# =============================================================================
# MODEL DEFINITION
# =============================================================================

struct MiyamotoModel <: AbstractGalacticModel end

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

"""
    create_miyamoto_model(config::PMEConfig) -> ModelResults

Create a Miyamoto model from configuration parameters.
"""
function create_miyamoto_model(config)
    n_M = config.model.n_M  # Default to n_M = 3
    unit_mass = config.model.unit_mass
    unit_length = config.model.unit_length
    selfgravity = config.physics.selfgravity

    return setup_model(MiyamotoModel, n_M;
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
For n=0: returns 1 (by convention)
For n=1: returns 1
For n=2: returns 3
For n=3: returns 15
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
    # Handle edge cases
    if s < 0 || s > n_M
        return 0.0
    end

    # Compute factorials
    n_factorial = factorial(n_M)
    s_factorial = factorial(s)
    n_minus_s_factorial = factorial(n_M - s)

    # Compute odd double factorial
    double_fact = double_factorial_odd(s)

    # Compute gamma functions
    gamma_num = gamma(2*n_M + 4)
    gamma_den = gamma(2*n_M - s + 3)

    # Combine all terms
    coefficient = n_factorial * gamma_num * (2.0^s) / (s_factorial * n_minus_s_factorial * double_fact * gamma_den)

    return coefficient
end

"""
    compute_radial_action(E::Real, L::Real, V::Function) -> Float64

Compute the radial action Jr for given energy E and angular momentum L.
Jr = (1/π) ∫[r₁ to r₂] vᵣ dr

where vᵣ = √(2(E - V(r)) - L²/r²) and r₁, r₂ are turning points.
"""
function compute_radial_action(E::Real, L::Real, V::Function)
    if E >= 0.0 || L < 0.0
        return 0.0
    end

    L2 = L^2

    # Find turning points (roots of vᵣ² = 0)
    # 2(E - V(r)) - L²/r² = 0
    # This is solved numerically

    # Effective potential: V_eff(r) = V(r) + L²/(2r²)
    V_eff(r) = V(r) + L2 / (2 * r^2)

    # Find approximate bounds for turning points
    # For Plummer potential V(r) = -1/√(1+r²)
    # At large r: V_eff ≈ -1/r + L²/(2r²) ≈ L²/(2r²)
    # At small r: V_eff ≈ -1 + L²/(2r²)

    # Initial guess for inner turning point
    r_min = 1e-6
    r_max = 100.0

    # Find where E = V_eff(r)
    function find_turning_point(r_low, r_high, target_E)
        for _ in 1:50  # Newton-Raphson iterations
            r_mid = (r_low + r_high) / 2
            V_mid = V_eff(r_mid)
            if abs(V_mid - target_E) < 1e-10
                return r_mid
            end
            if V_mid > target_E
                r_low = r_mid
            else
                r_high = r_mid
            end
        end
        return (r_low + r_high) / 2
    end

    # Find turning points more carefully
    # Start from a reasonable guess
    r_test = range(r_min, r_max, length=1000)
    V_test = V_eff.(r_test)

    # Find indices where V_eff crosses E
    crossings = findall(diff(sign.(V_test .- E)) .!= 0)

    if length(crossings) < 2
        return 0.0  # No valid orbit
    end

    r1 = find_turning_point(r_test[crossings[1]], r_test[crossings[1]+1], E)
    r2 = find_turning_point(r_test[crossings[end]], r_test[crossings[end]+1], E)

    # Integrate vᵣ = √(2(E - V(r)) - L²/r²) from r₁ to r₂
    function integrand(r)
        vr2 = 2 * (E - V(r)) - L2 / r^2
        return vr2 > 0 ? sqrt(vr2) : 0.0
    end

    Jr_val, _ = quadgk(integrand, r1, r2, rtol=1e-8)

    return Jr_val / π
end

"""
    compute_frequencies(E::Real, L::Real, V::Function, dV::Function) -> (Float64, Float64)

Compute the radial and azimuthal frequencies (Ω₁, Ω₂) for given E and L.

Returns (Omega_1, Omega_2) where:
- Omega_1 = π / T_radial (radial frequency)
- Omega_2 = Omega_1 * Δφ / π (azimuthal frequency)
"""
function compute_frequencies(E::Real, L::Real, V::Function, dV::Function)
    if E >= 0.0 || L < 0.0
        return (0.0, 0.0)
    end

    L2 = L^2
    V_eff(r) = V(r) + L2 / (2 * r^2)

    # Find turning points (same as in compute_radial_action)
    r_min = 1e-6
    r_max = 100.0

    function find_turning_point(r_low, r_high, target_E)
        for _ in 1:50
            r_mid = (r_low + r_high) / 2
            V_mid = V_eff(r_mid)
            if abs(V_mid - target_E) < 1e-10
                return r_mid
            end
            if V_mid > target_E
                r_low = r_mid
            else
                r_high = r_mid
            end
        end
        return (r_low + r_high) / 2
    end

    r_test = range(r_min, r_max, length=1000)
    V_test = V_eff.(r_test)
    crossings = findall(diff(sign.(V_test .- E)) .!= 0)

    if length(crossings) < 2
        return (0.0, 0.0)
    end

    r1 = find_turning_point(r_test[crossings[1]], r_test[crossings[1]+1], E)
    r2 = find_turning_point(r_test[crossings[end]], r_test[crossings[end]+1], E)

    # Compute radial period: T_r = 2 ∫[r₁ to r₂] dr/vᵣ
    function time_integrand(r)
        vr2 = 2 * (E - V(r)) - L2 / r^2
        return vr2 > 0 ? 1.0 / sqrt(vr2) : 0.0
    end

    T_radial, _ = quadgk(time_integrand, r1, r2, rtol=1e-8)
    T_radial *= 2  # Full period (out and back)

    Omega_1 = 2π / T_radial

    # Compute azimuthal advance: Δφ = ∫[r₁ to r₂] (L/r²) dt = ∫ (L/r²) (dr/vᵣ)
    function phi_integrand(r)
        vr2 = 2 * (E - V(r)) - L2 / r^2
        return vr2 > 0 ? L / (r^2 * sqrt(vr2)) : 0.0
    end

    delta_phi, _ = quadgk(phi_integrand, r1, r2, rtol=1e-8)
    delta_phi *= 2  # Full period

    Omega_2 = Omega_1 * delta_phi / (2π)

    return (Omega_1, Omega_2)
end

# =============================================================================
# SETUP FUNCTION
# =============================================================================

"""
    setup_model(::Type{MiyamotoModel}, n_M::Int; kwargs...) -> ModelResults

Set up a Miyamoto model with Plummer potential for Kuzmin-Toomre disk.
Uses normalized units where GM = 1 and scale length a = 1.
"""
function setup_model(::Type{MiyamotoModel}, n_M::Int;
                     unit_mass::Real=1.0,
                     unit_length::Real=1.0,
                     selfgravity::Real=1.0)

    # Validate parameters
    if n_M < 0
        error("n_M must be non-negative, got $n_M")
    end

    # =============================================================================
    # PLUMMER POTENTIAL FUNCTIONS (normalized: GM=1, a=1)
    # =============================================================================

    # Potential: Φ(r) = -GM/√(a² + r²) = -1/√(1 + r²)
    V(r) = -1.0 / sqrt(1.0 + r^2)

    # Derivative: dΦ/dr = GM·r/(a² + r²)^(3/2) = r/(1 + r²)^(3/2)
    dV(r) = r / (1.0 + r^2)^(3/2)

    # Circular frequency: Ω(r) = √(r·dΦ/dr)/r = √(1/(1 + r²)^(3/2))
    Omega(r) = 1.0 / (1.0 + r^2)^(3/4)

    # Epicyclic frequency: κ²(r) = 3Ω² + r·dΩ²/dr
    # For Plummer: κ²(r) = (r² + 4)/(1 + r²)^(5/2)
    function kappa(r)
        r2 = r^2
        return sqrt((r2 + 4.0) / (1.0 + r2)^(5/2))
    end

    # Surface density: Σ(r) = Ma/(2π(a² + r²)^(3/2)) = 1/(2π(1 + r²)^(3/2))
    # (normalized with M=1, a=1)
    Sigma_d(r) = 1.0 / (2π * (1.0 + r^2)^(3/2))

    # =============================================================================
    # DISTRIBUTION FUNCTION (Hunter's unidirectional formulation)
    # =============================================================================

    # Pre-compute coefficients for the finite sum
    coefficients = [compute_coefficient(n_M, s) for s in 0:n_M]

    """
    Distribution function F^(n_M)(E,L) from Expression (3.129)
    Unidirectional model (L > 0 only) with factor 2/(2π)²
    """
    function DF(E::Real, L::Real)::Real
        # Compute the finite sum from s=0 to n_M
        # f_even^(n_M)(E,L) = Σ_{s=0}^{n_M} coeff[s] * (L²/(-2E))^s * (-E)^(2n_M+2)
        sum_value = 0.0
        L2_over_neg2E = L^2 / (-2.0*E)

        # Build the sum
        for s in 0:n_M
            term = coefficients[s+1] * L2_over_neg2E^s
            sum_value += term
        end

        # Multiply by (-E)^(2n_M+2) and the normalization factor
        # Hunter's formulation: F = 2 * f_even * Θ(L)
        # Factor: 2/(2π)² for unidirectional model
        result = (2.0 / (2π)^2) * (-E)^(2*n_M + 2) * sum_value

        return result
    end

    """
    Energy derivative of distribution function: ∂F/∂E
    """
    function DF_dE(E::Real, L::Real)::Real
        # We need to compute d/dE[(-E)^(2n_M+2) * Σ coeff[s] * (L²/(-2E))^s]
        # Using product rule and chain rule

        sum_value = 0.0
        sum_deriv = 0.0
        L2_over_neg2E = L^2 / (-2.0*E)

        for s in 0:n_M
            term = coefficients[s+1] * L2_over_neg2E^s
            sum_value += term

            # Derivative of (L²/(-2E))^s w.r.t. E
            # d/dE[(L²/(-2E))^s] = s * (L²/(-2E))^(s-1) * L²/(2E²)
            if s > 0
                deriv_term = coefficients[s+1] * s * L2_over_neg2E^(s-1) * L^2 / (2.0*E^2)
                sum_deriv += deriv_term
            end
        end

        # Apply product rule:
        # d/dE[(-E)^(2n_M+2) * sum] = (2n_M+2)*(-E)^(2n_M+1)*(-1)*sum + (-E)^(2n_M+2)*sum_deriv
        factor1 = -(2*n_M + 2) * (-E)^(2*n_M + 1) * sum_value
        factor2 = (-E)^(2*n_M + 2) * sum_deriv

        result = (2.0 / (2π)^2) * (factor1 + factor2)

        return result
    end

    """
    Angular momentum derivative of distribution function: ∂F/∂L
    """
    function DF_dL(E::Real, L::Real)::Real
        # We need to compute d/dL[Σ coeff[s] * (L²/(-2E))^s]
        # d/dL[(L²/(-2E))^s] = s * (L²/(-2E))^(s-1) * 2L/(-2E) = s * (L²/(-2E))^(s-1) * L/(-E)

        sum_deriv = 0.0
        L2_over_neg2E = L^2 / (-2.0*E)

        for s in 1:n_M  # Start from s=1 since s=0 term has no L dependence
            deriv_term = coefficients[s+1] * s * L2_over_neg2E^(s-1) * (2.0*L) / (-2.0*E)
            sum_deriv += deriv_term
        end

        # Multiply by (-E)^(2n_M+2) and normalization
        result = (2.0 / (2π)^2) * (-E)^(2*n_M + 2) * sum_deriv

        return result
    end

    # =============================================================================
    # ACTION-ANGLE FUNCTIONS
    # =============================================================================

    # Radial action (computed numerically)
    Jr(E, L) = compute_radial_action(E, L, V)

    # Frequencies (computed numerically)
    function Omega_1(E, L)
        omega1, _ = compute_frequencies(E, L, V, dV)
        return omega1
    end

    function Omega_2(E, L)
        _, omega2 = compute_frequencies(E, L, V, dV)
        return omega2
    end

    # =============================================================================
    # CREATE MODEL RESULTS
    # =============================================================================

    # Helper functions (model-specific)
    helpers = (
        coefficients = coefficients,
        n_M = n_M,
        Jr = Jr,
        Omega_1 = Omega_1,
        Omega_2 = Omega_2
    )

    # Model parameters
    params = Dict{String,Any}(
        "n_M" => n_M,
        "unit_mass" => unit_mass,
        "unit_length" => unit_length,
        "selfgravity" => selfgravity,
        "potential" => "Plummer",
        "disk_type" => "Kuzmin-Toomre",
        "DF_type" => 0,
    )

    return ModelResults{Float64}(
        V,                    # potential
        dV,                   # potential_derivative
        Omega,                # rotation_frequency
        kappa,                # epicyclic_frequency
        DF,                   # distribution_function
        DF_dE,                # df_energy_derivative
        DF_dL,                # df_angular_derivative
        Sigma_d,              # surface_density
        nothing,              # velocity_dispersion (not implemented)
        helpers,              # helper_functions
        "Miyamoto",           # model_type
        params                # parameters
    )
end

end # module Miyamoto
