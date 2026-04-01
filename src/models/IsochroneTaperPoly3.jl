# src/models/IsochroneTaperPoly3.jl

"""
Isochrone model with poly3 taper in circulation space
Based on Kalnajs isochrone with poly3 taper T(v) = 1/2 + 3x/4 - x^3/4, x = v/v_0
Implements equations from LinearMatrix.pdf with poly3 taper.
"""
module IsochroneTaperPoly3

using QuadGK: quadgk
using Dierckx: Spline1D, derivative
using ..AbstractModel: AbstractGalacticModel, ModelResults

export IsochroneTaperPoly3Model, create_isochrone_taper_poly3_model, setup_model

# =============================================================================
# MODEL DEFINITION
# =============================================================================

struct IsochroneTaperPoly3Model <: AbstractGalacticModel end

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

"""
    create_isochrone_taper_poly3_model(config) -> ModelResults

Create an Isochrone model with poly3 taper in circulation space from configuration parameters.
"""
function create_isochrone_taper_poly3_model(config)
    mk = config.model.mk
    v_0 = hasproperty(config.model, :v_0) ? getfield(config.model, :v_0) : 0.2
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity

    return setup_model(IsochroneTaperPoly3Model, mk, v_0;
                       unit_mass=unit_mass,
                       unit_length=unit_length,
                       selfgravity=selfgravity)
end

# =============================================================================
# G-FUNCTION COMPUTATION (Isochrone)
# =============================================================================

"""
    compute_g_function(m::Int) -> Spline1D

Compute the g(x) function for isochrone model via numerical integration.
Returns a cubic spline interpolation for efficient evaluation.
"""
function compute_g_function(m::Int)
    # Grid setup (optimized spacing)
    x_eps = 1e-5
    x = reverse(1.0 .- 10.0 .^ range(log10(x_eps), 0, length=1001))

    # Coordinate transformation
    r = 2.0 * x ./ (1.0 .- x.^2)
    d_r = 2.0 * (x.^2 .+ 1.0) ./ (1.0 .- x.^2).^2

    # Isochrone helper functions
    s = sqrt.(r.^2 .+ 1.0)
    L = log.(r .+ s) .- r ./ s
    tau = L .* (s .+ 1.0).^m ./ (2π) ./ r.^3

    # Derivatives
    d_L_dr = r.^2 ./ s.^3
    d_s_dr = r ./ s
    d_tau_dr = tau .* (d_L_dr ./ L .+ m ./ (1.0 .+ s) .* r ./ s .- 3.0 ./ r)

    # Components
    g1 = m * tau .+ x .* d_tau_dr .* d_r
    g2 = m * (m - 1) / 2.0 * tau

    # Compute g3 via numerical integration
    P_coeffs, coeffs_desc = get_legendre_derivatives(m - 1)

    g3 = zeros(length(x))
    for i in 2:length(x)
        xi = x[i]
        integrand(t) = integrand_function(t, xi, m, coeffs_desc)
        try
            g3[i], _ = quadgk(integrand, 0.0, 1.0, rtol=1e-10, atol=1e-12)
        catch
            g3[i] = 0.0
        end
    end

    # Final result
    g_total = (g1 .- g2 .+ g3) ./ π

    # Set boundary value analytically: g_total(1) = m/(3pi^2)
    g_total[1] = m / (3 * π^2)

    # Create cubic spline interpolation
    g_spline = cubic_spline_interp(x, g_total)

    return g_spline
end

"""
Compute integrand for g3 calculation.
"""
function integrand_function(t::Real, xi::Real, m::Int, coeffs_desc::Vector{Float64})
    if t >= 1.0
        return 0.0
    end

    u = t * xi
    if abs(u) >= 1.0 - 1e-12
        return 0.0
    end

    # Calculate r = 2u/(1-u^2)
    r_val = 2.0 * u / (1.0 - u^2)

    # Calculate tau(r)
    s_val = sqrt(r_val^2 + 1.0)
    L_val = log(r_val + s_val) - r_val / s_val

    if r_val >= 1e-3
        tau_val = L_val * (s_val + 1.0)^m / (2π) / r_val^3
    else
        # Taylor expansion for small r
        tau_val = (-35*r_val^6/144 + 15*r_val^4/56 - 3*r_val^2/10 + 1/3) / 2π * 2^m
    end

    # Legendre polynomial second derivative
    P_double_prime_val = evalpoly(t, coeffs_desc)

    return tau_val * t^m * P_double_prime_val
end

"""
    get_legendre_derivatives(n::Int) -> (Vector{Float64}, Vector{Float64})

Compute Legendre polynomial coefficients and second derivative coefficients.
"""
function get_legendre_derivatives(n::Int)
    if n == 0
        return [1.0], [0.0]
    elseif n == 1
        return [0.0, 1.0], [0.0]
    end

    # Recurrence relation for Legendre polynomials
    P0 = [1.0]
    P1 = [0.0, 1.0]

    for k in 1:(n-1)
        xPk = vcat([0.0], P1)
        target_length = k + 2

        xPk_padded = vcat(xPk, zeros(target_length - length(xPk)))
        P0_padded = vcat(P0, zeros(target_length - length(P0)))

        P_new = ((2*k + 1) * xPk_padded - k * P0_padded) / (k + 1)

        P0 = P1
        P1 = P_new[1:target_length]
    end
    P_coeffs = P1

    # Compute second derivative
    P_double_prime_coeffs = compute_second_derivative_coeffs(P_coeffs)

    return P_coeffs, P_double_prime_coeffs
end

"""
    compute_second_derivative_coeffs(poly_coeffs::Vector{Float64}) -> Vector{Float64}

Compute second derivative coefficients from polynomial coefficients.
"""
function compute_second_derivative_coeffs(poly_coeffs::Vector{Float64})
    if length(poly_coeffs) < 3
        return [0.0]
    end

    n = length(poly_coeffs)
    second_deriv_coeffs = zeros(max(1, n - 2))

    for k in 3:n
        power = k - 1
        coeff = poly_coeffs[k]
        new_power = power - 2
        new_coeff = coeff * power * (power - 1)

        if new_power >= 0 && new_power + 1 <= length(second_deriv_coeffs)
            second_deriv_coeffs[new_power + 1] = new_coeff
        end
    end

    # Remove trailing zeros
    while length(second_deriv_coeffs) > 1 && abs(second_deriv_coeffs[end]) < 1e-15
        pop!(second_deriv_coeffs)
    end

    if all(abs.(second_deriv_coeffs) .< 1e-15)
        return [0.0]
    end

    return second_deriv_coeffs
end

# =============================================================================
# SIGMA_U FUNCTION COMPUTATION
# =============================================================================

"""
    tau_f(u, m) -> Float64

Calculate tau function used for sigma_u calculation.
"""
function tau_f(u::Real, m::Int)
    if abs(u) >= 1.0 - 1e-12
        return 0.0
    end

    # Calculate r = 2u/(1-u^2)
    r_val = 2.0 * u / (1.0 - u^2)

    # Calculate tau(r)
    s_val = sqrt(r_val^2 + 1.0)
    L_val = log(r_val + s_val) - r_val / s_val

    if r_val >= 1e-3
        tau_val = L_val * ((s_val + 1.0)/2.0)^m / (2π) / r_val^3
    else
        # Taylor expansion for small r
        tau_val = (-35.0*r_val^6/144.0 + 15.0*r_val^4/56.0 - 3.0*r_val^2/10.0 + 1/3.0) / 2π
    end

    return tau_val
end

"""
    compute_sigma_u_function(m::Int) -> Spline1D

Compute sigma_u(r) function for velocity dispersion via numerical integration.
"""
function compute_sigma_u_function(m::Int)
    # Grid setup
    xu_eps = 1e-6
    logspace_values = 10.0 .^ range(log10(xu_eps), 0, length=10001)
    xu_temp = 1.0 .- logspace_values
    xu = reverse(xu_temp)
    dxu = diff(xu)

    # Calculate tau_f for each xu value
    t_tau = [tau_f(u, m) for u in xu] .* (2.0 .* xu).^m

    # Numerical integration using trapezoidal rule
    gu = zeros(length(xu))
    gu[2:end] .= (t_tau[1:end-1] .+ t_tau[2:end]) .* dxu ./ 2.0

    # Cumulative sum for the integral
    cumulative_gu = cumsum(gu)

    sigma_u_spline = cubic_spline_interp(xu, cumulative_gu)

    return sigma_u_spline
end

# =============================================================================
# SETUP MODEL
# =============================================================================

"""
    setup_model(::Type{IsochroneTaperPoly3Model}, mk, v_0; kwargs...) -> ModelResults

Initialize Isochrone model with poly3 taper in circulation space.
Taper function: T(v) = 1/2 + 3x/4 - x^3/4, where x = v/v_0
"""
function setup_model(::Type{IsochroneTaperPoly3Model}, mk::Int, v_0::Real;
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    if mk < 1; error("mk must be >= 1, got $mk"); end
    if v_0 <= 0; error("v_0 must be positive, got $v_0"); end

    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model with poly3 taper (mk=$mk, v_0=$v_0, selfgravity=$selfgravity)")
    end

    # Isochrone potential
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt(sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0))
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))

    # Surface density
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)

    # Poly3 Taper: T(v) = 1/2 + 3x/4 - x^3/4, x = v/v_0
    # For |x| <= 1; T(x) = 0 for x < -1; T(x) = 1 for x > 1
    function taper(v)
        x = v / v_0
        if x < -1.0
            return 0.0
        elseif x > 1.0
            return 1.0
        else
            return 0.5 + 0.75*x - 0.25*x^3
        end
    end

    # Derivative: dT/dv = (3/4 - 3x^2/4) / v_0 for |x| <= 1; 0 elsewhere
    function taper_deriv(v)
        x = v / v_0
        if x < -1.0 || x > 1.0
            return 0.0
        else
            return (0.75 - 0.75*x^2) / v_0
        end
    end

    # Compute g function
    g_interp = compute_g_function(mk)

    # Compute sigma_u function
    ppu_interp = compute_sigma_u_function(mk)

    # Velocity dispersion (not tapered)
    sigma_u(r) = sqrt( ppu_interp(-r.*V(r)) ./ r.^(mk+1) ./ Sigma_d(r) )

    # DF with circulation taper
    function DF(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        if E >= 0.0
            return 0.0
        end

        v = grids.v[iR, iv]

        xi = sqrt(-2.0 * E) * abs(L)
        F_base = (-2.0 * E)^(mk - 1) * g_interp(xi)

        return F_base * taper(v)
    end

    # Energy derivative
    function DF_dE_inner(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        if E >= 0.0
            return 0.0
        end

        r1 = grids.R1[iR, iv]; r2 = grids.R2[iR, iv]; Rc = grids.Rc[iR, iv]
        sign_L = grids.SGNL[iR, iv]
        v = grids.v[iR, iv]

        # Compute dv/dE
        if abs(L) < 1e-10
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

        xi = sqrt(-2.0 * E) * abs(L)
        g_val = g_interp(xi)

        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end

        # d/dE[(-2E)^(mk-1) * g(xi)] where xi = sqrt(-2E) * |L|
        term1 = (mk - 1) * (-2.0 * E)^(mk - 2) * (-2.0) * g_val
        term2 = (-2.0 * E)^(mk - 1) * g_prime * abs(L) / sqrt(-2.0 * E) * (-1.0)

        dF_base_dE = term1 + term2

        T_val = taper(v)
        dT_dv = taper_deriv(v)

        F_base = (-2.0 * E)^(mk - 1) * g_val

        return dF_base_dE * T_val + F_base * dT_dv * dv_dE
    end

    # L derivative
    function DF_dL_inner(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        if E >= 0.0
            return 0.0
        end

        r1 = grids.R1[iR, iv]; r2 = grids.R2[iR, iv]; Rc = grids.Rc[iR, iv]
        sign_L = grids.SGNL[iR, iv]
        v = grids.v[iR, iv]

        # Compute dv/dL
        if abs(L) < 1e-10
            # Proper limit as L -> 0
            V0 = V(0.0)
            dv_dL = 1.0 / (Rc * sqrt(2.0 * (E - V0)))
        else
            V1 = V(r1); V2 = V(r2); dV1 = dV(r1); dV2 = dV(r2)
            t1 = 2*(E - V1)*r1 - dV1*r1^2
            t2 = 2*(E - V2)*r2 - dV2*r2^2
            dr1_dL = L / t1
            dr2_dL = L / t2
            dRc_dL = 0.5*(dr1_dL + dr2_dL)
            dv_dL = sign_L * (dr1_dL/Rc - r1*dRc_dL/Rc^2)
        end

        xi = sqrt(-2.0 * E) * abs(L)

        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end

        # d/dL[(-2E)^(mk-1) * g(xi)] where xi = sqrt(-2E) * |L|
        dF_base_dL = (-2.0 * E)^(mk - 1) * g_prime * sqrt(-2.0 * E) * sign(L)

        T_val = taper(v)
        dT_dv = taper_deriv(v)

        F_base = (-2.0 * E)^(mk - 1) * g_interp(xi)

        return dF_base_dL * T_val + F_base * dT_dv * dv_dL
    end

    # Wrapper functions for grid-based evaluation
    function DF_dE(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        return DF_dE_inner(E, L, iR, iv, grids)
    end

    function DF_dL(E::Real, L::Real, iR::Int, iv::Int, grids)::Real
        return DF_dL_inner(E, L, iR, iv, grids)
    end

    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk, "v_0" => v_0,
        "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "DF_type" => 0, "taper_type" => "Poly3"
    )

    helpers = (
        model_name = "Kalnajs Isochrone - Poly3 Taper",
        title = "Isochrone Poly3 mk=$mk, v_0=$v_0",
        V = V, dV = dV, Omega = Omega, kappa = kappa,
        Sigma_d = Sigma_d, sigma_u = sigma_u,
        taper = taper, taper_deriv = taper_deriv
    )

    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers,
        "IsochroneTaperPoly3",
        params,
    )
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    cubic_spline_interp(x, y) -> Spline1D

Create cubic spline interpolation that supports derivative evaluation.
"""
function cubic_spline_interp(x::Vector{T}, y::Vector{T}) where T
    return Spline1D(x, y, k=3)
end

end # module IsochroneTaperPoly3
