# src/models/IsochroneTaperJH.jl
"""
    IsochroneTaperJH

Isochrone disc model with the Jalali & Hunter (2005) taper applied to the
distribution function. The JH taper subtracts a term proportional to
1/(1 + Jr + |L|)^(2m-2) to smoothly suppress retrograde orbits.

Reference potential:  V(r) = -1 / (1 + sqrt(r^2 + 1))
"""
module IsochroneTaperJH

using SpecialFunctions
using QuadGK
using Dierckx: Spline1D, derivative

using ..AbstractModel
using ..Configuration

export IsochroneTaperJHModel, create_isochrone_taper_jh_model, setup_model

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------
struct IsochroneTaperJHModel <: AbstractGalacticModel end

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
function create_isochrone_taper_jh_model(config::PMEConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity
    return setup_model(IsochroneTaperJHModel, mk; unit_mass, unit_length, selfgravity)
end

# ===========================================================================
# Shared helper functions
# ===========================================================================

function cubic_spline_interp(x::Vector{T}, y::Vector{T}) where T
    return Spline1D(x, y, k=3)
end

function compute_second_derivative_coeffs(poly_coeffs::Vector{Float64})
    if length(poly_coeffs) < 3; return [0.0]; end
    n = length(poly_coeffs)
    second_deriv_coeffs = zeros(max(1, n - 2))
    for k in 3:n
        power = k - 1; coeff = poly_coeffs[k]; new_power = power - 2
        new_coeff = coeff * power * (power - 1)
        if new_power >= 0 && new_power + 1 <= length(second_deriv_coeffs)
            second_deriv_coeffs[new_power + 1] = new_coeff
        end
    end
    while length(second_deriv_coeffs) > 1 && abs(second_deriv_coeffs[end]) < 1e-15
        pop!(second_deriv_coeffs)
    end
    if all(abs.(second_deriv_coeffs) .< 1e-15); return [0.0]; end
    return second_deriv_coeffs
end

function get_legendre_derivatives(n::Int)
    if n == 0; return [1.0], [0.0]; end
    if n == 1; return [0.0, 1.0], [0.0]; end
    P0 = [1.0]; P1 = [0.0, 1.0]
    for k in 1:(n-1)
        xPk = vcat([0.0], P1)
        target_length = k + 2
        xPk_padded = vcat(xPk, zeros(target_length - length(xPk)))
        P0_padded = vcat(P0, zeros(target_length - length(P0)))
        P_new = ((2*k + 1) * xPk_padded - k * P0_padded) / (k + 1)
        P0 = P1; P1 = P_new[1:target_length]
    end
    P_coeffs = P1
    P_double_prime_coeffs = compute_second_derivative_coeffs(P_coeffs)
    return P_coeffs, P_double_prime_coeffs
end

function integrand_function(t::Real, xi::Real, m::Int, coeffs_desc::Vector{Float64})
    if t >= 1.0; return 0.0; end
    u = t * xi
    if abs(u) >= 1.0 - 1e-12; return 0.0; end
    r_val = 2.0 * u / (1.0 - u^2)
    s_val = sqrt(r_val^2 + 1.0)
    L_val = log(r_val + s_val) - r_val / s_val
    if r_val >= 1e-3
        tau_val = L_val * (s_val + 1.0)^m / (2π) / r_val^3
    else
        tau_val = (-35*r_val^6/144 + 15*r_val^4/56 - 3*r_val^2/10 + 1/3) / 2π * 2^m
    end
    P_double_prime_val = evalpoly(t, coeffs_desc)
    return tau_val * t^m * P_double_prime_val
end

function compute_g_function(m::Int)
    x_eps = 1e-5
    x = reverse(1.0 .- 10.0 .^ range(log10(x_eps), 0, length=1001))
    r = 2.0 * x ./ (1.0 .- x.^2)
    d_r = 2.0 * (x.^2 .+ 1.0) ./ (1.0 .- x.^2).^2
    s = sqrt.(r.^2 .+ 1.0)
    L = log.(r .+ s) .- r ./ s
    tau = L .* (s .+ 1.0).^m ./ (2π) ./ r.^3
    d_L_dr = r.^2 ./ s.^3
    d_s_dr = r ./ s
    d_tau_dr = tau .* (d_L_dr ./ L .+ m ./ (1.0 .+ s) .* r ./ s .- 3.0 ./ r)
    g1 = m * tau .+ x .* d_tau_dr .* d_r
    g2 = m * (m - 1) / 2.0 * tau
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
    g_total = (g1 .- g2 .+ g3) ./ π
    g_total[1] = m / (3 * π^2)
    g_spline = cubic_spline_interp(x, g_total)
    return g_spline
end

function tau_f(u::Real, m::Int)
    if abs(u) >= 1.0 - 1e-12; return 0.0; end
    r_val = 2.0 * u / (1.0 - u^2)
    s_val = sqrt(r_val^2 + 1.0)
    L_val = log(r_val + s_val) - r_val / s_val
    if r_val >= 1e-3
        tau_val = L_val * ((s_val + 1.0)/2.0)^m / (2π) / r_val^3
    else
        tau_val = (-35.0*r_val^6/144.0 + 15.0*r_val^4/56.0 - 3.0*r_val^2/10.0 + 1/3.0) / 2π
    end
    return tau_val
end

function compute_sigma_u_function(m::Int)
    xu_eps = 1e-6
    logspace_values = 10.0 .^ range(log10(xu_eps), 0, length=10001)
    xu_temp = 1.0 .- logspace_values
    xu = reverse(xu_temp)
    dxu = diff(xu)
    t_tau = [tau_f(u, m) for u in xu] .* (2.0 .* xu).^m
    gu = zeros(length(xu))
    gu[2:end] .= (t_tau[1:end-1] .+ t_tau[2:end]) .* dxu ./ 2.0
    cumulative_gu = cumsum(gu)
    sigma_u_spline = cubic_spline_interp(xu, cumulative_gu)
    return sigma_u_spline
end

# ===========================================================================
# Distribution function (JH taper)
# ===========================================================================

function create_jh_distribution_functions(m::Int, g_interp)
    function DF(E::Real, h::Real)::Real
        if E >= 0.0; return 0.0; end
        Jr = 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
        taper_term = m / (6.0*π^2) / (1.0 + Jr + abs(h))^(2*m - 2)
        if h >= 0.0
            xi = sqrt(-2.0 * E) * h
            g_val = g_interp(xi)
            main_term = (-2.0 * E)^(m - 1) * g_val
            return main_term - taper_term
        else
            return taper_term
        end
    end
    function DF_dE(E::Real, h::Real)::Real
        if E >= 0.0; return 0.0; end
        Jr = 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
        dJr_dE = 1.0 / ((-2.0*E)^(3/2))
        if h > 0.0
            xi = sqrt(-2.0 * E) * h
            g_val = g_interp(xi)
            g_prime = try; derivative(g_interp, xi); catch; 0.0; end
            term1a = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
            term1b = (-2.0*E)^(m - 1) * g_prime * (-h / sqrt(-2.0*E))
            term2 = m/(6.0*π^2) * (2*m - 2) * dJr_dE / (1.0 + Jr + abs(h))^(2*m - 1)
            return term1a + term1b + term2
        elseif h < 0.0
            return -m/(6.0*π^2) * (2*m - 2) * dJr_dE / (1.0 + Jr + abs(h))^(2*m - 1)
        else
            xi = 0.0; g_val = g_interp(xi)
            term1a = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
            Jr_zero = 1.0/sqrt(-2.0*E) - 1.0
            term2 = m/(6.0*π^2) * (2*m - 2) * dJr_dE / (1.0 + Jr_zero)^(2*m - 1)
            return term1a + term2
        end
    end
    function DF_dL(E::Real, h::Real)::Real
        if E >= 0.0; return 0.0; end
        Jr = 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
        if h > 0.0
            dJr_dh = -(1.0 + h/sqrt(h^2 + 4.0))/2.0
            dabs_h_dh = 1.0
        else
            dJr_dh = -(-1.0 + h/sqrt(h^2 + 4.0))/2.0
            dabs_h_dh = -1.0
        end
        if h > 0.0
            xi = sqrt(-2.0 * E) * h
            g_prime = try; derivative(g_interp, xi); catch; 0.0; end
            term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E)
            term2 = m/(6.0*π^2) * (2*m - 2) * (dJr_dh + dabs_h_dh) / (1.0 + Jr + abs(h))^(2*m - 1)
            return term1 + term2
        elseif h < 0.0
            term2 = m/(6.0*π^2) * (2*m - 2) * (dJr_dh + dabs_h_dh) / (1.0 + Jr + abs(h))^(2*m - 1)
            return -term2
        else
            g_prime = try; derivative(g_interp, 0.0); catch; 0.0; end
            term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E)
            term2 = m/(6.0*π^2) * (2*m - 2) * 0.5 / (1.0 + Jr)^(2*m - 1)
            return term1 + term2
        end
    end
    return DF, DF_dE, DF_dL
end

# ===========================================================================
# Setup
# ===========================================================================

function setup_model(::Type{IsochroneTaperJHModel}, mk::Int;
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model with JH taper (mk=$mk, selfgravity=$selfgravity)")
    end
    g_interp = compute_g_function(mk)
    ppu_interp = compute_sigma_u_function(mk)
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt(sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0))
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    sigma_u(r) = sqrt(ppu_interp(-r.*V(r)) ./ r.^(mk+1) ./ Sigma_d(r))
    DF, DF_dE, DF_dL = create_jh_distribution_functions(mk, g_interp)
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "DF_type" => 0, "taper_type" => "JH"
    )
    helpers = (
        model_name = "Kalnajs Isochrone - JH Taper (Jalali & Hunter 2005)",
        title = "JH Isochrone mk=$mk",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
    )
    return ModelResults{Float64}(
        V, dV, Omega, kappa, DF, DF_dE, DF_dL,
        Sigma_d, sigma_u, helpers, "IsochroneTaperJH", params
    )
end

end # module IsochroneTaperJH
