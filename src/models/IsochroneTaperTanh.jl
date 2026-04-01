# src/models/IsochroneTaperTanh.jl
"""
    IsochroneTaperTanh

Isochrone disc model with a tanh-based taper applied to the distribution function.
The taper is energy-dependent: at each energy, the circular-orbit angular momentum
Lc(E) sets the scale, and the DF is multiplied by (1 + tanh(L / (eta * Lc))) / 2.

Reference potential:  V(r) = -1 / (1 + sqrt(r^2 + 1))
"""
module IsochroneTaperTanh

using SpecialFunctions
using QuadGK
using Dierckx: Spline1D, derivative

using ..AbstractModel
using ..Configuration

export IsochroneTaperTanhModel, create_isochrone_taper_tanh_model, setup_model

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------
struct IsochroneTaperTanhModel <: AbstractGalacticModel end

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
function create_isochrone_taper_tanh_model(config::PMEConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity
    eta = config.model.eta
    return setup_model(IsochroneTaperTanhModel, mk; unit_mass, unit_length, selfgravity, eta)
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
# Distribution function (Tanh taper, energy-dependent Lc)
# ===========================================================================

function create_tanh_distribution_functions_energy_dependent(m::Int, g_interp, eta::Real)
    taper(x) = (1.0 + tanh(x)) / 2.0
    taper_deriv(x) = 0.5 * (1.0 - tanh(x)^2)
    function DF_core(E::Real, h::Real, Lc_val::Real, eta::Real)::Real
        if E >= 0.0; return 0.0; end
        xi = sqrt(-2.0 * E) * abs(h)
        main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        return main_term * taper_val
    end
    function DF_dE_core(E::Real, h::Real, Lc_val::Real, omega_inv::Real, eta::Real)::Real
        if E >= 0.0; return 0.0; end
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        g_prime = try; derivative(g_interp, xi); catch; 0.0; end
        term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
        term2 = (-2.0*E)^(m - 1) * g_prime * (-abs(h) / sqrt(-2.0*E))
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        taper_d = taper_deriv(x_arg)
        dtaper_dE = taper_d * (-h / (eta * Lc_val^2)) * omega_inv
        return (term1 + term2) * taper_val + (-2.0*E)^(m - 1) * g_val * dtaper_dE
    end
    function DF_dL_core(E::Real, h::Real, Lc_val::Real, eta::Real)::Real
        if E >= 0.0; return 0.0; end
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        g_prime = try; derivative(g_interp, xi); catch; 0.0; end
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        taper_d = taper_deriv(x_arg)
        dtaper_dL = taper_d / (eta * Lc_val)
        term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * sign(h) * taper_val
        term2 = (-2.0*E)^(m - 1) * g_val * dtaper_dL
        return term1 + term2
    end
    return DF_core, DF_dE_core, DF_dL_core
end

# ===========================================================================
# Setup
# ===========================================================================

function setup_model(::Type{IsochroneTaperTanhModel}, mk::Int;
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0, eta::Real=0.1)
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model with Tanh taper (energy-dependent Lc) (mk=$mk, selfgravity=$selfgravity, eta=$eta)")
    end
    g_interp = compute_g_function(mk)
    ppu_interp = compute_sigma_u_function(mk)
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt(sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0))
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    sigma_u(r) = sqrt(ppu_interp(-V(r) * r) / r^(mk+1) / Sigma_d(r))
    DF_core, DF_dE_core, DF_dL_core = create_tanh_distribution_functions_energy_dependent(mk, g_interp, eta)
    # Wrappers with grid access
    function DF(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing; error("Grid data is required for Tanh model"); end
        Lc_val = grids.L_m[iR, end]
        return DF_core(E, h, Lc_val, eta)
    end
    function DF_dE(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing; error("Grid data is required for Tanh model"); end
        Lc_val = grids.L_m[iR, end]
        Rc_val = grids.Rc[iR, 1]
        omega_inv = 1.0 / Omega(Rc_val)
        return DF_dE_core(E, h, Lc_val, omega_inv, eta)
    end
    function DF_dL(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing; error("Grid data is required for Tanh model"); end
        Lc_val = grids.L_m[iR, end]
        return DF_dL_core(E, h, Lc_val, eta)
    end
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "eta" => eta, "taper_type" => "TanhEnergy"
    )
    helpers = (
        model_name = "Kalnajs Isochrone - Tanh Taper (E-dependent)",
        title = "Tanh Isochrone mk=$mk, eta=$eta",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
    )
    return ModelResults{Float64}(
        V, dV, Omega, kappa, DF, DF_dE, DF_dL,
        Sigma_d, sigma_u, helpers, "IsochroneTaperTanh", params
    )
end

end # module IsochroneTaperTanh
