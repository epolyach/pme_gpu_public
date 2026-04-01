# src/models/Kuzmin.jl

"""
Kuzmin-Toomre model with sharp edge at L=0 (no taper).
Unidirectional formulation (L >= 0 only) for use with sharp_df=true.
Based on Kalnajs (1976) Kuzmin-Toomre disk, equations A14-A17 from LinearMatrix.pdf.
"""
module Kuzmin

# NOTE: Sharp-edge model (no taper). Requires the generalized matrix method.

using QuadGK: quadgk
using Dierckx: Spline1D, derivative
using CSV: write as csv_write
using DataFrames: DataFrame
using ..AbstractModel: AbstractGalacticModel, ModelResults

export KuzminModel, create_kuzmin_model, setup_model

# =============================================================================
# MODEL DEFINITION
# =============================================================================

struct KuzminModel <: AbstractGalacticModel end

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

"""
    create_kuzmin_model(config::PMEConfig) -> ModelResults

Create a plain (untapered) Kuzmin-Toomre model from configuration.
Unidirectional formulation (L >= 0) for use with sharp_df=true.
"""
function create_kuzmin_model(config)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity

    return setup_model(KuzminModel, mk;
                       unit_mass=unit_mass,
                       unit_length=unit_length,
                       selfgravity=selfgravity)
end

# =============================================================================
# G-FUNCTION COMPUTATION
# =============================================================================

"""
    compute_g_function(m::Int) -> Interpolation

Compute g(x) for Kuzmin-Toomre model from scratch using equations A16-A17.
"""
function compute_g_function(m::Int)
    # Grid setup - logarithmic near x=0, linear near x=1
    x_eps = 1e-5
    a = 10 .^ range(log10(x_eps), 0, length=1001)
    x = 1.0 .- a
    x = reverse(x)

    # Tau and derivative
    p = (3.0 - m) / 2.0
    dp = (1.0 - m) / 2.0
    tau = exp.(p .* log1p.(-x.^2)) ./ (2π)
    d_tau_dx = -(3.0 - m) .* x .* exp.(dp .* log1p.(-x.^2)) ./ (2π)

    # Components
    g1 = x .* d_tau_dx
    g2 = m * (m - 3) / 2.0 .* tau

    # Get Legendre coefficients
    P_coeffs, coeffs_desc = get_legendre_derivatives(m - 1)

    # Third term via quadrature
    g3 = zeros(length(x))
    for i in 2:length(x)
        xi = x[i]
        function integrand(t)
            u = t * xi
            tau_val = exp(p * log1p(-u^2)) / (2π)
            Ppp = evalpoly(t, coeffs_desc)
            return tau_val * t^m * Ppp
        end
        try
            g3[i], _ = quadgk(integrand, 0.0, 1.0, rtol=1e-12, atol=1e-14)
        catch
            g3[i] = 0.0
        end
    end

    # Total g(x)
    g = (2.0 / (2π)) .* (g1 .- g2 .+ g3)

    # Set g(0) to exact value
    g[1] = 3.0 / (32.0 * π^2)

    # Optional CSV dump for verification
    fn = "g_kuzmin_m$(m).csv"
    df = DataFrame(x=x, g=g)
    csv_write(fn, df)
    println("Wrote $fn ($(length(x)) rows)")

    # Create cubic spline interpolation
    g_spline = Spline1D(x, g, k=3, bc="nearest")

    return g_spline
end

"""
    compute_sigma_u_function(m::Int) -> Interpolation

Compute sigma_u^2 integration for velocity dispersion calculation.
"""
function compute_sigma_u_function(m::Int)
    xu = collect(range(0.0, 0.999, length=2001))
    dxu = xu[2] - xu[1]

    t_tau = [(1.0 - u^2)^((3.0 - m)/2.0) / (2π) for u in xu] .* (2.0 .* xu).^m

    gu = zeros(length(xu))
    gu[2:end] .= (t_tau[1:end-1] .+ t_tau[2:end]) .* dxu ./ 2.0

    cumulative_gu = cumsum(gu)
    sigma_u_spline = Spline1D(xu, cumulative_gu, k=3, bc="nearest")

    return sigma_u_spline
end

# =============================================================================
# SETUP MODEL
# =============================================================================

"""
    setup_model(::Type{KuzminModel}, mk; kwargs...) -> ModelResults

Initialize plain (untapered) Kuzmin-Toomre model.
Unidirectional DF (L >= 0 only) with sharp edge at L=0.
For use with sharp_df=true in the pipeline.
"""
function setup_model(::Type{KuzminModel}, mk::Int;
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    if mk < 1; error("mk must be >= 1, got $mk"); end

    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Kuzmin-Toomre model (plain, no taper) (mk=$mk, selfgravity=$selfgravity)")
    end

    # Kuzmin-Toomre potential (equation A14)
    V(r) = -1.0 / sqrt(1.0 + r^2)
    dV(r) = r / (1.0 + r^2)^(3/2)
    Omega(r) = 1.0 / (1.0 + r^2)^(3/4)
    function kappa(r)
        r2 = r^2
        return sqrt((r2 + 4.0) / (1.0 + r2)^(5/2))
    end
    Sigma_d(r) = 1.0 / (2π * (1.0 + r^2)^(3/2))

    # Compute g function
    g_interp = compute_g_function(mk)

    # Compute sigma_u function
    ppu_interp = compute_sigma_u_function(mk)

    # Velocity dispersion
    sigma_u(r) = sqrt(ppu_interp(-r .* V(r)) ./ r.^(mk+1) ./ Sigma_d(r))

    # DF without taper - simple 2-arg signature (E, L)
    # Unidirectional: returns 0 for L < 0
    function DF(E::Real, L::Real)::Real
        if E >= 0.0 || L < 0.0
            return 0.0
        end

        xi = sqrt(-2.0 * E) * L
        F_base = (-2.0 * E)^(mk - 1) * g_interp(xi)

        return F_base
    end

    # Energy derivative
    function DF_dE(E::Real, L::Real)::Real
        if E >= 0.0 || L < 0.0
            return 0.0
        end

        xi = sqrt(-2.0 * E) * L
        g_val = g_interp(xi)

        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end

        # d/dE[(-2E)^(mk-1) * g(xi)] where xi = sqrt(-2E) * L
        term1 = (mk - 1) * (-2.0 * E)^(mk - 2) * (-2.0) * g_val
        term2 = (-2.0 * E)^(mk - 1) * g_prime * L / sqrt(-2.0 * E) * (-1.0)

        return term1 + term2
    end

    # L derivative
    function DF_dL(E::Real, L::Real)::Real
        if E >= 0.0 || L < 0.0
            return 0.0
        end

        xi = sqrt(-2.0 * E) * L

        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end

        # d/dL[(-2E)^(mk-1) * g(xi)] = (-2E)^(mk-1) * g'(xi) * sqrt(-2E)
        return (-2.0 * E)^(mk - 1) * g_prime * sqrt(-2.0 * E)
    end

    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk,
        "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "DF_type" => 0, "taper_type" => "None"
    )

    helpers = (
        model_name = "Kalnajs Kuzmin-Toomre - Plain (no taper)",
        title = "Plain Kuzmin mk=$mk",
        V = V, dV = dV, Omega = Omega, kappa = kappa,
        Sigma_d = Sigma_d, sigma_u = sigma_u
    )

    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers,
        "Kuzmin",
        params,
    )
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    get_legendre_derivatives(n::Int) -> (Vector, Vector)

Compute coefficients for Legendre polynomial and its second derivative.
"""
function get_legendre_derivatives(n::Int)
    P_coeffs = zeros(n + 1)
    if n == 0
        P_coeffs[1] = 1.0
    elseif n == 1
        P_coeffs[2] = 1.0
    else
        P_prev = [1.0]
        P_curr = [0.0, 1.0]

        for k in 2:n
            P_next = zeros(k + 1)
            for i in 1:length(P_curr)
                if i + 1 <= length(P_next)
                    P_next[i + 1] += (2k - 1) * P_curr[i] / k
                end
            end
            for i in 1:length(P_prev)
                P_next[i] -= (k - 1) * P_prev[i] / k
            end
            P_prev = P_curr
            P_curr = P_next
        end
        P_coeffs = P_curr
    end

    # Compute second derivative coefficients
    coeffs_desc = zeros(max(0, length(P_coeffs) - 2))
    for i in 3:length(P_coeffs)
        coeffs_desc[i - 2] = P_coeffs[i] * (i - 1) * (i - 2)
    end

    return P_coeffs, coeffs_desc
end

end # module Kuzmin
