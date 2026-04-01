# src/models/KuzminTaperPoly3L.jl

"""
Kuzmin-Toomre model with poly3 taper in angular momentum (L) space
Based on Kalnajs (1976) Kuzmin-Toomre disk with poly3 taper T(L) = 1/2 + 3x/4 - x^3/4, x = L/L_*
Equation (1) from LinearMatrix.pdf adapted to L space.
Much simpler derivatives compared to circulation (v) taper.
Implements equations A14-A17 from LinearMatrix.pdf with poly3 L-taper.
"""
module KuzminTaperPoly3L

using SpecialFunctions
using QuadGK
using Interpolations
using Dierckx
using ..AbstractModel: AbstractGalacticModel, ModelResults
using CSV
using DataFrames

export KuzminTaperPoly3LModel, create_kuzmin_taper_poly3L_model, setup_model

# =============================================================================
# MODEL DEFINITION
# =============================================================================

struct KuzminTaperPoly3LModel <: AbstractGalacticModel end

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

"""
    create_kuzmin_taper_poly3L_model(config::PMEConfig) -> ModelResults

Create a Kuzmin-Toomre model with poly3 taper in angular momentum (L) space.
"""
function create_kuzmin_taper_poly3L_model(config)
    mk = config.model.mk

    # Helper: read from model.parameters (if present), then struct fields, with synonyms
    has_params = hasproperty(config.model, :parameters)
    function pget(keys::Vector{String}, default)
        for k in keys
            if has_params && haskey(getfield(config.model, :parameters), k)
                return getfield(config.model, :parameters)[k]
            end
            if hasproperty(config.model, Symbol(k))
                return getfield(config.model, Symbol(k))
            end
        end
        return default
    end

    # Support both L_star and legacy L_0
    L_star = pget(["L_star", "L_0"], 0.2)
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity

    return setup_model(KuzminTaperPoly3LModel, mk, L_star;
                       unit_mass=unit_mass,
                       unit_length=unit_length,
                       selfgravity=selfgravity)
end

# =============================================================================
# G-FUNCTION COMPUTATION (Same as base Kuzmin model)
# =============================================================================

"""
    compute_g_function(m::Int; dump_csv::Bool=false) -> Interpolation

Compute g(x) for Kuzmin-Toomre model from scratch using equations A16-A17.
If dump_csv=true, writes g_kuzmin_poly3L_m<m>.csv for verification.
"""
function compute_g_function(m::Int; dump_csv::Bool=false)
    # Grid setup
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
    # if dump_csv
        fn = "g_kuzmin_poly3L_m$(m).csv"
        df = DataFrame(x=x, g=g)
        CSV.write(fn, df)
        println("Wrote $fn ($(length(x)) rows)")
    # end
    
    # Create cubic spline interpolation
    g_spline = cubic_spline_interp(x, g)
    
    return g_spline
end



"""
Compute integrand for g3 calculation (equation A9).
"""
function integrand_function(t::Real, xi::Real, m::Int)
    if t >= 1.0
        return 0.0
    end
    
    u = t * xi
    if abs(u) >= 1.0 - 1e-12
        return 0.0
    end
    
    tau_val = (1.0 - u^2)^((3.0 - m)/2.0) / (2π)
    
    P_coeffs, coeffs_desc = get_legendre_derivatives(m - 1)
    P_double_prime_val = evalpoly(t, coeffs_desc)
    
    return tau_val * t^m * P_double_prime_val
end

"""
    compute_sigma_u_function(m::Int) -> Interpolation

Compute σ_u² integration for velocity dispersion calculation.
"""
function compute_sigma_u_function(m::Int)
    xu = collect(range(0.0, 0.999, length=2001))
    dxu = xu[2] - xu[1]
    
    t_tau = [(1.0 - u^2)^((3.0 - m)/2.0) / (2π) for u in xu] .* (2.0 .* xu).^m
    
    gu = zeros(length(xu))
    gu[2:end] .= (t_tau[1:end-1] .+ t_tau[2:end]) .* dxu ./ 2.0
    
    cumulative_gu = cumsum(gu)
    sigma_u_spline = cubic_spline_interp(xu, cumulative_gu)
    
    return sigma_u_spline
end

# =============================================================================
# SETUP MODEL
# =============================================================================

"""
    setup_model(::Type{KuzminTaperPoly3LModel}, mk, L_star; kwargs...) -> ModelResults

Initialize Kuzmin-Toomre model with poly3 taper in angular momentum (L) space.
Taper function: T(L) = 1/2 + 3x/4 - x^3/4, where x = L/L_*
From Equation (1) of LinearMatrix.pdf with v → L, v_* → L_*.
"""
function setup_model(::Type{KuzminTaperPoly3LModel}, mk::Int, L_star::Real; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    if mk < 1; error("mk must be >= 1, got $mk"); end
    if L_star <= 0; error("L_star must be positive, got $L_star"); end
    
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Kuzmin-Toomre model with poly3 L-taper (mk=$mk, L_*=$L_star, selfgravity=$selfgravity)")
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

    # Poly3 Taper in L space: T(L) = 1/2 + 3x/4 - x^3/4, x = L/L_*
    # For L < 0: T = 0; for x > 1: T = 1
    function taper(L)
        x = L / L_star
        if x < -1.0
            return 0.0
        elseif x > 1.0
            return 1.0
        else
            return 0.5 + 0.75*x - 0.25*x^3
        end
    end
    
    # Derivative: dT/dL = (3/4 - 3x^2/4) / L_* for 0 <= x <= 1; 0 elsewhere
    function taper_deriv(L)
        x = L / L_star
        if x < -1.0 || x > 1.0
            return 0.0
        else
            return (0.75 - 0.75*x^2) / L_star
        end
    end

    # Compute g function
    g_interp = compute_g_function(mk)
    
    # Compute sigma_u function
    ppu_interp = compute_sigma_u_function(mk)
    
    # Velocity dispersion
    sigma_u(r) = sqrt( ppu_interp(-r.*V(r)) ./ r.^(mk+1) ./ Sigma_d(r) )

    # DF with L-taper - simpler than v-taper, no grid coordinates needed
    function DF(E::Real, L::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(L)
        F_base = (-2.0 * E)^(mk - 1) * g_interp(xi)
        
        return F_base * taper(L)
    end

    # Energy derivative - simpler than v-taper version
    function DF_dE(E::Real, L::Real)::Real
        if E >= 0.0
            return 0.0
        end

        xi = sqrt(-2.0 * E) * abs(L)
        g_val = g_interp(xi)
        
        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end
        
        # d/dE[(-2E)^(mk-1) * g(ξ)] where ξ = √(-2E) * |L|
        term1 = (mk - 1) * (-2.0 * E)^(mk - 2) * (-2.0) * g_val
        term2 = (-2.0 * E)^(mk - 1) * g_prime * abs(L) / sqrt(-2.0 * E) * (-1.0)
        
        dF_base_dE = term1 + term2
        
        T_val = taper(L)
        
        # No dT/dE term since taper depends only on L
        return dF_base_dE * T_val
    end

    # L derivative - includes taper derivative
    function DF_dL(E::Real, L::Real)::Real
        if E >= 0.0
            return 0.0
        end

        xi = sqrt(-2.0 * E) * abs(L)
        
        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end
        
        # d/dL[(-2E)^(mk-1) * g(ξ)] where ξ = √(-2E) * |L| and d|L|/dL = sign(L) (0 at L=0)
        dF_base_dL = (-2.0 * E)^(mk - 1) * g_prime * sqrt(-2.0 * E) * sign(L)
        
        T_val = taper(L)
        dT_dL = taper_deriv(L)
        
        F_base = (-2.0 * E)^(mk - 1) * g_interp(xi)
        
        # Product rule: d/dL[F_base * T] = dF_base/dL * T + F_base * dT/dL
        return dF_base_dL * T_val + F_base * dT_dL
    end

    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk, "L_star" => L_star,
        "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "DF_type" => 0, "taper_type" => "Poly3L"
    )
    
    helpers = (
        model_name = "Kalnajs Kuzmin-Toomre - Poly3 L-Taper",
        title = "Kuzmin Poly3L mk=$mk, L_*=$L_star",
        V = V, dV = dV, Omega = Omega, kappa = kappa,
        Sigma_d = Sigma_d, sigma_u = sigma_u,
        taper = taper, taper_deriv = taper_deriv
    )
    
    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers,
        "KuzminTaperPoly3L",
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

"""
    cubic_spline_interp(x, y) -> Interpolation

Create cubic spline interpolation that supports derivative evaluation.
"""
function cubic_spline_interp(x::Vector, y::Vector)
    spline = Spline1D(x, y, k=3, bc="nearest")
    return spline
end

end # module KuzminTaperPoly3L
