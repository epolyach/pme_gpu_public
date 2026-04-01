# src/models/ExpDisk.jl
"""
Implements the cored exponential disk model, including its potential,
kinematics, and distribution function, based on GModel_expdisk.m.
"""
module ExpDisk

using SpecialFunctions
using Symbolics
using ..AbstractModel
using ..Configuration

export ExpDiskModel, create_expdisk_model

"""
A type representing the exponential disk model.
"""
struct ExpDiskModel <: AbstractGalacticModel end

"""
    create_expdisk_model(config::PMEConfig) -> ModelResults

Factory function to create an ExpDisk model instance from a configuration object.
Extracts necessary parameters from `config.model`.
"""
function create_expdisk_model(config::PMEConfig)
    # Get parameters from model configuration
    RC = config.model.RC
    N = config.model.N
    lambda = config.model.lambda
    alpha = config.model.alpha
    L0 = config.model.L0
    selfgravity = config.physics.selfgravity

    return setup_model(ExpDiskModel, RC; N=N, lambda=lambda, alpha=alpha, L0=L0, selfgravity=selfgravity)
end

"""
    setup_model(::Type{ExpDiskModel}, RC; N, lambda, alpha, L0) -> ModelResults

Initializes the ExpDisk model with specified parameters and returns a `ModelResults`
struct containing all necessary functions for the calculation.
"""
function setup_model(::Type{ExpDiskModel}, RC::Real;
                     N::Int, lambda::Real, alpha::Real, L0::Real, selfgravity::Real=1.0)

    # Derived model parameters
    RC2 = RC^2
    RD = RC / lambda
    Sigma0 = alpha / RD
    v0 = 1.0  # Normalization constant
    v02 = v0^2

    # Core kinematic functions
    V(x) = v02/2 * log(1 + x^2/RC2)
    dV(x) = x * v02 / (RC2 + x^2)
    Omega(x) = v0 / sqrt(RC2 + x^2)
    kappa(x) = v0 / (RC2 + x^2) * sqrt(4*RC2 + 2*x^2)
    Sigma_d(x) = Sigma0 * exp(-lambda * sqrt(1 + x^2/RC2))

    # Create the distribution function and its derivatives
    DF, DF_dE, DF_dL = create_distribution_functions(v02, N, lambda, L0, RC, Sigma0)

    # Package parameters for reference
    params = Dict{String,Any}(
        "RC" => RC, "N" => N, "lambda" => lambda, "alpha" => alpha, "L0" => L0,
        "selfgravity" => selfgravity, "DF_type" => 0, "RD" => RD, "Sigma0" => Sigma0, "v0" => v0
    )

    # Helper functions and metadata
    helpers = (
        model_name = "Cored Exponential Disk",
        title = "Cored exp model; α=$alpha, λ=$lambda",
    )

    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, nothing, # velocity_dispersion is not defined
        helpers, "ExpDisk", params
    )
end

"""
    create_distribution_functions(v02, N, lambda, L0, RC, Sigma0)

Constructs the distribution function `DF(E,L)` and its derivatives for the ExpDisk model.
Uses symbolic computation to derive DF and its derivatives, then evaluates numerically.
"""
function create_distribution_functions(v02, N, lambda, L0, RC, Sigma0)

    # === SYMBOLIC COMPUTATION PHASE ===
    println("Computing symbolic distribution function for N=$N...")

    # Define symbolic variables - make them local to ensure proper capture
    @variables E_sym L_sym

    # Base exponential function f(E) = exp(-2*N*E/v02 - lambda*exp(E/v02))
    u_sym = -2*N*E_sym/v02 - lambda*exp(E_sym/v02)
    f_sym = exp(u_sym)

    # Compute symbolic derivatives of f(E) up to order N+1
    println("  Computing symbolic derivatives of f(E) up to order $(N+1)...")
    f_derivatives_sym = Vector{Num}(undef, N+2)
    f_derivatives_sym[1] = f_sym  # 0th derivative

    current_deriv = f_sym
    for i in 1:(N+1)
        current_deriv = Symbolics.derivative(current_deriv, E_sym)
        # println("    Derivative $i: $(Symbolics.simplify(current_deriv))")
        f_derivatives_sym[i+1] = current_deriv
    end

    # Helper function for binomial coefficients
    function binomial_coeff(n::Int, k::Int)::Float64
        if k < 0 || k > n
            return 0.0
        end
        return exp(lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1))
    end

    # Build the symbolic distribution function using binomial expansion
    println("  Building symbolic binomial expansion...")
    L_over_RC = L_sym / RC
    G_sym = 0

    for n in 0:N
        # Coefficient C_n = (-1)^(n+1) / (2^n * Γ(n+0.5))
        C_n = (-1.0)^(n+1) / (2.0^n * gamma(n + 0.5))

        # Binomial coefficient (numerical value)
        binom_coeff = binomial_coeff(N, n)

        # L^(2n) term
        L_power = L_over_RC^(2*n)

        # (n+1)th derivative of f
        f_deriv = f_derivatives_sym[n+2]

        G_sym = G_sym + binom_coeff * L_power * C_n * f_deriv
        # println("    G_sym $n: $(Symbolics.simplify(G_sym))")
    end

    # Apply taper function conditionally
    if L0 > 0
        println("  Applying taper function with L0=$L0...")
        # Apply taper function: 1 - exp(-(L/L0)^2)
        taper_sym = 1 - exp(-(L_sym/L0)^2)
        G_with_taper_sym = G_sym * taper_sym
    else
        println("  No taper function (L0=0)...")
        G_with_taper_sym = G_sym
    end

    # Final normalization
    CONST = Sigma0 / sqrt(π)
    DF_sym = CONST * G_with_taper_sym

    # Compute symbolic derivatives of DF
    println("  Computing symbolic derivatives of DF...")
    DF_dE_sym = Symbolics.derivative(DF_sym, E_sym)
    DF_dL_sym = Symbolics.derivative(DF_sym, L_sym)

    println("✓ Symbolic computation complete")

    # === NUMERICAL EVALUATION FUNCTIONS ===

    # Pre-build the substitution dictionaries to avoid repeated symbolic operations
    # Convert symbolic expressions to compiled functions for better performance
    DF_func = Symbolics.build_function(DF_sym, E_sym, L_sym, expression=Val{false})
    DF_dE_func = Symbolics.build_function(DF_dE_sym, E_sym, L_sym, expression=Val{false})
    DF_dL_func = Symbolics.build_function(DF_dL_sym, E_sym, L_sym, expression=Val{false})

    # Helper function to safely evaluate compiled functions
    function safe_evaluate(func, E_val::Real, L_val::Real)
        try
            result = func(E_val, L_val)
            return isfinite(result) ? Float64(result) : 0.0
        catch e
            return 0.0
        end
    end

    # Base distribution function
    function DF_base(E::Real, L::Real)::Real
        if E <= 0.0
            return 0.0
        end
        result = safe_evaluate(DF_func, E, L)
        return max(0.0, result)
    end

    # E-derivative function
    function DF_dE(E::Real, L::Real)::Real
        if E <= 0.0
            return 0.0
        end
        result = safe_evaluate(DF_dE_func, E, L)
        if result == 0.0  # If evaluation failed, use finite differences
            h = max(1e-8, abs(E) * 1e-8)
            return (DF_base(E + h, L) - DF_base(E - h, L)) / (2*h)
        end
        return result
    end

    # L-derivative function
    function DF_dL(E::Real, L::Real)::Real
        if E <= 0.0
            return 0.0
        end
        result = safe_evaluate(DF_dL_func, E, L)
        if result == 0.0  # If evaluation failed, use finite differences
            h = max(1e-8, abs(L) * 1e-8)
            return (DF_base(E, L + h) - DF_base(E, L - h)) / (2*h)
        end
        return result
    end

    return DF_base, DF_dE, DF_dL
end


end # module ExpDisk
