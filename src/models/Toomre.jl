# src/models/Toomre.jl

"""
Toomre-Zang Model

This module implements the Toomre-Zang galactic disk model with logarithmic potential.
The model features:
- Logarithmic potential: V(r) = log(r)
- Flat rotation curve: Ω(r) = 1/r
- Zang taper factor for angular momentum: [1 + (L₀/L)^n]^{-1}
- Distribution function with power-law in angular momentum

Reference: Toomre (1981) in "Structure and Evolution of Normal Galaxies"
          Zang (1976) PhD thesis
"""
module Toomre

using SpecialFunctions
using ..AbstractModel
using ..Configuration

export ToomreModel, create_toomre_model

"""
Concrete type for Toomre-Zang model
"""
struct ToomreModel <: AbstractGalacticModel end

"""
    create_toomre_model(config::PMEConfig)

Create a Toomre-Zang model from configuration.

# Model Parameters
- `L0`: Angular momentum scale (default: 1.0)
- `n_zang`: Exponent for Zang taper (default: 4)
- `q1`: Radial velocity dispersion parameter (default: 7)
- `selfgravity`: Self-gravity scaling factor (default: 1.0)
"""
function create_toomre_model(config::PMEConfig)
    # Extract model parameters from structured config (no params dict)
    L0 = config.model.L0
    n_zang = config.model.n_zang
    q1 = config.model.q1
    selfgravity = config.physics.selfgravity
    
    return setup_model(ToomreModel; L0=L0, n_zang=n_zang, q1=q1, selfgravity=selfgravity)
end

"""
    setup_model(::Type{ToomreModel}; L0=1.0, n_zang=4, q1=7, selfgravity=1.0)

Setup the Toomre-Zang model with specified parameters.

# Arguments
- `L0::Float64`: Angular momentum scale (typically 1.0)
- `n_zang::Int`: Exponent for Zang taper (typically 4)
- `q1::Int`: Parameter controlling velocity dispersion (typically 7)
- `selfgravity::Float64`: Self-gravity scaling factor (0.0 to 1.0)

# Returns
- `ModelResults`: Structure containing all model functions and metadata
"""
function setup_model(::Type{ToomreModel}; 
                     L0::Real=1.0, 
                     n_zang::Integer=4, 
                     q1::Integer=7, 
                     selfgravity::Real=1.0)
    
    # Derived parameters
    sigma_r0 = 1.0 / sqrt(q1)
    q_zang = q1 - 1
    
    # Normalization constant for distribution function
    CONST = 1.0 / (2.0 * π * 2.0^(q_zang/2) * sqrt(π) * 
                   gamma((q_zang + 1)/2) * sigma_r0^(q_zang + 2))
    
    # ============================================================================
    # Core potential and kinematic functions
    # ============================================================================
    
    """Logarithmic potential: V(r) = log(r)"""
    V(r) = log(r)
    
    """Potential derivative: dV/dr = 1/r"""
    dV(r) = 1.0 / r
    
    """Rotation frequency: Ω(r) = 1/r (flat rotation curve)"""
    Omega(r) = 1.0 / r
    
    """Epicyclic frequency: κ(r) = √2/r"""
    kappa(r) = sqrt(2.0) / r
    
    """Surface density: Σ(r) = 1/(2πr)"""
    Sigma_d(r) = 1.0 / (2.0 * π * r)
    
    # ============================================================================
    # Distribution function and derivatives
    # ============================================================================
    
    """
    Zang taper function: [1 + (L₀/L)^n]^{-1}
    
    For n_zang = 0, no taper is applied (taper = 1)
    For n_zang > 0, the taper suppresses small angular momentum
    """
    function taper(L::Real)
        if n_zang == 0
            return 1.0
        else
            # Guard against division by zero
            if abs(L) < 1e-12
                return 0.0
            end
            return 1.0 / (1.0 + (L0 / L)^n_zang)
        end
    end
    
    """
    Distribution function: DF(E, L)
    
    DF(E,L) = CONST × exp(-E/σᵣ₀²) × L^q_zang × taper(L)
    """
    function DF(E::Real, L::Real)
       
        # Angular momentum must be positive (working with |L|)
        L_abs = abs(L)
        if L_abs < 1e-12
            return 0.0
        end
        
        # Compute distribution function
        exp_factor = exp(-E / sigma_r0^2)
        L_power = L_abs^q_zang
        taper_factor = taper(L_abs)
        
        result = CONST * exp_factor * L_power * taper_factor
        
        # Guard against NaN or Inf
        if !isfinite(result)
            return 0.0
        end
        
        return result
    end
    
    """
    Energy derivative of distribution function: ∂DF/∂E
    
    ∂DF/∂E = -DF(E,L) / σᵣ₀²
    """
    function DF_dE(E::Real, L::Real)
        df_val = DF(E, L)
        return -df_val / sigma_r0^2
    end
    
    """
    Angular momentum derivative of distribution function: ∂DF/∂L
    
    ∂DF/∂L = DF(E,L) × (q_zang + n_zang - n_zang × taper(L)) / L
    """
    function DF_dL(E::Real, L::Real)
        # Guard against zero angular momentum
        L_abs = abs(L)
        if L_abs < 1e-12
            return 0.0
        end
        
        df_val = DF(E, L)
        if df_val == 0.0
            return 0.0
        end
        
        taper_factor = taper(L_abs)
        coeff = (q_zang + n_zang - n_zang * taper_factor) / L_abs
        
        result = df_val * coeff
        
        
        return result
    end
    
    # ============================================================================
    # Velocity dispersion (constant in this model)
    # ============================================================================
    
    """Radial velocity dispersion: σᵣ(r) = σᵣ₀"""
    sigma_r(r) = sigma_r0
    
    # ============================================================================
    # Package parameters and create result structure
    # ============================================================================
    
    params = Dict{String,Any}(
        "L0" => L0,
        "n_zang" => n_zang,
        "q1" => q1,
        "q_zang" => q_zang,
        "sigma_r0" => sigma_r0,
        "selfgravity" => selfgravity,
        "DF_type" => 0  # Standard DF type
    )
    
    # Helper functions
    helpers = (
        model_name = "Toomre-Zang",
        title = "Toomre-Zang model; L₀=$(L0), n=$(n_zang), q₁=$(q1)",
        taper = taper,
        sigma_r0 = sigma_r0
    )
    
    # Return complete model results
    return ModelResults{Float64}(
        V,                    # potential
        dV,                   # potential_derivative
        Omega,                # rotation_frequency
        kappa,                # epicyclic_frequency
        DF,                   # distribution_function
        DF_dE,               # df_energy_derivative
        DF_dL,               # df_angular_derivative
        Sigma_d,             # surface_density
        sigma_r,             # velocity_dispersion
        helpers,             # helper_functions
        "Toomre",            # model_type
        params               # parameters
    )
end

end # module Toomre
