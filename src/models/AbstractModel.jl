# src/models/AbstractModel.jl
"""
Abstract base type for all galactic models
"""
module AbstractModel

export AbstractGalacticModel, ModelResults, setup_model

"""
Abstract base type for all galactic models
"""
abstract type AbstractGalacticModel end

"""
Results structure containing all model functions
"""
struct ModelResults{T<:Real}
    # Core potential and kinematics
    potential::Function              # V(r)
    potential_derivative::Function   # dV/dr  
    rotation_frequency::Function     # Ω(r)
    epicyclic_frequency::Function    # κ(r)
    
    # Distribution function and derivatives
    distribution_function::Function     # DF(E,L)
    df_energy_derivative::Function      # ∂DF/∂E
    df_angular_derivative::Function     # ∂DF/∂L
    
    # Physical profiles
    surface_density::Function       # Σ(r)
    velocity_dispersion::Union{Function,Nothing} # σᵣ(r), optional
    
    # Helper functions (model-specific)
    helper_functions::NamedTuple
    
    # Model metadata
    model_type::String
    parameters::Dict{String,Any}
end

"""
Base setup function that all models should implement
"""
function setup_model(::Type{<:AbstractGalacticModel}, args...; kwargs...)
    error("setup_model not implemented for this model type")
end

end # module AbstractModel
