# src/io/BinaryIO.jl
"""
Binary I/O functions for PME data exchange, including saving and loading
of intermediate and final results. Supports both native Julia and
MATLAB-compatible formats.
"""
module BinaryIO

using Dates
using ..OrbitCalculator
using ..PME # To get the config type
using ..Configuration: setup_data_directory

export save_orbit_data_binary, setup_data_directory


"""
    write_binary_float32(filename, data)

Write an array to a binary file as Float32 in column-major order.
"""
function write_binary_float32(filename::String, data::AbstractArray{<:Real})
    open(filename, "w") do io
        data_f32 = Float32.(data)
        write(io, data_f32)
    end
end

"""
    write_binary_float64(filename, data)

Write an array to a binary file as Float64 in column-major order.
"""
function write_binary_float64(filename::String, data::AbstractArray{<:Real})
    open(filename, "w") do io
        data_f64 = Float64.(data)
        write(io, data_f64)
    end
end

"""
    write_binary_data(filename, data, single_precision)

Write an array to a binary file with configurable precision.
"""
function write_binary_data(filename::String, data::AbstractArray{<:Real}, single_precision::Bool)
    if single_precision
        write_binary_float32(filename, data)
    else
        write_binary_float64(filename, data)
    end
end

"""
    save_orbit_data_binary(orbit_data, config)

Save all orbital data to binary files for inter-process communication or caching.
"""
function save_orbit_data_binary(orbit_data::OrbitData, config)
    # Use the data directory path directly - directory should already be set up
    data_dir = config.io.data_path
    binary_dir = joinpath(data_dir, "binary")
    # Ensure binary subdirectory exists
    mkpath(binary_dir)
    
    single_precision = config.io.single_precision
    precision_str = single_precision ? "single (Float32)" : "double (Float64)"
    #     println("Saving orbital data to binary files in $precision_str precision...")
    
    # Core frequency and momentum data (column-major order for MATLAB/C compatibility)
    write_binary_data(joinpath(binary_dir, "Omega_1.bin"), orbit_data.Omega_1, single_precision)
    write_binary_data(joinpath(binary_dir, "Omega_2.bin"), orbit_data.Omega_2, single_precision)
    write_binary_data(joinpath(binary_dir, "jacobian.bin"), orbit_data.jacobian, single_precision)
    write_binary_data(joinpath(binary_dir, "L_m.bin"), orbit_data.grids.L_m, single_precision)
    
    # Orbital trajectory data
    write_binary_data(joinpath(binary_dir, "r.bin"), orbit_data.ra, single_precision)
    write_binary_data(joinpath(binary_dir, "ph.bin"), orbit_data.pha, single_precision)
    write_binary_data(joinpath(binary_dir, "w1.bin"), orbit_data.w1, single_precision)
    
    # Distribution function data
    write_binary_data(joinpath(binary_dir, "F0.bin"), orbit_data.F0, single_precision)
    write_binary_data(joinpath(binary_dir, "FE.bin"), orbit_data.FE, single_precision)
    write_binary_data(joinpath(binary_dir, "FL.bin"), orbit_data.FL, single_precision)
    
    # Action variables and grid info
    write_binary_data(joinpath(binary_dir, "Ir.bin"), orbit_data.Ir, single_precision)
    write_binary_data(joinpath(binary_dir, "Rc.bin"), orbit_data.grids.Rc, single_precision)
    write_binary_data(joinpath(binary_dir, "E.bin"), orbit_data.grids.E, single_precision)
    
    # Circulation data
    write_binary_data(joinpath(binary_dir, "v.bin"), orbit_data.grids.eccentricity.circulation_grid, single_precision)
    
    # Grid weights product (S_RC_expanded .* S_e) for K-matrix construction
    NR, Nv = size(orbit_data.grids.Rc)
    S_RC_expanded = repeat(orbit_data.grids.radial.weights, 1, Nv)
    grid_weights_product = S_RC_expanded .* orbit_data.grids.eccentricity.weights
    write_binary_data(joinpath(binary_dir, "grid_weights.bin"), grid_weights_product, single_precision)
    
    # Debug output: save additional arrays if debug mode is enabled
    if config.io.debug
#         println("Debug mode enabled: saving additional debug data...")
        
        # Save eccentricity data
        write_binary_data(joinpath(binary_dir, "eccentricity.bin"), orbit_data.grids.eccentricity.points, single_precision)
        
        # Save R1 and R2 arrays
        write_binary_data(joinpath(binary_dir, "R1.bin"), orbit_data.grids.R1, single_precision)
        write_binary_data(joinpath(binary_dir, "R2.bin"), orbit_data.grids.R2, single_precision)
        
#         println("Debug data saved: eccentricity.bin, R1.bin, R2.bin")
    end
    
    #     println("Binary files saved to: $binary_dir")
    
    #     # Create a summary file for external tools (like the C code)
    create_binary_summary(binary_dir, config, orbit_data)
end

"""
    create_binary_summary(binary_dir, config, orbit_data)

Create a human-readable text file summarizing the grid parameters.
"""
function create_binary_summary(binary_dir::String, config, orbit_data::OrbitData)
    summary_file = joinpath(binary_dir, "grid_info.txt")
    
    precision_str = config.io.single_precision ? "Float32" : "Float64"
    bytes_per_element = config.io.single_precision ? 4 : 8
    
    open(summary_file, "w") do io
        println(io, "# PME Binary Data Summary")
        println(io, "# Generated by: PME.jl")
        println(io, "# Date: $(now())")
        println(io, "")
        println(io, "NR = $(config.grid.NR)")
        println(io, "Ne = $(config.grid.Nv)")
        println(io, "nwa = $(config.grid.NW)")
        println(io, "KRES = $(config.core.kres)")
        println(io, "m = $(config.core.m)")
        println(io, "beta = $(config.physics.beta)")
        println(io, "precision = $(precision_str)")
        println(io, "bytes_per_element = $(bytes_per_element)")
        println(io, "debug_mode = $(config.io.debug)")
        println(io, "")
        println(io, "# Binary files are $precision_str, column-major order (Fortran/MATLAB)")
        println(io, "# Dimensions are provided in the comments below.")
        println(io, "# Omega_1.bin  - Radial frequencies (NR × Ne)")
        println(io, "# Omega_2.bin  - Azimuthal frequencies (NR × Ne)")
        println(io, "# jacobian.bin - Analytic Jacobian matrix (NR × Ne)")
        println(io, "# L_m.bin      - Angular momenta (NR × Ne)")
        println(io, "# r.bin        - Radial positions (nwa × NR × Ne)")
        println(io, "# ph.bin       - Azimuthal phases (nwa × NR × Ne)")
        println(io, "# w1.bin       - Radial phases (nwa × NR × Ne)")
        println(io, "# F0.bin       - Distribution function (NR × Ne)")
        println(io, "# FE.bin       - ∂DF/∂E (NR × Ne)")
        println(io, "# FL.bin       - ∂DF/∂L (NR × Ne)")
        println(io, "# Ir.bin       - Radial actions (NR × Ne)")
        println(io, "# Rc.bin       - Guiding-center radii (NR × Ne)")
        println(io, "# E.bin        - Energies (NR × Ne)")
        println(io, "# v.bin        - Circulation (NR × Ne)")
        
        if config.io.debug
#             println(io, "")
#             println(io, "# Debug mode files:")
#             println(io, "# pi_elements.bin - Pi matrix elements (NR×Nv × NR×Nv × KRES × KRES)")
#             println(io, "# eccentricity.bin - Eccentricity grid points (NR × Ne)")
#             println(io, "# R1.bin       - Inner radial boundaries (NR × Ne)")
#             println(io, "# R2.bin       - Outer radial boundaries (NR × Ne)")
        end
    end
    
    #     println("Grid summary saved to: $summary_file")
end


end # module BinaryIO
