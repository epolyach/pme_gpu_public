# PME GPU - Polyachenko Matrix Equation Solver

GPU-accelerated solver for galactic disk normal modes using the Polyachenko Matrix Equation (PME) method. Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs.

## Prerequisites

### Julia Installation

1. Download Julia **1.9 or later** from <https://julialang.org/downloads/>.
2. Follow the platform-specific installation instructions:
   - **Linux**: extract the tarball and add `julia` to your `PATH`:
     ```bash
     tar xzf julia-1.10.x-linux-x86_64.tar.gz
     export PATH="$PATH:$(pwd)/julia-1.10.x/bin"
     ```
   - **macOS**: drag the `.app` bundle to Applications, then add to PATH:
     ```bash
     export PATH="$PATH:/Applications/Julia-1.10.app/Contents/Resources/julia/bin"
     ```
   - **Windows**: run the installer and ensure "Add Julia to PATH" is checked.
3. Verify the installation:
   ```bash
   julia --version
   ```

### CUDA Installation (NVIDIA GPUs)

1. Install the **NVIDIA driver** appropriate for your GPU from <https://www.nvidia.com/drivers>.
2. Verify the driver:
   ```bash
   nvidia-smi
   ```
3. CUDA.jl (the Julia CUDA package) will automatically download a compatible CUDA toolkit on first use. No separate CUDA Toolkit installation is required.
4. In Julia, test GPU availability:
   ```julia
   using CUDA
   CUDA.functional()   # should return true
   CUDA.versioninfo()  # prints driver and toolkit info
   ```

### ROCm Installation (AMD GPUs)

1. Install ROCm following the official guide at <https://rocm.docs.amd.com/>.
2. Verify:
   ```bash
   rocm-smi
   ```
3. In Julia:
   ```julia
   using AMDGPU
   AMDGPU.functional()  # should return true
   ```

### Julia Dependencies

This project ships with a `Manifest.toml` that pins all dependency versions to a tested, compatible combination. From the project root, install all dependencies:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

> **Important:** Do **not** run `Pkg.update()` unless you know what you are doing. Updating may pull newer versions of CUDA.jl that require a newer NVIDIA driver than the one installed on your system. If you do update and encounter CUDA errors, restore the original `Manifest.toml`:
> ```bash
> git checkout Manifest.toml
> julia --project=. -e 'using Pkg; Pkg.instantiate()'
> ```

#### Troubleshooting: CUDA Version Mismatch

If you see:
```
Error: This version of CUDA.jl requires an NVIDIA driver for CUDA 12.x or higher
```

This means your `Manifest.toml` has resolved to a CUDA.jl version that is too new for your driver. Restore the shipped Manifest:
```bash
git checkout Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If the shipped Manifest is also incompatible with your driver, check your CUDA version with `nvidia-smi` and set a compatible CUDA.jl version in `Project.toml`:
```toml
[compat]
CUDA = "5.8"    # for CUDA 11.x drivers
```
Then regenerate the Manifest:
```bash
rm Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

#### Proxy Configuration

If you are behind a corporate proxy or firewall, Julia may fail to download packages. Configure the proxy settings before installing dependencies:

**Linux/macOS:**
```bash
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
export JULIA_PKG_SERVER=""  # Disable pkg server if it is blocked

julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Windows (Command Prompt):**
```cmd
set HTTP_PROXY=http://proxy.example.com:8080
set HTTPS_PROXY=http://proxy.example.com:8080
set JULIA_PKG_SERVER=

julia --project=. -e "using Pkg; Pkg.instantiate()"
```

**Windows (PowerShell):**
```powershell
$env:HTTP_PROXY="http://proxy.example.com:8080"
$env:HTTPS_PROXY="http://proxy.example.com:8080"
$env:JULIA_PKG_SERVER=""

julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Permanent Configuration:**

Add proxy settings to Julia's startup file (`~/.julia/config/startup.jl`):
```julia
ENV["HTTP_PROXY"] = "http://proxy.example.com:8080"
ENV["HTTPS_PROXY"] = "http://proxy.example.com:8080"
ENV["JULIA_PKG_SERVER"] = ""
```

Replace `proxy.example.com:8080` with your actual proxy server and port.

## Quick Start

```bash
julia run_pme_gpu.jl configs/miyamoto.toml models/miyamoto.toml --gpu=0 --threads=4
```

### Command-line Arguments

| Argument | Description |
|---|---|
| `config.toml` | Grid, physics, and solver parameters |
| `model.toml` | Model type and model-specific parameters |
| `--gpu=0\|1\|01` | GPU device selection (0, 1, or both) |
| `--threads=N` | Number of BLAS threads (default: 4) |

## Available Models

All models are in `src/models/`. Each file is a self-contained Julia module.

### Naming Convention

- **`<Potential>.jl`** -- base (untapered) model for a given potential
- **`<Potential>Taper<Type>.jl`** -- tapered variant with smooth DF cutoff

### Sharp-Edge Models (Generalized Matrix Method)

The following models have distribution functions with a **sharp edge** (discontinuity at `L = 0`). They require the **generalized matrix method** (`sharp_df = true` in config) to handle the singular DF boundary correctly.

| File | Model type string | Potential | Key parameter |
|---|---|---|---|
| `Isochrone.jl` | `"Isochrone"` | Kalnajs isochrone | `mk` |
| `Miyamoto.jl` | `"Miyamoto"` | Plummer (Kuzmin-Toomre disk) | `n_M` |
| `Kuzmin.jl` | `"Kuzmin"` | Kuzmin-Toomre | `mk` |

### Tapered Models (Standard Matrix Method)

These models have a smooth taper function that regularises the DF near `L = 0`, so the standard PME matrix method applies directly.

#### Isochrone potential (Kalnajs isochrone)

| File | Model type string | Taper | Key parameters |
|---|---|---|---|
| `IsochroneTaperJH.jl` | `"IsochroneTaperJH"` | Jalali & Hunter (2005) | `mk` |
| `IsochroneTaperZH.jl` | `"IsochroneTaperZH"` | Zang & Hohl (1978) poly3 in L | `mk`, `Jc` or `Rc`, `eta` |
| `IsochroneTaperTanh.jl` | `"IsochroneTaperTanh"` | tanh, energy-dependent Lc | `mk`, `eta` |
| `IsochroneTaperExp.jl` | `"IsochroneTaperExp"` | exponential, energy-dependent Lc | `mk`, `eta` |
| `IsochroneTaperPoly3.jl` | `"IsochroneTaperPoly3"` | poly3 in circulation space | `mk`, `v_0` |

#### Plummer potential (Kuzmin-Toomre disk, Miyamoto DF)

| File | Model type string | Taper | Key parameters |
|---|---|---|---|
| `MiyamotoTaperExp.jl` | `"MiyamotoTaperExp"` | exponential in L | `n_M`, `L_0` |
| `MiyamotoTaperTanh.jl` | `"MiyamotoTaperTanh"` | tanh in circulation space | `n_M`, `v_0` |
| `MiyamotoTaperPoly3.jl` | `"MiyamotoTaperPoly3"` | poly3 in circulation space | `n_M`, `v_0` |

#### Kuzmin-Toomre potential (Kalnajs DF)

| File | Model type string | Taper | Key parameters |
|---|---|---|---|
| `KuzminTaperPoly3.jl` | `"KuzminTaperPoly3"` | poly3 in circulation space | `mk`, `v_0` |
| `KuzminTaperPoly3L.jl` | `"KuzminTaperPoly3L"` | poly3 in angular momentum L | `mk`, `L_star` |

#### Logarithmic potential

| File | Model type string | Taper | Key parameters |
|---|---|---|---|
| `Toomre.jl` | `"Toomre"` | Zang power-law | `L0`, `n_zang`, `q1` |

### Other Models

| File | Model type string | Description | Key parameters |
|---|---|---|---|
| `ExpDisk.jl` | `"ExpDisk"` | Cored exponential disk | `RC`, `N`, `lambda`, `alpha`, `L0` |

## Configuration Files

- **`configs/*.toml`** -- grid, physics, and solver settings (NR, Nv, kres, beta, etc.)
- **`models/*.toml`** -- model type and model-specific parameters

Example model TOML:
```toml
[model]
type = "MiyamotoTaperPoly3"
n_M = 3
v_0 = 0.2

[physics]
m = 2
beta = 0.1
selfgravity = 1.0
```

## Project Structure

```
pme_gpu_public/
  run_pme_gpu.jl          # Main GPU entry point
  Project.toml            # Julia project dependencies
  Manifest.toml           # Pinned dependency versions (tested)
  configs/                # Solver configuration files
  models/                 # Model parameter files
  src/
    PME.jl                # Main module
    PMEWorkflow.jl        # High-level calculation pipeline
    GPUWorkflow.jl        # GPU-specific workflow
    config/               # Configuration loading
    models/               # Galactic model definitions (one file per model)
    grids/                # Computational grid construction
    orbits/               # Orbit integration
    matrix/               # PME matrix and eigenvalue solvers
    io/                   # Binary I/O
    utils/                # GPU backend abstraction, progress tracking
```

## References

- Polyachenko E.V. (2005), *MNRAS*, 357, 559. doi:10.1111/j.1365-2966.2005.08660.x -- PME method
- Kalnajs A.J. (1976), *ApJ* 205, 745 -- Kuzmin-Toomre disk models
- Miyamoto M. (1971), *PASJ* 23, 21 -- Miyamoto disk models
- Jalali M.A. & Hunter C. (2005), *AJ* 130, 576 -- JH taper
- Zang T.A. & Hohl F. (1978), *ApJ* 226, 521 -- ZH taper
- Toomre A. (1981), in *Structure and Evolution of Normal Galaxies* -- Toomre-Zang model

## License

TBD
