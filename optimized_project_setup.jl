# =============================================================================
# Project.toml - Optimized for MetaMorphic ML Workflow
# =============================================================================

name = "MetaMorphicML"
uuid = "12345678-1234-5678-9abc-123456789abc"
version = "1.0.0"
authors = ["Your Name <your.email@example.com>"]

[deps]
# Core ML Framework
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
Transformers = "21ca0261-441d-5938-ace7-c90938fde4d4"

# Data Handling
MLDataPattern = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
MLDataUtils = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"

# Mathematical Computing
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

# GPU Computing (Optional but recommended)
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

# Interactive Development
Pluto = "c3e4b0f8-55cb-11ea-2926-15256bba5781"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"  # For Jupyter support

# Visualization
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
GraphMakie = "1ecd5474-83a3-4783-bb4f-06765db800d2"

# Development Tools  
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ProfileView = "c46f51b8-102a-5cf2-8d2c-8597cb0e0da7"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

# Metaprogramming & Code Generation
MacroTools = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"

# Utilities
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
OrderedCollections = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"

# Agent-Based Modeling
Agents = "46ada45e-f475-11e8-01d0-f70cc89e6671"

# Optimization
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"

# Text Processing
TextAnalysis = "a2db99b7-8b79-58f8-94bf-bbc811eef33d"
MLJText = "5e27fcf9-6bac-46ba-8580-b5712f3d6387"

# Web/HTML for rich output
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"

# Testing & Quality Assurance
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
SafeTestsets = "1bc83da4-3b8d-516f-aca4-4fe02f6d838f"

[compat]
julia = "1.9"
Flux = "0.14"
MLJ = "0.20"
Pluto = "0.19"
Plots = "1.38"
CUDA = "5.0"

# =============================================================================
# LocalPreferences.toml - For package-specific configurations
# =============================================================================

[Plots]
default_backend = "plotlyjs"
inspectdr_display = "gui"

[PlotlyJS]
use_kaleido = true

[CUDA]
version = "12.2"

[Pluto]
auto_reload_from_file = true
workspace_use_distributed = false

# =============================================================================
# .gitignore for MetaMorphic ML Project
# =============================================================================

# Julia
*.jl.cov
*.jl.*.cov
*.jl.mem
/Manifest.toml
/docs/build/
/docs/site/

# Jupyter Notebooks
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Pluto.jl
*.pluto.jl~

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (adjust as needed)
/data/raw/
/data/processed/*.csv
/models/checkpoints/
/logs/

# MetaMorphic specific
/metamorphic_cache/
/injection_logs/
*.metamorphic.bak

# =============================================================================
# Makefile for convenient project management
# =============================================================================

# Makefile
.PHONY: setup install dev test clean pluto jupyter profile

# Environment setup
setup:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	@echo "✅ MetaMorphic ML environment ready"

# Install additional packages
install:
	julia --project=. -e 'using Pkg; Pkg.add(ARGS)' $(PACKAGES)

# Development mode - start with hot reload
dev:
	julia --project=. --startup-file=yes -i -e 'using Revise; using MetaMorphicML'

# Start Pluto notebook server
pluto:
	julia --project=. -e 'using Pluto; Pluto.run(auto_reload_from_file=true)'

# Start Jupyter with Julia kernel
jupyter:
	jupyter lab --allow-root

# Run tests
test:
	julia --project=. -e 'using Pkg; Pkg.test()'

# Profile metamorphic operations
profile:
	julia --project=. scripts/profile_metamorphic.jl

# Clean generated files
clean:
	rm -rf .julia_cache/
	rm -rf metamorphic_cache/
	rm -rf injection_logs/
	find . -name "*.jl.cov" -delete
	find . -name "*.jl.mem" -delete

# Generate documentation
docs:
	julia --project=docs/ docs/make.jl

# Benchmarking suite
benchmark:
	julia --project=. scripts/benchmark_suite.jl

# =============================================================================
# docker-compose.yml - Containerized development environment
# =============================================================================

version: '3.8'

services:
  julia-metamorphic:
    build:
      context: .
      dockerfile: Dockerfile.julia
    ports:
      - "8888:8888"  # Jupyter
      - "1234:1234"  # Pluto
      - "8000:8000"  # General purpose
    volumes:
      - .:/workspace
      - julia-depot:/root/.julia
    environment:
      - JULIA_NUM_THREADS=auto
      - JULIA_PROJECT=/workspace
      - JUPYTER_ENABLE_LAB=yes
    command: tail -f /dev/null  # Keep container running
    
  gpu-julia:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    runtime: nvidia
    ports:
      - "8889:8888"
      - "1235:1234"
    volumes:
      - .:/workspace
      - julia-depot-gpu:/root/.julia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - JULIA_NUM_THREADS=auto
      - JULIA_CUDA_USE_BINARYBUILDER=false
    command: tail -f /dev/null

volumes:
  julia-depot:
  julia-depot-gpu:

# =============================================================================
# Dockerfile.julia - CPU development container
# =============================================================================

FROM julia:1.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for Jupyter integration
RUN pip3 install jupyter jupyterlab ipywidgets plotly bokeh

# Set up Julia environment
WORKDIR /workspace
COPY Project.toml ./
COPY Manifest.toml ./

# Precompile Julia packages
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Install IJulia for Jupyter integration
RUN julia --project=. -e 'using Pkg; Pkg.add("IJulia"); using IJulia; IJulia.installkernel("Julia", "--project=/workspace")'

# Install Pluto
RUN julia --project=. -e 'using Pkg; Pkg.add("Pluto")'

# Copy startup configuration
COPY config/startup.jl /root/.julia/config/
COPY config/metamorphic_utils.jl /root/.julia/config/

# Expose ports
EXPOSE 8888 1234 8000

# Default command
CMD ["julia", "--project=.", "--startup-file=yes"]

# =============================================================================
# Dockerfile.gpu - GPU-enabled development container  
# =============================================================================

FROM nvcr.io/nvidia/julia:1.9-cuda12.2-ubuntu22.04

# Install additional packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyter jupyterlab ipywidgets plotly

WORKDIR /workspace
COPY Project.toml ./
COPY Manifest.toml ./

# Install Julia packages with CUDA support
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA"); Pkg.precompile()'
RUN julia --project=. -e 'using Pkg; Pkg.add("IJulia"); using IJulia; IJulia.installkernel("Julia GPU", "--project=/workspace")'

COPY config/ /root/.julia/config/

EXPOSE 8888 1234

CMD ["julia", "--project=.", "--startup-file=yes"]

# =============================================================================
# scripts/profile_metamorphic.jl - Performance profiling script
# =============================================================================

#!/usr/bin/env julia

using Profile, ProfileView
using BenchmarkTools
using Plots

# Include the metamorphic framework
include("../src/metamorphic_injection.jl")

function profile_metamorphic_operations()
    println("🔬 Profiling MetaMorphic Operations")
    println("=" ^ 50)
    
    # Create test model
    base_model = Chain(
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 10)
    )
    
    test_input = randn(Float32, 128, 100)
    
    # Benchmark baseline
    baseline_time = @benchmark $base_model($test_input)
    println("Baseline inference: $(minimum(baseline_time.times) / 1e6) ms")
    
    # Create metamorphic version
    meta_model = build_constructivist_architecture(base_model, Dict(:target_types => [Dense]))
    
    # Benchmark metamorphic (no injections)
    meta_time = @benchmark $meta_model($test_input)
    println("MetaMorphic (no injections): $(minimum(meta_time.times) / 1e6) ms")
    
    # Apply injections and benchmark
    modified_model, _ = constructivist_training_session!(
        deepcopy(meta_model),
        "optimize and boost performance",
        [:optimization_boost],
        Dict(:boost_factor => 1.1)
    )
    
    injection_time = @benchmark $modified_model($test_input)
    println("MetaMorphic (with injections): $(minimum(injection_time.times) / 1e6) ms")
    
    # Profile detailed execution
    println("\n🔍 Detailed Profiling...")
    Profile.clear()
    @profile for i in 1:100
        modified_model(test_input)
    end
    
    # Save profile results
    ProfileView.view()
    Profile.print(IOBuffer())  # Capture profile data
    
    # Memory profiling
    println("\n💾 Memory Usage Analysis")
    baseline_allocs = @allocated base_model(test_input)
    meta_allocs = @allocated meta_model(test_input)
    injection_allocs = @allocated modified_model(test_input)
    
    println("Baseline allocations: $(baseline_allocs) bytes")
    println("MetaMorphic allocations: $(meta_allocs) bytes")
    println("Injection allocations: $(injection_allocs) bytes")
    
    # Generate performance report
    generate_performance_report(baseline_time, meta_time, injection_time)
end

function generate_performance_report(baseline, meta, injection)
    # Create performance comparison plot
    times = [
        minimum(baseline.times) / 1e6,
        minimum(meta.times) / 1e6,
        minimum(injection.times) / 1e6
    ]
    
    labels = ["Baseline", "MetaMorphic", "With Injections"]
    
    p = bar(labels, times, 
           title="MetaMorphic Performance Impact",
           ylabel="Time (ms)",
           color=[:blue, :orange, :red])
    
    savefig(p, "metamorphic_performance.png")
    println("📊 Performance plot saved as metamorphic_performance.png")
end

# Run if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    profile_metamorphic_operations()
end

# =============================================================================
# scripts/benchmark_suite.jl - Comprehensive benchmarking
# =============================================================================

#!/usr/bin/env julia

using BenchmarkTools
using DataFrames
using CSV
using Dates

include("../src/metamorphic_injection.jl")

const BENCHMARK_SUITE = BenchmarkGroup()

# Model size variations
model_configs = [
    ("small", Chain(Dense(64, 32, relu), Dense(32, 10))),
    ("medium", Chain(Dense(128, 64, relu), Dense(64, 32, relu), Dense(32, 10))),
    ("large", Chain(Dense(256, 128, relu), Dense(128, 64, relu), Dense(64, 32, relu), Dense(32, 10)))
]

# Injection patterns
injection_patterns = [
    ("none", "", Symbol[]),
    ("optimization", "optimize performance", [:optimization_boost]),
    ("bypass", "create bypass paths", [:logic_bypass]),
    ("full", "optimize bypass and restructure", [:optimization_boost, :logic_bypass, :pattern_override])
]

function setup_benchmarks()
    for (size_name, base_model) in model_configs
        BENCHMARK_SUITE[size_name] = BenchmarkGroup()
        
        # Test input
        input_size = size(base_model.layers[1].weight, 2)
        test_input = randn(Float32, input_size, 32)
        
        # Baseline benchmark
        BENCHMARK_SUITE[size_name]["baseline"] = @benchmarkable $base_model($test_input)
        
        # MetaMorphic benchmarks
        for (pattern_name, injection_text, break_points) in injection_patterns
            if pattern_name == "none"
                meta_model = build_constructivist_architecture(base_model, Dict(:target_types => [Dense]))
                BENCHMARK_SUITE[size_name]["metamorphic_$pattern_name"] = @benchmarkable $meta_model($test_input)
            else
                meta_model = build_constructivist_architecture(base_model, Dict(:target_types => [Dense]))
                modified_model, _ = constructivist_training_session!(
                    meta_model, injection_text, break_points, Dict(:boost_factor => 1.1)
                )
                BENCHMARK_SUITE[size_name]["metamorphic_$pattern_name"] = @benchmarkable $modified_model($test_input)
            end
        end
    end
end

function run_benchmark_suite()
    println("🚀 Running MetaMorphic ML Benchmark Suite")
    println("=" ^ 60)
    
    setup_benchmarks()
    results = run(BENCHMARK_SUITE)
    
    # Process results into DataFrame
    df_data = []
    
    for (size_name, size_results) in results
        for (benchmark_name, bench_result) in size_results
            push!(df_data, (
                model_size = size_name,
                benchmark_type = benchmark_name,
                min_time_ms = minimum(bench_result.times) / 1e6,
                median_time_ms = median(bench_result.times) / 1e6,
                mean_time_ms = mean(bench_result.times) / 1e6,
                allocations = bench_result.allocs,
                memory_mb = bench_result.memory / 1024^2,
                timestamp = now()
            ))
        end
    end
    
    df = DataFrame(df_data)
    
    # Save results
    timestamp_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "benchmark_results_$timestamp_str.csv"
    CSV.write(filename, df)
    
    println("📊 Benchmark results saved to $filename")
    
    # Print summary
    println("\n📈 Performance Summary:")
    for size_name in ["small", "medium", "large"]
        size_data = filter(row -> row.model_size == size_name, df)
        baseline_time = filter(row -> row.benchmark_type == "baseline", size_data)[1, :median_time_ms]
        
        println("\n$size_name Model:")
        for row in eachrow(size_data)
            overhead = round((row.median_time_ms / baseline_time - 1) * 100, digits=1)
            println("  $(row.benchmark_type): $(round(row.median_time_ms, digits=3))ms ($(overhead > 0 ? "+" : "")$(overhead)%)")
        end
    end
    
    return df
end

# Run if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark_suite()
end

# =============================================================================
# scripts/setup_development.jl - Complete development environment setup
# =============================================================================

#!/usr/bin/env julia

using Pkg

function setup_metamorphic_environment()
    println("🛠️  Setting up MetaMorphic ML Development Environment")
    println("=" ^ 60)
    
    # Activate project environment
    Pkg.activate(".")
    
    # Update and precompile packages
    println("📦 Installing and precompiling packages...")
    Pkg.instantiate()
    Pkg.precompile()
    
    # Setup directory structure
    println("📁 Creating directory structure...")
    dirs = [
        "src",
        "test", 
        "docs",
        "scripts",
        "config",
        "notebooks",
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "logs",
        "metamorphic_cache"
    ]
    
    for dir in dirs
        if !isdir(dir)
            mkpath(dir)
            println("  Created: $dir/")
        end
    end
    
    # Copy configuration files if they don't exist
    config_files = [
        ("config/startup.jl", create_startup_config),
        ("config/metamorphic_utils.jl", create_utils_config),
        ("notebooks/metamorphic_template.jl", create_pluto_template)
    ]
    
    for (filepath, creator_func) in config_files
        if !isfile(filepath)
            println("📝 Creating $filepath...")
            creator_func(filepath)
        end
    end
    
    # Install Julia kernel for Jupyter if IJulia is available
    try
        using IJulia
        IJulia.installkernel("MetaMorphic Julia", "--project=$(pwd())")
        println("✅ Jupyter kernel installed")
    catch
        println("⚠️  IJulia not available - Jupyter kernel not installed")
    end
    
    # Create example scripts
    create_example_scripts()
    
    println("\n🎉 MetaMorphic ML environment setup complete!")
    println("\nNext steps:")
    println("1. Copy config/startup.jl to ~/.julia/config/startup.jl")
    println("2. Start development with: make dev")
    println("3. Launch Pluto: make pluto") 
    println("4. Launch Jupyter: make jupyter")
    println("5. Run benchmarks: make benchmark")
end

function create_startup_config(filepath)
    # This would contain the startup.jl content from earlier
    write(filepath, "# MetaMorphic ML startup configuration\nprintln(\"🧬 MetaMorphic ML loaded\")")
end

function create_utils_config(filepath)
    # This would contain the utils content
    write(filepath, "# MetaMorphic utilities\n# Add your custom utilities here")
end

function create_pluto_template(filepath)
    # This would contain the Pluto template
    write(filepath, "### A Pluto.jl notebook ###\n# MetaMorphic ML Template")
end

function create_example_scripts()
    examples = [
        "examples/basic_injection.jl" => """
        # Basic MetaMorphic Injection Example
        using MetaMorphicML
        
        # Create a simple model
        model = Chain(Dense(10, 5, relu), Dense(5, 1))
        
        # Apply metamorphic transformation
        meta_model = build_constructivist_architecture(model, Dict(:target_types => [Dense]))
        
        # Test injection
        modified_model, patterns = constructivist_training_session!(
            meta_model,
            "optimize the network performance",
            [:optimization_boost]
        )
        
        println("Injection complete. Detected patterns: ", patterns)
        """,
        
        "examples/advanced_injection.jl" => """
        # Advanced MetaMorphic Operations
        using MetaMorphicML
        
        # Complex model with attention
        model = Chain(
            Dense(128, 64),
            TransformerBlock(8, 64),
            Dense(64, 10)
        )
        
        # Multiple injection strategies
        strategies = [
            ("attention_redirect", "redirect attention to important features"),
            ("logic_bypass", "create fast bypass paths"),
            ("pattern_override", "override standard processing patterns")
        ]
        
        for (strategy, description) in strategies
            println("Testing strategy: $strategy")
            
            meta_model = build_constructivist_architecture(model, Dict(:target_types => [Dense, TransformerBlock]))
            modified_model, patterns = constructivist_training_session!(
                meta_model, description, [Symbol(strategy)]
            )
            
            # Test performance
            test_input = randn(Float32, 128, 32)
            @time result = modified_model(test_input)
            
            println("Strategy $strategy completed successfully\\n")
        end
        """
    ]
    
    for (filepath, content) in examples
        if !isfile(filepath)
            mkpath(dirname(filepath))
            write(filepath, content)
            println("📝 Created $filepath")
        end
    end
end

# Run if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    setup_metamorphic_environment()
end