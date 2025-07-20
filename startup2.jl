# =============================================================================
# STARTUP.JL CONFIGURATION FOR METAMORPHIC ML WORKFLOW
# Place this in ~/.julia/config/startup.jl
# =============================================================================

# Performance and precompilation optimizations
ENV["JULIA_NUM_THREADS"] = string(Sys.CPU_THREADS)
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"  # Better for interactive development
ENV["GKSwstype"] = "100"  # For headless plotting

# Enable logging for metamorphic operations
ENV["JULIA_DEBUG"] = "Main,MetaMorphic"

# Interactive development settings
Base.active_repl_backend.ast_transforms[end] = 
    Base.active_repl_backend.ast_transforms[end] ∘ 
    (x -> x isa Expr && x.head == :toplevel ? Expr(:block, x.args...) : x)

# Preload essential packages for faster startup
println("🚀 Loading MetaMorphic ML Environment...")

try
    # Core ML stack
    @eval using Pkg
    @eval using Revise  # For hot-reloading during development
    
    # Precompile commonly used packages in background
    @async begin
        @eval using Flux, MLJ, Transformers
        @eval using Plots, PlutoUI, HypertextLiteral
        @eval using CUDA  # If available
        @eval using MLDataPattern, MLUtils
        println("✅ Core ML packages loaded")
    end
    
    # Interactive development tools
    @eval using BenchmarkTools
    @eval using ProfileView, Profile
    @eval using InteractiveUtils
    
    # Custom metamorphic utilities
    include(joinpath(homedir(), ".Metamorphic_Utils.jl.jl"))
    
    println("✅ MetaMorphic environment ready!")
    
catch e
    println("⚠️  Startup warning: ", e)
end

# Custom REPL utilities for metamorphic development
macro inject(expr)
    quote
        @info "Injecting logic: $($(string(expr)))"
        $(esc(expr))
    end
end

macro morph(model_expr, modification_expr)
    quote
        original = $(esc(model_expr))
        modified = $(esc(modification_expr))(original)
        @info "Morphed model structure"
        modified
    end
end

# Enhanced show methods for metamorphic layers
function Base.show(io::IO, ::MIME"text/plain", layer::Any)
    if hasfield(typeof(layer), :active_modifications)
        println(io, "🧬 MetaMorphic ", typeof(layer))
        println(io, "   Active modifications: ", length(layer.active_modifications))
        if !isempty(layer.active_modifications)
            for (i, mod) in enumerate(layer.active_modifications)
                println(io, "   [$i] $(typeof(mod))")
            end
        end
    else
        invoke(show, Tuple{IO, MIME"text/plain", Any}, io, MIME"text/plain"(), layer)
    end
end

# =============================================================================
# METAMORPHIC_UTILS.JL - Custom utilities
# Place this in ~/.julia/config/Metamorphic_Utils.jl
# =============================================================================

# Quick metamorphic layer creation
function quick_morph(layer)
    MetaMorphicLayer(layer, Dict{Symbol,Any}())
end

# Interactive injection helpers
function inject_debug(layer::MetaMorphicLayer)
    debug_fn = function(input, context)
        @info "Layer processing" input_size=size(get_hidden_state(input)) context_keys=keys(context)
        return input
    end
    inject_logic!(layer, debug_fn)
end

function inject_visualizer(layer::MetaMorphicLayer; plot_every=10)
    counter = Ref(0)
    viz_fn = function(input, context)
        counter[] += 1
