[[LLVM QUEEN JULIA]] 

# Julia Artifacts & Executable Usage Guide

## What are Artifacts?

Artifacts in Julia are **immutable, versioned containers** for:
- Platform-specific binaries (executables, libraries)
- Datasets and data files
- Text files, documentation
- Any external dependencies your package needs

They solve the problem of **distributing non-Julia dependencies** reliably across platforms.

## Basic Artifact Usage

### 1. Using Existing Artifacts

```julia
using Artifacts

# Access an artifact (downloads if needed)
artifact_path = @artifact"MyArtifact"
println("Artifact location: $artifact_path")

# Access specific files within artifact
binary_path = @artifact"FFMPEG/bin/ffmpeg"
executable = @artifact"MyTool/bin/mytool"
```

### 2. Checking Artifact Status

```julia
using Artifacts

# Check if artifact exists locally
hash = artifact_hash("MyArtifact", "Artifacts.toml")
if artifact_exists(hash)
    println("Artifact is available")
    path = artifact_path(hash)
else
    println("Artifact needs to be downloaded")
end

# Find artifacts.toml file
toml_path = find_artifacts_toml(@__FILE__)
println("Artifacts defined in: $toml_path")
```

## Artifacts.toml Structure

Create an `Artifacts.toml` file in your package root:

```toml
# Simple artifact
[MyDataset]
git-tree-sha1 = "43563e7a588cd231043dbf93a5b74a2b91c55e6f"
lazy = true

    [[MyDataset.download]]
    url = "https://example.com/dataset.tar.gz"
    sha256 = "ab40..."

# Platform-specific artifact (for executables)
[[MyExecutable]]
arch = "x86_64"
os = "linux"
git-tree-sha1 = "539108cf0cd4c9ea6cb5af6ec8a4e6cbb2e0b1e4"

    [[MyExecutable.download]]
    url = "https://releases.example.com/tool-linux-x64.tar.gz"
    sha256 = "cd21..."

[[MyExecutable]]
arch = "x86_64" 
os = "windows"
git-tree-sha1 = "8ba89a4cc31af79bafbfc9ce77b0e2c4ea905496"

    [[MyExecutable.download]]
    url = "https://releases.example.com/tool-windows-x64.zip"
    sha256 = "ef43..."
```

## Running Executables from Artifacts

### Method 1: Direct Execution

```julia
using Artifacts

# Get path to executable
exec_path = @artifact"MyTool/bin/mytool"

# Make executable (Unix systems)
if Sys.isunix()
    chmod(exec_path, 0o755)
end

# Run the executable
result = read(`$exec_path --version`, String)
println("Tool version: $result")

# Run with arguments
output = read(`$exec_path process input.txt`, String)
```

### Method 2: Using Cmd Objects

```julia
using Artifacts

function run_mytool(args...)
    tool_path = @artifact"MyTool/bin/mytool"
    
    # Create command
    cmd = Cmd([tool_path, args...])
    
    # Run and capture output
    io = IOBuffer()
    run(pipeline(cmd, stdout=io, stderr=io))
    return String(take!(io))
end

# Usage
result = run_mytool("--help")
processed = run_mytool("process", "data.txt", "--output", "result.txt")
```

### Method 3: Interactive Execution

```julia
using Artifacts

function interactive_tool()
    tool_path = @artifact"MyTool/bin/mytool"
    
    # Run interactively (inherits stdin/stdout)
    run(`$tool_path --interactive`)
end

# Launch interactive session
interactive_tool()
```

## Advanced Artifact Patterns

### 1. Lazy Loading with Error Handling

```julia
function get_tool_safely()
    try
        return @artifact"MyTool"
    catch e
        @warn "Tool not available: $e"
        @info "Install with: using Pkg; Pkg.artifact_install()"
        return nothing
    end
end
```

### 2. Multi-Platform Executable Wrapper

```julia
function get_platform_executable(tool_name)
    artifact_path = @artifact"MultiPlatformTool"
    
    if Sys.iswindows()
        return joinpath(artifact_path, "bin", "$tool_name.exe")
    else
        return joinpath(artifact_path, "bin", tool_name)
    end
end

function run_cross_platform_tool(args...)
    exec_path = get_platform_executable("mytool")
    run(`$exec_path $args`)
end
```

### 3. Environment Setup for Complex Tools

```julia
function setup_tool_environment()
    tool_root = @artifact"ComplexTool"
    
    # Set up environment variables
    ENV["TOOL_HOME"] = tool_root
    ENV["TOOL_LIB"] = joinpath(tool_root, "lib") 
    ENV["PATH"] = joinpath(tool_root, "bin") * ":" * ENV["PATH"]
    
    return joinpath(tool_root, "bin", "main_executable")
end

function run_with_environment(args...)
    exec_path = setup_tool_environment()
    run(`$exec_path $args`)
end
```

## Package Development with Artifacts

### Creating Your Own Artifact

```julia
# In your package development
using ArtifactUtils, Artifacts

function create_my_artifact()
    # Create artifact directory
    artifact_dir = mktempdir()
    
    # Download/build your executable
    download("https://example.com/tool.tar.gz", joinpath(artifact_dir, "tool.tar.gz"))
    run(`tar -xzf $(joinpath(artifact_dir, "tool.tar.gz")) -C $artifact_dir`)
    
    # Create artifact
    hash = create_artifact() do artifact_dir
        # Copy your files to artifact_dir
        cp("my_built_tool", joinpath(artifact_dir, "bin", "mytool"))
        chmod(joinpath(artifact_dir, "bin", "mytool"), 0o755)
    end
    
    # Bind to artifacts.toml
    bind_artifact!("Artifacts.toml", "MyTool", hash; 
                   download_info=[("https://example.com/tool.tar.gz", sha256_hash)])
end
```

## Common Patterns in Your Enhanced REPL

Add these functions to your startup.jl:

```julia
# Artifact inspection utilities
function list_artifacts(toml_path="Artifacts.toml")
    if isfile(toml_path)
        artifacts = load_artifacts_toml(toml_path)
        println("Available artifacts:")
        for (name, _) in artifacts
            println("  - $name")
        end
    else
        println("No Artifacts.toml found")
    end
end

function inspect_artifact(name::String)
    try
        path = @artifact_str name
        println("Artifact '$name' location: $path")
        if isdir(path)
            println("Contents:")
            for item in readdir(path)
                println("  $item")
            end
        end
        return path
    catch e
        println("Artifact '$name' not found: $e")
        return nothing
    end
end

function run_artifact_executable(artifact_name::String, executable_name::String, args...)
    try
        exec_path = @artifact_str "$artifact_name/bin/$executable_name"
        if !isfile(exec_path)
            exec_path = @artifact_str "$artifact_name/$executable_name"
        end
        
        if Sys.isunix() && !isexecutable(exec_path)
            chmod(exec_path, 0o755)
        end
        
        run(`$exec_path $args`)
    catch e
        @error "Failed to run executable: $e"
    end
end
```

## Best Practices

1. **Always use platform-specific artifacts** for executables
2. **Set lazy=true** for large artifacts that aren't always needed  
3. **Handle missing artifacts gracefully** with try-catch
4. **Make executables executable** on Unix systems with chmod
5. **Use absolute paths** when calling executables
6. **Set up proper environment variables** for complex tools
7. **Version your artifacts** by using different git-tree-sha1 hashes

This system lets you **distribute platform-specific executables reliably** while keeping your Julia package pure Julia code.
