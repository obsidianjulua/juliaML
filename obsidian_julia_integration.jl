# Obsidian-Julia Data Pipeline
# Creates .md files for Obsidian and .jl drop files for Julia

using Dates, JSON3, Markdown

# Configuration
const OBSIDIAN_VAULT = "/home/grim/Vaults/queen-julia"
const DATA_FOLDER = "ML_Training_Data"
const PROFILES_FOLDER = "Behavioral_Profiles" 
const JULIA_DROPS = "Julia_Drops"

# Create folders if they don't exist
function setup_obsidian_folders()
    folders = [DATA_FOLDER, PROFILES_FOLDER, JULIA_DROPS]
    for folder in folders
        path = joinpath(OBSIDIAN_VAULT, folder)
        !isdir(path) && mkpath(path)
    end
end

# Generate training session markdown
function create_training_session_md(session_data::Dict)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "training_session_$timestamp.md"
    
    md_content = """
# Training Session - $(Dates.format(now(), "yyyy-mm-dd HH:MM"))

## Session Summary
- **User ID**: $(session_data["user_id"])
- **Duration**: $(session_data["duration_minutes"]) minutes
- **Total Interactions**: $(session_data["total_interactions"])
- **Success Rate**: $(round(session_data["success_rate"], digits=3))
- **Model Used**: $(session_data["model"])

## Performance Metrics
- **Accuracy**: $(session_data["accuracy"])
- **Confidence**: $(session_data["avg_confidence"])
- **Adaptations**: $(session_data["adaptations"])

## Behavioral Variables Added
$(join(["- " * var for var in session_data["new_variables"]], "\n"))

## Recommendations
$(join(["- " * rec for rec in session_data["recommendations"]], "\n"))

## Tags
#training #$(session_data["user_id"]) #$(session_data["model"]) #behavioral-ml

## Raw Data
```json
$(JSON3.write(session_data, indent=2))
```
"""
    
    filepath = joinpath(OBSIDIAN_VAULT, DATA_FOLDER, filename)
    write(filepath, md_content)
    return filepath
end

# Generate profile evolution markdown
function create_profile_md(profile::BehavioralProfile)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "profile_$(profile.individual_id)_$timestamp.md"
    
    # Calculate profile stats
    var_count = length(profile.variables)
    categories = unique([var.category for var in values(profile.variables)])
    avg_confidence = var_count > 0 ? mean([var.confidence for var in values(profile.variables)]) : 0.0
    
    md_content = """
# Behavioral Profile: $(profile.individual_id)

**Last Updated**: $(Dates.format(now(), "yyyy-mm-dd HH:MM"))
**Profile Age**: $(Dates.value(now() - profile.creation_date) ÷ (1000*60*60*24)) days

## Profile Overview
- **Variables**: $var_count
- **Categories**: $(join(categories, ", "))
- **Complexity Score**: $(round(profile.complexity_score, digits=3))
- **Stability Score**: $(round(profile.stability_score, digits=3))
- **Average Confidence**: $(round(avg_confidence, digits=3))

## Variable Categories
$(join(["### " * cat * "\n" * join(["- **" * name * "**: " * string(round(var.value, digits=3)) * " (conf: " * string(round(var.confidence, digits=2)) * ")" for (name, var) in profile.variables if var.category == cat], "\n") for cat in categories], "\n\n"))

## Recent Evolution
$(length(profile.variable_history) > 0 ? "Last $(min(5, length(profile.variable_history))) changes:" : "No recent changes")
$(join(["- " * h["type"] * ": " * h["variable"] * " (" * string(h["timestamp"]) * ")" for h in profile.variable_history[max(1, end-4):end]], "\n"))

## Model Transitions
$(join(["- " * t["source_model"] * " → " * t["target_model"] * " (fidelity: " * string(round(t["fidelity"], digits=3)) * ")" for t in profile.model_transitions], "\n"))

## Tags
#profile #$(profile.individual_id) #behavioral-variables

## Links
- [[Training Sessions for $(profile.individual_id)]]
- [[Model Performance Analysis]]

## Raw Profile Data
```julia
# Generated $(now())
profile_data = $(repr(profile))
```
"""
    
    filepath = joinpath(OBSIDIAN_VAULT, PROFILES_FOLDER, filename)
    write(filepath, md_content)
    return filepath
end

# Generate Julia drop file
function create_julia_drop(data::Dict, drop_type::String)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "$(drop_type)_drop_$timestamp.jl"
    
    julia_content = """
# Julia Data Drop - $drop_type
# Generated: $(now())
# Auto-import for behavioral ML pipeline

using Dates, Statistics

# Drop metadata
const DROP_TIMESTAMP = "$(now())"
const DROP_TYPE = "$drop_type"

# Data payload
const DROP_DATA = $(repr(data))

# Quick accessor functions
get_drop_data() = DROP_DATA
get_drop_timestamp() = DROP_TIMESTAMP
get_drop_type() = DROP_TYPE

# Auto-execution on include
if @isdefined(BEHAVIORAL_TRAINING_ACTIVE) && BEHAVIORAL_TRAINING_ACTIVE
    println("🔄 Processing $drop_type drop from \$DROP_TIMESTAMP")
    
    if DROP_TYPE == "training_session"
        process_training_drop(DROP_DATA)
    elseif DROP_TYPE == "profile_update"  
        process_profile_drop(DROP_DATA)
    elseif DROP_TYPE == "model_performance"
        process_performance_drop(DROP_DATA)
    end
    
    println("✅ Drop processed successfully")
end

# Individual processing functions (customize as needed)
function process_training_drop(data)
    # Add to global training tracker
    global TRAINING_SESSIONS = get(TRAINING_SESSIONS, Vector{Dict}(), [])
    push!(TRAINING_SESSIONS, data)
end

function process_profile_drop(data) 
    # Update profile registry
    global PROFILE_REGISTRY = get(PROFILE_REGISTRY, Dict(), Dict())
    PROFILE_REGISTRY[data["individual_id"]] = data
end

function process_performance_drop(data)
    # Update model performance tracking
    global MODEL_PERFORMANCE = get(MODEL_PERFORMANCE, Dict(), Dict())
    merge!(MODEL_PERFORMANCE, data)
end

# Export for manual inspection
export get_drop_data, get_drop_timestamp, get_drop_type
"""
    
    filepath = joinpath(OBSIDIAN_VAULT, JULIA_DROPS, filename)
    write(filepath, julia_content)
    return filepath
end

# Obsidian template for manual note creation
function create_obsidian_templates()
    template_folder = joinpath(OBSIDIAN_VAULT, "Templates")
    !isdir(template_folder) && mkpath(template_folder)
    
    # Training session template
    training_template = """
# Training Session - {{date:yyyy-MM-dd HH:mm}}

## Session Summary
- **User ID**: 
- **Duration**: minutes
- **Model Used**: 
- **Success Rate**: 

## Key Observations
- 
- 
- 

## Behavioral Changes
- 
- 

## Next Steps
- 
- 

## Tags
#training #behavioral-ml

## Data Link
[[Julia_Drops/]]
"""
    
    write(joinpath(template_folder, "Training_Session_Template.md"), training_template)
    
    # Profile analysis template  
    profile_template = """
# Profile Analysis - {{date:yyyy-MM-dd}}

## Individual: 
**Profile Age**: days
**Last Update**: 

## Current State
### Strengths
- 
- 

### Areas for Development
- 
- 

### Behavioral Patterns
- 
- 

## Model Performance
- **Best Model**: 
- **Success Rate**: 
- **Confidence Level**: 

## Recommendations
- 
- 

## Tags
#profile #analysis #behavioral-ml

## Related
- [[Behavioral_Profiles/]]
- [[Training Sessions]]
"""
    
    write(joinpath(template_folder, "Profile_Analysis_Template.md"), profile_template)
    
    println("📝 Obsidian templates created")
end

# Batch export for existing data
function export_to_obsidian(trainer::AutomatedTrainer)
    setup_obsidian_folders()
    
    exported_files = String[]
    
    # Export all active sessions
    for (user_id, session) in trainer.active_sessions
        session_data = Dict(
            "user_id" => user_id,
            "duration_minutes" => Dates.value(now() - session.session_start) / 1000 / 60,
            "total_interactions" => session.total_interactions,
            "success_rate" => session.successful_predictions / max(1, session.total_interactions),
            "model" => "ensemble",
            "accuracy" => session.total_interactions > 0 ? session.successful_predictions / session.total_interactions : 0.0,
            "avg_confidence" => length(session.training_history) > 0 ? mean([h["confidence"] for h in session.training_history]) : 0.0,
            "adaptations" => session.adaptation_events,
            "new_variables" => collect(keys(session.current_profile.variables)),
            "recommendations" => generate_recommendations(session)
        )
        
        # Create markdown
        md_file = create_training_session_md(session_data)
        push!(exported_files, md_file)
        
        # Create Julia drop
        jl_file = create_julia_drop(session_data, "training_session")
        push!(exported_files, jl_file)
        
        # Create profile markdown
        profile_file = create_profile_md(session.current_profile)
        push!(exported_files, profile_file)
    end
    
    create_obsidian_templates()
    
    println("📊 Exported $(length(exported_files)) files to Obsidian")
    return exported_files
end

# Watch folder for new Julia drops (auto-import)
function watch_julia_drops()
    drop_folder = joinpath(OBSIDIAN_VAULT, JULIA_DROPS)
    
    # Simple file watcher (replace with FileWatching.jl for production)
    known_files = Set(readdir(drop_folder))
    
    while true
        current_files = Set(readdir(drop_folder))
        new_files = setdiff(current_files, known_files)
        
        for file in new_files
            if endswith(file, ".jl")
                filepath = joinpath(drop_folder, file)
                try
                    include(filepath)
                    println("🔄 Auto-imported: $file")
                catch e
                    println("⚠️ Failed to import $file: $e")
                end
            end
        end
        
        known_files = current_files
        sleep(5)  # Check every 5 seconds
    end
end

# Quick setup function
function setup_obsidian_integration(vault_path::String)
    global OBSIDIAN_VAULT = vault_path
    setup_obsidian_folders()
    create_obsidian_templates()
    
    println("✅ Obsidian integration configured")
    println("📁 Vault: $vault_path")
    println("📝 Use export_to_obsidian(trainer) to export data")
    println("👀 Use watch_julia_drops() to auto-import")
end
