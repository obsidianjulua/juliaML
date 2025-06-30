# Julia Enhanced REPL Configuration v1.0 - Performance Focus
# Save this file as: ~/.julia/config/startup.jl
# === ALL IMPORTS ===
using Revise
using FuzzyCompletions
using Term
using InteractiveUtils
using BenchmarkTools
using REPL
using LinearAlgebra  # For BLAS.get_num_threads() in env_info()
using Profile        # For Profile.clear() and Profile.print() in @profile macro
using StatsBase      # For sample(topics, 2, replace=false) in create_social_network()
using Random         # For rand() calls in environment rules
using Agents

# === SETUP ===
atreplinit() do repl
    try
        REPL.numbered_prompt!(repl)
    catch e
        # Silently fail if numbered prompts can't be enabled
    end
end

# === MACROS ===
macro t(ex)
    quote
        @btime $(esc(ex))
    end
end

macro mem(ex)
    quote
        local stats = Base.gc_num()
        local result = @time $(esc(ex))
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        println("Memory: $(Base.prettytime(diff.total_time/1e9)) gc time, $(diff.allocd) bytes allocated")
        result
    end
end

macro typeof(ex)
    quote
        val = $(esc(ex))
        println("Type: ", typeof(val))
        println("Supertype: ", supertype(typeof(val)))
        if hasmethod(fieldnames, (typeof(val),))
            println("Fields: ", fieldnames(typeof(val)))
        end
        val
    end
end

macro methods(ex)
    quote
        methods($(esc(ex)))
    end
end

macro src(ex)
    quote
        @which $(esc(ex)) |> println
        @edit $(esc(ex))
    end
end

macro help(ex)
    quote
        @doc $(esc(ex))
    end
end

# === FUNCTIONS ===
function whos()
    vars = []
    total_size = 0
    
    for name in sort(collect(names(Main, all=false, imported=false)))
        if name ∉ [:Base, :Core, :Main, :InteractiveUtils] && 
           isdefined(Main, name) && 
           !startswith(string(name), "#")
            try
                val = getfield(Main, name)
                size_bytes = Base.summarysize(val)
                total_size += size_bytes
                size_str = Base.format_bytes(size_bytes)
                push!(vars, (name, typeof(val), size_str))
            catch
                push!(vars, (name, "undefined", "0 bytes"))
            end
        end
    end
    
    if isempty(vars)
        println("No user-defined variables")
    else
        println("Variables in workspace (Total: $(Base.format_bytes(total_size))):")
        for (name, type, size) in vars
            println("  $name :: $type ($size)")
        end
    end
end

function clear_vars()
    cleared = []
    for name in names(Main, all=false, imported=false)
        if name ∉ [:Base, :Core, :Main, :InteractiveUtils] && 
           !startswith(string(name), "#") &&
           isdefined(Main, name)
            try
                val = getfield(Main, name)
                if !(val isa Function || val isa Module || val isa Type)
                    @eval Main $(name) = nothing
                    push!(cleared, name)
                end
            catch
            end
        end
    end
    println("Cleared: ", join(cleared, ", "))
    GC.gc()
end

function find_functions(pattern::String)
    matching = []
    for name in names(Main, all=true)
        if occursin(pattern, string(name)) && isdefined(Main, name)
            try
                val = getfield(Main, name)
                if val isa Function
                    push!(matching, name)
                end
            catch
            end
        end
    end
    
    for mod in [Base, Core]
        for name in names(mod, all=true)
            if occursin(pattern, string(name))
                try
                    val = getfield(mod, name)
                    if val isa Function
                        push!(matching, Symbol("$mod.$name"))
                    end
                catch
                end
            end
        end
    end
    
    return sort(unique(matching))
end

# === WELCOME ===
println("Julia Enhanced REPL v1.0 - Performance Edition")
println("Commands: whos(), clear_vars(), find_functions(\"pattern\")")
println("Macros: @t, @mem, @typeof, @methods, @src, @help")

# === ADDITIONAL UTILITIES ===

# Quick package operations
function pkg_status()
    run(`julia -e "using Pkg; Pkg.status()"`)
end

function pkg_update()
    run(`julia -e "using Pkg; Pkg.update()"`)
end

# Enhanced directory navigation
function ls(path=".")
    for item in readdir(path)
        full_path = joinpath(path, item)
        if isdir(full_path)
            println("📁 $item/")
        else
            println("📄 $item")
        end
    end
end

function pwd_info()
    p = pwd()
    println("Current directory: $p")
    println("Contents:")
    ls()
end

# Julia file operations
function find_jl_files(dir=".")
    jl_files = []
    for (root, dirs, files) in walkdir(dir)
        for file in files
            if endswith(file, ".jl")
                push!(jl_files, joinpath(root, file))
            end
        end
    end
    return jl_files
end

function include_all(dir=".")
    jl_files = find_jl_files(dir)
    for file in jl_files
        try
            include(file)
            println("✓ Included: $file")
        catch e
            println("✗ Failed to include $file: $e")
        end
    end
end

# Performance shortcuts
macro profile(ex)
    quote
        using Profile
        Profile.clear()
        @profile $(esc(ex))
        Profile.print()
    end
end

# Quick data inspection
function inspect(x)
    println("Value: $x")
    println("Type: $(typeof(x))")
    println("Size: $(sizeof(x)) bytes")
    if hasmethod(length, (typeof(x),))
        println("Length: $(length(x))")
    end
    if hasmethod(size, (typeof(x),))
        println("Dimensions: $(size(x))")
    end
    if hasmethod(fieldnames, (typeof(x),))
        fields = fieldnames(typeof(x))
        if !isempty(fields)
            println("Fields: $fields")
        end
    end
end

# Environment info
function env_info()
    println("Julia version: $(VERSION)")
    println("CPU threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("DEPOT_PATH: $(DEPOT_PATH)")
    println("LOAD_PATH: $(LOAD_PATH)")
end

# Quick benchmarking comparison
function compare_performance(expr1, expr2, name1="expr1", name2="expr2")
    println("Comparing $name1 vs $name2:")
    print("$name1: ")
    t1 = @time eval(expr1)
    print("$name2: ")
    t2 = @time eval(expr2)
    return (t1, t2)
end

# Git shortcuts (if in git repo)
function git_status()
    try
        run(`git status --short`)
    catch
        println("Not in a git repository")
    end
end

function git_log(n=5)
    try
        run(`git log --oneline -n $n`)
    catch
        println("Not in a git repository")
    end
end

# Memory utilities
function memory_usage()
    stats = Base.gc_num()
    println("GC stats:")
    println("  Total allocations: $(stats.total_allocd) bytes")
    println("  GC time: $(stats.total_time/1e9) seconds")
    println("  Collections: $(stats.pause)")
end

function force_gc()
    println("Running garbage collection...")
    @time GC.gc()
    memory_usage()
end

# Extended workspace tools
function save_workspace(filename="workspace.jl")
    open(filename, "w") do io
        for name in names(Main, all=false, imported=false)
            if name ∉ [:Base, :Core, :Main, :InteractiveUtils] && 
               isdefined(Main, name) && 
               !startswith(string(name), "#")
                try
                    val = getfield(Main, name)
                    if !(val isa Function || val isa Module)
                        println(io, "$name = $val")
                    end
                catch
                end
            end
        end
    end
    println("Workspace saved to $filename")
end

function load_workspace(filename="workspace.jl")
    if isfile(filename)
        include(filename)
        println("Workspace loaded from $filename")
    else
        println("File $filename not found")
    end
end

println("\nExtended commands: ls(), pwd_info(), inspect(x), env_info(), git_status()")
println("Utilities: save_workspace(), load_workspace(), memory_usage(), force_gc()")

# === BEHAVIORAL ML UTILITIES ===

# Simple intent-based decision system
mutable struct Intent
    goal::String
    priority::Float64
    context::Dict{String, Any}
    actions::Vector{Function}
    conditions::Vector{Function}
end

# Decision agent with behavioral weights
mutable struct BehaviorAgent
    name::String
    intents::Vector{Intent}
    memory::Dict{String, Any}
    action_history::Vector{Tuple{String, Any, Float64}}  # action, result, reward
    learning_rate::Float64
end

function BehaviorAgent(name::String)
    BehaviorAgent(name, Intent[], Dict(), Tuple{String, Any, Float64}[], 0.1)
end

# Add intent to agent
function add_intent!(agent::BehaviorAgent, goal::String, priority::Float64=1.0)
    intent = Intent(goal, priority, Dict(), Function[], Function[])
    push!(agent.intents, intent)
    return intent
end

# Add action to intent
function add_action!(intent::Intent, action_name::String, action_func::Function)
    labeled_action = () -> (action_name, action_func())
    push!(intent.actions, labeled_action)
end

# Add condition for intent activation
function add_condition!(intent::Intent, condition_func::Function)
    push!(intent.conditions, condition_func)
end

# Evaluate which intents are active based on context
function active_intents(agent::BehaviorAgent, context::Dict=Dict())
    active = Intent[]
    for intent in agent.intents
        # Check if all conditions are met
        if all(cond -> cond(context, agent.memory), intent.conditions)
            push!(active, intent)
        end
    end
    # Sort by priority (higher first)
    return sort(active, by=i -> i.priority, rev=true)
end

# Execute highest priority intent
function decide_and_act!(agent::BehaviorAgent, context::Dict=Dict())
    actives = active_intents(agent, context)
    
    if isempty(actives)
        println("$(agent.name): No active intents")
        return nothing
    end
    
    top_intent = first(actives)
    println("$(agent.name): Pursuing goal '$(top_intent.goal)'")
    
    if isempty(top_intent.actions)
        println("$(agent.name): No actions defined for goal")
        return nothing
    end
    
    # Execute first available action (could add selection logic)
    action_name, result = first(top_intent.actions)()
    
    # Store in history for learning
    push!(agent.action_history, (action_name, result, 1.0))  # Default reward
    
    println("$(agent.name): Executed '$action_name' → $result")
    return result
end

# Simple reinforcement learning from outcomes
function learn_from_outcome!(agent::BehaviorAgent, reward::Float64)
    if !isempty(agent.action_history)
        action_name, result, old_reward = agent.action_history[end]
        # Update last action's reward
        agent.action_history[end] = (action_name, result, reward)
        
        # Simple learning: adjust intent priorities based on success
        for intent in agent.intents
            for action in intent.actions
                action_label = action()[1]  # Get action name
                if action_label == action_name
                    if reward > 0
                        intent.priority += agent.learning_rate * reward
                    else
                        intent.priority -= agent.learning_rate * abs(reward)
                    end
                    intent.priority = max(0.1, intent.priority)  # Keep positive
                end
            end
        end
        
        println("$(agent.name): Learned from outcome (reward: $reward)")
    end
end

# Quick teaching/training functions
function teach_behavior(agent_name::String="Student")
    agent = BehaviorAgent(agent_name)
    println("Created agent: $agent_name")
    println("Use: add_intent!(agent, \"goal\", priority)")
    println("     add_action!(intent, \"name\", () -> your_action)")
    println("     add_condition!(intent, (ctx, mem) -> condition)")
    println("     decide_and_act!(agent, context_dict)")
    println("     learn_from_outcome!(agent, reward)")
    return agent
end

# Example behavioral patterns
function demo_behavioral_agent()
    # Create a simple foraging agent
    agent = BehaviorAgent("Forager")
    
    # Add survival intent
    survive = add_intent!(agent, "survive", 10.0)
    add_condition!(survive, (ctx, mem) -> get(ctx, "energy", 100) < 50)
    add_action!(survive, "find_food", () -> begin
        println("  → Searching for food...")
        return "found_berries"
    end)
    
    # Add exploration intent  
    explore = add_intent!(agent, "explore", 5.0)
    add_condition!(explore, (ctx, mem) -> get(ctx, "energy", 100) > 70)
    add_action!(explore, "move_forward", () -> begin
        println("  → Moving to new area...")
        return "discovered_cave"
    end)
    
    # Add social intent
    social = add_intent!(agent, "socialize", 3.0)
    add_condition!(social, (ctx, mem) -> get(ctx, "others_nearby", false))
    add_action!(social, "communicate", () -> begin
        println("  → Attempting communication...")
        return "shared_information"
    end)
    
    println("Demo: Behavioral agent with survival, exploration, and social goals")
    println("Try different contexts:")
    println("  decide_and_act!(agent, Dict(\"energy\" => 30))")
    println("  decide_and_act!(agent, Dict(\"energy\" => 80))")
    println("  decide_and_act!(agent, Dict(\"energy\" => 60, \"others_nearby\" => true))")
    
    return agent
end

# Multi-agent interaction
function create_teaching_environment()
    teacher = BehaviorAgent("Teacher") 
    student = BehaviorAgent("Student")
    
    # Teacher's intent: educate
    educate = add_intent!(teacher, "educate", 8.0)
    add_condition!(educate, (ctx, mem) -> get(ctx, "student_confused", false))
    add_action!(educate, "explain_concept", () -> begin
        println("  → Teacher: Breaking down concept...")
        return "explanation_given"
    end)
    
    # Student's intent: learn
    learn = add_intent!(student, "learn", 7.0)
    add_condition!(learn, (ctx, mem) -> get(ctx, "new_material", false))
    add_action!(learn, "ask_question", () -> begin
        println("  → Student: What does this mean?")
        return "question_asked" 
    end)
    
    println("Teaching environment created!")
    println("Context examples:")
    println("  ctx = Dict(\"student_confused\" => true, \"new_material\" => true)")
    println("  decide_and_act!(teacher, ctx)")
    println("  decide_and_act!(student, ctx)")
    
    return (teacher, student)
end

println("\nBehavioral ML:")
println("  teach_behavior(name)     - Create learning agent")
println("  demo_behavioral_agent()  - Example foraging behavior")
println("  create_teaching_environment() - Teacher/student agents")

# === STAGING ENVIRONMENTS ===

mutable struct Environment
    name::String
    state::Dict{String, Any}
    rules::Vector{Function}
    agents::Vector{BehaviorAgent}
    history::Vector{Dict{String, Any}}
    step_count::Int
end

function Environment(name::String, initial_state::Dict=Dict())
    Environment(name, copy(initial_state), Function[], BehaviorAgent[], Dict[], 0)
end

# Add environmental rules (how the world changes)
function add_rule!(env::Environment, rule_func::Function)
    push!(env.rules, rule_func)
end

# Add agents to environment
function add_agent!(env::Environment, agent::BehaviorAgent)
    push!(env.agents, agent)
end

# Run one simulation step
function step!(env::Environment)
    env.step_count += 1
    
    println("\n=== Step $(env.step_count) in $(env.name) ===")
    println("Environment state: $(env.state)")
    
    # Let each agent act based on current environment
    for agent in env.agents
        result = decide_and_act!(agent, env.state)
        
        # Store result in agent's memory
        if result !== nothing
            agent.memory["last_action_result"] = result
            agent.memory["step"] = env.step_count
        end
    end
    
    # Apply environmental rules (world changes)
    for rule in env.rules
        env.state = rule(env.state, env.agents)
    end
    
    # Record history
    snapshot = Dict(
        "step" => env.step_count,
        "state" => copy(env.state),
        "agents" => [(a.name, length(a.action_history)) for a in env.agents]
    )
    push!(env.history, snapshot)
    
    return env.state
end

# Run multiple steps
function simulate!(env::Environment, steps::Int=5)
    println("Running $(steps) simulation steps...")
    for i in 1:steps
        step!(env)
        sleep(0.5)  # Pause to observe
    end
    
    println("\n=== Simulation Complete ===")
    println("Final state: $(env.state)")
    
    # Simple analysis
    for agent in env.agents
        println("$(agent.name): $(length(agent.action_history)) actions taken")
        if !isempty(agent.action_history)
            last_action = agent.action_history[end]
            println("  Last: $(last_action[1]) → $(last_action[2])")
        end
    end
end

# Staging environment templates
function create_resource_world()
    env = Environment("ResourceWorld", Dict(
        "food_available" => 100,
        "water_available" => 80,
        "predator_nearby" => false,
        "season" => "spring"
    ))
    
    # Environmental rules
    add_rule!(env, (state, agents) -> begin
        # Food depletes over time
        state["food_available"] = max(0, state["food_available"] - 5)
        
        # Random predator appearances
        state["predator_nearby"] = rand() < 0.2
        
        # Seasonal changes
        if state["step_count"] % 10 == 0
            seasons = ["spring", "summer", "fall", "winter"]
            current_idx = findfirst(==(state["season"]), seasons)
            state["season"] = seasons[mod1(current_idx + 1, 4)]
        end
        
        return state
    end)
    
    println("Resource world created!")
    println("Try: add_agent!(env, your_agent); simulate!(env, 10)")
    return env
end

function create_classroom()
    env = Environment("Classroom", Dict(
        "lesson_difficulty" => 1,
        "student_attention" => 100,
        "material_covered" => 0,
        "questions_asked" => 0
    ))
    
    # Classroom dynamics
    add_rule!(env, (state, agents) -> begin
        # Attention decreases over time
        state["student_attention"] = max(0, state["student_attention"] - 10)
        
        # Difficulty increases with progress
        if state["material_covered"] > 50
            state["lesson_difficulty"] = 2
        end
        
        # Reset attention if questions are asked
        if state["questions_asked"] > 0
            state["student_attention"] = min(100, state["student_attention"] + 20)
            state["questions_asked"] = 0
        end
        
        return state
    end)
    
    println("Classroom environment created!")
    return env
end

function create_social_network()
    env = Environment("SocialNetwork", Dict(
        "network_activity" => 50,
        "trending_topics" => ["AI", "Climate"],
        "misinformation_level" => 10,
        "user_engagement" => 75
    ))
    
    # Social dynamics
    add_rule!(env, (state, agents) -> begin
        # Viral content spreads
        if state["user_engagement"] > 80
            state["network_activity"] += 15
        end
        
        # Misinformation can grow
        if rand() < 0.3
            state["misinformation_level"] += 5
        end
        
        # Topics change randomly
        if rand() < 0.4
            topics = ["Tech", "Politics", "Sports", "Entertainment", "Science"]
            state["trending_topics"] = sample(topics, 2, replace=false)
        end
        
        return state
    end)
    
    println("Social network environment created!")
    return env
end

# Quick staging utilities
function reset_environment!(env::Environment)
    env.step_count = 0
    env.history = Dict[]
    for agent in env.agents
        agent.action_history = Tuple{String, Any, Float64}[]
        agent.memory = Dict()
    end
    println("Environment $(env.name) reset")
end

function analyze_run(env::Environment)
    if isempty(env.history)
        println("No simulation data to analyze")
        return
    end
    
    println("\n=== Analysis of $(env.name) ===")
    println("Total steps: $(length(env.history))")
    
    # Show state progression
    first_state = env.history[1]["state"]
    last_state = env.history[end]["state"]
    
    println("\nState changes:")
    for key in keys(first_state)
        if haskey(last_state, key)
            initial = first_state[key]
            final = last_state[key]
            if initial != final
                println("  $key: $initial → $final")
            end
        end
    end
    
    # Agent activity summary
    println("\nAgent activity:")
    for agent in env.agents
        actions = [a[1] for a in agent.action_history]
        if !isempty(actions)
            action_counts = Dict()
            for action in actions
                action_counts[action] = get(action_counts, action, 0) + 1
            end
            println("  $(agent.name): $action_counts")
        end
    end
end

println("  create_resource_world()  - Survival simulation")
println("  create_classroom()       - Learning environment") 
println("  create_social_network()  - Social dynamics")
println("\nEnvironment commands:")
println("  add_agent!(env, agent)   - Add agent to environment")
println("  simulate!(env, steps)    - Run simulation")
println("  analyze_run(env)         - Analyze results")
println("  reset_environment!(env)  - Reset for new run")
