# Training Command Center Integration
# Add to your CommandCenter.jl for automated training control

module TrainingCommandCenter
using ..BehavioralTrainingRegiment
using Printf, Dates

# Global training state
const TRAINING_STATE = Dict{Symbol,Any}()

function init_training_system()
    TRAINING_STATE[:trainer] = AutomatedTrainer()
    TRAINING_STATE[:auto_mode] = false
    TRAINING_STATE[:training_log] = []
    println("🧠 Training system initialized")
end

function get_trainer()
    if !haskey(TRAINING_STATE, :trainer)
        init_training_system()
    end
    return TRAINING_STATE[:trainer]
end

# Command implementations
function cmd_train(args)
    length(args) < 3 && error("Usage: train <user_id> <action> [feedback]")
    
    user_id = args[1]
    action = join(args[2:end-1], " ")
    feedback = length(args) > 2 ? tryparse(Float64, args[end]) : nothing
    
    trainer = get_trainer()
    context = Dict("timestamp" => now(), "interactive" => true)
    
    result = adaptive_learning_loop(trainer, user_id, action, context, feedback)
    
    @printf("Intent: %s (%.3f confidence)\n", result.intent, result.confidence)
    @printf("Success rate: %.3f | Profile vars: %d\n", 
            result.session_stats["success_rate"], 
            result.session_stats["profile_complexity"])
    
    if !isempty(result.recommendations)
        println("Recommendations: ", join(result.recommendations, "; "))
    end
end

function cmd_auto_train(args)
    if isempty(args)
        TRAINING_STATE[:auto_mode] = !get(TRAINING_STATE, :auto_mode, false)
        status = TRAINING_STATE[:auto_mode] ? "ON" : "OFF"
        println("Auto-training: $status")
    else
        duration = parse(Int, args[1])
        auto_training_session(duration)
    end
end

function cmd_training_stats(args)
    trainer = get_trainer()
    stats = trainer.training_statistics
    
    if isempty(stats)
        println("No training data yet")
        return
    end
    
    println("🔬 Training Statistics")
    println("=" ^ 30)
    @printf("Total predictions: %d\n", get(stats, "total_predictions", 0))
    @printf("Active users: %d\n", length(get(stats, "active_individuals", [])))
    
    if haskey(stats, "confidence_history") && !isempty(stats["confidence_history"])
        avg_conf = mean(stats["confidence_history"])
        @printf("Average confidence: %.3f\n", avg_conf)
    end
    
    println("\nIntent distribution:")
    for (intent, count) in get(stats, "intent_distribution", Dict())
        @printf("  %s: %d\n", intent, count)
    end
end

function cmd_session_info(args)
    length(args) < 1 && error("Usage: session_info <user_id>")
    
    user_id = args[1]
    trainer = get_trainer()
    
    if !haskey(trainer.active_sessions, user_id)
        println("No session found for user: $user_id")
        return
    end
    
    session = trainer.active_sessions[user_id]
    stats = get_session_stats(session)
    
    println("👤 Session Info: $user_id")
    println("=" ^ 30)
    @printf("Interactions: %d\n", stats["total_interactions"])
    @printf("Success rate: %.3f\n", stats["success_rate"])
    @printf("Learning rate: %.4f\n", stats["current_learning_rate"])
    @printf("Profile variables: %d\n", stats["profile_complexity"])
    @printf("Duration: %.1f minutes\n", stats["session_duration"])
    
    if !isempty(session.training_history)
        println("\nRecent history:")
        for event in session.training_history[max(1, end-2):end]
            @printf("  %s: %s → %s (%.3f)\n", 
                    Dates.format(event["timestamp"], "HH:MM:SS"),
                    event["action"][1:min(30, end)],
                    event["predicted_intent"],
                    event["confidence"])
        end
    end
end

function cmd_batch_train(args)
    length(args) < 1 && error("Usage: batch_train <n_iterations>")
    
    n = parse(Int, args[1])
    trainer = get_trainer()
    
    # Predefined training scenarios
    scenarios = [
        ("alice", "I need help understanding this concept", Dict("learning" => 1.0), 0.8),
        ("bob", "Let me analyze the performance data", Dict("analytical" => 0.9), 0.7),
        ("charlie", "I want to create a new visualization", Dict("creative" => 0.8), 0.9),
        ("alice", "Can you optimize this algorithm?", Dict("technical" => 0.9), 0.6),
        ("bob", "I'm exploring different approaches", Dict("exploration" => 0.7), 0.8)
    ]
    
    println("🔄 Running batch training ($n iterations)...")
    
    for i in 1:n
        user, action, context, feedback = rand(scenarios)
        result = adaptive_learning_loop(trainer, user, action, context, feedback)
        
        if i % 10 == 0 || i <= 5
            @printf("[%d/%d] %s: %s → %s\n", i, n, user, result.intent, round(result.confidence, digits=3))
        end
    end
    
    println("✅ Batch training complete")
    cmd_training_stats([])
end

function auto_training_session(duration_minutes::Int)
    trainer = get_trainer()
    start_time = now()
    iteration = 0
    
    println("🤖 Auto-training for $duration_minutes minutes...")
    
    while Dates.value(now() - start_time) / 1000 / 60 < duration_minutes
        # Simulate realistic user interactions
        user = rand(["user1", "user2", "user3", "user4"])
        
        actions = [
            "I need assistance with this task",
            "Let me understand how this works", 
            "I want to create something innovative",
            "Help me solve this problem",
            "Can you analyze this information?",
            "I'm trying to optimize performance",
            "Let me explore different options",
            "I need to communicate these findings"
        ]
        
        action = rand(actions)
        context = Dict(
            "auto_generated" => true,
            "complexity" => rand(),
            "urgency" => rand(),
            "familiarity" => rand()
        )
        
        # Random feedback (simulating user satisfaction)
        feedback = rand() > 0.3 ? rand(0.4:0.1:1.0) : rand(-0.5:0.1:0.3)
        
        adaptive_learning_loop(trainer, user, action, context, feedback)
        iteration += 1
        
        # Progress update every 50 iterations
        if iteration % 50 == 0
            elapsed = Dates.value(now() - start_time) / 1000 / 60
            @printf("Auto-training: %d iterations, %.1f minutes\n", iteration, elapsed)
        end
        
        sleep(0.1)  # Realistic interaction pace
    end
    
    @printf("🎯 Auto-training complete: %d iterations\n", iteration)
end

# Extended command dispatch
const TRAINING_COMMANDS = Dict(
    "train" => cmd_train,
    "auto" => cmd_auto_train,
    "stats" => cmd_training_stats,
    "session" => cmd_session_info,
    "batch" => cmd_batch_train,
    "init_training" => args -> init_training_system()
)

function handle_training_command(cmd::String, args::Vector{SubString{String}})
    if haskey(TRAINING_COMMANDS, cmd)
        try
            TRAINING_COMMANDS[cmd](args)
        catch e
            println("⚠️  Training error: ", e)
        end
        return true
    end
    return false
end

# Integration helper for main command center
function extend_command_center()
    println("🔧 Training commands available:")
    println("  train <user> <action> [feedback]  - Train behavioral intent")
    println("  auto [duration]                   - Toggle/run auto-training")
    println("  stats                             - View training statistics")
    println("  session <user>                    - View user session info")
    println("  batch <n>                         - Run batch training")
    println("  init_training                     - Initialize training system")
end

end # module

# Usage in your main CommandCenter.jl:
# 1. Add: using .TrainingCommandCenter
# 2. In main command loop, before "Unknown command":
#    if TrainingCommandCenter.handle_training_command(cmd, args)
#        continue
#    end