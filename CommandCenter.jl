# CommandCenter.jl – lightweight interactive shell for Agents.jl simulations
# ----------------------------------------------------------------------
# Save in src/CommandCenter.jl and from the project root do:
#   include("src/CommandCenter.jl"); using .CommandCenter; CommandCenter.main()
# ----------------------------------------------------------------------

module CommandCenter
using Agents, Random, Printf

# === AGENT & MODEL DEFINITION ===============================================
mutable struct Learner <: AbstractAgent
    id::Int
    pos::Tuple{Int,Int}
    score::Float64
end

function learner_step!(agent::Learner, model)
    agent.score += randn()*0.1
    if agent.score < -5      # auto‑remove bad performers
        kill_agent!(agent, model)
    end
end

space(size) = GridSpace((size, size), periodic=false)

function init_model(size::Int, n_agents::Int)
    m = ABM(Learner, space(size); properties=Dict(:tick=>0), scheduler=by_id)
    for _ in 1:n_agents
        add_agent!(rand(Point2(1:size,1:size)), m, score=0.0)
    end
    return m
end

# === COMMAND CENTER STATE ====================================================
const STATE = Dict{Symbol,Any}()

# Helper to fetch active model or throw
getmodel() = get(STATE, :model, nothing) === nothing ? error("No active simulation. Use `new`.") : STATE[:model]

# Commands --------------------------------------------------------------------
function cmd_new(args)
    length(args) < 2 && error("Usage: new <size> <n_agents>")
    size = parse(Int, args[1]); n = parse(Int, args[2])
    STATE[:model] = init_model(size, n)
    println("Simulation created with $n agents on a $size×$size grid.")
end

function cmd_step(args)
    n = length(args)==0 ? 1 : parse(Int, args[1])
    m = getmodel()
    for _ in 1:n
        m.tick += 1
        Agents.step!(m, learner_step!, 1)
    end
    @printf("Stepped %d ticks → current tick %d\n", n, m.tick)
end

function cmd_status(_args)
    m = getmodel()
    avg = length(allagents(m))==0 ? 0.0 : mean(a.score for a in allagents(m))
        @printf("Tick: %-4d | Agents: %-3d | Avg score: %.2f\n", m.tick, length(allagents(m)), avg)
    end

    function cmd_add(args)
        length(args)==0 && error("Usage: add <n>")
        n = parse(Int, args[1])
        m = getmodel()
        sz = m.space.extent[1]
        for _ in 1:n
            add_agent!(rand(Point2(1:sz,1:sz)), m, score=0.0)
        end
        println("Added $n agents → total ", length(allagents(m)))
    end

    function cmd_kill(args)
        length(args)==0 && error("Usage: kill <n>")
        n = parse(Int, args[1])
        m = getmodel()
        victims = first(allagents(m), min(n, length(allagents(m))))
        foreach(a -> kill_agent!(a, m), victims)
        println("Killed $(length(victims)) agents → remaining ", length(allagents(m)))
    end

    function cmd_help(_args)
        println("""
                Commands:
                new <size> <n_agents>     create fresh simulation
                step [n]                  advance n ticks (default 1)
                status                    view current stats
                add <n>                   add n agents
                kill <n>                  remove n agents
                quit / exit               leave command center
                """)
    end

    # Command dispatch table
    const DISPATCH = Dict(
        "new"    => cmd_new,
        "step"   => cmd_step,
        "status" => cmd_status,
        "add"    => cmd_add,
        "kill"   => cmd_kill,
        "help"   => cmd_help,
        )

    function main()
        println("🕹  Command Center ready. Type `help` for commands.")
            while true
                print("cmd> "); flush(stdout)
                input = readline(stdin)
                isempty(input) && continue
                words = split(strip(input))
                cmd   = lowercase(words[1])
                args  = words[2:end]
                if cmd in ("quit","exit")
                    println("Bye!"); break
                end
                if haskey(DISPATCH, cmd)
                    try
                        DISPATCH[cmd](args)
                        catch e
                        println("⚠️  Error: ", e)
                    end
                else
                    println("Unknown command `$(cmd)`. Type `help`.")
                end
            end
        end

    end # module
