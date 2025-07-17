using POMDPs
using QuickPOMDPs
using POMDPTools
using POMCPOW
using Random
using Combinatorics
using Plots
using Printf

# --- Observation labels ---
observations = [:empty, :landing_zone, :obstacle]
roles = [:empty, :landing_zone, :obstacle]

# --- Generate state space ---
all_grid_states = [Tuple(s) for s in Iterators.product(fill(roles, 9)...)]
valid_true_states = [s for s in all_grid_states if count(==( :landing_zone), s) == 1 && count(==( :obstacle), s) == 3]

# --- Action Types ---
abstract type LandingAction end
struct Inspect <: LandingAction
    cell::Int
end
struct Land <: LandingAction
    cell::Int
end

# --- Transition Function ---
function transition_fn(s, a)
    if a isa Inspect
        return Deterministic(s)
    elseif a isa Land
        return Deterministic(s)
        # return Uniform(all_grid_states)
    end
end

# --- Define POMDP ---
pomdp = QuickPOMDP(
    states = all_grid_states,
    actions = vcat([Inspect(i) for i in 1:9], [Land(i) for i in 1:9]),
    observations = observations,
    discount = 0.95,

    transition = transition_fn,

    observation = function (s, a, sp)
        if a isa Inspect
            if sp[a.cell] == :landing_zone
                return SparseCat(observations, [0.1, 0.8, 0.1])
            elseif sp[a.cell] == :obstacle
                return SparseCat(observations, [0.1, 0.1, 0.8])
            else
                return SparseCat(observations, [0.8, 0.1, 0.1])
            end
        else
            return Deterministic(:empty)
        end
    end,

    reward = function (s, a)
        if a isa Inspect
            return -1
        elseif a isa Land
            if s[a.cell] == :landing_zone
                return 10.0
            elseif s[a.cell] == :obstacle
                return -100.0
            else
                return -50.0
            end
        end
    end,

    initialstate = Uniform(all_grid_states),
)


struct ExploreOnlyRollout <: Policy end

function POMDPs.action(::ExploreOnlyRollout, b)
    return Inspect(rand(1:9))  # Always inspect
end




# --- Solver and belief updater ---
solver = POMCPOWSolver(max_depth=20, tree_queries=10000, criterion=MaxUCB(5), enable_action_pw=false, check_repeat_act=true, check_repeat_obs=true)
updater = DiscreteUpdater(pomdp)

# --- True state ---
true_state = rand(valid_true_states)
landing_idx = findfirst(x -> x == :landing_zone, true_state)
obstacle_idxs = findall(x -> x == :obstacle, true_state)

println("==== Ground Truth ====")
println("Landing Zone Grid Cell: $landing_idx")
println("Obstacle Grid Cells: ", obstacle_idxs)
println("======================")

# --- Belief initialization ---
belief = initialize_belief(updater, Uniform(states(pomdp)))

# --- Helper for class confidence per cell ---
function cellwise_class_belief(belief, pomdp)
    belief_grid = Dict{Int, Dict{Symbol, Float64}}()
    for c in 1:9
        belief_grid[c] = Dict(:empty => 0.0, :landing_zone => 0.0, :obstacle => 0.0)
    end
    all_states = states(pomdp)
    for (i, prob) in enumerate(belief.b)
        s = all_states[i]
        for c in 1:9
            belief_grid[c][s[c]] += prob
        end
    end
    return belief_grid
end

# --- Main loop with online action selection ---
cumulative_reward = 0.0
action_trace = String[]
reward_for_plot = Float64[]
anim = Animation()
done = false

for t in 1:100
    global belief, true_state, cumulative_reward, done
    println("======== Time Step $t ========")

    # ✅ Use solver online at each step
    planner = POMCPOWPlanner(solver, pomdp)
    a = action(planner, belief)    
    println("Selected Action: $a")
    push!(action_trace, string(a))
    next_state_dist = transition_fn(true_state, a)
    true_state = rand(next_state_dist)
    obs = rand(observation(pomdp, true_state, a, true_state))

    cell_beliefs = cellwise_class_belief(belief, pomdp)

    if a isa Land
        conf = cell_beliefs[a.cell][:landing_zone]
        if conf < 0.88
            println("⚠️ Confidence too low (only $(round(conf, digits=3))). Skipping landing.")
            a = Inspect(a.cell)
        end
    end

    belief = update(updater, belief, a, obs)

    cell_beliefs = cellwise_class_belief(belief, pomdp)
    println("Per-cell class confidence:")
    for c in 1:9
        confs = cell_beliefs[c]
        println("  Cell $c: ", join(["$(k) → $(round(v, digits=3))" for (k, v) in confs], ", "))
    end




    r = reward(pomdp, true_state, a)
    cumulative_reward += r
    push!(reward_for_plot, cumulative_reward)
    println("Received Observation:$obs")
    println("Reward: $r | Cumulative Reward: $cumulative_reward")



    if a isa Land
        println("✈️ Landing attempted at timestep $t. Ending episode.")
        break
    end
    




    colors = [begin
        confs = cell_beliefs[i]
        r = confs[:obstacle]
        g = confs[:landing_zone]
        b = confs[:empty]
        RGB(r, g, b)
    end for i in 1:9]


    # --- Set emoji annotations based on true state ---
    annotations = ["" for _ in 1:9]
    annotations[landing_idx] = "H"  # landing zone

    for idx in obstacle_idxs
        annotations[idx] = "O"      # obstacle
    end

    # --- Belief heatmap ---
    heatmap(reshape(colors, (3, 3)),
        title="Belief Map (t=$t)\nAction: $a\nCumulative Reward: $(round(cumulative_reward, digits=2))",
        yflip=true, size=(400, 400), axis=false)

    # --- Overlay emoji annotations on heatmap ---
    for i in 1:9
        col = div(i - 1, 3) + 1  # Julia fills columns first
        row = mod(i - 1, 3) + 1
        if annotations[i] != ""
            annotate!(col, row, text(annotations[i], :black, :bold, 16))
        end
    end


    # --- Add frame to animation ---
    frame(anim)


end

# --- Save GIF ---
gif(anim, "landing_belief_1.gif", fps=5)

# --- Print Summary ---
println("==== Action Trace ====")
for (i, act) in enumerate(action_trace)
    println("Step $i: $act")
end
println("======================")

println("==== Final Cumulative Reward ====")
println(cumulative_reward)
println("=================================")

println("==== Ground Truth ====")
println("Landing Zone Grid Cell: $landing_idx")
println("Obstacle Grid Cells: ", obstacle_idxs)
println("======================")
