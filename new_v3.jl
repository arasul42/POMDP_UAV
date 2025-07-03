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
        #return Deterministic(s)
        return Uniform(all_grid_states)
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
            return -1.0
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

# --- Solver and belief updater ---
solver = POMCPOWSolver(max_depth=15, tree_queries=5000, criterion=MaxUCB(5.0))
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

function summarize_belief(belief, target_type::Symbol, pomdp)
    scores = zeros(9)
    all_states = states(pomdp)
    for (i, prob) in enumerate(belief.b)
        s = all_states[i]
        for j in 1:9
            if s[j] == target_type
                scores[j] += prob
            end
        end
    end
    return scores ./ sum(scores)
end






# --- Main loop with online action selection ---
cumulative_reward = 0.0
action_trace = String[]
reward_for_plot = Float64[]
anim = Animation()
done = false

for t in 1:40
    global belief, true_state, cumulative_reward, done
    println("======== Time Step $t ========")

    # ✅ Use solver online at each step
    planner = POMCPOWPlanner(solver, pomdp)
    a = action(planner, belief)
    
    println("Selected Action: $a")
    push!(action_trace, string(a))

    r = reward(pomdp, true_state, a)
    cumulative_reward += r
    push!(reward_for_plot, cumulative_reward)

    if a isa Land
        println("✈️ Landing attempted at timestep $t. Ending episode.")
        break
    end




    obs = rand(observation(pomdp, true_state, a, true_state))
    println("Received Observation:$obs")
    println("Reward: $r | Cumulative Reward: $cumulative_reward")

    next_state_dist = transition_fn(true_state, a)
    true_state = rand(next_state_dist)

    belief = update(updater, belief, a, obs)


    bz = summarize_belief(belief, :landing_zone, pomdp)
    bo = summarize_belief(belief, :obstacle, pomdp)

    println("Per-cell landing_zone belief:")
    for c in 1:9
        println("  Cell $c: ", round(bz[c], digits=3))
    end

    
    colors = [RGB(bo[i], bz[i], 1 - bo[i] - bz[i]) for i in 1:9]
    
    heatmap(reshape(colors, (3, 3)),
        title="Belief Map (t=$t)\nAction: $a\nCumulative Reward: $(round(cumulative_reward, digits=2))",
        yflip=true, size=(400, 400), axis=false)

    # --- Set emoji annotations based on true state ---
    annotations = ["" for _ in 1:9]
    annotations[landing_idx] = "H"  # landing zone
    for idx in obstacle_idxs
        annotations[idx] = "O"      # obstacle
    end

    

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
gif(anim, "landing_belief_3.gif", fps=2)

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
