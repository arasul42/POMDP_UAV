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

# --- Generate all valid states ---
all_grid_states = []  # renamed from 'states' to avoid shadowing 'states()' function

for i in 1:9  # landing zone at cell i
    for obs_cells in Iterators.filter(x -> i ∉ x, collect(combinations(setdiff(1:9, [i]), 3)))
        s = fill(:empty, 9)
        s[i] = :landing_zone
        for j in obs_cells
            s[j] = :obstacle
        end
        push!(all_grid_states, s)
    end
end

# --- Define POMDP ---
pomdp = QuickPOMDP(
    states = all_grid_states,
    actions = 1:9,
    observations = observations,
    discount = 0.95,

    transition = (s, a) -> Deterministic(s),

    observation = function (s, a, sp)
        cell = sp[a]
        if cell == :landing_zone
            return SparseCat(observations, [0.05, 0.90, 0.05])
        elseif cell == :obstacle
            return SparseCat(observations, [0.05, 0.05, 0.9])
        else
            return SparseCat(observations, [0.9, 0.05, 0.05])
        end
    end,

    reward = function (s, a)
        if s[a] == :landing_zone
            return 5.0
        elseif s[a] == :obstacle
            return +1.0  # Changed from -5.0 to +2.0 to reward informative discovery
        else
            return -2.0
        end
    end,

    initialstate = Uniform(all_grid_states)
)

# --- Solver and belief updater ---
solver = POMCPOWSolver(max_depth=10, tree_queries=1000, criterion=MaxUCB(5.0))
planner = solve(solver, pomdp)
updater = DiscreteUpdater(pomdp)

# --- True state for simulation ---
true_state = rand(all_grid_states)

# Print ground truth
landing_idx = findfirst(x -> x == :landing_zone, true_state)
obstacle_idxs = findall(x -> x == :obstacle, true_state)

println("==== Ground Truth ====")
println("Landing Zone Grid Cell: $landing_idx")
println("Obstacle Grid Cells: ", obstacle_idxs)
println("======================")

# --- Initialize belief ---
belief = initialize_belief(updater, Uniform(all_grid_states))

# --- Helper: compute belief over cell content ---
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

# --- Prepare animation ---
anim = @animate for t in 1:20
    global belief

    println("======== Time Step $t ========")

    a = action(planner, belief)
    println("Selected Action (Grid Cell): $a")

    obs = rand(observation(pomdp, true_state, a, true_state))
    println("Observation Received: $obs")

    r = reward(pomdp, true_state, a)

    # Penalize redundant inspection of high-confidence cells
    bz = summarize_belief(belief, :landing_zone, pomdp)
    bo = summarize_belief(belief, :obstacle, pomdp)

    if bz[a] > 0.9 || bo[a] > 0.3
        println("  🔁 Penalty: Re-inspecting confident cell $a")
        r -= 1
    end

    println("Reward (after penalty): $r")

    belief = update(updater, belief, a, obs)

    bz = summarize_belief(belief, :landing_zone, pomdp)
    bo = summarize_belief(belief, :obstacle, pomdp)

    top_lz = sortperm(bz, rev=true)[1:3]
    top_obs = sortperm(bo, rev=true)[1:3]

    println("Top Landing Zone Beliefs:")
    for i in top_lz
        println("  Cell $i → $(round(bz[i], digits=3))")
    end

    println("Top Obstacle Beliefs:")
    for i in top_obs
        println("  Cell $i → $(round(bo[i], digits=3))")
    end

    landing = reshape(bz, (3,3))
    obstacle = reshape(bo, (3,3))
    p1 = heatmap(landing, c=:blues, title="Landing Zone Belief (t=$t)", clims=(0,1), yflip=true)
    p2 = heatmap(obstacle, c=:reds, title="Obstacle Belief (t=$t)", clims=(0,.3), yflip=true)
    plot(p1, p2, layout=(1,2), size=(800,400))

    if maximum(bz) > 0.95 && count(x -> x > 0.7, bo) >= 3
        println("🎯 Confident belief found. Stopping early at timestep $t.")
        break
    end
end

gif(anim, "belief_evolution.gif", fps=5)

println("==== Ground Truth ====")
println("Landing Zone Grid Cell: $landing_idx")
println("Obstacle Grid Cells: ", obstacle_idxs)
println("======================")
