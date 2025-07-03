using POMDPs
using QuickPOMDPs
using POMDPTools
using POMDPPolicies
using POMDPSimulators
using POMCPOW

# Define POMDP model
uav_model = QuickPOMDP(
    states = 1:9,  # true location of target in the 3x3 grid
    actions = 1:9, # which grid cell to inspect
    observations = [:target, :nothing],
    discount = 0.95,

    initialstate = Uniform(1:9),  # assume target could be anywhere

    # Transition: target stays in same cell since UAV is static
    transition = (s, a) -> Deterministic(s),

    # Observation: 
    # if inspecting correct cell, 90% chance to observe target
    # else, 10% chance due to noise
    observation = function(s, a, sp)
        if s == a
            return SparseCat([:target, :nothing], [0.9, 0.1])
        else
            return SparseCat([:target, :nothing], [0.1, 0.9])
        end
    end,

    # Reward: high reward for correctly detecting target
    reward = (s, a) -> s == a ? 10.0 : -1.0
)

# Create the solver
solver = POMCPOWSolver(criterion = MaxUCB(5.0), tree_queries = 1000)
policy = solve(solver, uav_model)


# Run and record one simulation episode
sim = HistoryRecorder(max_steps=15)  # max 15 inspection steps
history = simulate(sim, uav_model, policy)

# Print out episode details
for (i, step) in enumerate(history)
    println("Step $i")
    println("  Action (Inspected Grid): ", step.a)
    println("  Observation: ", step.o)
    println("  Reward: ", step.r)
    println()
end

