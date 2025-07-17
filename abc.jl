#= 
POMDP model for visual search on a high-resolution aerial image using a YOLO model.
Agent selects continuous rectangular regions (InspectRegion) as actions.
Observations are bounding boxes (YOLO detections) with confidence and class.
Belief is updated over a 2D grid map using detection likelihoods.
Uses POMCPOW for planning. 
=#

using POMDPs
using POMCPOW
using POMDPTools
using Distributions
using Plots
using Images
using Colors
using Random
using PyCall

# Python YOLO interface (replace with actual Python function or mock)
@pyimport yolo_wrapper as yw  # Assume Python has a callable yolo_wrapper.crop_detect(x, y, w, h) -> detections

# ---------------------- Types ----------------------

struct BoundingBox
    x::Float64
    y::Float64
    w::Float64
    h::Float64
end

struct Detection
    box::BoundingBox
    class::Symbol
    confidence::Float64
end

struct Observation
    detections::Vector{Detection}
end

struct InspectRegion
    x::Float64
    y::Float64
    w::Float64
    h::Float64
end

struct HelipadState
    box::BoundingBox  # true helipad box
end

struct VisualSearchPOMDP <: POMDP{HelipadState, InspectRegion, Observation}
    image::Array{RGB{N0f8},2}  # full high-res image
    image_width::Int
    image_height::Int
end

# ---------------------- POMDP Interface ----------------------

POMDPs.actions(p::VisualSearchPOMDP) = []  # POMCPOW will sample actions
POMDPs.states(p::VisualSearchPOMDP) = []
POMDPs.discount(p::VisualSearchPOMDP) = 0.95

function POMDPs.transition(p::VisualSearchPOMDP, s::HelipadState, a::InspectRegion)
    return Deterministic(s)  # static image
end

function POMDPs.observation(p::VisualSearchPOMDP, s::HelipadState, a::InspectRegion)
    # Convert crop to image coordinates
    crop_img = view(p.image, Int(a.y):Int(a.y+a.h), Int(a.x):Int(a.x+a.w))
    results = yw.crop_detect(Int(a.x), Int(a.y), Int(a.w), Int(a.h))  # returns list of (x, y, w, h, class, conf)

    dets = Detection[]
    for r in results
        cls = Symbol(r[4])
        push!(dets, Detection(BoundingBox(r[0], r[1], r[2], r[3]), cls, r[5]))
    end
    return Observation(dets)
end

function POMDPs.reward(p::VisualSearchPOMDP, s::HelipadState, a::InspectRegion)
    obs = POMDPs.observation(p, s, a)
    for det in obs.detections
        if det.class == :helipad && iou(det.box, s.box) > 0.5 && det.confidence > 0.5
            return 10.0
        end
    end
    return -1.0
end

# ---------------------- IOU ----------------------

function iou(bb1::BoundingBox, bb2::BoundingBox)
    x1 = max(bb1.x, bb2.x)
    y1 = max(bb1.y, bb2.y)
    x2 = min(bb1.x + bb1.w, bb2.x + bb2.w)
    y2 = min(bb1.y + bb1.h, bb2.y + bb2.h)
    if x2 <= x1 || y2 <= y1
        return 0.0
    end
    inter_area = (x2 - x1) * (y2 - y1)
    union_area = bb1.w * bb1.h + bb2.w * bb2.h - inter_area
    return inter_area / union_area
end

# ---------------------- Belief Grid ----------------------

struct BeliefGrid
    grid::Matrix{Float64}
    resolution::Int
end

function initialize_belief_grid(res::Int)
    return BeliefGrid(fill(1.0 / (res^2), res, res), res)
end

function update_belief_grid!(belief::BeliefGrid, obs::Observation, region::InspectRegion)
    for i in 1:belief.resolution, j in 1:belief.resolution
        cx = (i - 0.5) * (1000 / belief.resolution)
        cy = (j - 0.5) * (1000 / belief.resolution)
        for d in obs.detections
            if d.class == :helipad && d.confidence > 0.3
                b = BoundingBox(cx, cy, 20.0, 20.0)
                belief.grid[i, j] += iou(b, d.box)
            end
        end
    end
    belief.grid ./= sum(belief.grid)
end

# ---------------------- Visualization ----------------------

function visualize_belief(belief::BeliefGrid)
    heatmap(belief.grid', c=:viridis, aspect_ratio=1, title="Helipad Belief Map", xlabel="x", ylabel="y")
end

# ---------------------- Example Usage ----------------------

# Load image (replace with your real aerial image)
img = rand(RGB{N0f8}, 1000, 1000)  # Dummy image
p = VisualSearchPOMDP(img, 1000, 1000)

# Define true state (hidden)
s = HelipadState(BoundingBox(480.0, 520.0, 40.0, 40.0))

# Initialize belief grid
belief = initialize_belief_grid(50)

# Planner
solver = POMCPOWSolver(max_depth=10, tree_queries=1000)
planner = POMCPOWPlanner(solver, p)

anim = Animation()

for t in 1:10
    a = InspectRegion(rand(1:800), rand(1:800), 200.0, 200.0)  # sample random region
    obs = POMDPs.observation(p, s, a)
    update_belief_grid!(belief, obs, a)
    visualize_belief(belief)
    frame(anim)
end

gif(anim, "yolo_search_belief.gif", fps=2)
