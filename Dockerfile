FROM julia:1.9

RUN julia -e 'using Pkg; \
    Pkg.add("POMDPs"); \
    Pkg.add("POMDPModels"); \
    Pkg.add("POMDPSimulators"); \
    Pkg.add("POMDPTools"); \
    Pkg.add("POMCPOW"); \
    Pkg.add("QuickPOMDPs"); \
    Pkg.add("POMDPTools"); \
    Pkg.add("Combinatorics"); \
    Pkg.add("Plots"); \
    Pkg.add("Printf"); \
    Pkg.precompile()'

WORKDIR /workspace
CMD ["julia"]