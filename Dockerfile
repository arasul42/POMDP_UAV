FROM julia:1.9

RUN julia -e 'using Pkg; \
    Pkg.add("POMDPs"); \
    Pkg.add("POMDPModels"); \
    Pkg.add("POMDPSimulators"); \
    Pkg.add("POMDPTools"); \
    Pkg.add("POMCPOW"); \
    Pkg.precompile()'

WORKDIR /workspace
CMD ["julia"]