# UAV Landing Zone Search with POMDPs

This project implements a **Partially Observable Markov Decision Process (POMDP)** framework for actively locating a safe landing zone in a 3D environment using onboard camera observations.

## 🧠 Problem Overview

A UAV (e.g., drone) equipped with a limited field-of-view (FOV) camera must find a designated **landing zone** hidden among other cells that may be **empty** or contain **obstacles**. The challenge is compounded by:
- Uncertain observations,
- Large state and observation spaces.

## ✨ Key Features

- **OO-POMDP Formulation:** Object-oriented POMDP model for grid-based 3D search.

## 📚 References
- POMDPs.jl documentation: https://juliapomdp.github.io

## 📄 License

MIT License. See `LICENSE` file.
