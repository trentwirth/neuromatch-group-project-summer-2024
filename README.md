# Neuromatch, Summer 2024 - Exploring Ring Attractors

> Main Contributors:
> 
> Drs.
> - Trenton Wirth
> - PB Aneesh
> - Farhana Yasmin
> - Antoine Madar


The point of this repository is to explore ring attractor models for our group project for Neuromatch's course in Computational Neuroscience (2024). 

For our project, we decided to explore the Bayesian perception task found in [Laquitaine & Gardner, 2018](https://www.cell.com/neuron/fulltext/S0896-6273(17)31134-0). 

This code contains a few important files:
- `group_project/`
    - `main.py` -> the core file of this repo, it uses parallel processing to simulate ring attractor models based on a stimulus of "input_bias"es
    - `slow_simulation.py` -> this doesn't use parallel processing, and it would theoretically be the basis for future simulations that would rely on the trial order. This file is also used for debugging, because debugging is so hard in parallel processing...
    - `fit_parameters.py` -> this script is the "rough draft" of a procedure for fitting paramters. We wanted to fit `k_constant` (the weight of the motion coherence) and `neuron_bump_width` (how many neurons wide the activity spread of the coherent motion stimulus was in the code)
 
    - The best starting place to explore is in `slow_simulation.py`. All important functions are in the `group_project/utilities/` directory. Functions are imported regularly throughtout the project.
    - all "hard-coded" values are located in `input_values.py`. Use this file to change baseline functionality of the codebase.
 
The current result of the ring simulation is a "percept_decision" angle, 0-360 degrees (integer value). This is used to simulate the task in [Laquitaine & Gardner, 2018](https://www.cell.com/neuron/fulltext/S0896-6273(17)31134-0) (same paper link above).

Our group split into two subgroups, this code represents the work that went into the ring attractor simulation, but the other subgroup focused on creating a GLM to explore how trial feedback error might be combined with participant estimate history to predict future decisions.
Here is a link to our groups presentation slides: [Google Slides Link](https://docs.google.com/presentation/d/1J_PKW3DwD1ANmqQlLWMaXDaM1PZooryL3v2jaMFuLBI/edit?usp=sharing)

