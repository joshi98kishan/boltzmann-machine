# boltzmann-machine

[Work In Progress! 🚧]

This repo implements the original boltzmann machine as presented in the paper "A Learning Algorithm for Boltzmann Machines" by Ackley, Hinton and Sejnowski.
The corresponding blog will be published here (probably next month): [joshi98kishan.github.io](https://joshi98kishan.github.io).

`boltzmann_machine.py` contains the code for Boltzmann Machine.

Experiments:
- exp 1, `bm_exp1_parity_problem` notebook contains code to train BM for parity problem. It is messy in the current state, need to clean it.
- exp 2, encoder problem:
    - 4-2-4 encoder: check `bm_424_encoder` notebook (yet to make it clean).
    - 8-3-8 encoder (under progress)
    
- exp 3 will be on shifter problem.

All these experiments are taken from [extended paper](https://www.cs.utoronto.ca/~hinton/absps/bmtr.pdf) on BM. Here I am trying to reproduce the results in the paper.
