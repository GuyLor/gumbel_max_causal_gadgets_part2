# Learning Generalized Gumbel-Max Causal Mechanisms

## Overview

This repository contains the code for the second part of the NeurIPS 2021 paper "Learning Generalized Gumbel Max-Causal Mechanisms"(The code for the first part can be found [here](https://github.com/google-research/google-research/tree/master/gumbel_max_causal_gadgets)).  

* The second part of experiment 2 is in 'experiment_2b.ipynb' (section 7.2, the two right columns of Table 1).
* To run the third experiment (section 7.3, sepsis MDP) 'run_mdp_experiment.py' using the commands in runs.sh (By default, it extracts and uses MDP parameters that are already learned. Alternatively, you can run `learn_mdp_parameters.ipynb`, which will save the learned parameters in the `data` folder. This takes ~2 hours.)



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Acknowledgements
The code for the MDP environment on experiment 3 is based on Michael Oberst and David Sontag [code](https://github.com/clinicalml/gumbel-max-scm) for their paper: "Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models". Thanks!

In addition, we would like to thank Christina X Ji and [Fredrik D. Johansson](http://www.mit.edu/~fredrikj/) for their work on an earlier version of the sepsis simulator we use in this paper.
