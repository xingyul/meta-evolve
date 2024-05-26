## DexYCB data Experiments

This document contains information for running DexYCB dataset experiments reported in the paper. The current directory contains code and command scripts for our Meta-Evolve.

### Initial Expert Policies on Source Robot

We provide the expert policies trained from on the source robot in [log\_expert\_008\_pudding\_box\_20200918\_114346](https://github.com/xingyul/meta-evolve/blob/master/dex_ycb/log_expert_008_pudding_box_20200918_114346).

### Definition of MuJoCo Source and Target Models

The definition of the MuJoCo robot models of the source and target robot are in `make_generalized_envs.py`.
The instantiation of robot instances is shown in `test_env_unified.py`.
To visualize different robot models, please change the values of `robot_name` variable in `test_env_unified.py`.

### Launch Meta-Evolve

The command script for launching Meta-Evolve is the `command_train_evolve_tree.sh` file. 

Change `exp` variable in the script for different types of evolution strategy. For example, setting `exp` to be `herd` will launch vanilla HERD, and `steiner` will launch Steiner Tree version of Meta-Evolve.

### Visualize Trained Policies

The code for visualizing the trained policies is provided in `viz_policy.py`.
Set `--exp_dir` to be the root directory of the log files (e.g. `log_generalized-ycb-v2-unified__008_pudding_box_20200918_114346__steiner__l1__seed_1005`), `--path_name` to be the name of the path in the evolution (e.g. `log__steiner_2__kinova` for the pash from steiner #2 to Kinova3 target robot), and `--round_idx` to be the step index of evolution (e.g. 218 to be the final one).







