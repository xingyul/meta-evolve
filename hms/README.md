## Hand Manipulation Suite Experiments

This document contains information for running Hand Manipulation Suite experiments reported in the paper. The current directory contains code and command scripts for our Meta-Evolve.

### Initial Expert Policies on Source Robot

We provide the expert policies trained from on the source robot in [log\_door-v0](https://github.com/xingyul/meta-evolve/blob/master/hms/log_door-v0), [log\_hammer-v0](https://github.com/xingyul/meta-evolve/blob/master/hms/log_hammer-v0) and [log\_relocate-v0](https://github.com/xingyul/meta-evolve/blob/master/hms/log_relocate-v0). The RL algorithms and tasks can be inferred from the directory names.

### Definition of MuJoCo Source and Target Models

The definition of the MuJoCo robot models of the source and target robot are in `make_generalized_envs.py`.
The instantiation of robot instances is shown in `test_env.py`.
To visualize the models and expert policies, please change the values of `target_robot`, `generalized_env` and `policy_file` variables in `test_env.py`.

### Launch Meta-Evolve

The command scripts for launching Meta-Evolve are the `command_train_evolve_tree.sh` files. 

Change `generalized_env` variable in the script for different tasks. For example, setting `generalized_env` to be `generalized-hammer-v0` will launch Meta-Evolve on hammer task.

Change `exp` variable in the script for different types of evolution strategy. For example, setting `exp` to be `herd` will launch vanilla HERD, `geom_med` will launch Meta-Evolve with only one meta robot, and `steiner` Steiner Tree version of Meta-Evolve.

### Visualize Trained Policies

The code for visualizing the trained policies is provided in `viz_policy.py`.
Set `--env_name` to be the name of the task (e.g. `generalized-hammer-v0` for Hammer Task), `--exp_dir` to be the root directory of the log files (e.g. `log_generalized-hammer-v0_steiner__l1__seed_1025`), `--path_name` to be the name of the path in the evolution (e.g. `log_steiner_2_two_finger` for the pash from steiner #2 to two-finger target robot), and `--round_idx` to be the step index of evolution (e.g. 383 to be the final one).







