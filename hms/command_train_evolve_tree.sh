

algorithm=NPG
exp=herd
exp=steiner
# exp=geom_med

metric=l1
# metric=l2

seed=1025
gpu=0

init_log_std=-1.5


generalized_env=generalized-relocate-v0
interp_vector_dir=log_relocate-v0/iterations/
obs_expansion=0
act_expansion=0

# generalized_env=generalized-hammer-v0
# interp_vector_dir=log_hammer-v0/iterations/
# obs_expansion=0
# act_expansion=0

# generalized_env=generalized-door-v0
# interp_vector_dir=log_door-v0/iterations/
# obs_expansion=0
# act_expansion=0

round_id=0


interp_progression=0.03
interp_sample_range=0.03
eval_success_rate_thres=0.6
expl_success_rate_thres=0.5
sample_mode=trajectories
num_traj=12
num_jac=96
horizon=200
num_cpu=12
save_freq=1
step_size=0.0001
gamma=0.995
gae=0.97
env_sample_method=directional

log_root_dir=log_${generalized_env}_${exp}__${metric}__seed_${seed}
mkdir ${log_root_dir}


initial_dir_random=0
attract_lambda=1.05
attract_lambda=1e6
log_dir=${log_root_dir}

# log_dir=log_debug


#### launch meta
python train_evolve_tree.py \
    --generalized_env $generalized_env \
    --gpu $gpu \
    --exp $exp \
    --metric $metric \
    --algorithm $algorithm \
    --interp_progression $interp_progression \
    --round_id $round_id \
    --interp_vector_dir $interp_vector_dir \
    --obs_expansion $obs_expansion \
    --act_expansion $act_expansion \
    --init_log_std $init_log_std \
    --eval_success_rate_thres $eval_success_rate_thres \
    --expl_success_rate_thres $expl_success_rate_thres \
    --initial_dir_random $initial_dir_random \
    --interp_sample_range $interp_sample_range \
    --attract_lambda $attract_lambda \
    --seed $seed \
    --horizon $horizon \
    --sample_mode $sample_mode \
    --env_sample_method $env_sample_method \
    --num_jac $num_jac \
    --num_traj $num_traj \
    --num_cpu $num_cpu \
    --save_freq $save_freq \
    --step_size $step_size \
    --gamma $gamma \
    --gae $gae \
    --log_dir $log_dir \
    > $log_dir.txt 2>&1 &
