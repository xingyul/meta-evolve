
algorithm=NPG
exp=herd
exp=steiner

metric=l1
# metric=l2

seed=1005
gpu=0

init_log_std=-1.5

generalized_env=generalized-ycb-v2-unified

ycb_scene_name=008_pudding_box_20200918_114346

ycb_scene_dir=$(pwd)/${ycb_scene_name}
interp_vector_dir=log_expert_${ycb_scene_name}/iterations
obs_expansion=1
act_expansion=7


interp_progression=0.05
interp_sample_range=0.05
eval_success_rate_thres=0.59
expl_success_rate_thres=0.5
num_eval=10
sample_mode=trajectories
num_traj=12
num_jac=96
horizon=250
num_cpu=12
save_freq=1
step_size=0.0002
gamma=0.995
gae=0.97
env_sample_method=directional

log_root_dir=log_${generalized_env}__${ycb_scene_name}__${exp}__${metric}__seed_${seed}
# log_root_dir=log_debug
mkdir ${log_root_dir}


initial_dir_random=0
attract_lambda=1.05
attract_lambda=1e6
log_dir=${log_root_dir}


#### launch meta
python train_evolve_tree.py \
    --generalized_env $generalized_env \
    --gpu $gpu \
    --exp $exp \
    --metric $metric \
    --algorithm $algorithm \
    --interp_progression $interp_progression \
    --interp_vector_dir $interp_vector_dir \
    --obs_expansion $obs_expansion \
    --act_expansion $act_expansion \
    --ycb_scene_dir $ycb_scene_dir \
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
    --num_eval $num_eval \
    --num_cpu $num_cpu \
    --save_freq $save_freq \
    --step_size $step_size \
    --gamma $gamma \
    --gae $gae \
    --log_dir $log_dir \
    > $log_dir.txt 2>&1 &
