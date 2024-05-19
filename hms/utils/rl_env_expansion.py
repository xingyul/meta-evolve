


import copy
import numpy as np
import torch.nn as nn
import torch


def policy_expansion( \
        original_policy, \
        num_obs_expansion=0, \
        num_act_expansion=0, \
        init_log_std=-1.5):

    policy_expanded = copy.deepcopy(original_policy)
    original_in_dim = policy_expanded.model.fc_layers[0].in_features
    original_out_dim = policy_expanded.model.fc_layers[-1].out_features

    if num_obs_expansion > 0:
        ##### for policy.model
        input_module = policy_expanded.model.fc_layers[0]
        new_input_module = nn.Linear(original_in_dim + num_obs_expansion, \
                input_module.out_features)
        new_input_module.weight.data *= 0.
        new_input_module.weight.data[:, -original_in_dim:] = \
                input_module.weight.data
        new_input_module.bias.data = input_module.bias.data

        new_in_shift = np.concatenate([np.zeros([num_obs_expansion], dtype='float32'), \
                policy_expanded.model.in_shift.numpy()])
        new_in_shift = torch.from_numpy(new_in_shift)
        new_in_scale = np.concatenate([np.ones([num_obs_expansion], dtype='float32'), \
                policy_expanded.model.in_scale.numpy()])
        new_in_scale = torch.from_numpy(new_in_scale)

        policy_expanded.model.fc_layers[0] = new_input_module
        policy_expanded.model.in_shift = new_in_shift
        policy_expanded.model.in_scale = new_in_scale
        policy_expanded.model.obs_dim = original_in_dim + num_obs_expansion

        ##### for policy.old_model
        input_module = policy_expanded.old_model.fc_layers[0]
        new_input_module = nn.Linear(original_in_dim + num_obs_expansion, \
                input_module.out_features)
        new_input_module.weight.data *= 0.
        new_input_module.weight.data[:, -original_in_dim:] = \
                input_module.weight.data
        new_input_module.bias.data = input_module.bias.data

        new_in_shift = np.concatenate([np.zeros([num_obs_expansion], dtype='float32'), \
                policy_expanded.old_model.in_shift.numpy()])
        new_in_shift = torch.from_numpy(new_in_shift)
        new_in_scale = np.concatenate([np.ones([num_obs_expansion], dtype='float32'), \
                policy_expanded.old_model.in_scale.numpy()])
        new_in_scale = torch.from_numpy(new_in_scale)

        policy_expanded.old_model.fc_layers[0] = new_input_module
        policy_expanded.old_model.in_shift = new_in_shift
        policy_expanded.old_model.in_scale = new_in_scale
        policy_expanded.old_model.obs_dim = original_in_dim + num_obs_expansion

        ##### other vars
        policy_expanded.n = original_in_dim + num_obs_expansion
        policy_expanded.obs_var = torch.autograd.Variable( \
                torch.randn(policy_expanded.n), requires_grad=False)

    if num_act_expansion > 0:
        ##### for policy.model
        output_module = policy_expanded.model.fc_layers[-1]
        new_output_module = nn.Linear(output_module.in_features, \
                original_out_dim + num_act_expansion)
        new_output_module.weight.data[-original_out_dim:] = \
                output_module.weight.data
        new_output_module.weight.data[:-original_out_dim] *= 0.
        new_output_module.bias.data[-original_out_dim:] = output_module.bias.data
        new_output_module.bias.data[:-original_out_dim] *= 0.

        new_out_shift = np.concatenate([np.zeros([num_act_expansion], dtype='float32'), \
                policy_expanded.model.out_shift.numpy()])
        new_out_shift = torch.from_numpy(new_out_shift)
        new_out_scale = np.concatenate([np.ones([num_act_expansion], dtype='float32'), \
                policy_expanded.model.out_scale.numpy()])
        new_out_scale = torch.from_numpy(new_out_scale)

        policy_expanded.model.fc_layers[-1] = new_output_module
        policy_expanded.model.out_shift = new_out_shift
        policy_expanded.model.out_scale = new_out_scale
        policy_expanded.model.act_dim = original_out_dim + num_act_expansion

        ##### for policy.old_model
        output_module = policy_expanded.old_model.fc_layers[-1]
        new_output_module = nn.Linear(output_module.in_features, \
                original_out_dim + num_act_expansion)
        new_output_module.weight.data[-original_out_dim:] = \
                output_module.weight.data
        new_output_module.weight.data[:-original_out_dim] *= 0.
        new_output_module.bias.data[-original_out_dim:] = output_module.bias.data
        new_output_module.bias.data[:-original_out_dim] *= 0.

        new_out_shift = np.concatenate([np.zeros([num_act_expansion], dtype='float32'), \
                policy_expanded.old_model.out_shift.numpy()])
        new_out_shift = torch.from_numpy(new_out_shift)
        new_out_scale = np.concatenate([np.ones([num_act_expansion], dtype='float32'), \
                policy_expanded.old_model.out_scale.numpy()])
        new_out_scale = torch.from_numpy(new_out_scale)

        policy_expanded.old_model.fc_layers[-1] = new_output_module
        policy_expanded.old_model.out_shift = new_out_shift
        policy_expanded.old_model.out_scale = new_out_scale
        policy_expanded.old_model.act_dim = original_out_dim + num_act_expansion

        ##### other vars
        policy_expanded.m = original_out_dim + num_act_expansion

        policy_expanded.log_std = torch.autograd.Variable( \
                torch.ones(original_out_dim + num_act_expansion) * init_log_std, \
                requires_grad=True)
        policy_expanded.log_std.data[-original_out_dim:] = \
                original_policy.log_std.data
        policy_expanded.log_std_val = np.float64(policy_expanded.log_std.data.numpy().ravel())
        policy_expanded.old_log_std = torch.autograd.Variable( \
                torch.ones(original_out_dim + num_act_expansion) * init_log_std)
        policy_expanded.old_log_std.data[-original_out_dim:] = \
                original_policy.old_log_std.data

    policy_expanded.trainable_params = \
            list(policy_expanded.model.parameters()) + \
            [policy_expanded.log_std]
    policy_expanded.old_params = \
            list(policy_expanded.old_model.parameters()) + \
            [policy_expanded.old_log_std]

    policy_expanded.model.layer_sizes = \
            (original_in_dim + num_obs_expansion, ) + \
            policy_expanded.model.layer_sizes[1:-1] + \
            (original_out_dim + num_act_expansion, )

    policy_expanded.param_shapes = [p.data.cpu().numpy().shape for p in policy_expanded.trainable_params]
    policy_expanded.param_sizes = [p.data.cpu().numpy().size for p in policy_expanded.trainable_params]
    policy_expanded.d = np.sum(policy_expanded.param_sizes)

    return policy_expanded


def baseline_expansion( \
        original_baseline, \
        num_obs_expansion=0):

    baseline_expanded = copy.deepcopy(original_baseline)
    original_in_dim = baseline_expanded.model[0].in_features

    if num_obs_expansion > 0:
        input_module = baseline_expanded.model[0]
        new_input_module = nn.Linear(original_in_dim + num_obs_expansion, \
                input_module.out_features)
        new_input_module.weight.data *= 0.
        new_input_module.weight.data[:, -original_in_dim:] = \
                input_module.weight.data[:, -original_in_dim:]
        new_input_module.bias.data = input_module.bias.data

        baseline_expanded.model[0] = new_input_module
        baseline_expanded.n = original_in_dim + num_obs_expansion

    return baseline_expanded
