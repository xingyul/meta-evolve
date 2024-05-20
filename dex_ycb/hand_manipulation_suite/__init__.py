from gym.envs.registration import register

register(
    id='relocate-v0',
    entry_point='hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.relocate_v0 import RelocateEnvV0

register(
    id='generalized-ycb-v2-unified',
    entry_point='hand_manipulation_suite:GeneralizedYCBEnvV2Unified',
    max_episode_steps=200,
)
from hand_manipulation_suite.generalized_ycb_v2_unified import GeneralizedYCBEnvV2Unified

