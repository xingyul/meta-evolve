# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

import os
import numpy as np

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4

_ADROIT_TO_MANO_MAP = {
    # 'WRJ0': 'wrist',
    # 'THJ2': 'thumb_mcp',
    'THJ1': 'thumb_pip',
    # 'THJ0': 'thumb_dip',
    'S_thtip': 'thumb_tip',
    # 'FFJ2': 'index_mcp',
    'FFJ1': 'index_pip',
    # 'FFJ0': 'index_dip',
    'S_fftip': 'index_tip',
    # 'MFJ2': 'middle_mcp',
    'MFJ1': 'middle_pip',
    # 'MFJ0': 'middle_dip',
    'S_mftip': 'middle_tip',
    # 'RFJ2': 'ring_mcp',
    'RFJ1': 'ring_pip',
    # 'RFJ0': 'ring_dip',
    'S_rftip': 'ring_tip',
    # 'LFJ2': 'little_mcp',
    'LFJ1': 'little_pip',
    # 'LFJ0': 'little_dip',
    'S_lftip': 'little_tip'
}
