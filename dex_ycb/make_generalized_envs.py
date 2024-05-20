
import os

try:
    import mjcf_utils
except:
    import utils.mjcf_utils as mjcf_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from make_generalized_envs_unified import *


generalized_envs = {
        'relocate-v0-unified': generalized_relocate_unified,
        }


