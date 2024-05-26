### *Meta-Evolve*: Continuous Robot Evolution for One-to-many Policy Transfer

**ICLR 2024**

Created by <a href="http://xingyul.github.io">Xingyu Liu</a>, <a href="https://www.cs.cmu.edu/~dpathak" target="_blank">Deepak Pathak</a> and <a href="https://www.meche.engineering.cmu.edu/directory/bios/zhao-ding.html" target="_blank">Ding Zhao</a> from Carnegie Mellon University.

[[arXiv]](https://arxiv.org/abs/2405.03534) [[project]](https://sites.google.com/view/meta-evolve)

<img src="https://github.com/xingyul/meta-evolve/blob/master/doc/meta_evolve.png" width="100%">

### Citation
If you find our work useful in your research, please cite:

        @inproceedings{liu:2024:meta:evolve,
          title={Meta-Evolve: Continuous Robot Evolution for One-to-many Policy Transfer},
          author={Liu, Xingyu and Pathak, Deepak and Zhao, Ding},
          booktitle={International Conference on Learning Representations (ICLR)},
          year={2024}
        }

### Abstract

We investigate the problem of transferring an expert policy from a source robot to multiple different robots. To solve this problem, we propose a method named Meta-Evolve that uses continuous robot evolution to efficiently transfer the policy to each target robot through a set of tree-structured evolutionary robot sequences. The robot evolution tree allows the robot evolution paths to be shared, so our approach can significantly outperform naive one-to-one policy transfer. We present a heuristic approach to determine an optimized robot evolution tree. Experiments have shown that our method is able to improve the efficiency of one-to-three transfer of manipulation policy by up to 3.2x and one-to-six transfer of agile locomotion policy by 2.4x in terms of simulation cost over the baseline of launching multiple independent one-to-one policy transfers.


### Installation

Our implementation uses MuJoCo as simulation engine and PyTorch as deep learning framework. The code is tested under Ubuntu 20.04, Python 3.8, [mujoco-py](https://github.com/openai/mujoco-py) 2.1, and [PyTorch](https://pytorch.org/get-started/locally/) 2.2.2.

### Code for Hand Manipulation Suite Experiments

The code and scripts for our Hand Manipulation Suite experiments are in [hms/](https://github.com/xingyul/meta-evolve/blob/master/hms/). Please refer to [hms/README.md](https://github.com/xingyul/meta-evolve/blob/master/hms/README.md) for more details on how to use our code.

### Code for DexYCB dataset Experiments

The code and scripts for our Hand Manipulation Suite experiments are in [dex\_ycb/](https://github.com/xingyul/meta-evolve/blob/master/dex_ycb/). Please refer to [dex\_ycb/README.md](https://github.com/xingyul/meta-evolve/blob/master/dex_ycb/README.md) for more details on how to use our code.

### LICENSE

Please refer to `LICENSE` file.


