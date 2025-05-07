# ADAC: Actor-Double-Attention-Critic for Multi-agent Cooperation in Mixed Cooperative-competitive Environments
This is the source code of ADAC based on [*R-MADDPG*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/774412967f19ea61d448977ad9749078-Abstract.html) and our paper is accepted by journal "*IEEE Transactions on Intelligent Transportation Systems*".
# Prerequisites
## Install dependencies
See ``requirments.txt`` file for more information about how to install the dependencies.
## Environments
The environments used in our paper are based on the [multi-agent particle environment (MPE)](https://github.com/openai/multiagent-particle-envs), including *Predator-prey*, [*Adversarial*, and *Grassland*](https://github.com/qian18long/epciclr2020).
# Usages
(Option A) In these tasks, the opponents or good agents should be pre-trained in 20k episodes. 

(Option B) Or, optionally, you can use the files in folder **``/pretrained/``** where opponents or good agents have pre-trained in 20k episodes by *MADDPG*. 

For example, you can execute the *ADAC* as follows based on Option B.
#### Adversarial
```python
python ADAC.py  --scenario=adversarial --max-episode-len=50 --lr-actor=0.001 --lr-critic=0.001 --adv-policy=adac  --good-policy=maddpg --resume=/pretrained/Adversarial_resume_8V8/ --n-food=6 --n-good=8  --n-adv=8 --exp-run-num=0
```

#### Grassland
```python
python ADAC.py  --scenario=grassland --max-episode-len=25 --lr-actor=0.001 --lr-critic=0.001 --adv-policy=adac  --good-policy=maddpg --resume=/pretrained/Grassland_resume_4V6/ --alpha=0.0 --n-food=4 --n-good=4 --n-adv=6  --exp-run-num=0
```

#### Predator-prey
```python
python ADAC.py  --scenario=simple_tag --max-episode-len=25 --lr-actor=0.001 --lr-critic=0.001 --adv-policy=adac  --good-policy=maddpg --resume=/pretrained/Predator_prey_resume_5V3/ --exp-run-num=0
```

# Acknowledgments
We want to express our gratitude to the authors of [*R-MADDPG*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/774412967f19ea61d448977ad9749078-Abstract.html) for publishing the source code.
# Citation
```
@ARTICLE{10979333,
  author={Kong, He and Xing, Qianli and Wang, Qi and Niu, Runliang and Chen, Hechang and Wang, Yu and Wang, Shiqi and Duan, Zhiyi and Chang, Yi},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={ADAC: Actor-Double-Attention-Critic for Multi-Agent Cooperation in Mixed Cooperative-Competitive Environments}, 
  year={2025},
  pages={1-14},
  doi={10.1109/TITS.2025.3562302}}
```

