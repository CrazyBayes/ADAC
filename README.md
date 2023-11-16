# ADAC: Actor-Double-Attention-Critic for Multi-agent Reinforcement  Learning
This is the source code of ADAC based on [*R-MADPG*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/774412967f19ea61d448977ad9749078-Abstract.html) and our paper is submitted into journal "".
# Prerequisites
## Install dependencies
See ''requirments.txt'' file for more information about how to install the dependencies.
## Environments
The environments used in our paper are based on the [multi-agent particle environment (MPE)](https://github.com/openai/multiagent-particle-envs), including *Predator-prey*, *Adversarial*, and *Grassland*.
# Usages
(Option A) In these tasks, the agents of opponents or good agents should be pre-trained in 20k episodes. 

(Option B) Or, optional, you can use the files in fold **/pretrained/** where have pre-trained opponenets or good agens in 20k episodes by *MADDPG*.

For example, you can execute the *ADAC* as follows based on Option B.

```python
python ADAC.py  --scenario=adversarial --max-episode-len=50 --lr-actor=0.001 --lr-critic=0.001 --adv-policy=adac  --good-policy=maddpg --resume=/pretrained/ADAC/Adversarial_resume_8V8/ --n-food=6 --n-good=8 --n-adv=8 --exp-run-num=0
```

# Acknowledgments
We want to express our gratitude to the authors of [*R-MADDPG*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/774412967f19ea61d448977ad9749078-Abstract.html) for publishing the source code.
