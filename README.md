# ADAC
This is the source code of ADAC based on [R-MADPG](https://proceedings.neurips.cc/paper_files/paper/2020/hash/774412967f19ea61d448977ad9749078-Abstract.html) and our paper is submitted into journal "".

(Option A) In these tasks, the agents of opponents or good agents should be pre-trained in 20k episodes. 

(Option B) Or, optional, you can use the files in fold /pretrained/.

For example, you can execute the ADAC as follows based on Option B.

'''python
python ADAC.py  --scenario=adversarial --max-episode-len=50 --lr-actor=0.001 --lr-critic=0.001 --adv-policy=adac  --good-policy=maddpg --resume=/pretrained/ADAC/Adversarial_resume8V8/ --n-food=6 --n-good=8 --n-adv=8 --exp-run-num=0
'''

We want to express our gratitude to the authors of [R-MADDPG](https://proceedings.neurips.cc/paper_files/paper/2020/hash/774412967f19ea61d448977ad9749078-Abstract.html) for providing the source code.
