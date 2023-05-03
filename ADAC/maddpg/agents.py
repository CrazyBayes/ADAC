import itertools as it
import os

import numpy as np
import torch
import torch.nn.functional as F
from ADAC.common.memory import ReplayMemory
from ADAC.maddpg.models import Actor
from ADAC.common.distributions import truncated_normal_
from torch.distributions import RelaxedOneHotCategorical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device used is %s' % DEVICE)

class Agent:
    def act(self, obs, **kwargs):
        raise NotImplementedError

    def experience(self, episode_count, obs, action, reward, new_obs, done):
        pass

    def update(self, agents):
        pass


class SpreadScriptedAgent(Agent):
    def __init__(self, index, name, env):
        self.env = env
        self.index = index
        self.name = name

    @staticmethod
    def length(a):
        return np.sqrt(np.sum(a**2))

    @staticmethod
    def acc2action(acc):
        action = np.zeros(5)
        for i, a in enumerate(acc):
            if a >= 0:
                action[1+i*2] = a
            else:
                action[2+i*2] = -a
        if abs(np.sum(action)) > 0:
            action = action / np.sum(action)
        return action

    def get_target(self, agents, landmarks):
        matchings = [list(zip(agents, p)) for p in it.permutations(landmarks)]
        dists = [sum(self.length(l - a) for a, l in m) for m in matchings]
        best_matching = matchings[np.argmin(dists)]
        return best_matching[self.index][1]

    def act(self, obs, **kwargs):
        # vel = obs[:2]
        l1 = obs[2:4]
        l2 = obs[4:6]
        l3 = obs[6:8]
        a1 = obs[8:10]
        a2 = obs[10:12]
        # target = self.get_target([l1, l2, l3], [a1, a2])
        landmarks = [l1, l2, l3]
        agents = [a1, a2]
        agents.insert(self.index, [0, 0])
        target = self.get_target(agents, landmarks)
        return self.acc2action(target)


class RandomAgent(Agent):
    def __init__(self, index, name, env):
        self.env = env
        self.index = index
        self.name = name
        self.num_actions = self.env.action_space[self.index].n

    def act(self, obs, **kwargs):
        logits = np.random.sample(self.num_actions)
        return logits / np.sum(logits)


class MADDPGAgent(Agent):
    def __init__(self, index, name, env, actor, critic, params, policy_name):
        self.index = index
        self.name = name
        self.typename = env.agents[index].typename
        self.env = env
        self.num_agents = env.n
        try:
            self.num_adversaries = len(env.world.adversaries)
        except:
            self.num_adversaries = 0
        self.local = policy_name == 'ddpg' ##这里local表示是否使用DDPG策略

        self.actor = actor.to(DEVICE)
        self.actor_target = actor.clone().to(DEVICE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params.lr_actor)

        self.critic = critic.to(DEVICE)
        self.critic_target = critic.clone().to(DEVICE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=params.lr_critic)

        self.memory = ReplayMemory(params.memory_size, params.max_episode_len,
                                   self.actor.n_outputs, self.actor.n_inputs)
        self.mse = torch.nn.MSELoss()

        # params
        self.batch_size = params.batch_size
        self.tau = params.tau      # for soft update of target networks
        self.gamma = params.gamma  # discount


    # def init_agent_models(self, agents):
    #     for agent in agents:
    #         if agent is self:
    #             continue
    #         agent_model = self.model_class.from_actor(agent.actor).to(DEVICE)
    #         self.agent_models[agent.index] = agent_model
    #         optim = torch.optim.Adam(agent_model.parameters(), lr=self.model_lr)
    #         self.model_optims[agent.index] = optim

    def update_params(self, target, source):
        zipped = zip(target.parameters(), source.parameters())
        for target_param, source_param in zipped:
            updated_param = target_param.data * (1.0 - self.tau) + \
                source_param.data * self.tau
            target_param.data.copy_(updated_param)

    def act(self, obs, explore=True):
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(DEVICE)
        actions = self.actor.select_action(obs, explore=explore).detach()
        return actions.to('cpu').numpy()

    def experience(self, episode_count, obs, action, reward, new_obs, done):
        self.memory.add(episode_count, obs, action, reward, new_obs, float(done))

    def train_actor(self, batch):

        ### forward pass ###
        pred_actions = self.actor.select_action(batch.observations[self.index])
        actions = list(batch.actions)

        q_obs = [batch.observations[self.index]] if self.local else batch.observations

        actions[self.index] = pred_actions
        q_actions = [actions[self.index]] if self.local else actions
        pred_q = self.critic(q_obs, q_actions)

        ### backward pass ###源码原本的样子 regularize magnitude of logits
        p_reg = torch.mean(self.actor(batch.observations[self.index])**2) ##
        loss = -pred_q.mean() + 1e-3 * p_reg
        self.actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()


        return loss

    def train_critic(self, batch, agents):
        """Train critic with TD-target."""
        ### forward pass ###
        # (a_1', ..., a_n') = (mu'_1(o_1'), ..., mu'_n(o_n'))
        self_obs = batch.next_observations[self.index]
        self_action = self.actor_target.select_action(self_obs).detach()
        if self.local:
            pred_next_actions = [self_action]
        else:
            pred_next_actions = [a.actor_target.select_action(o).detach()
                                 for o, a in zip(batch.next_observations, agents)]

        q_next_obs = [batch.next_observations[self.index]] if self.local else batch.next_observations


        q_next = self.critic_target(q_next_obs, pred_next_actions)

        reward = batch.rewards[self.index]
        done = batch.dones[self.index]

        # y = r + (1-done) * gamma * Q'(o_1, ..., o_n, a_1', ..., a_n')
        q_target = reward + (1.0 - done) * self.gamma * q_next


        q_obs = [batch.observations[self.index]] if self.local else batch.observations
        q_actions = [batch.actions[self.index]] if self.local else batch.actions
        loss = self.mse(self.critic(q_obs, q_actions), q_target.detach())

        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        return loss


    def update(self, agents):
        # collect transition memories form all agents
        memories = [a.memory for a in agents]


        # sample minibatch
        batch = self.memory.sample_transitions_from(memories, self.batch_size)
        critic_loss = self.train_critic(batch, agents)
        actor_loss = self.train_actor(batch)

    # update target network params
        self.update_params(self.actor_target, self.actor)
        self.update_params(self.critic_target, self.critic)
        return actor_loss, critic_loss

    def get_state(self):
        model_pair = None
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, model_pair  # TODO: also nature actor if available

    def load_state(self, state):
        for key, value in state['state_dicts'].items():
            getattr(self, key).load_state_dict(value)
        if 'models' in state:
            models, optims = state['models']
            for i, m in models.items():
                self.agent_models[i].load_state_dict(m)
            for i, o in optims.items():
                self.model_optims[i].load_state_dict(o)


class M_MADDPGAgent(Agent):
    def __init__(self, index, name, env, actor, critic, params, policy_name):
        self.index = index
        self.name = name
        self.env = env
        self.typename= env.agents[index].typename
        self.num_agents = env.n
        try:
            self.num_adversaries = len(env.world.adversaries)
        except:
            self.num_adversaries = 0
        self.policy_name = policy_name
        self.local = policy_name == 'ddpg' ##这里local表示是否使用DDPG策略

        self.actor = actor.to(DEVICE)
        self.actor_target = actor.clone().to(DEVICE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params.lr_actor)

        self.critic = critic.to(DEVICE)
        self.critic_target = critic.clone().to(DEVICE)



        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=params.lr_critic)

        self.memory = ReplayMemory(params.memory_size, params.max_episode_len,
                                   self.actor.n_outputs, self.actor.n_inputs)
        self.mse = torch.nn.MSELoss()

        # params
        self.batch_size = params.batch_size
        self.tau = params.tau      # for soft update of target networks
        self.gamma = params.gamma  # discount


    # def init_agent_models(self, agents):
    #     for agent in agents:
    #         if agent is self:
    #             continue
    #         agent_model = self.model_class.from_actor(agent.actor).to(DEVICE)
    #         self.agent_models[agent.index] = agent_model
    #         optim = torch.optim.Adam(agent_model.parameters(), lr=self.model_lr)
    #         self.model_optims[agent.index] = optim

    def update_params(self, target, source):
        zipped = zip(target.parameters(), source.parameters())
        for target_param, source_param in zipped:
            updated_param = target_param.data * (1.0 - self.tau) + \
                source_param.data * self.tau
            target_param.data.copy_(updated_param)

    def act(self, obs, explore=True):
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(DEVICE)
        actions = self.actor.select_action(obs, explore=explore).detach()
        return actions.to('cpu').numpy()

    def experience(self, episode_count, obs, action, reward, new_obs, done):
        self.memory.add(episode_count, obs, action, reward, new_obs, float(done))

    ###
    def train_actor(self, batch, agents):

        ### forward pass ###
        pred_actions = self.actor.select_action(batch.observations[self.index])
        actions = list(batch.actions)

        q_obs =  batch.observations

        actions[self.index] = pred_actions
        q_actions = actions
            ####


        pred_q = self.critic(q_obs, q_actions, agents)

            ### backward pass ###源码原本的样子 regularize magnitude of logits
        p_reg = torch.mean(self.actor(batch.observations[self.index]) ** 2)  ##
        loss = -pred_q.mean() + 1e-3 * p_reg
        self.actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()


        return loss


    def train_critic(self, batch, agents):
        """Train critic with TD-target."""
        ### forward pass ###
        # (a_1', ..., a_n') = (mu'_1(o_1'), ..., mu'_n(o_n'))

        obs_next = batch.next_observations
        pred_actons_next = [a.actor_target.select_action(o).detach()
                                for o, a in zip(batch.next_observations, agents)]
              # embeding_current__type[a.typename].append(em_in)
        q_next = self.critic_target(obs_next, pred_actons_next, agents)
            ####end
            ###########
            #*current_state*
        obs = batch.observations
        actions = batch.actions


            ##########
        reward = batch.rewards[self.index]
        done = batch.dones[self.index]

            # y = r + (1-done) * gamma * Q'(o_1, ..., o_n, a_1', ..., a_n')
        q_target = reward + (1.0 - done) * self.gamma * q_next
        critic_q = self.critic(obs, actions, agents)

        loss = self.mse(critic_q, q_target.detach())

        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()


        return loss



    def update(self, agents):
        # collect transition memories form all agents
        memories = [a.memory for a in agents]


        # sample minibatch
        batch = self.memory.sample_transitions_from(memories, self.batch_size)
        critic_loss = self.train_critic(batch, agents)
        actor_loss = self.train_actor(batch, agents)

    # update target network params
        self.update_params(self.actor_target, self.actor)
        self.update_params(self.critic_target, self.critic)
        return actor_loss, critic_loss

    def get_state(self):
        model_pair = None
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, model_pair  # TODO: also nature actor if available

    def load_state(self, state):
        for key, value in state['state_dicts'].items():
            getattr(self, key).load_state_dict(value)
        if 'models' in state:
            models, optims = state['models']
            for i, m in models.items():
                self.agent_models[i].load_state_dict(m)
            for i, o in optims.items():
                self.model_optims[i].load_state_dict(o)