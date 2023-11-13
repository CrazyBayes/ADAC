import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Clonable:
    def clone(self, requires_grad=False):
        clone = copy.deepcopy(self)
        for param in clone.parameters():
            param.requires_grad = requires_grad
        return clone


class Actor(nn.Module, Clonable):
    @classmethod
    def from_actor(cls, actor):
        return cls(actor.n_inputs, actor.action_split, actor.n_hidden)

    def __init__(self, n_inputs, action_split, n_hidden):
        super().__init__()
        self.action_split = tuple(action_split)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = int(sum(action_split))
        self.act_net= nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.n_outputs)
        )


    def forward(self, x):

        logits = self.act_net(x)
        return logits

    def prob_dists(self, obs, temperature=1.0):
        logits = self.forward(obs)
        split_logits = torch.split(logits, self.action_split, dim=-1)
        temperature = torch.tensor(temperature).to(DEVICE)
        return [RelaxedOneHotCategorical(temperature, logits=l) for l in split_logits]

    def select_action(self, obs, explore=False, temp=1.0):
        distributions = self.prob_dists(obs, temp)
        if explore:
            actions = [d.rsample() for d in distributions]
        else:
            actions = [d.probs for d in distributions]
        return torch.cat(actions, dim=-1)


class Critic(nn.Module, Clonable):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.critic_net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )


    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=-1)

        return self.critic_net(x)


class M_Actor(nn.Module, Clonable):
    @classmethod
    def from_actor(cls, actor):
        return cls(actor.n_inputs, actor.action_split, actor.n_hidden)

    def __init__(self, n_inputs, action_split, n_hidden):
        super().__init__()
        self.action_split = tuple(action_split)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = int(sum(action_split))
        self.act_net= nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.n_outputs)
        )


    def forward(self, x):

        logits = self.act_net(x)
        return logits

    def prob_dists(self, obs, temperature=1.0):
        logits = self.forward(obs)
        split_logits = torch.split(logits, self.action_split, dim=-1)
        temperature = torch.tensor(temperature).to(DEVICE)
        return [RelaxedOneHotCategorical(temperature, logits=l) for l in split_logits]

    def select_action(self, obs, explore=False, temp=1.0):
        distributions = self.prob_dists(obs, temp)
        if explore:
            actions = [d.rsample() for d in distributions]
        else:
            actions = [d.probs for d in distributions]
        return torch.cat(actions, dim=-1)


class M_Critic(nn.Module, Clonable):
    def __init__(self, i,type_length, n_embedding_Layer_inputs,
                 embedding_hidden_size, embedding_output_size, ##embeding_layer的输出
                 n_hidden, n_head_cri, n_attention_dim): ##n_head_cri 是n_head
        super().__init__()
        #n_inputs 需要调整
        self.belongto = i
        self.type_length = type_length
        self.embedding_hidden_size = embedding_hidden_size
        self.embedding_output_size = embedding_output_size
        self.n_embedding_Layer_inputs = n_embedding_Layer_inputs
        ####Attention部分
        self.attend_heads = n_head_cri
        self.n_attention_dim = n_attention_dim
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.key_Q = nn.ModuleList()
        self.selector_Q = nn.ModuleList()


        self.critic_net = nn.ModuleList()
        for type in range(self.type_length):
            self.key_extractors.append(nn.ModuleList())
            self.selector_extractors.append(nn.ModuleList())
            for i in range(self.attend_heads):
                self.key_extractors[type].append(nn.Linear(self.embedding_output_size,
                                                 self.n_attention_dim, bias=False))
                self.selector_extractors[type].append(nn.Linear(self.embedding_output_size,
                                                          self.n_attention_dim, bias=False))
            self.critic_net.append(
                nn.Sequential(
                    nn.Linear(self.n_attention_dim * self.attend_heads, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, 1)
                )
            )

        for i in range(self.attend_heads):
            self.key_Q.append(nn.Linear(1,1, bias=False))
            self.selector_Q.append(nn.Linear(1,1, bias=False))


        self.critic_self_net = nn.Sequential(
            nn.Linear(self.embedding_output_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
    #### embedding
        self.embeddingLayers = nn.ModuleList()
        self.embed_trs = nn.ModuleList()
        for index in range(len(self.n_embedding_Layer_inputs)):
            self.embeddingLayers.append(
                nn.Sequential(
                    nn.Linear(self.n_embedding_Layer_inputs[index], self.embedding_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.embedding_hidden_size, self.embedding_output_size)
                )
                #embeddingLayer(self.n_embedding_Layer_inputs[index], self.embedding_hidden_size,
                 #             self.embedding_output_size)
            )
            self.embed_trs.append(
                nn.Sequential(
                    nn.Linear(self.embedding_output_size, self.embedding_output_size),
                    nn.LeakyReLU()
                )
            )




    #other_embed: dict{"typename':[]} 本层的agent形式 #self_embed, embeding_current__type
    def forward(self, observations, actions, agents):#(self, )

        embeding_current = []
        ##收集不同类型的embedding_current_type1
        embeding_current__type = {}
        embed_trs_ctype = {}

        for a in agents:
            embeding_current__type.setdefault(a.typename, [])
            embed_trs_ctype.setdefault(a.typename, [])
        self_embed = None
        for index, a in enumerate(agents):
            if index != self.belongto:
                em_in = self.embeddingLayers[index](
                    torch.cat([observations[index], actions[index]], dim=-1))
                embeding_current.append(
                    em_in
                )
                embeding_current__type[a.typename].append(em_in)
                embed_trs_ctype[a.typename].append(self.embed_trs[index](em_in))
            else:
                em_in = self.embeddingLayers[index](torch.cat([observations[index],
                                                               actions[index]], dim=-1))
                embeding_current.append(
                    em_in
                )
                self_embed = em_in

        #x = torch.cat([*observations, *actions], dim=-1)
        self_Q_critic = self.critic_self_net(self_embed)
        Q_values = []
        Q_values.append(self_Q_critic)
        type_count = 0

        for key in embeding_current__type:
            ##第一步e_jW_K 与 W_qe_i
            other_embed = embeding_current__type[key]
            other_v_embed_tr = embed_trs_ctype[key]
            self_head_selectors = [[sel_ext(self_embed)]
                                   for sel_ext in self.selector_extractors[type_count]]
            other_head_keys = [[k_ext(enc) for enc in other_embed] for
                               k_ext in self.key_extractors[type_count]]
            # extract sa values for each head for each agent
            ##第二步 weight=exp(e_jW_K * W_qe_i)
            sum_embed = []
            for current_sh_selec, curr_oh_keys in zip(self_head_selectors, other_head_keys):
                curr_attend_weights = []
                sh_selec = current_sh_selec[0].reshape(current_sh_selec[0].shape[0], -1, 1)
                for oh_keys in curr_oh_keys:
                    oh_keys = oh_keys.reshape(current_sh_selec[0].shape[0], 1, -1)
                    weight = torch.matmul(oh_keys, sh_selec).reshape(oh_keys.shape[0], 1)
                    curr_attend_weights.append(weight)
                curr_attend_weights = F.softmax(torch.stack(curr_attend_weights))
                sum_cur_embed = 0
                for w_o, enc in zip(curr_attend_weights, other_v_embed_tr):
                    sum_cur_embed += w_o * enc
                sum_embed.append(sum_cur_embed)
            sum_embed = torch.stack(sum_embed).reshape(self_embed.shape[0], -1)
            #sum_embed = torch.cat([self_embed, sum_embed], dim = -1)
            cu_Q_v = self.critic_net[type_count](sum_embed)
            Q_values.append(cu_Q_v)
            type_count += 1
        ###层间注意力机制
        Q_key = [[k_ext(enc) for enc in Q_values] for k_ext in self.key_Q]
        # extract sa values for each head for each agent
        Q_selectors = [[sel_ext(self_Q_critic)] for sel_ext
                       in self.selector_Q]
        sum_Q = []
        for current_sh_selec, curr_oh_keys in zip(Q_selectors, Q_key):
            curr_attend_weights = []
            sh_selec = current_sh_selec[0].reshape(current_sh_selec[0].shape[0], -1, 1)
            for oh_keys in curr_oh_keys:
                oh_keys = oh_keys.reshape(current_sh_selec[0].shape[0], 1, -1)
                weight = torch.matmul(oh_keys, sh_selec).reshape(current_sh_selec[0].shape[0], 1)
                curr_attend_weights.append(weight)
            curr_attend_weights = F.softmax(torch.stack(curr_attend_weights))
            sum_cur_embed = 0
            for w_o, enc in zip(curr_attend_weights, Q_values):
                sum_cur_embed += w_o * enc
            sum_Q.append(sum_cur_embed)
        return sum(sum_Q) / len(sum_Q)

        #return sum(Q_values)/len(Q_values)
        #return self.critic_net(x)


class NatureActor(nn.Module, Clonable):
    """ Similar to Q function, input obs and actions, output scalar """

    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, 1)

    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=-1)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        return self.lin_3(x)

#informatoin_i = g_i(o_i, a_i)

class embeddingLayer(nn.Module):

    ### n_inputs 等于一个智能体的(o_i, a_i)
    ### n_outpus等于输出尺寸，简单起见等于hidden size 但是 critic的输入应该是 n个 hidden_size
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin_1 = nn.Linear(n_inputs, n_hidden).to(DEVICE)
        #self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, n_outputs).to(DEVICE)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=-1)
        #x = x.to(DEVICE)
        x = F.relu(self.lin_1(x)).to(DEVICE)
        #x = F.relu(self.lin_2(x))
        return self.lin_3(x)

'''class embedingLayer(nn.Module):

    ### n_inputs 等于一个智能体的(o_i, a_i)
    ### n_outpus等于输出尺寸，简单起见等于hidden size 但是 critic的输入应该是 n个 hidden_size
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin_1 = nn.Linear(n_inputs, n_hidden).to(DEVICE)
        #self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, n_outputs).to(DEVICE)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=-1)
        #x = x.to(DEVICE)
        x = F.relu(self.lin_1(x)).to(DEVICE)
        #x = F.relu(self.lin_2(x))
        return self.lin_3(x)'''





