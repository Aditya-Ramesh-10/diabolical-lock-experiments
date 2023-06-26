import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space, ac_hidden_dim_size=64,
                 use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.number_of_actions = action_space.n

        obs_dim = obs_space["image"][0]
        self.image_embedding_size = obs_dim
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, ac_hidden_dim_size),
            nn.ReLU(),
            nn.Linear(ac_hidden_dim_size, ac_hidden_dim_size),
            nn.ReLU(),
            nn.Linear(ac_hidden_dim_size, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, ac_hidden_dim_size),
            nn.ReLU(),
            nn.Linear(ac_hidden_dim_size, ac_hidden_dim_size),
            nn.ReLU(),
            nn.Linear(ac_hidden_dim_size, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs):
        x = obs.image
        x = x.reshape(x.shape[0], -1)

        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class ACModelShared(nn.Module, torch_ac.ACModel):

    def __init__(self, obs_space, action_space, ac_hidden_dim_size=64,
                 use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.number_of_actions = action_space.n

        obs_dim = obs_space["image"][0]
        self.image_embedding_size = obs_dim

        self.feature_net = nn.Sequential(nn.Linear(obs_dim,
                                                   ac_hidden_dim_size),
                                         nn.ReLU(),
                                         nn.Linear(ac_hidden_dim_size,
                                                   ac_hidden_dim_size),
                                         nn.ReLU(),
                                         nn.Linear(ac_hidden_dim_size,
                                                   ac_hidden_dim_size),
                                         nn.ReLU())

        self.actor = nn.Linear(ac_hidden_dim_size, action_space.n)

        self.critic = nn.Linear(ac_hidden_dim_size, 1)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs):
        x = obs.image
        x = x.reshape(x.shape[0], -1)

        embedding = self.feature_net(x)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class RandomFeatureNetwork(nn.Module):

    def __init__(self, obs_space, output_embedding_size,
                 rnd_hidden_dim_size=64, extra_layer=False):

        super().__init__()

        # Define image embedding
        self.recurrent = False
        self.output_embedding_size = output_embedding_size
        self.extra_layer = extra_layer

        obs_dim = obs_space["image"][0]

        if self.extra_layer:
            self.rnd_net = nn.Sequential(
                nn.Linear(obs_dim, rnd_hidden_dim_size),
                nn.ReLU(),
                nn.Linear(rnd_hidden_dim_size, rnd_hidden_dim_size),
                nn.ReLU(),
                nn.Linear(rnd_hidden_dim_size, self.output_embedding_size),
                )
        else:
            self.rnd_net = nn.Sequential(
                nn.Linear(obs_dim, rnd_hidden_dim_size),
                nn.ReLU(),
                nn.Linear(rnd_hidden_dim_size, self.output_embedding_size),
                )

    def forward(self, obs):

        x = obs.image
        x = x.reshape(x.shape[0], -1)

        x = self.rnd_net(x)

        return x


class EnsembledFeatureNetwork(nn.Module):

    def __init__(self, obs_space, output_embedding_size,
                 rnd_hidden_dim_size=64, n_ensemble=5,
                 extra_layer=False):

        super().__init__()

        # Define image embedding
        self.recurrent = False
        self.num_heads = n_ensemble
        self.output_embedding_size = output_embedding_size

        self.net_list = nn.ModuleList([RandomFeatureNetwork(obs_space, output_embedding_size, rnd_hidden_dim_size, extra_layer=extra_layer) for k in range(n_ensemble)])

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, obs):
        x = self._heads(obs)
        return x
