import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Gating_Net(nn.Module):

    def __init__(self, embedding_dim, mlp_dims):
        super(Gating_Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.softmax =  nn.LogSoftmax(dim=1)
        fc_layers = []
        for i in range(len(mlp_dims)):
            if i == 0:
                fc_layers.append(nn.Linear(embedding_dim*2, mlp_dims[i]))
            else:
                fc_layers.append(nn.Linear(mlp_dims[i-1], mlp_dims[i]))	
            if i != len(mlp_dims) - 1:
                fc_layers.append(nn.BatchNorm1d(mlp_dims[i]))
                fc_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*fc_layers)

    def gumbel_softmax(self, logits, temperature, division_noise, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature, division_noise) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature, division_noise):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        noise = self.sample_gumbel(logits)
        y = (logits + (noise/division_noise)) / temperature
        return F.softmax(y)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise.float()).cuda()

    def forward(self, feature, temperature, hard, division_noise): #z= batch x z_dim // #feature =  batch x num_gen x 256*8*8
        x = self.mlp(feature)
        out = self.gumbel_softmax(x, temperature, division_noise, hard)
        out_value = out.unsqueeze(2)
        out = out_value.repeat(1, 1, self.embedding_dim)
                
        return out, torch.sum(out_value[:,0]), torch.sum(out_value[:,1])