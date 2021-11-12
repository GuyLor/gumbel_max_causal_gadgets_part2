import torch
from torch import nn

class JointPredictorGivenLogits(nn.Module):
    def __init__(self, k, hidden_dim):
        super().__init__()

        self.pq = nn.Sequential(nn.Linear(k, hidden_dim),
                                nn.Sigmoid(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.Sigmoid(),
                                nn.Linear(hidden_dim, k*k))

        self.qp = nn.Sequential(nn.Linear(k, hidden_dim),
                                nn.Sigmoid(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.Sigmoid(),
                                nn.Linear(hidden_dim, k*k))

    def forward(self, logits, is_pq=True):
        bs, k = logits.size(0), logits.size(1)

        x = torch.softmax(logits, -1)
        joint_logits = self.pq(x) if is_pq else self.qp(x)
        joint_logits = joint_logits.view(bs, k, k)

        c = logits - torch.logsumexp(joint_logits, dim=-1)

        joint_logits = joint_logits + c.view(bs, -1, 1)
        return joint_logits



class JointPredictor(nn.Module):
    def __init__(self, k, hidden_dim):
        super().__init__()

        self.embed = nn.Embedding(k, hidden_dim)

        self.pq = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                nn.ReLU(),
                                nn.Linear(2*hidden_dim, k*k))

        self.qp = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                nn.ReLU(),
                                nn.Linear(2*hidden_dim, k*k))

    def forward(self, s, logits, is_pq=True):
        bs, k = logits.size(0), logits.size(1)
        s = self.embed(s)
        joint_logits = self.pq(s) if is_pq else self.qp(s)
        joint_logits = joint_logits.view(bs, k, k)

        c = logits - torch.logsumexp(joint_logits, dim=-1)

        joint_logits = joint_logits + c.view(bs, -1, 1)  # joint_logits + c.view(bs, -1, 1)

        # Enforce coupling constraint so that joint rows sum to desired marginal.
        #
        #       exp(logits_p[i]) = sum_j exp(joint_logits[i, j] + c_i)
        #                      = exp(c_i) * sum_j exp(joint_logits[i, j])
        #   =>  logits_p[i] = c_i + log(sum_j exp(joint_logits[i, j]))
        #             c_i = logits_p[i] - log(sum_j exp(joint_logits[i, j]))
        # for all i.

        return joint_logits


class FreeParamsPredictor(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.pq_logits = nn.Parameter(.001 * torch.randn(1, k, k))
        self.qp_logits = nn.Parameter(.001 * torch.randn(1, k, k))

    def forward(self, s, logits, is_pq):
        bs = logits.size(0)
        if is_pq:
            joint_logits = self.pq_logits
        else:
            joint_logits = self.qp_logits
        c = logits - torch.logsumexp(joint_logits, dim=-1)

        result = joint_logits.repeat([bs, 1, 1]) + c.view(bs, -1, 1)
        return result

class Gadget2Model(nn.Module):
    def __init__(self, s_dim, z_dim, hidden_dim):
        super().__init__()
        self.theta = nn.Sequential(nn.Linear(s_dim, hidden_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim, s_dim * z_dim))

    def forward(self, x):
        return self.theta(x)


