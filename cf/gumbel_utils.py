'''
Tools for sampling efficiently from a Gumbel posterior

Original code taken from https://cmaddis.github.io/gumbel-machinery, and then
modified to work as numpy arrays, and to fit our nomenclature, e.g.
* np.log(alpha) is replaced by log probabilities (which we refer to as logits)
* np.log(sum(alphas)) is removed, because it should always equal zero
'''
import numpy as np
import torch


def sample_gumbel(u, eps=1e-10):
    return -torch.log(eps - torch.log(u + eps))


def sample_truncated_gumbel(mu, b, eps=1e-10):
    """Sample a Gumbel(mu) truncated to be less than b."""
    u = torch.rand_like(mu)
    return -torch.log(eps - torch.log(u + eps) + torch.exp(-b + mu)) + mu


def batched_topdown(logits, k, topgumbel=None):
    """topdown

    Top-down sampling from the Gumbel posterior

    :param logits: log probabilities of each outcome
    :param k: Index of observed maximum
    :param nsamp: Number of samples from gumbel posterior
    """

    bs, ncat = logits.shape[0], logits.shape[1]
    device = logits.device
    k = torch.tensor(k).repeat([bs, 1]) if np.isscalar(k) else k
    k = k.to(device).view(-1, 1)

    # Sample top gumbels
    if topgumbel is None:
        loc = torch.logsumexp(logits, -1)
        topgumbel = sample_gumbel(torch.rand_like(loc)) + loc  # m.sample()

    gumbels = sample_truncated_gumbel(logits.reshape(-1), topgumbel[:, None].expand(-1, ncat).reshape(-1)).view(bs, ncat)
    return gumbels.scatter(dim=-1, index=k, src=topgumbel.unsqueeze(-1)) - logits
