import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def marginal_gumbel_log_softmax(joint_logits, gumbel_noise, rows=True, tau=1.0, hard=False, eps=1e-10):
    assert joint_logits.shape == gumbel_noise.shape, 'joint_logits and gumbel_noise must have the same shape'
    y = joint_logits + gumbel_noise
    marginal_logits, _ = torch.max(y, -1) if rows else torch.max(y, -2)
    marginal_logits = marginal_logits / tau
    if hard:
        raise NotImplementedError('TODO: implement if needed')
        k = torch.argmax(marginal_logits, dim=-1)
        return F.one_hot(k, num_classes=joint_logits.size(-1)).log()

    else:
        return marginal_logits - torch.logsumexp(marginal_logits, dim=-1, keepdims=True)


def logsumexp(x):
    c = x.max(axis=-1, keepdims=True)
    return c + np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))


def convert_p_q_and_sp_sq_to_pytorch(p, q, s_prime_obs=None, n_draws=32, device='cpu'):
    logits_p = torch.clamp(torch.tensor(p).log(), min=-80.0).to(device)
    logits_q = torch.clamp(torch.tensor(q).log(), min=-80.0).to(device)

    s_prime_obs_n = torch.tensor(s_prime_obs).long().to(device)

    if n_draws > 1:
        logits_p = logits_p.unsqueeze(0).repeat([n_draws, 1])
        logits_q = logits_q.unsqueeze(0).repeat([n_draws, 1])
        s_prime_obs_n = s_prime_obs_n.unsqueeze(0).repeat([n_draws, 1])

    return logits_p, logits_q, s_prime_obs_n


def compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, s_prime_obs=None):
    """ Compute the variance over batch of the treatment effects """
    if s_prime_obs is not None:
        # In the CF case, the sampled next state under the physician policy is observed by definition.
        if not isinstance(s_prime_obs, torch.Tensor):
            s_prime_obs = torch.from_numpy(s_prime_obs).to(s_prime_p.device)
        assert (s_prime_p == s_prime_obs.reshape(-1)).sum() == s_prime_p.shape[0],\
            'Warning:  despite the top down s_prime_p != s_prime_obs. num of diff {}'.format(s_prime_p.shape[0] - (s_prime_p == s_prime_obs.reshape(-1)).sum())

    if type(reward_vector) != type(s_prime_p):
        s_prime_p = s_prime_p.cpu().numpy()
        s_prime_q = s_prime_q.cpu().numpy()
        #    'type of s_prime_p {} must be the same as reward_vector {}'.format(s_prime_p,reward_vector)

    rp = reward_vector[s_prime_p.reshape(-1)]
    rq = reward_vector[s_prime_q.reshape(-1)]
    ate = rp - rq
    return ate.var()


def plot_mdp_variances(gmx, icdf, gd1, gd2, cf=True, generalized=True, figpath=''):
    def conv_to_np(this_list):
        this_arr = np.array(this_list)[:, np.newaxis]
        this_arr = this_arr.squeeze()[:, np.newaxis]
        return this_arr

    gmx = conv_to_np(gmx); icdf = conv_to_np(icdf); gd1 = conv_to_np(gd1); gd2 = conv_to_np(gd2)
    plt.rcParams.update({'font.size': 15})
    reward = np.concatenate([
        gmx,
        icdf,
        gd1,
        gd2
    ], axis=1)
    reward_df = pd.DataFrame(reward, columns=['Gumbel-Max',
                                              'Inverse-CDF',
                                              'Gadget-1',
                                              'Gadget-2',
                                              ])

    plt.figure(figsize=(7.7, 5))
    sns.barplot(data=reward_df, ci=68)
    type_coup = 'counterfactual' if cf else 'joint sampling'
    gen = 'generalized' if generalized else 'fixed'
    plt.title('{}, {} (p, q)'.format(type_coup, gen))
    plt.ylabel("Variance ATE")
    plt.savefig("{}-both_gadgets.pdf".format(figpath)) if figpath != '' else None
    plt.show()
