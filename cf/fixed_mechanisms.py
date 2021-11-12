import torch
import cf.gumbel_utils as gt


def gumbel_max_coupling(logits_p, logits_q, s_prime_obs=None, counterfactual=True, device='cpu'):
    if not isinstance(logits_p, torch.Tensor):
        logits_p, logits_q = torch.from_numpy(logits_p).to(device).float(),\
                                          torch.from_numpy(logits_q).to(device).float()
        s_prime_obs = torch.from_numpy(s_prime_obs).to(device) if s_prime_obs is not None else None
    if counterfactual:
        assert s_prime_obs is not None, 'in the topdown setup s_prime_obs must be given'
        gumbels = gt.batched_topdown(logits_p, s_prime_obs)
    else:
        u = torch.rand_like(logits_p)
        gumbels = gt.sample_gumbel(u)
    s_prime_p = torch.argmax(gumbels + logits_p, -1)
    s_prime_q = torch.argmax(gumbels + logits_q, -1)

    if counterfactual:
        eq = (s_prime_p == s_prime_obs.view(-1))
        assert eq.sum() == s_prime_p.size(0), 'Error: s_prime_p != s_prime_obs {}'.format(eq)
    return s_prime_p, s_prime_q


def inverse_cdf_coupling(logits_p, logits_q, s_prime_obs=None, counterfactual=True, device='cpu'):
    if not isinstance(logits_p, torch.Tensor):
        logits_p, logits_q = torch.from_numpy(logits_p).to(device).float(),\
                                          torch.from_numpy(logits_q).to(device).float()
        s_prime_obs = torch.from_numpy(s_prime_obs).to(device) if s_prime_obs is not None else None
    def sample_inverse_cdf(logits, u):
        p = torch.exp(logits - torch.logsumexp(logits, -1, keepdim=True))
        pr_cs = p.cumsum(-1)
        ans = pr_cs > u #.view(u.size(0), 1)
        ans = (ans[:, 1:] ^ ans[:, :-1]).long()
        return torch.where(ans.sum(-1) == 1, torch.argmax(ans, -1) + 1, 0)

    def rejection_sampling(logits):
        bs = logits.size(0)
        all_accepted = False
        current_u = torch.rand(bs, device=logits.device).unsqueeze(-1)
        to_stop = 0
        while not all_accepted:
            s_prime = sample_inverse_cdf(logits, current_u)
            comp = s_prime == s_prime_obs.view(-1)
            all_accepted = comp.sum() == bs
            current_u = torch.where(comp.unsqueeze(-1),
                                    current_u,
                                    torch.rand(bs, device=logits_p.device).unsqueeze(-1))

            if to_stop > bs*bs:
                print('icdf cf has failed')
                break
            to_stop += 1
        return current_u

    if counterfactual:
        assert s_prime_obs is not None, 'in the topdown setup s_prime_obs must be given'
        u = rejection_sampling(logits_p)
    else:
        u = torch.rand(logits_p.size(0), device=logits_p.device).unsqueeze(-1)

    s_prime_p = sample_inverse_cdf(logits_p, u)
    s_prime_q = sample_inverse_cdf(logits_q, u)
    return s_prime_p, s_prime_q
