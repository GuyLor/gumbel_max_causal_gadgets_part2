import torch
import numpy as np
from cf import utils
from cf.gadget_1 import Gadget1
from cf.gadget_2_jax import Gadget2Jax
from cf.gadget_2_pytorch import GadgetTwoMLPPredictor as Gadget2Pytorch


class Coupler:
    def __init__(self, s_dim, z_dim=20, hidden_features=[1024, 1024], tmp=1.0, seed=59, gadget2_jax=False):

        self.gadget_1 = Gadget1(s_dim=s_dim, hidden_dim=hidden_features[0], tmp=tmp, seed=seed)
        self.gadget_2 = Gadget2Pytorch(s_dim=s_dim, z_dim=z_dim, hidden_dim=hidden_features[0],
                                       relaxation_temperature=tmp, seed=seed) if not gadget2_jax else\
            Gadget2Jax(s_dim=s_dim, z_dim=z_dim, hidden_features=hidden_features, tmp=tmp)


        self.gadget2_jax = gadget2_jax
        self.s_dim = s_dim  # size of states space
        self.seed = seed

    def train_gadget_1(self,p, q, s_prime_obs, reward_vector, batch_size=64, counterfactual=True, num_iter=1000,
                       noise_scale=0, verbose=True):

        every100, every400 = (100, 400)

        logits_p, logits_q, s_prime_obs = utils.convert_p_q_and_sp_sq_to_pytorch(p, q, s_prime_obs,
                                                                                 n_draws=batch_size, device=self.gadget_1.device)

        r_ij = torch.square(torch.from_numpy(reward_vector[None, :] - reward_vector[:, None])).to(logits_p.device).float()

        losses = []
        for i in range(0, num_iter+1):
            if i % every400 == 0:
                self.gadget_1.estimate(logits_p, logits_q, s_prime_obs, noise_scale=noise_scale, counterfactual=counterfactual,
                                       reward_vector=reward_vector, n=200)
            for inner in range(16):
                _logits_p, _logits_q = self.gadget_1.add_noise(logits_p, logits_q, noise_scale) if noise_scale > 0 else (logits_p, logits_q)
                (_, _), loss = self.gadget_1.sample_from_joint(_logits_p, _logits_q, s_prime_obs=s_prime_obs, counterfactual=counterfactual,
                                                               train=True, rewards=r_ij)
                losses.append(loss)

            if verbose and (i % 200 == 0 or np.remainder(np.log2(i), 1) == 0):
                print('{}: {}'.format(i, {'loss': torch.stack(losses).mean().item()}))
                losses = []
        self.gadget_1.model.load_state_dict(self.gadget_1.best_model_state_dict)

    def train_gadget_2_pytorch(self, p, q, s_prime_obs, reward_vector, batch_size=64, counterfactual=True, num_iter=1000,
                               noise_scale=0, verbose=True):

        logits_p, logits_q, s_prime_obs = utils.convert_p_q_and_sp_sq_to_pytorch(p, q, s_prime_obs,
                                                                                 n_draws=batch_size, device=self.gadget_1.device)

        r_ij = torch.square(torch.from_numpy(reward_vector[None, :] - reward_vector[:, None])).to(logits_p.device).float()

        losses = []
        for i in range(0, num_iter+1):
            for inner in range(16):
                _logits_p, _logits_q = self.gadget_2.add_noise(logits_p, logits_q, noise_scale) if noise_scale > 0 else (logits_p, logits_q)
                (_, _), loss = self.gadget_2.sample_from_joint(_logits_p, _logits_q, s_prime_obs=s_prime_obs, counterfactual=counterfactual,
                                                               train=True, rewards=r_ij)
                losses.append(loss)

            if verbose and (i % 200 == 0 or np.remainder(np.log2(i), 1) == 0):
                print('{}: {}'.format(i, {'loss': torch.stack(losses).mean().item()}))
                losses = []

    def train_gadget_2_jax(self, p, q, s_prime_obs, reward_vector,
                           counterfactual, batch_size, num_iter, noise_scale):
        self.gadget_2.train(p, q, s_prime_obs=s_prime_obs, policy_reward_vector=reward_vector,
                            cf=counterfactual, batch_size=batch_size, num_iter=num_iter, noise_scale=noise_scale, seed=self.seed)

    def train_gadget_2(self, *args, **kwargs):
        if self.gadget2_jax:
            self.train_gadget_2_jax(*args, **kwargs)
        else:
            self.train_gadget_2_pytorch(*args, **kwargs)


