import torch
import cf.gumbel_utils as gt
from cf import models
from cf import utils
from cf import fixed_mechanisms as fm


class Gadget1:
    def __init__(self, s_dim, hidden_dim=1024, model_type='given_logits', tmp=1.0, seed=0):
        #self.rng = torch.manual_seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(
            "cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

        self.model = models.JointPredictorGivenLogits(s_dim, hidden_dim=hidden_dim) if model_type == 'given_logits' \
            else models.FreeParamsPredictor(s_dim)

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                   gamma=0.8, verbose=False)
        self.s_dim = s_dim  # k is the size of the states space
        self.tau = tmp
        self.min_var = float('inf')
        self.best_model_state_dict = self.model.state_dict()
        #self.rng = torch.random.get_rng_state()

    def add_noise(self, p, q, std):
        return (torch.clamp(p + std*torch.randn_like(p), max=0.0), torch.clamp(q + std*torch.randn_like(q), max=0.0))
        #return (p + std * torch.randn_like(p), q + std * torch.randn_like(q))

    def get_joints(self, logits_p, logits_q):
        joint_pq = self.model(logits_p, is_pq=True)
        joint_qp = self.model(logits_q, is_pq=False)
        return joint_pq, joint_qp

    def sample_from_joint(self, logits_p, logits_q, s_prime_obs=None, counterfactual=True,
                          train=True, rewards=None):
        # logits_p: prev_interv_probs (state tx under behavior policy):  log p(s' | sp, ap)
        # logits_q: new_interv_probs (state tx under target policy)      log q(s' | sq, aq)
        # sp: observed state
        # sq: previous counterfactual state
        # ap: behavior action
        # aq: cf action
        if not isinstance(logits_p, torch.Tensor):
            logits_p, logits_q = torch.from_numpy(logits_p).to(self.device).float(),\
                                              torch.from_numpy(logits_q).to(self.device).float()
            s_prime_obs = torch.from_numpy(s_prime_obs).to(self.device) if s_prime_obs is not None else None

        with torch.set_grad_enabled(train):
            joint_pq, joint_qp = self.get_joints(logits_p, logits_q)
            if counterfactual:
                gumbel_max_i = (gt.batched_topdown(logits_p, s_prime_obs) + logits_p).view(-1)  # shape: [n_draws * num_states, 1]
                sm_i = torch.softmax(joint_pq.view(-1, self.s_dim), -1)
                argmax_i = torch.multinomial(sm_i, 1).view(-1)
                # argmax_i = torch.argmax(
                #     joint_pq.view(-1, self.s_dim) + gt.sample_gumbel(torch.rand_like(joint_pq.view(-1, self.s_dim))),
                #     dim=-1)
                gumbels = gt.batched_topdown(joint_pq.view(-1, self.s_dim), argmax_i, topgumbel=gumbel_max_i).view(-1, self.s_dim, self.s_dim)
            else:
                u = torch.rand_like(joint_pq)
                gumbels = gt.sample_gumbel(u)

            logprobs_p = utils.marginal_gumbel_log_softmax(joint_pq, gumbels, rows=True, tau=self.tau, hard=False)
            logprobs_q = utils.marginal_gumbel_log_softmax(joint_qp, gumbels.transpose(-2, -1), rows=True, tau=self.tau, hard=False)

            s_prime_p = logprobs_p.argmax(-1)
            s_prime_q = logprobs_q.argmax(-1)
            obj = None
            if train:
                obj = (torch.exp(logprobs_p[:, :, None] + logprobs_q[:, None, :]) * rewards).detach()
                loss = torch.sum((logprobs_p[:, :, None] + logprobs_q[:, None, :]) * obj)/obj.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                obj = torch.sum(obj, dim=(1, 2))

        return (s_prime_p, s_prime_q), obj

    def estimate(self, prev_interv_probs, new_interv_probs, s_prime_obs=None, counterfactual=True,
                 noise_scale=1.0, n=100, reward_vector=None):
        ates_gm, ates_icdf, ates_gd1 = [],[],[]
        reward_vector = torch.from_numpy(reward_vector).to(self.device)
        s_prime_obs = None if not counterfactual else s_prime_obs
        with torch.no_grad():
            for _ in range(n):
                p, q = self.add_noise(prev_interv_probs, new_interv_probs, noise_scale) if noise_scale > 0 else (prev_interv_probs, new_interv_probs)

                (s_prime_p, s_prime_q), _ = self.sample_from_joint(p, q, s_prime_obs, counterfactual=counterfactual, train=False)

                var_ate_gd1 = utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, s_prime_obs)

                s_prime_p, s_prime_q = fm.gumbel_max_coupling(p, q, s_prime_obs, counterfactual=counterfactual)

                var_ate_gm = utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, s_prime_obs)

                s_prime_p, s_prime_q = fm.inverse_cdf_coupling(p, q, s_prime_obs, counterfactual=counterfactual)

                var_ate_icdf = utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, s_prime_obs)

                ates_gd1.append(var_ate_gd1)
                ates_gm.append(var_ate_gm)
                ates_icdf.append(var_ate_icdf)

            var_ates_gm, var_ates_icdf, var_ates_jp = torch.stack(ates_gm).mean(), torch.stack(ates_icdf).mean(), torch.stack(ates_gd1).mean()
            #print(f"Var-treatment-effect: \n Gumbel-max: {var_ates_gm.item()}, iCDF: {var_ates_icdf.item()}, Gadget1: {var_ates_jp.item()}")
            if var_ates_jp < self.min_var:
                self.min_var = var_ates_jp
                # print("New best variance ATE: ", var_ates_jp)
                self.best_model_state_dict = self.model.state_dict()
