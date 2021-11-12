import torch
from torch import nn
import cf.gumbel_utils as gt
from cf import models

def fix_coupling_sinkhorn(log_coupling, log_p1, log_p2, iterations=10):
    """Adjust a coupling to approximately match marginals using Sinkhorn iteration.
    Args:
    log_coupling: Log of a coupling, possibly unnormalized, of shape [M,M]
    log_p1: Log probabilities for first dimension logits, of shape [M].
    log_p2: Log probabilities for second dimension logits, of shape [M].
    iterations: How many Sinkhorn iterations to use.
    Returns:
    Matrix of shape [M,M]. Will always exactly match the marginals of `log_p2`
    along the second axis, and will attempt to match marginals of `log_p1` along
    the first axis as well, but may not reach it depending on iteration count.
    """
    for _ in range(iterations):
        log_coupling = log_coupling + log_p1[:, :, None] - torch.logsumexp(log_coupling, dim=2, keepdim=True)
        log_coupling = log_coupling + log_p2[:, None, :] - torch.logsumexp(log_coupling, dim=1, keepdim=True)
    return log_coupling


def fix_coupling_rejection(log_coupling, log_p1, log_p2):
    """Apply final correction step to ensure a coupling satisfies both marginals.
    Args:
    log_coupling: Log of a coupling, possibly unnormalized, of shape [M,M].
      Assumed to approximately be a coupling of log_p1 and log_p2.
    log_p1: Log probabilities for first dimension logits, of shape [M].
    log_p2: Log probabilities for second dimension logits, of shape [M].
    Returns:
    A matrix of shape [M,M] representing a valid coupling of log_p1 and log_p2.
    If the original coupling was already close to a valid coupling, it will
    be only changed slightly.
    """
    # Normalize so that it matches p1. Then consider mixing with a p-independent
    # distribution so that it also matches p2.
    log_coupling_fixed_p1 = (
      log_coupling + log_p1[:, :, None] -
      torch.logsumexp(log_coupling, dim=2, keepdim=True))
    approx_log_p2 = torch.logsumexp(log_coupling_fixed_p1, dim=1)
    # How much more often do we sample each value of s2 than we should?
    # accept rate = C p2(x)/p2tilde(x)
    accept_rate = log_p2 - approx_log_p2
    m, _ = torch.max(accept_rate, dim=-1, keepdim=True)
    accept_rate = accept_rate - m
    # Downweight rejections.
    log_coupling_accept_s2_given_s1 = torch.nn.functional.log_softmax(
      log_coupling_fixed_p1, dim=-1) + accept_rate[:, None, :]
    # Compensate by then drawing from p2 exactly if we failed.
    log_prob_keep = torch.logsumexp(
      log_coupling_accept_s2_given_s1, dim=-1)
    # print(accept_rate, log_prob_keep)
    certainly_keep = torch.exp(log_prob_keep) >= 1.0
    resample_log_p1rob = torch.where(certainly_keep, - torch.tensor(float("Inf"), device=log_p2.device),
                                     torch.log1p(-torch.where(certainly_keep, torch.tensor(0.0, device=log_p2.device), torch.exp(log_prob_keep))))
    compensation = resample_log_p1rob[:, :, None] + log_p2[:, None, :]
    return log_p1[: ,: , None] + torch.logaddexp(log_coupling_accept_s2_given_s1, compensation)

class GadgetTwoMLPPredictor:
    """Gadget 2 coupling layer, supporting counterfactual inference.
    Attributes:
    S_dim: Number of possible outcomes.
    Z_dim: Size of the latent space. May need to be much larger than S_dim to
      obtain good results.
    hidden_features: List of dimensions for hidden layers.
    sinkhorn_iterations: Number of Sinkhorn iterations to use before final
      correction step.
    learn_prior: Whether to create parameters for the prior. Note that we only
      compute derivatives with respect to these when using
      `counterfactual_sample_relaxed`, which is not supported by gadget 1. We
      set this to False for all of our experiments.
    relaxation_temperature: Default temperature used when training.
    """
    def __init__(self, s_dim, z_dim, hidden_dim=1024, sinkhorn_iterations=10,
                 learn_prior=False, relaxation_temperature=1.0, seed=0):
        super(GadgetTwoMLPPredictor, self).__init__()

        #self.rand_gen = torch.manual_seed(seed)
        rng = torch.random.get_rng_state()
        self.device = torch.device(
            "cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
        self.S_dim = s_dim  # pylint: disable=invalid-name
        self.Z_dim = z_dim  # pylint: disable=invalid-name
        self.hidden_features = hidden_dim  #List[int]
        self.sinkhorn_iterations = sinkhorn_iterations
        self.learn_prior = learn_prior
        self.relaxation_temperature = relaxation_temperature

        self.theta = models.Gadget2Model(s_dim, z_dim, hidden_dim)

        if self.learn_prior:
            self.prior = nn.Parameter(torch.randn(z_dim, device=self.device))

        self.theta = self.theta.to(self.device)
        self.optimizer = torch.optim.Adam(self.theta.parameters(), lr=0.00001)
        self.batch_size = 1
        #self.rng = torch.random.get_rng_state()
        torch.random.set_rng_state(rng)

    def get_prior(self):
        """Returns the learned or static prior distribution."""
        if self.learn_prior:
          return nn.functional.log_softmax(self.prior, dim=-1).repeat(self.batch_size, 1)
        else:
          return nn.functional.log_softmax(torch.zeros([self.batch_size, self.Z_dim], device=self.device), dim=-1)

    def get_joint(self, s_logits):
        """Returns a joint that agrees with s_logits when axis 0 is marginalized out."""
        z_logits = self.get_prior()
        value = torch.softmax(s_logits, dim=-1)

        log_joint = self.theta(value).view(-1, self.Z_dim, self.S_dim)
        # Sinkhorn has nicer gradients.
        log_joint = fix_coupling_sinkhorn(log_joint, z_logits, s_logits,
                                          self.sinkhorn_iterations)
        # ... but rejection is harder to exploit inaccuracy for
        log_joint = fix_coupling_rejection(log_joint, z_logits, s_logits)
        return log_joint

    def get_forward(self, s_logits):
        """Returns a matrix corresponding to `log pi(x | z, s_logits)`."""
        return nn.functional.log_softmax(self.get_joint(s_logits), dim=-1)

    def sample(self, s_logits, u_z, u_s):
        """Draws a single sample from `s_logits`.
        Args:
          s_logits: Logits to sample from.
          u_z: uniform. Sharing this across multiple calls produces an implicit
            coupling.
          u_s: uniform. Sharing this across multiple calls produces an implicit
            coupling.
        Returns:
          Sampled integer index from s_logits.
        """
        log_z = self.get_prior()
        log_s_given_z = self.get_forward(s_logits)
        z = torch.argmax(log_z + gt.sample_gumbel(u_z))
        s = torch.argmax(log_s_given_z[:, z] + gt.sample_gumbel(u_s))
        return s

    def sample_relaxed(self, s_logits,  u_z, u_s, temperature=None):
        """Sample a relaxed Gumbel-softmax output.
        TODO: update doc
        Args:
          s_logits: Logits to sample from.
          rng: PRNGKey. Sharing this across multiple calls produces an implicit
            coupling.
          temperature: Relaxation temperature to use. Defaults to the value from the
            class.
        Returns:
          Vector float32[S_dim] of relaxed outputs that sum to 1. The argmax of
          this vector will always be the same as the result of the equivalent call
          to `sample`.
        """
        if temperature is None:
            assert self.relaxation_temperature
            temperature = self.relaxation_temperature
        log_z = self.get_prior()
        log_s_given_z = self.get_forward(s_logits)
        z = torch.argmax(log_z + gt.sample_gumbel(u_z), dim=-1)
        log_s_given_z_ = torch.gather(log_s_given_z, 1, z.unsqueeze(-1).repeat(1, log_s_given_z.size(-1))[:, None, :]).squeeze(1)
        s = torch.softmax(log_s_given_z_ + gt.sample_gumbel(u_s)/ temperature, dim=-1)
        # Gumbel-softmax instead of argmax here
        return s

    def counterfactual_sample(self, p_logits, q_logits, p_observed,  u_z):
        """Sample a single sample from q conditioned on observing p_observed.
        Args:
          p_logits: Logits describing the original distribution of p_observed.
          q_logits: Logits describing a new counterfactual intervention.
          p_observed: Sample index we observed.
          rng: PRNGKey. Sharing this across multiple calls produces an implicit
            coupling.
        Returns:
          Sampled integer index from q_logits, conditioned on observing p_observed
          under p_logits.
        """
        log_z = self.get_prior()
        log_ps_given_z = self.get_forward(p_logits)
        log_qs_given_z = self.get_forward(q_logits)
        # Infer z from p_observed
        log_z_given_ps = (log_z[:, None] + log_ps_given_z)[:, :, p_observed]

        z = torch.argmax(log_z_given_ps + gt.sample_gumbel(u_z))
        # Infer Gumbels from p_observed and z
        gumbels = gt.batched_topdown(log_ps_given_z[:,z], p_observed)
        # Choose accordingly
        qs = torch.argmax(gumbels + log_qs_given_z[:,z])
        return qs

    def counterfactual_sample_relaxed(self, p_logits, q_logits, p_observed, u_z, temperature=1.0):
        """Sample a relaxed sample from q conditioned on observing p_observed.
        This essentially continuously relaxes the counterfactual outcome without
        relaxing the observed outcome. This should enable learning the prior over z
        in addition to the transformation. We did not use this in our experiments.
        Args:
          p_logits: Logits describing the original distribution of p_observed.
          q_logits: Logits describing a new counterfactual intervention.
          p_observed: Sample index we observed.
          rng: PRNGKey. Sharing this across multiple calls produces an implicit
            coupling.
          temperature: Relaxation temperature to use. Defaults to the value from the
            class.
        Returns:
          Vector float32[S_dim] of relaxed outputs that sum to 1.
        """
        log_z = self.get_prior()
        log_ps_given_z = self.get_forward(p_logits)
        log_qs_given_z = self.get_forward(q_logits)
        # Infer distn of z from p_observed.
        log_z_given_ps = (log_z[:, :, None] + log_ps_given_z)[:, :, p_observed[0][0]]
        z_given_ps = torch.softmax(log_z_given_ps, dim=-1)

        distns = torch.zeros_like(log_ps_given_z)
        for z in torch.arange(self.Z_dim, device=self.device):
            g = gt.batched_topdown(log_ps_given_z[:, z], p_observed)
            distns[:, z, :] = torch.softmax((g + log_qs_given_z[:, z]) / temperature, dim=-1)
        # For each z, counterfactually sample some Gumbels, then apply softmax.
        # def relaxed_for_fixed_z(z):
        #     # (again, uses common random numbers here)
        #     g = coupling_util.counterfactual_gumbels(log_ps_given_z[z], p_observed, rng)
        #       # Gumbel-softmax instead of argmax here
        #     soft_q = jax.nn.softmax((g + log_qs_given_z[z]) / temperature)
        #     return soft_q

        #distns = jax.vmap(relaxed_for_fixed_z)(jnp.arange(self.Z_dim))
        avg_distn = torch.sum(z_given_ps[:, :, None] * distns, dim=1)
        return avg_distn

    def add_noise(self, p, q, std):
        return (torch.clamp(p + std*torch.randn_like(p), max=0.0), torch.clamp(q + std*torch.randn_like(q), max=0.0))

    def sample_from_joint(self, logits_p, logits_q, s_prime_obs=None, counterfactual=True,
                          train=True, rewards=None):
        # logits_p: prev_interv_probs (state tx under behavior policy):  log p(s' | sp, ap)
        # logits_q: new_interv_probs (state tx under target policy)      log q(s' | sq, aq)
        # sp: observed state
        # sq: previous counterfactual state
        # ap: behavior action
        # aq: cf action
        rng = torch.random.get_rng_state()
        if not isinstance(logits_p, torch.Tensor):
            logits_p, logits_q = torch.from_numpy(logits_p).to(self.device).float(),\
                                              torch.from_numpy(logits_q).to(self.device).float()
            s_prime_obs = torch.from_numpy(s_prime_obs).to(self.device) if s_prime_obs is not None else None

        self.batch_size = logits_p.size(0)
        with torch.set_grad_enabled(train):
            if counterfactual:
                u_z = torch.rand((logits_p.size(0), self.Z_dim), device=self.device)
                soft_q = self.counterfactual_sample_relaxed(logits_p, logits_q, s_prime_obs, u_z)
                soft_p = nn.functional.one_hot(s_prime_obs.squeeze(), num_classes=soft_q.size(-1))
                coupling_loss = torch.sum(soft_q * rewards[s_prime_obs[0][0], :])/self.batch_size if train else None


            else: # crn
                u_z = torch.rand((logits_p.size(0), self.Z_dim), device=self.device)
                u_s = torch.rand((logits_p.size(0), self.S_dim), device=self.device)
                soft_p = self.sample_relaxed(logits_p, u_z, u_s, temperature=None)
                soft_q = self.sample_relaxed(logits_q, u_z, u_s, temperature=None)
                coupling_loss = torch.sum(soft_p[:, :, None] * soft_q[:, None, :] * rewards)/self.batch_size if train else None

            s_prime_p = soft_p.argmax(-1)
            s_prime_q = soft_q.argmax(-1)
            if train:
                self.optimizer.zero_grad()
                coupling_loss.backward()
                self.optimizer.step()
        torch.random.set_rng_state(rng) if not train else None  # keep repreducability without gadget 2 (gadget 2 was implemented in pytorch after canera-ready)
        return (s_prime_p, s_prime_q), coupling_loss


