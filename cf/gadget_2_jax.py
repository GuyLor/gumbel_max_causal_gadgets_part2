
import numpy as np
import jax
import jax.numpy as jnp
import optax

import flax.linen as nn

import matplotlib.pyplot as plt

plt.ion()

np.set_printoptions(linewidth=150)


import sys
import time

from typing import *


def sample_multinomial(key, total_count, logits, n):
    # https://github.com/deepmind/distrax/blob/master/distrax/_src/distributions/multinomial.py#L150
    key, = jax.random.split(key, 1)  # for consistent rngs with distrax, not strictly necessary
    dtype = logits.dtype

    def cond_func(args):
        i, _, _ = args
        return jnp.less(i, total_count)

    def body_func(args):
        i, key_i, sample_aggregator = args
        key_i, current_key = jax.random.split(key_i)
        sample_i = jax.random.categorical(current_key, logits=logits, shape=(n,))
        one_hot_i = jax.nn.one_hot(sample_i, logits.shape[0]).astype(dtype)
        return i + 1, key_i, sample_aggregator + one_hot_i

    init_aggregator = jnp.zeros((n, logits.shape[0]), dtype=dtype)
    return jax.lax.while_loop(cond_func, body_func, (0, key, init_aggregator))[2]


def fix_coupling_sinkhorn(log_coupling, log_p1, log_q2, iterations=10):
    for _ in range(iterations):
        log_coupling = (log_coupling + log_p1[:, None]
                        - jax.scipy.special.logsumexp(log_coupling, axis=1, keepdims=True))
        log_coupling = (log_coupling + log_q2[None, :]
                        - jax.scipy.special.logsumexp(log_coupling, axis=0, keepdims=True))
    return log_coupling


def fix_coupling_rejection(log_coupling, log_p1, log_p2):
    # Normalize so that it matches p1. Then consider mixing with a p-independent
    # distribution so that it also matches p2.
    log_coupling_fixed_p1 = (log_coupling + log_p1[:, None]
                             - jax.scipy.special.logsumexp(log_coupling, axis=1, keepdims=True))
    approx_log_p2 = jax.scipy.special.logsumexp(log_coupling_fixed_p1, axis=0)
    # How much more often do we sample each value of s2 than we should?
    # accept rate = C p2(x)/p2tilde(x)
    accept_rate = log_p2 - approx_log_p2
    accept_rate = accept_rate - jnp.max(accept_rate)
    # Downweight rejections.
    log_coupling_accept_s2_given_s1 = jax.nn.log_softmax(log_coupling_fixed_p1, axis=-1) + accept_rate[None, :]
    # Compensate by then drawing from p2 exactly if we failed.
    log_prob_keep = jax.scipy.special.logsumexp(log_coupling_accept_s2_given_s1, axis=-1)
    # print(accept_rate, log_prob_keep)
    certainly_keep = jnp.exp(log_prob_keep) >= 1.0
    resample_log_p1rob = jnp.where(
        certainly_keep, -jnp.inf,
        jnp.log1p(-jnp.where(certainly_keep, 0.0, jnp.exp(log_prob_keep))))
    compensation = resample_log_p1rob[:, None] + log_p2[None, :]
    return log_p1[:, None] + jnp.logaddexp(log_coupling_accept_s2_given_s1, compensation)


"""# Unbiased relaxation of a Gumbel-Max coupling, without Gumbel-softmax

Suppose we want to compute unbiased gradients with respect to the expected reward variance $\mathbb{E}[(R(s_1) - R(s_2))^2]$ under our leraned coupling. This would be easy if we had a closed form for the coupling joint
$$
p_\theta(s_1, s_2 | \ell_1, \ell_2)
$$
because we could then either
- just compute the exact expectation by summing over all pairs
- use something like REINFORCE with this as the log prob term.

Unfortunately, $p_\theta$ isn't defined in terms of a joint; it's defined in terms of a perturbation-argmax process. So we don't just immediately obtain a (differentiable) estimate of the joint.

One thing you could try would be to Gumbel-softmax relax the argmax steps. This seems fine, but introduces a bit of bias, and I couldn't get it to work with my parameterization for some reason.

Another thing you could do, though, is to consider partially marginalizing out subsets of Gumbels. This would give a sparse but differentiable estimate of the joint distribution, such that averaged over the (non-marginalized) Gumbels we still obtain exactly the desired joint.

Let's consider a Gumbel-max coupling of two things:
"""


def joint_from_samples(sampler, logits_1, logits_2, rng, num_samples, loop_size=None):
    # JAX note: jax.vmap computes many samples in parallel
    logits_1 = jnp.array(logits_1)
    logits_2 = jnp.array(logits_2)
    if loop_size is None:
        return jnp.mean(jax.vmap(lambda key: sampler(logits_1, logits_2, key))(
            jax.random.split(rng, num_samples)), axis=0)
    else:
        assert num_samples % loop_size == 0

        def go(i, counts):
            return counts + jnp.sum(jax.vmap(lambda key: sampler(logits_1, logits_2, key))(
                jax.random.split(jax.random.fold_in(rng, i), loop_size)), axis=0)

        counts = jax.lax.fori_loop(0, num_samples // loop_size, go, jnp.zeros([10, 10]))
        return counts / num_samples


def gumbel_max_sampler(logits_1, logits_2, rng):
    gumbels = jax.random.gumbel(rng, logits_1.shape)
    x = jnp.argmax(gumbels + logits_1)
    y = jnp.argmax(gumbels + logits_2)
    return jnp.zeros([10, 10]).at[x, y].set(1.)


"""Now let's sample unbiased estimates of our joint by sampling $N - 1$ Gumbels, then marginalizing out the last one:"""


def unbiased_gumbel_max_simple(log_pA, log_pB, rng):
    """Unbiased estimator of Gumbel-max coupling probabilities.

    This can be used to differentiate through samples from a Gumbel-max coupling,
    by giving a closed form unbiased estimate for the probability of any given
    event happening.
    """
    dim, = log_pA.shape
    gumbel_key, marginalize_key = jax.random.split(rng, 2)
    # Idea: Analytically marginalize out a single random Gumbel.
    base_gumbels = jax.random.gumbel(gumbel_key, [dim])
    x = jax.random.categorical(marginalize_key, jnp.zeros([dim]))
    # Four cases:
    # - If this new Gumbel is the max for both A and B, x becomes
    #   the sample for both.
    # - If this new Gumbel is the max for A but not B, it becomes the sample for
    #   A only, and the sample for B is B's old max.
    # - If this new Gumbel is the max for B but not A, it becomes the sample for
    #   B only, and the sample for A is A's old max.
    # - If it isn't the max of either, use the old maxes for A and B.
    log_pA_x = log_pA[x]
    log_pB_x = log_pB[x]
    fallback_A = jnp.argmax((base_gumbels + log_pA).at[x].set(-np.inf))
    fallback_A_val = (base_gumbels + log_pA)[fallback_A]
    fallback_B = jnp.argmax((base_gumbels + log_pB).at[x].set(-np.inf))
    fallback_B_val = (base_gumbels + log_pB)[fallback_B]
    cutoff_A = fallback_A_val - log_pA_x
    cutoff_B = fallback_B_val - log_pB_x
    prob_replace_both = 1 - jnp.exp(-jnp.exp(-jnp.maximum(cutoff_A, cutoff_B)))
    prob_replace_neither = jnp.exp(-jnp.exp(-jnp.minimum(cutoff_A, cutoff_B)))
    prob_replace_A_or_B = 1 - prob_replace_neither
    prob_replace_A_not_B = jnp.where(
        cutoff_A < cutoff_B,
        prob_replace_A_or_B - prob_replace_both,
        0.0)
    prob_replace_B_not_A = jnp.where(
        cutoff_B < cutoff_A,
        prob_replace_A_or_B - prob_replace_both,
        0.0)
    result = jnp.zeros([dim, dim])
    result = result.at[x, x].set(prob_replace_both)
    result = result.at[fallback_A, x].set(prob_replace_B_not_A)
    result = result.at[x, fallback_B].set(prob_replace_A_not_B)
    result = result.at[fallback_A, fallback_B].set(prob_replace_neither)
    return result


def counterfactual_gumbels(log_p, observed_sample, rng):
    dim, = log_p.shape
    uniforms = jax.random.uniform(
        rng, shape=(dim,), minval=jnp.finfo(log_p.dtype).tiny, maxval=1.)
    max_gumbel_shifted = -jnp.log(-jnp.log(uniforms[observed_sample]))
    gumbels_shifted = log_p - jnp.log(jnp.exp(log_p - max_gumbel_shifted) - jnp.log(uniforms))
    gumbels_shifted = gumbels_shifted.at[observed_sample].set(max_gumbel_shifted)
    gumbels_unshifted = gumbels_shifted - log_p
    return gumbels_unshifted


def unbiased_counterfactual_gumbel_max(log_pA, sample_A, log_pB, rng):
    """Unbiased counterfactual estimator of Gumbel-max coupling probabilities
    """
    dim, = log_pA.shape
    log_pA = jnp.array(log_pA)
    log_pB = jnp.array(log_pB)

    # Top down sampling with log_pA.
    base_gumbels = counterfactual_gumbels(log_pA, sample_A, rng)
    gumbels_for_A = base_gumbels + log_pA
    max_gumbel_for_A = gumbels_for_A[sample_A]

    # uniforms = jax.random.uniform(
    #     rng, shape=(dim,), minval=jnp.finfo(log_pA.dtype).tiny, maxval=1.)
    # max_gumbel_for_A = -jnp.log(-jnp.log(uniforms[sample_A]))
    # gumbels_for_A = log_pA - jnp.log(jnp.exp(log_pA - max_gumbel_for_A) - jnp.log(uniforms))
    # gumbels_for_A = gumbels_for_A.at[sample_A].set(max_gumbel_for_A)
    # base_gumbels = gumbels_for_A - log_pA

    # Closest alternative sample for A.
    max_gumbel_closest_alternative_for_A = jnp.max(gumbels_for_A.at[sample_A].set(-np.inf))

    # Now, consider each index i, and analytically
    # marginalize out the truncated Gumbel for it.
    def marginalize_b_prob(marginalize_index):
        # What's the maximum in B ignoring marginalize_index?
        otherwise_B_gumbels = (base_gumbels + log_pB).at[marginalize_index].set(-np.inf)
        otherwise_max_B_index = jnp.argmax(otherwise_B_gumbels)
        # otherwise_max_B_value = (base_gumbels + log_pB)[otherwise_max_B_index]
        otherwise_max_B_value = jnp.max(otherwise_B_gumbels)

        # What's the probability that a resampled Gumbel will be higher than this?
        unshifted_threshold = otherwise_max_B_value - log_pB[marginalize_index]
        p_cross_threshold_for_B = 1 - jnp.exp(-jnp.exp(-unshifted_threshold))

        # Now, correct for our observation of sample_A.
        # Situation 1: we are resampling the Gumbel for the active sample. Our
        # constraint is that it must be greater than the cutoff point where we
        # would have picked something else for A. It's possible that satisfying
        # the constraint necessarily forces this sample to be the sample for B.
        # Here sample_A == marginalize_index.
        lower_constraint_here_unshifted = (
                max_gumbel_closest_alternative_for_A - log_pA[sample_A])
        p_sample_still_good_for_A = 1 - jnp.exp(-jnp.exp(-lower_constraint_here_unshifted))
        is_possible = p_cross_threshold_for_B < p_sample_still_good_for_A
        p_b_choose_sample_index = jnp.where(
            is_possible,
            p_cross_threshold_for_B / jnp.where(is_possible, p_sample_still_good_for_A, 1),
            # double where trick for stability
            1.0)

        # Situation 2: we are resampling some other Gumbel. Our constraint is that
        # it must be less than the threshold. It's possible that there's no way
        # to make this sample to be the sample for B.
        upper_constraint_here_unshifted = (
                max_gumbel_for_A - log_pA[marginalize_index])
        p_no_exceed_observed_A = jnp.exp(-jnp.exp(-upper_constraint_here_unshifted))
        p_exceed_observed_A = 1 - p_no_exceed_observed_A
        is_possible = p_cross_threshold_for_B > p_exceed_observed_A
        p_b_choose_other_marginalize_index = jnp.where(
            is_possible,
            (p_cross_threshold_for_B - p_exceed_observed_A) / jnp.where(is_possible, p_no_exceed_observed_A, 1),
            0.0)
        # Combine our situations
        p_choose_marginalize_index = jnp.where(
            sample_A == marginalize_index,
            p_b_choose_sample_index,
            p_b_choose_other_marginalize_index)
        # Construct probs for B based on this.
        # b_probs = jnp.zeros((dim,))
        # b_probs = b_probs.at[marginalize_index].set(p_choose_marginalize_index)
        # b_probs = b_probs.at[otherwise_max_B_index].set(1 - p_choose_marginalize_index)
        b_probs = (
                jax.nn.one_hot(marginalize_index, dim) * p_choose_marginalize_index
                + jax.nn.one_hot(otherwise_max_B_index, dim) * (1 - p_choose_marginalize_index)
        )
        return b_probs

    # Now, average over all of the indices we could choose to resample.
    alternatives = jax.vmap(marginalize_b_prob)(jnp.arange(dim))
    result = jnp.mean(alternatives, axis=0)
    return result


def unbiased_gumbel_max_using_counterfactual(log_pA, log_pB, rng):
    """Gumbel-max coupling by sampling A, then choosing B conditionally on that."""
    dim, = log_pA.shape
    k1, k2 = jax.random.split(rng)
    sample_A = jax.random.categorical(k1, log_pA)
    return jnp.zeros((dim, dim)).at[sample_A].set(
        unbiased_counterfactual_gumbel_max(log_pA, sample_A, log_pB, k2))


"""# Assembling a joint predictor
### Model
"""


class MLPCausalPriorPredictor(nn.Module):
    S_dim: int
    Z_dim: int
    hidden_features: List[int]
    sinkhorn_iterations: int = 10
    learn_prior: bool = True
    relaxation_temperature: Optional[float] = None

    def setup(self):
        self.hidden_layers = [nn.Dense(feats) for feats in self.hidden_features]
        self.output_layer = nn.DenseGeneral((self.Z_dim, self.S_dim))
        if self.learn_prior:
            self.prior = self.param("prior", nn.zeros, (self.Z_dim,))

    def get_prior(self):
        if self.learn_prior:
            return jax.nn.log_softmax(self.prior)
        else:
            return jax.nn.log_softmax(jnp.zeros([self.Z_dim]))

    def get_joint(self, s_logits):
        z_logits = self.get_prior()
        value = jax.nn.softmax(s_logits)
        for layer in self.hidden_layers:
            value = nn.sigmoid(layer(value))

        log_joint = self.output_layer(value)
        # Sinkhorn has nicer gradients.
        log_joint = fix_coupling_sinkhorn(log_joint, z_logits, s_logits, self.sinkhorn_iterations)
        # ... but rejection is harder to exploit inaccuracy for
        log_joint = fix_coupling_rejection(log_joint, z_logits, s_logits)
        return log_joint

    def get_forward(self, s_logits):
        return jax.nn.log_softmax(self.get_joint(s_logits), axis=-1)

    __call__ = get_joint

    def sample(self, s_logits, rng):
        log_z = self.get_prior()
        log_s_given_z = self.get_forward(s_logits)
        k1, k2 = jax.random.split(rng)
        z = jax.random.categorical(k1, log_z)
        # jax categorical does Gumbel-max internally, so given common random numbers
        # this will produce a Gumbel-max coupling of s given z.
        s = jax.random.categorical(k2, log_s_given_z[z])
        return s

    def sample_relaxed(self, s_logits, rng, temperature=None):
        """Sample a relaxed Gumbel-softmax output."""
        if temperature is None:
            assert self.relaxation_temperature
            temperature = self.relaxation_temperature
        log_z = self.get_prior()
        log_s_given_z = self.get_forward(s_logits)
        k1, k2 = jax.random.split(rng)
        z = jax.random.categorical(k1, log_z)
        g = jax.random.gumbel(k2, [self.S_dim])
        # Gumbel-softmax instead of argmax here
        return jax.nn.softmax((g + log_s_given_z[z]) / temperature)

    def counterfactual_sample(self, p_logits, q_logits, p_observed, rng):
        """Sample a single sample from q conditioned on observing p_observed."""
        k1, k2 = jax.random.split(rng)
        log_z = self.get_prior()
        log_ps_given_z = self.get_forward(p_logits)
        log_qs_given_z = self.get_forward(q_logits)
        # Infer z from p_observed
        log_z_given_ps = (log_z[:, None] + log_ps_given_z)[:, p_observed]
        z = jax.random.categorical(k1, log_z_given_ps)
        # Infer Gumbels from p_observed and z
        gumbels = counterfactual_gumbels(log_ps_given_z[z], p_observed, k2)
        # Choose accordingly
        qs = jnp.argmax(gumbels + log_qs_given_z[z])
        return qs

    def counterfactual_sample_diffble(self, p_logits, q_logits, p_observed, rng):
        """Sample an unbiased differentiable estimate of q given p.

        Returns an array of size [S_dim] that sums to 1.
        Sampling from the result should give something distributed identically to
        `counterfactual_sample`. But the probabilities in the table will be
        differentiable with respect to the parameters.

        WARNING: Doesn't actually produce unbiased gradients! Might not be useful.
        """
        log_z = self.get_prior()
        log_ps_given_z = self.get_forward(p_logits)
        log_qs_given_z = self.get_forward(q_logits)
        # Infer distn of z from p_observed.
        log_z_given_ps = (log_z[:, None] + log_ps_given_z)[:, p_observed]
        z_given_ps = jax.nn.softmax(log_z_given_ps)
        # Infer differentiable distns from p_observed and each possible z.
        # (todo: better to use common random numbers here or no? Currently uses the
        # same RNG for each possible z.)
        distns = jax.vmap(
            lambda log_ps_given_zi, log_qs_given_zi:
            unbiased_counterfactual_gumbel_max(log_ps_given_zi, p_observed, log_qs_given_zi, rng))(
            log_ps_given_z, log_qs_given_z)
        avg_distn = jnp.sum(z_given_ps[:, None] * distns, axis=0)
        return avg_distn

    def counterfactual_sample_relaxed(self, p_logits, q_logits, p_observed, rng, temperature=1.0):
        """Sample a Gumbel-Softmax estimate of q given p.

        Conceptually similar to counterfactual_sample_diffble but more
        straightforward to explain.
        """
        log_z = self.get_prior()
        log_ps_given_z = self.get_forward(p_logits)
        log_qs_given_z = self.get_forward(q_logits)
        # Infer distn of z from p_observed.
        log_z_given_ps = (log_z[:, None] + log_ps_given_z)[:, p_observed]
        z_given_ps = jax.nn.softmax(log_z_given_ps)

        # For each z, counterfactually sample some Gumbels, then apply softmax.
        def relaxed_for_fixed_z(z):
            # (again, uses common random numbers here)
            g = counterfactual_gumbels(log_ps_given_z[z], p_observed, rng)
            # Gumbel-softmax instead of argmax here
            soft_q = jax.nn.softmax((g + log_qs_given_z[z]) / temperature)
            return soft_q

        distns = jax.vmap(relaxed_for_fixed_z)(jnp.arange(self.Z_dim))
        avg_distn = jnp.sum(z_given_ps[:, None] * distns, axis=0)
        return avg_distn


# model = MLPCausalPriorPredictor(10, 20, [1024, 1024])

"""### Task: couple two distns"""

# @title
def gumbel_max_sampling(logits_1, logits_2, reward_f, key):
    gumbels = jax.random.gumbel(key, logits_1.shape)
    x = jnp.argmax(gumbels + logits_1, axis=-1)
    y = jnp.argmax(gumbels + logits_2, axis=-1)

    ate = reward_f[x] - reward_f[y]
    return ate


def inverse_cdf_sampling(logits_1, logits_2, reward_f, key):
    def inverse_cdf(logits, u):
        probs = jax.nn.softmax(logits)
        pr_cs = jnp.cumsum(probs, axis=-1)
        ans = pr_cs > u  # .view(u.size(0), 1)
        ans = (ans[:, 1:] ^ ans[:, :-1])
        return jnp.where(ans.sum(-1) == 1, jnp.argmax(ans, -1) + 1, 0)

    uni = jax.random.uniform(key, (logits_1.shape[0], 1))  # or 1
    x = inverse_cdf(logits_1, uni)
    y = inverse_cdf(logits_2, uni)
    ate = reward_f[x] - reward_f[y]
    return ate

def get_single_ate(model, params, logits_1, logits_2, reward_f, seed, p_observed=None, method='crn'):
    """Sample with common random numbers, obtaining the induced coupling."""
    logits_1, logits_2 = jnp.asarray(logits_1), jnp.asarray(logits_2)
    reward_f = jnp.asarray(reward_f)
    rng = jax.random.PRNGKey(seed)
    if p_observed is not None:
        p_observed = jnp.asarray(p_observed)

    if method == 'crn':
        x = model.bind(params).sample(logits_1, rng)
        y = model.bind(params).sample(logits_2, rng)

    elif method == 'cf':
        x = p_observed
        y = model.bind(params).counterfactual_sample(logits_1, logits_2, p_observed, rng)

    ate = reward_f[x] - reward_f[y]
    return (np.array(x), np.array(y)), np.array(ate)


# counterfactual_sample(self, p_logits, q_logits, p_observed, rng)

def wrap_joint_predictor_sampling(model, params, p_logits_batch, q_logits_batch,
                                  policy_reward_vector_batch, rng, p_observed=None, method='crn'):
    def joint_predictor_sampling(logits_1, logits_2, reward_f, key):
        """Sample with common random numbers, obtaining the induced coupling."""
        x = model.bind(params).sample(logits_1, key)
        y = model.bind(params).sample(logits_2, key)
        ate = reward_f[x] - reward_f[y]
        return ate

    def joint_predictor_sampling_counterfactual(logits_1, logits_2, reward_f, p_observed, key):
        """Sample with common random numbers, obtaining the induced coupling."""
        # x = model.bind(params).sample(logits_1, key)
        y = model.bind(params).counterfactual_sample(logits_1, logits_2, p_observed, key)
        ate = reward_f[p_observed] - reward_f[y]
        return ate

    if method == 'crn':
        return jax.tree_map(jnp.var, jax.vmap(joint_predictor_sampling)(p_logits_batch, q_logits_batch,
                                                                        policy_reward_vector_batch,
                                                                        jax.random.split(rng,
                                                                                         p_logits_batch.shape[
                                                                                             0])))
    elif method == 'counterfactual':
        return jax.tree_map(jnp.var, jax.vmap(joint_predictor_sampling_counterfactual)(p_logits_batch, q_logits_batch,
                                                                                       policy_reward_vector_batch, p_observed,
                                                                                       jax.random.split(rng,
                                                                                                        p_logits_batch.shape[
                                                                                                            0])))


class Gadget2Jax:
    def __init__(self, s_dim=146, z_dim=20, hidden_features=[1024,1024], tmp=1.0):
        """ this is a wrapper class to expose the trained model outside and to train using jit"""
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.hidden_features = hidden_features
        self.tmp = tmp

        self.model = MLPCausalPriorPredictor(S_dim=146, Z_dim=20, hidden_features=[1024, 1024], relaxation_temperature=1.0)
        rng = jax.random.PRNGKey(0)

        rng, init_key = jax.random.split(rng)
        self.params = self.model.init(init_key, jnp.zeros([self.model.S_dim]))


    def get_var_ate(self, logits_p, logits_q, s_prime_obs, policy_reward_vector, num_iter=500, bs_infrence=512, seed=42,
                    method="counterfactual", generalized=0.0):

        rng = jax.random.PRNGKey(seed)

        vars = []
        for i in range(num_iter):
            rng, subkey = jax.random.split(rng)

            policy_reward_vector_batch = jnp.broadcast_to(policy_reward_vector, (bs_infrence, self.model.S_dim))
            p_logits_batch = jnp.broadcast_to(logits_p, (bs_infrence, self.model.S_dim))
            q_logits_batch = jnp.broadcast_to(logits_q, (bs_infrence, self.model.S_dim))
            sample_p_batch = jnp.broadcast_to(s_prime_obs, (bs_infrence,))
            # if generalized > 0:
            #     p_logits_batch += generalized * jax.random.normal(rng, shape=p_logits_batch.shape)
            #     q_logits_batch += generalized * jax.random.normal(subkey, shape=q_logits_batch.shape)

            var_ate_jpz = wrap_joint_predictor_sampling(self.model, self.params, p_logits_batch, q_logits_batch,
                                                        policy_reward_vector_batch, subkey, sample_p_batch, method)
            vars.append(var_ate_jpz)

        return np.mean(vars)

    def get_var_ate_batch(self, p_logits_batch, q_logits_batch, s_prime_obs, policy_reward_vector, rng=42,
                          method="counterfactual"):

        #rng = jax.random.PRNGKey(seed)
        rng, subkey = jax.random.split(rng)
        bs_infrence = p_logits_batch.shape[0]
        policy_reward_vector_batch = jnp.broadcast_to(policy_reward_vector, (bs_infrence, self.model.S_dim))
        p_logits_batch = jnp.array(p_logits_batch)

        q_logits_batch = jnp.array(q_logits_batch)
        s_prime_obs = jnp.broadcast_to(jnp.array(s_prime_obs), (bs_infrence,))
        # if generalized > 0:
        #     p_logits_batch += generalized * jax.random.normal(rng, shape=p_logits_batch.shape)
        #     q_logits_batch += generalized * jax.random.normal(subkey, shape=q_logits_batch.shape)

        var_ate_jpz = wrap_joint_predictor_sampling(self.model, self.params, p_logits_batch, q_logits_batch,
                                                    policy_reward_vector_batch, subkey, s_prime_obs, method)

        return var_ate_jpz, rng

    def train(self, p, q, s_prime_obs, policy_reward_vector, cf=True,
              batch_size=64, num_iter=2048, noise_scale=0, verbose=True, seed=0):  # relaxed-counterfactual

        if isinstance(p, list):
            np_p, np_q, s_prime_obs = np.array(p), np.array(q), np.array(s_prime_obs)
            logits_p = np.log(np_p+1e-10).clip(min=-80.0)
            logits_q = np.log(np_q+1e-10).clip(min=-80.0)
        else:
            logits_p = p
            logits_q = q

        method = "counterfactual" if cf else "crn"
        # self.model will be override at the end of each iteration
        model = MLPCausalPriorPredictor(S_dim=self.s_dim,
                                        Z_dim=self.z_dim,
                                        hidden_features=self.hidden_features, relaxation_temperature=self.tmp)

        def loss_and_metrics(params, coupling_loss_matrix, p_logits, q_logits, sample_p, rng, method,
                             inner_num_samples):
            def sample_loss(key):
                if method == "crn":
                    soft_p = model.apply(params, p_logits, key, method=model.sample_relaxed)
                    soft_q = model.apply(params, q_logits, key, method=model.sample_relaxed)
                    coupling_loss = jnp.sum(soft_p[:, None] * soft_q[None, :] * coupling_loss_matrix)

                elif method == "counterfactual":
                    k1, k2 = jax.random.split(key)
                    # sample_p = jax.random.categorical(k1, p_logits)
                    soft_q = model.apply(params,
                                         p_logits=p_logits, q_logits=q_logits, p_observed=sample_p,
                                         method=model.counterfactual_sample_relaxed, rng=k2)
                    coupling_loss = jnp.sum(soft_q * coupling_loss_matrix[sample_p, :])

                independent_loss = jnp.sum(p_logits[:, None] * q_logits[None, :] * coupling_loss_matrix)
                return coupling_loss, independent_loss

            loss_samples, ref_samples = jax.vmap(sample_loss)(
                jax.random.split(rng, inner_num_samples))
            loss = jnp.mean(loss_samples)
            return loss, {"loss": loss}

        def batch_loss(params, coupling_loss_matrix_batch, p_logits_batch, q_logits_batch, sample_p_batch, rng, method,
                       inner_num_samples):
            def go(coupling_loss_matrix, p_logits, q_logits, sample_p, key):
                return loss_and_metrics(params, coupling_loss_matrix, p_logits, q_logits, sample_p, rng, method,
                                        inner_num_samples)

            return jax.tree_map(jnp.mean,
                                jax.vmap(go)(coupling_loss_matrix_batch, p_logits_batch, q_logits_batch, sample_p_batch,
                                             jax.random.split(rng, p_logits_batch.shape[0])))

        # important: rerun this cell whenever you change the model / learning rate / method / inner_num_samples
        @jax.jit
        def opt_step(opt_state, params, coupling_loss_matrix_batch, p_logits_batch, q_logits_batch, sample_p_batch,
                     rng):
            grads, metrics = jax.grad(batch_loss, has_aux=True)(params, coupling_loss_matrix_batch, p_logits_batch,
                                                                q_logits_batch, sample_p_batch, rng, method=method,
                                                                inner_num_samples=INNER_NUM_SAMPLES)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            any_was_nan = jax.tree_util.tree_reduce(jnp.logical_or,
                                                    jax.tree_map(lambda v: jnp.any(jnp.isnan(v)), grads))
            new_opt_state, new_params = jax.tree_multimap(lambda a, b: jnp.where(any_was_nan, a, b),
                                                          (opt_state, params),
                                                          (new_opt_state, new_params))
            return new_opt_state, new_params, metrics, grads, any_was_nan

        logits_p, logits_q = jnp.asarray(logits_p), jnp.asarray(logits_q)
        s_prime_obs = jnp.asarray(s_prime_obs)
        policy_reward_vector = jnp.asarray(policy_reward_vector)
        coupling_loss_matrix = jnp.square(policy_reward_vector[None, :] - policy_reward_vector[:, None])

        # "relaxed"  # "marginalized" # can also be "relaxed"
        INNER_NUM_SAMPLES = 16  # can be anything
        # You may want a larger learning rate. I think marginalized is more stable and can use higher learning rates.
        tx = optax.adam(0.00001)  # you may want to choose a different learning rate

        rng = jax.random.PRNGKey(seed)

        rng, init_key = jax.random.split(rng)
        params = model.init(init_key, jnp.zeros([model.S_dim]))
        opt_state = tx.init(params)

        # all_states = {}

        i = 1
        start_time = time.time()
        count_since_reset = 0
        training_vars = []
        while i < num_iter+1:  # can be higher if needed 2048
            rng, key = jax.random.split(rng)

            # Obtain a batch of logits and rewards. Here I just assume it's 64 random things with the same reward.
            coupling_loss_matrix_batch = jnp.broadcast_to(coupling_loss_matrix, (batch_size, model.S_dim, model.S_dim))
            # rng, key = jax.random.split(rng)
            # jax.vmap(sample_p_or_q_logits)('p')
            p_logits_batch = jnp.broadcast_to(logits_p, (batch_size, model.S_dim))
            # rng, key = jax.random.split(rng)
            # jax.vmap(sample_p_or_q_logits)('q')
            q_logits_batch = jnp.broadcast_to(logits_q, (batch_size, model.S_dim))
            policy_reward_vector_batch = jnp.broadcast_to(policy_reward_vector, (batch_size, model.S_dim))

            sample_p_batch = jnp.broadcast_to(s_prime_obs, (batch_size,))

            if noise_scale > 0:
                p_logits_batch += noise_scale * jax.random.normal(rng, shape=p_logits_batch.shape)
                q_logits_batch += noise_scale * jax.random.normal(key, shape=q_logits_batch.shape)
                #p_logits_batch -= jax.scipy.special.logsumexp(p_logits_batch, axis=-1, keepdims=True)
                #q_logits_batch -= jax.scipy.special.logsumexp(q_logits_batch, axis=-1, keepdims=True)


            if i % 400 == 0:
                var_ate_jpz = wrap_joint_predictor_sampling(model, params, p_logits_batch, q_logits_batch,
                                                            policy_reward_vector_batch, rng, sample_p_batch, method)
                var_ate_jpz = np.array(var_ate_jpz).astype(float)

                training_vars.append(var_ate_jpz)


            # Pass the inputs in and take a gradient step.
            opt_state, params, metrics, grads, bad = opt_step(opt_state, params, coupling_loss_matrix_batch,
                                                              p_logits_batch,
                                                              q_logits_batch, sample_p_batch, rng)
            self.model = model
            self.params = params
            if bad:
                raise RuntimeError("nan!")
            count_since_reset += 1
            if verbose and (i % 200 == 0 or np.remainder(np.log2(i), 1) == 0):
                now = time.time()
                rate = count_since_reset / (now - start_time)
                start_time = now
                count_since_reset = 0
                #print(f"{i} [{rate}/s]:", jax.tree_map(float, metrics))
                print(f"{i}:", jax.tree_map(float, metrics))
                sys.stdout.flush()
                time.sleep(0.02)
            i += 1

        return training_vars
