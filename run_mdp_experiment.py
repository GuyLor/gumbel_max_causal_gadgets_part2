from sepsis_mdp import SepsisMDP
import numpy as np
import cf.utils as utils
from joint_predictor import Coupler
from cf import fixed_mechanisms as fm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=59, help="seed (there is more inside the code)")
parser.add_argument('--noise_scale', type=float, default=1.0, help="std of a Gaussian for generalize (p,q)")
parser.add_argument('--counterfactual', action='store_true', help='whether counterfactual or common random numbers (joint sampling)')
parser.add_argument('--gadget2_jax', action='store_true', help='using (old) version of gadget-2 implemented in Jax for reproducing the paper plots')

params = parser.parse_args()

if params.gadget2_jax:
    import jax

def run_comparison(behavior_interv_probs, target_interv_probs, s_prime_obs=None, n=10, seed=0):
    # testing
    logits_p = np.log(np.array(behavior_interv_probs) + 1e-10).clip(min=-80.0)
    logits_q = np.log(np.array(target_interv_probs) + 1e-10).clip(min=-80.0)
    batch_size_test = 2000
    cf = 'counterfactual' if params.counterfactual else 'crn'
    batch_logits_p = np.tile(logits_p, (batch_size_test, 1)) + params.noise_scale * np.random.randn(batch_size_test,
                                                                                             logits_p.shape[-1])
    batch_logits_q = np.tile(logits_q, (batch_size_test, 1)) + params.noise_scale * np.random.randn(batch_size_test,
                                                                                             logits_q.shape[-1])

    batch_s_prime_obs = np.tile(s_prime_obs, (batch_size_test, 1)) if params.counterfactual else None
    rng = jax.random.PRNGKey(seed) if params.gadget2_jax else None
    vars_gm, vars_icdf, vars_gd1, vars_gd2 = [], [], [], []
    for i in range(n):
        (s_prime_p, s_prime_q), _ = c.gadget_1.sample_from_joint(batch_logits_p, batch_logits_q, batch_s_prime_obs,
                                                                 counterfactual=params.counterfactual, train=False)
        vars_gd1.append(utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, batch_s_prime_obs))

        if not params.gadget2_jax:
            (s_prime_p, s_prime_q), _ = c.gadget_2.sample_from_joint(batch_logits_p, batch_logits_q, batch_s_prime_obs,
                                                                     counterfactual=params.counterfactual, train=False)
            vars_gd2.append(utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, batch_s_prime_obs))
        else:
            var, rng = c.gadget_2.get_var_ate_batch(batch_logits_p, batch_logits_q, obs_next_states, reward_vector, rng, cf)
            vars_gd2.append(var)

        s_prime_p, s_prime_q = fm.gumbel_max_coupling(batch_logits_p, batch_logits_q, batch_s_prime_obs, counterfactual=params.counterfactual)
        vars_gm.append(utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, batch_s_prime_obs))

        s_prime_p, s_prime_q = fm.inverse_cdf_coupling(batch_logits_p, batch_logits_q, batch_s_prime_obs, counterfactual=params.counterfactual)
        vars_icdf.append(utils.compute_variance_treatment_effect(reward_vector, s_prime_p, s_prime_q, batch_s_prime_obs))

    return np.mean(vars_gm), np.mean(vars_icdf), np.mean(vars_gd1), np.mean(vars_gd2)


# Setup of the sepsis simulator:
sep = SepsisMDP()

# load an MDP that was trained over the simulator - represents the 'true' transition distributions of sepsis management
true_mdp = sep.load_mdp_from_simulator()

# Train a behavior policy over the true MDP using policy iteration algorithm
physician_policy = sep.get_physician_policy(true_mdp)

# Sample trajectories of patients by interacting with the MDP using the physician policy
# Using these trajectories (collected data), construct an estimated MDP
obs_samples, est_mdp = sep.simulate_patient_trajectories_and_construct_mdp(physician_policy,
                                                                           num_steps=20,
                                                                           num_samples=20000) #

# Train a policy over the estimated MDP
cf_policy = sep.train_rl_policy(est_mdp)

relevant_trajs_and_t = sep.search_for_relevant_tr_t(obs_samples, cf_policy, est_mdp,
                                                    num_of_diff_p_q=sep.n_proj_states, num_gt_zero_probs=4)

vars_gm, vars_icdf, vars_gd1, vars_gd2 = [], [], [], []
j = 0
for p_q in range(6):

    trajectory_idx, time_idx = relevant_trajs_and_t[p_q]

    current_state, obs_action, obs_next_states = sep.parse_samples(obs_samples, trajectory_idx, time_idx)
    cf_action = cf_policy[current_state, :].squeeze().argmax()

    behavior_interv_probs = est_mdp.tx_mat[0, obs_action, current_state, :].squeeze().tolist()
    target_interv_probs = est_mdp.tx_mat[0, cf_action, current_state, :].squeeze().tolist()

    # Unlike Oberst and Sontag, we couple between single time steps and therefore we consider rewards per state (instead of [0,1] rewards at trajectory completion)
    # The state is composed by 7 categorical variables, each with a different number of categories.
    # We sample a Gaussian noise for each category representing its energy.
    # The reward of a given state is obtained by summing the energies associated with its variables.
    for t in range(5):
        print('='*80)
        print(f'Trial {j}: sample new rewards')
        c = Coupler(s_dim=sep.n_proj_states, z_dim=20, hidden_features=[1024, 1024], tmp=1.0,
                    seed=params.seed+j, gadget2_jax=params.gadget2_jax)

        reward_vector = sep.randomize_states_rewards()
        print('---- Train gadget-1 -----')
        c.train_gadget_1(p=behavior_interv_probs, q=target_interv_probs, s_prime_obs=obs_next_states, reward_vector=reward_vector,
                         batch_size=64, counterfactual=params.counterfactual, num_iter=2000, noise_scale=params.noise_scale)

        print('---- Train gadget-2 -----')
        c.train_gadget_2(p=behavior_interv_probs, q=target_interv_probs, s_prime_obs=obs_next_states, reward_vector=reward_vector,
                         batch_size=64, counterfactual=params.counterfactual, num_iter=2000, noise_scale=params.noise_scale)

        gm, icdf, gd1, gd2 = run_comparison(behavior_interv_probs, target_interv_probs, obs_next_states, n=10)
        print(f'Gumbel-max: {gm}, inverse-CDF: {icdf}, gadget-1: {gd1}, gadget-2: {gd2}')
        vars_gm.append(gm); vars_icdf.append(icdf); vars_gd1.append(gd1); vars_gd2.append(gd2)
        j += 1

    print('5 rewards realizations, same p,q')
    print(f'Gumbel-max: {np.mean(vars_gm[-5:])}, inverse-CDF: {np.mean(vars_icdf[-5:])}, gadget-1: { np.mean(vars_gd1[-5:])}, gadget-2: {np.mean(vars_gd2[-5:])}')

cf_crn = 'cf' if params.counterfactual else 'crn'
gen = 'generalized' if params.noise_scale > 0 else 'fixed'
figpath = './figs/{}_{}_{}'.format(cf_crn, gen, params.noise_scale)

utils.plot_mdp_variances(vars_gm, vars_icdf, vars_gd1, vars_gd2, cf=params.counterfactual,
                         generalized=params.noise_scale > 0, figpath=figpath)
