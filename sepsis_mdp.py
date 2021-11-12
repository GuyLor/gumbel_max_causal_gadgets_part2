
"""

Most of the the code in this file was written by Oberst and Sontag, 2019. Thanks!

"""

import numpy as np

import torch
import pickle
from scipy.linalg import block_diag

# Sepsis Simulator code
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.DataGenerator import DataGenerator

import mdptoolboxSrc.mdp as mdptools


def format_dgen_samps(states, actions, rewards, hidden, NSTEPS, NSIMSAMPS):
    """format_dgen_samps
    Formats the output of the data generator (a batch of trajectories) in a way
    that the other functions will consume

    :param states: states
    :param actions: actions
    :param rewards: rewards
    :param hidden: hidden states
    :param NSTEPS: Maximum length of trajectory
    :param NSIMSAMPS: Number of trajectories
    """
    obs_samps = np.zeros((NSIMSAMPS, NSTEPS, 7))
    obs_samps[:, :, 0] = np.arange(NSTEPS)  # Time Index
    obs_samps[:, :, 1] = actions[:, :, 0]
    obs_samps[:, :, 2] = states[:, :-1, 0]  # from_states
    obs_samps[:, :, 3] = states[:, 1:, 0]  # to_states
    obs_samps[:, :, 4] = hidden[:, :, 0]  # Hidden variable
    obs_samps[:, :, 5] = hidden[:, :, 0]  # Hidden variable
    obs_samps[:, :, 6] = rewards[:, :, 0]

    return obs_samps


class SepsisMDP:
    def __init__(self):
        self.n_actions = Action.NUM_ACTIONS_TOTAL
        # Construct the projection matrix for obs->proj states
        # These are added as absorbing states
        self.n_states_abs = State.NUM_OBS_STATES + 2
        self.discStateIdx = self.n_states_abs - 1
        self.deadStateIdx = self.n_states_abs - 2

        self.n_proj_states = int((self.n_states_abs - 2) / 5) + 2
        self.proj_matrix = self.create_projection_matrix()
        self.proj_lookup = self.proj_matrix.argmax(axis=-1)

        self.discount_policy = 0.99    # discount policy

        self.states_vector = {}
        for i in range(self.n_states_abs - 2):
            this_state = State(state_idx=i, idx_type='obs',
                               diabetic_idx=1)
            j = this_state.get_state_idx('proj_obs')
            self.states_vector[j] = this_state.get_state_vector()

    def load_mdp_from_simulator(self, sim_mdp_path='diab_txr_mats-replication.pkl'):
        # loads the transitions and reward counts from the simulator. This represent the 'true' mdp.
        # WE use a pretrained mdp which is produced by running 'learn_mdp_parameters.ipynb'
        path = "./data/{}".format(sim_mdp_path)
        try:
            with open(path, "rb") as f:
                mdict = pickle.load(f)
        except FileNotFoundError:
            # check if zip:
            import zipfile
            if zipfile.is_zipfile(path+'.zip'):
                with zipfile.ZipFile(path+'.zip', 'r') as zip_ref:
                    zip_ref.extractall('./data')
                return self.load_mdp_from_simulator(sim_mdp_path)
            else:
                print('Transitions and reward counts from the simulator are missing.')
                print('It can be produced by running learn_mdp_parameters.ipynb')




        tx_mat = mdict["tx_mat"]
        r_mat = mdict["r_mat"]


        tx_mat_full = np.zeros((self.n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))
        r_mat_full = np.zeros((self.n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))

        for a in range(self.n_actions):
            tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a, ...])
            r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

        fullMDP = MatrixMDP(tx_mat_full, r_mat_full)
        return fullMDP

    def randomize_states_rewards(self, rng=None):
        categ_num = np.array([3, 3, 2, 5, 2, 2, 2])
        #torch.random.set_rng_state(rng)
        # print('rewards:', (torch.random.get_rng_state()[torch.random.get_rng_state() != 0],
        #                     torch.random.get_rng_state()[torch.random.get_rng_state() != 0].sum()))
        energies = [torch.randn(i).numpy() for i in categ_num]   #  a bit weird to use torch and convert it to np but it's for reproduce submitted results
        #energies = [torch.randn(i).numpy() for i in categ_num]
        #np.random.seed(seed)
        #energies = [np.random.randn(i) for i in categ_num]
        def get_reward(next_state):
            if next_state == 144:
                # death state
                reward = -4
            elif next_state == 145:
                reward = 4
            else:
                state_vector = self.states_vector[next_state]
                reward = np.sum([energies[enrg][i] for (enrg, i) in zip(range(len(energies)), state_vector)])
            return reward

        reward_vector = np.array([get_reward(next_state) for next_state in range(self.n_proj_states)])
        return reward_vector

    def create_projection_matrix(self):
        proj_matrix = np.zeros((self.n_states_abs, self.n_proj_states))
        states_vector = {}
        for i in range(self.n_states_abs - 2):
            this_state = State(state_idx=i, idx_type='obs',
                               diabetic_idx=1)  # Diab a req argument, no difference
            # assert this_state == State(state_idx = i, idx_type = 'obs', diabetic_idx = 0)
            j = this_state.get_state_idx('proj_obs')
            proj_matrix[i, j] = 1
            states_vector[j] = this_state.get_state_vector()

        # Add the projection to death and discharge`
        proj_matrix[self.deadStateIdx, -2] = 1
        proj_matrix[self.discStateIdx, -1] = 1
        return proj_matrix.astype(int)

    def get_physician_policy(self, fullMDP):
        # trained over the full MDP with a little variation
        PHYS_EPSILON = 0.05  # Used for sampling using physician pol as eps greedy
        fullPol = fullMDP.policyIteration(discount=self.discount_policy, eval_type=1)

        physPolSoft = np.copy(fullPol)
        physPolSoft[physPolSoft == 1] = 1 - PHYS_EPSILON
        physPolSoft[physPolSoft == 0] = PHYS_EPSILON / (self.n_actions- 1)
        return physPolSoft

    def simulate_patient_trajectories_and_construct_mdp(self, physician_policy, num_steps=20, num_samples=20000):
        # simulate patient trajectories using a physician policy (trained using policy iteration over the full mdp)
        np.random.seed(0)
        dgen = DataGenerator()
        states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals = dgen.simulate(
            num_samples, num_steps, policy=physician_policy, policy_idx_type='full',
            p_diabetes=0.2, use_tqdm=False)  # True, tqdm_desc='Behaviour Policy Simulation')


        ############## Construct the Transition Matrix w/proj states ##############
        proj_tx_cts = np.zeros((self.n_actions, self.n_proj_states, self.n_proj_states))
        proj_tx_mat = np.zeros_like(proj_tx_cts)

        # (1) NOTE: Previous code marginalized here, but now we are just getting observed quantities out, no components
        assert emp_tx_totals.ndim == 3

        # (2) Add new aborbing states, and a new est_tx_mat with Absorbing states
        death_states = (emp_r_totals.sum(axis=0).sum(axis=0) < 0)
        disch_states = (emp_r_totals.sum(axis=0).sum(axis=0) > 0)

        est_tx_cts_abs = np.zeros((self.n_actions, self.n_states_abs, self.n_states_abs))
        est_tx_cts_abs[:, :-2, :-2] = np.copy(emp_tx_totals)

        death_states = np.concatenate([death_states, np.array([True, False])])
        disch_states = np.concatenate([disch_states, np.array([False, True])])
        assert est_tx_cts_abs[:, death_states, :].sum() == 0
        assert est_tx_cts_abs[:, disch_states, :].sum() == 0

        est_tx_cts_abs[:, death_states, self.deadStateIdx] = 1
        est_tx_cts_abs[:, disch_states, self.discStateIdx] = 1

        # (3) Project the new est_tx_cts_abs to the reduced state space
        for a in range(self.n_actions):
            proj_tx_cts[a] = self.proj_matrix.T.dot(est_tx_cts_abs[a]).dot(self.proj_matrix)

        # Normalize
        nonzero_idx = proj_tx_cts.sum(axis=-1) != 0
        proj_tx_mat[nonzero_idx] = proj_tx_cts[nonzero_idx]

        proj_tx_mat[nonzero_idx] /= proj_tx_mat[nonzero_idx].sum(axis=-1, keepdims=True)

        proj_r_mat = np.zeros((self.n_actions, self.n_proj_states, self.n_proj_states))
        proj_r_mat[..., -2] = -1
        proj_r_mat[..., -1] = 1

        proj_r_mat[..., -2, -2] = 0  # No reward once in aborbing state
        proj_r_mat[..., -1, -1] = 0

        ############ Construct the empirical prior on the initial state ##################
        initial_state_arr = np.copy(states[:, 0, 0])
        initial_state_counts = np.zeros((self.n_states_abs, 1))
        for i in range(initial_state_arr.shape[0]):
            initial_state_counts[initial_state_arr[i]] += 1

        # Project initial state counts to new states
        proj_state_counts = self.proj_matrix.T.dot(initial_state_counts).T
        proj_p_initial_state = proj_state_counts / proj_state_counts.sum()

        # Because some SA pairs are never observed, assume they cause instant death
        zero_sa_pairs = proj_tx_mat.sum(axis=-1) == 0
        proj_tx_mat[zero_sa_pairs, -2] = 1  # Always insta-death if you take a never-taken action
        self.proj_tx_mat = proj_tx_mat  # for assertion
        # Construct an extra axis for the mixture component, of which there is only one
        projMDP = MatrixMDP(proj_tx_mat, proj_r_mat,
                               p_initial_state=proj_p_initial_state)

        def projection_func(obs_state_idx):
            if obs_state_idx == -1:
                return -1
            else:
                return self.proj_lookup[obs_state_idx]

        proj_f = np.vectorize(projection_func)

        states_proj = proj_f(states)

        assert states_proj.shape == states.shape

        obs_samps_proj = format_dgen_samps(states_proj, actions, rewards, diab, num_steps, num_samples)

        return obs_samps_proj, projMDP

    def train_rl_policy(self, projected_mdp):
        # train rl policy using policy iteration over the projected mdp
        try:
            RlPol = projected_mdp.policyIteration(discount=self.discount_policy)
        except:
            assert np.allclose(self.proj_tx_mat.sum(axis=-1), 1)
            RlPol = projected_mdp.policyIteration(discount=self.discount_policy, skip_check=True)
        return RlPol

    def parse_samples(self,obs_sampels, trajectory_idx, time_idx):
        obs_action = obs_sampels[trajectory_idx, time_idx, 1].astype(int).squeeze()
        current_state = obs_sampels[trajectory_idx, time_idx, 2].astype(int).squeeze()
        obs_next_states = obs_sampels[trajectory_idx, time_idx, 3].astype(int).squeeze()
        return current_state, obs_action, obs_next_states


    def search_for_relevant_tr_t(self, samples, cf_policy, mdp, num_of_diff_p_q=144, num_gt_zero_probs=3):
        component = 0
        relevants = []
        p_q_list = []
        for tr in range(samples.shape[0]):
            for t in range(samples.shape[1]):
                obs_action = samples[tr, t, 1].astype(int).squeeze()
                current_state = samples[tr, t, 2].astype(int).squeeze()
                obs_to_states = samples[tr, t, 3].astype(int).squeeze()

                cf_action = cf_policy[current_state, :].squeeze().argmax()
                new_interv_probs = \
                    mdp.tx_mat[component,
                    cf_action, current_state,
                    :].squeeze().tolist()

                prev_interv_probs = \
                    mdp.tx_mat[component,
                    obs_action, current_state,
                    :].squeeze().tolist()

                p, q = np.array(prev_interv_probs), np.array(new_interv_probs)

                if (p == q).sum() < num_of_diff_p_q and \
                        len(p[p != 0]) > num_gt_zero_probs and \
                        len(q[q != 0]) > num_gt_zero_probs and \
                        obs_action != -1:
                    p_q = np.concatenate((p, q))
                    p_q_list.append(p_q)
                    relevants.append((tr, t))

        p_q_array = np.array(p_q_list)
        relevant_full = np.array(relevants)
        _, relevant_idx = np.unique(p_q_array, axis=0, return_index=True)
        relevant_full = relevant_full[relevant_idx]
        np.random.shuffle(relevant_full)
        return relevant_full.tolist()


class MatrixMDP(object):
    def __init__(self, tx_mat, r_mat, p_initial_state=None, p_mixture=None):
        """__init__

        :param tx_mat:  Transition matrix of shape (n_components x n_actions x
        n_states x n_states) or (n_actions x n_states x n_states)
        :param r_mat:  Reward matrix of shape (n_components x n_actions x
        n_states x n_states) or (n_actions x n_states x n_states)
        :param p_initial_state: Probability over initial states
        :param p_mixture: Probability over "mixture" components, in this case
        diabetes status
        """
        # QA the size of the inputs
        assert tx_mat.ndim == 4 or tx_mat.ndim == 3, \
            "Transition matrix wrong dims ({} != 3 or 4)".format(tx_mat.ndim)
        assert r_mat.ndim == 4 or r_mat.ndim == 3, \
            "Reward matrix wrong dims ({} != 3 or 4)".format(tx_mat.ndim)
        assert r_mat.shape == tx_mat.shape, \
            "Transition / Reward matricies not the same shape!"
        assert tx_mat.shape[-1] == tx_mat.shape[-2], \
            "Last two dims of Tx matrix should be equal to num of states"

        # Get the number of actions and states
        n_actions = tx_mat.shape[-3]
        n_states = tx_mat.shape[-2]

        # Get the number of components in the mixture:
        # If no hidden component, add a dummy so the rest of the interface works
        if tx_mat.ndim == 3:
            n_components = 1
            tx_mat = tx_mat[np.newaxis, ...]
            r_mat = r_mat[np.newaxis, ...]
        else:
            n_components = tx_mat.shape[0]

        # Get the prior over initial states
        if p_initial_state is not None:
            if p_initial_state.ndim == 1:
                p_initial_state = p_initial_state[np.newaxis, :]

            assert p_initial_state.shape == (n_components, n_states), \
                ("Prior over initial state is wrong shape "
                 "{} != (C x S)").format(p_initial_state.shape)

        # Get the prior over components
        if n_components == 1:
            p_mixture = np.array([1.0])
        elif p_mixture is not None:
            assert p_mixture.shape == (n_components, ), \
                ("Prior over components is wrong shape "
                 "{} != (C)").format(p_mixture.shape)

        self.n_components = n_components
        self.n_actions = n_actions
        self.n_states = n_states
        self.tx_mat = tx_mat
        self.r_mat = r_mat
        self.p_initial_state = p_initial_state
        self.p_mixture = p_mixture

        self.current_state = None
        self.component = None

    def reset(self):
        """reset

        Reset the environment, and return the initial position

        :returns: Tuple of (initial state, component)
        """
        # Draw from the mixture
        if self.p_mixture is None:
            self.component = np.random.randint(self.n_components)
        else:
            self.component = np.random.choice(
                self.n_components, size=1, p=self.p_mixture.tolist())[0]

        # Draw an initial state
        if self.p_initial_state is None:
            self.current_state = np.random.randint(self.n_states)
        else:
            self.current_state = np.random.choice(
                self.n_states, size=1,
                p=self.p_initial_state[self.component, :].squeeze().tolist())[0]

        return self.current_state, self.component

    def step(self, action):
        """step

        Take a step with the given action

        :action: Integer of the action
        :returns: Tuple of (next_state, reward)
        """
        assert action in range(self.n_actions), "Invalid action!"
        is_term = False

        next_prob = self.tx_mat[
                self.component, action, self.current_state,
                :].squeeze()

        assert np.isclose(next_prob.sum(), 1), "Probs do not sum to 1!"

        next_state = np.random.choice(self.n_states, size=1, p=next_prob)[0]

        reward = self.r_mat[self.component, action,
                            self.current_state, next_state]
        self.current_state = next_state

        # In this MDP, rewards are only received at the terminal state
        if reward != 0:
            is_term = True

        return self.current_state, reward, is_term

    def policyIteration(self, discount=0.9, obs_pol=None, skip_check=False,
            eval_type=1):
        """Calculate the optimal policy for the marginal tx_mat and r_mat,
        using policy iteration from pymdptoolbox

        Note that this function marginalizes over any mixture components if
        they exist.

        :discount: Discount factor for rewards
        :returns: Policy matrix with deterministic policy

        """
        # Define the marginalized transition and reward matrix
        r_mat_obs = self.r_mat.T.dot(self.p_mixture).T
        tx_mat_obs = self.tx_mat.T.dot(self.p_mixture).T

        # Run Policy Iteration
        pi = mdptools.PolicyIteration(
            tx_mat_obs, r_mat_obs, discount=discount, skip_check=skip_check,
            policy0=obs_pol, eval_type=eval_type)
        pi.setSilent()
        pi.run()

        # Convert this (deterministic) policy pi into a matrix format
        pol_opt = np.zeros((self.n_states, self.n_actions))
        pol_opt[np.arange(len(pi.policy)), pi.policy] = 1

        return pol_opt


"""
true_mdp
physician_policy
sample_trajectories
construct_mdp
learn_policy
couple between p and q, where p is the transition distribution under the observed action and q under the rl policy
"""