'''
Implementation of Boltzmann Machine as mentioned in the paper - "A Learning Algorithm for Boltzmann Machines" by Ackley, Hinton and Sejnowski.
'''

import math
import numpy as np
from tqdm.notebook import tqdm
from utils import stringify_vec, get_env_states_ranks

class BoltzmannMachine:
    def __init__(self, 
                 env_states, 
                 num_hnodes, 
                 init_W=None, 
                 weight_mask=None, 
                 noisy_clamping=True, 
                 signed_update=True, 
                 logging=True, 
                 num_noisyEnvState_per_envState=20,
                 on_bit_noise = 0.15, 
                 off_bit_noise = 0.05):
        '''
        env_states     : (np array) array of environmental states
        num_hnodes     : (int) number of hidden nodes
        init_W         : (np array) initial weight array, helpful for continuing the learning
        weight_mask    : (np array) represents connections to avoid, e.g. in encoder problem
        noisy_clamping : (bool) whether to use noisy clamping
        signed_update  : (bool) whether to use signed update while updating weights  
        logging        : (bool) whether to log during learning
        num_noisyEnvState_per_envState: (int) number of noisy states to create from a single env states, if noisy clamping is True
        on_bit_noise   : (float) probability to toggle on bit 
        off_bit_noise  : (float) probability to toggle off bit

        Note: this code uses "state" term for two kinds of states - environmental states (corresponding to visible nodes)
        and network states (corresponding to all the nodes). And it should be clear from the context.  
        '''
    
        self.rng = np.random.default_rng()
        self.env_states = env_states
        self.num_vnodes = env_states.shape[1]
        self.num_hnodes = num_hnodes
        self.signed_update = signed_update
        self.logging = logging

        # variables for noisy clamping
        self.noisy_clamping = noisy_clamping
        self.num_noisyEnvState_per_envState = num_noisyEnvState_per_envState
        self.on_bit_noise = on_bit_noise
        self.off_bit_noise = off_bit_noise

        # as mentioned in the paper, "co-occurrences were then gathered for as ```many annealings``` as were used to estimate pij."
        self.num_freeRun_inits = self.num_noisyEnvState_per_envState*self.env_states.shape[0]
    
        # last node is for conversion of bias to weights
        self.num_nodes = self.num_vnodes + self.num_hnodes + 1 
        if init_W is None:
            self.W = np.zeros((self.num_nodes, self.num_nodes))
        else:
            self.W = init_W

        self.weight_mask = weight_mask
       
        # values taken from paper.
        self.sa_sched = [(2, 20), (2, 15), (2, 12), (4, 10)]  # Stochastic Annealing (SA) params - list of tuple of `time` and `temperature`.
        self.equil_time_temp = (10, 10)                       # time and temp to use after annealing
        self.weight_update_magnitude = 2

        # variables to store data for visualization
        self.energy_change_debug = []
        self.clamped_run_debug = []
        self.free_run_debug = []
        self.learning_debug = []

    def get_noisy_env_states(self):
        '''
        Generates the array of noisy env states. Returns different array everytime it is called.
        '''

        noisy_env_states = []
        for env_state in self.env_states:
            prob_arr = np.zeros(self.num_vnodes)
            ones_mask = (env_state==1)
            prob_arr[ones_mask] = self.on_bit_noise
            prob_arr[~ones_mask] = self.off_bit_noise
            
            toggled_env_state = 1-env_state

            for _ in range(self.num_noisyEnvState_per_envState):
                noisy_env_state = env_state.copy()
                rand_nos = self.rng.uniform(0, 1, size=self.num_vnodes)
                toggle_mask = rand_nos<prob_arr
                noisy_env_state[toggle_mask] = toggled_env_state[toggle_mask]

                noisy_env_states.append(noisy_env_state)

        noisy_env_states = np.array(noisy_env_states)
        return noisy_env_states
            
    def get_rand_init_state(self, env_state=None):
        '''
        Generate a random init network state for clamped or unclamped network.
        '''

        if env_state is not None:
            vec = np.zeros(self.num_nodes)
            vec[:self.num_vnodes] = env_state

            rand_hstate = self.rng.integers(0, 2, size=self.num_hnodes)
            vec[self.num_vnodes:-1] = rand_hstate
        else:
            vec = self.rng.integers(0, 2, size=self.num_nodes)
        
        vec[-1] = 1
        return vec

    def sigmoid(self, x, T):
        return 1 / (1 + math.exp(-x/T))
    
    def one_time_run(self, init_state, idxs, T):
        '''
        generates network states in a single unit of "time" for a given temperature T.
        '''

        one_time_states = []

        for _ in range(len(idxs)):
            i = self.rng.choice(idxs, 1).item()
            energy_change = np.dot(self.W[i][:], init_state)
            self.energy_change_debug.append(energy_change)

            p_of_1 = self.sigmoid(energy_change, T)
            
            if self.rng.uniform(0, 1)<p_of_1:
                init_state[i] = 1
            else:
                init_state[i] = 0

            one_time_states.append(init_state.copy())

        return one_time_states
    
    def run(self, init_state, idxs):
        '''
        does Simulated Annealing for the given set of unclamped nodes and then generate samples from the equilibrium distribution.
        init_state: (np array) initial random network state
        idxs      : (np array) indices of unclamped nodes

        "one_time_run()" method changes the "init_state" array in-place.
        '''

        equi_samples = []

        # as mentioned in the paper, network was allowed to reach equilibrium twice.
        for _ in range(2):
            # SA
            for times, T in self.sa_sched:
                for _ in range(times):
                    self.one_time_run(init_state, idxs, T)

            # now it is assumed that it has reached equilibrium, now getting the samples from equilibrium distribution.
            for t in range(self.equil_time_temp[0]):
                equi_samples.extend(self.one_time_run(init_state, idxs, self.equil_time_temp[1]))

        return equi_samples

    def free_run(self):
        '''
        returns p'_i_j for all the connections in the form of symmetric array
        '''

        #********* all the nodes are unclamped
        idxs = np.arange(self.num_vnodes + self.num_hnodes)
        #*********

        equi_samples = []
    
        for _ in range(self.num_freeRun_inits):
            init_state = self.get_rand_init_state()
            equi_samples.extend(self.run(init_state, idxs))    
        
        equi_samples = np.array(equi_samples)
        p_prime = (equi_samples.T@equi_samples)/equi_samples.shape[0]

        dist = self.create_states_dist(equi_samples)
        self.free_run_debug.append(dist)
        
        if self.logging:
            self.free_run_eval(dist)

        return p_prime

    def clamped_run(self):
        '''
        returns p_i_j for all the connections in the form of symmetric array
        '''

        equi_samples = []
        
        # this list stores the "dist" dictionary for each env state 
        debug_list = [] 

        if self.noisy_clamping:
            states_to_clamp = self.get_noisy_env_states()
        else:
            states_to_clamp = self.env_states

        for state_to_clamp in states_to_clamp:
            init_state = self.get_rand_init_state(state_to_clamp)
            #********* only hidden nodes are unclamped
            idxs = np.arange(self.num_vnodes, self.num_vnodes+self.num_hnodes)
            #*********
            clamped_equi_samples = self.run(init_state, idxs)
            equi_samples.extend(clamped_equi_samples)

            if not self.noisy_clamping:
                dist = self.create_states_dist(np.array(clamped_equi_samples))
                debug_list.append(dist)
                if self.logging:
                    self.clamped_run_eval(dist) # such print is particularly helpful for encoder problems


        if self.noisy_clamping:
            for state_to_clamp in self.env_states:
                init_state = self.get_rand_init_state(state_to_clamp)
                #*********
                idxs = np.arange(self.num_vnodes, self.num_vnodes+self.num_hnodes)
                #*********
                clamped_equi_samples = self.run(init_state, idxs)
                
                dist = self.create_states_dist(np.array(clamped_equi_samples))
                debug_list.append(dist)
                
                if self.logging:
                    self.clamped_run_eval(dist)


        equi_samples = np.array(equi_samples)
        p = (equi_samples.T@equi_samples)/equi_samples.shape[0]

        self.clamped_run_debug.append(debug_list)

        return p
    
    def search(self, query):
        '''
        `-1` in the query represents the visible nodes which needs to be searched for.
        return samples from the conditional distribution.
        '''
        mask = (query==-1)
        query[mask] = self.rng.integers(0, 2, size=mask.sum())
        init_state = self.get_rand_init_state(query)
        
        
        idxs_free_vis = np.nonzero(mask)[0] # indices of unclamped visible nodes (which needs to be search for)
        idxs_hid = np.arange(self.num_vnodes, self.num_vnodes+self.num_hnodes)
        
        #*********
        idxs = np.append(idxs_free_vis, idxs_hid)
        #*********

        equi_samples = self.run(init_state, idxs)
        equi_samples = np.array(equi_samples)
        return equi_samples
        
    def learn(self, learning_cycles):
        '''
        learning of weights
        '''
        
        for _ in tqdm(range(learning_cycles)):
            p = self.clamped_run()
            p_prime = self.free_run()
            
            #DIRECTION OF UPDATE - whether to inc or dec
            if self.signed_update:
                direction = np.sign(p-p_prime)
            else:
                direction = (p-p_prime)

            # removing the unwanted connections by filling with zero.
            np.fill_diagonal(direction, 0)
            if self.weight_mask is not None:
                direction[self.weight_mask] = 0

            self.learning_debug.append({
                'W': self.W,
                'p': p, 
                'p_prime': p_prime, 
                'direction': direction, 
            })
            
            #WEIGHT UPDATE
            self.W = self.W + direction*self.weight_update_magnitude
  
    def calc_energy(self, state):
        '''
        returns energy of the given state
        '''
        energy = -(state@(self.W@state))*0.5
        return energy
    
    def create_states_dist(self, equi_samples):
        '''
        Calculates counts of each unique states in the sample. Along with this, it also return energy of the unique states.
        "dist" here refers to the discrete distribution of states in terms of counts.
        '''
        
        equi_dist = np.unique(equi_samples, axis=0, return_counts=True)
        states, state_counts = equi_dist
        state_labels = [stringify_vec(state) for state in states]
        state_energies = [self.calc_energy(state) for state in states]

        return {
            'states': states,
            'state_labels': state_labels,
            'state_counts': state_counts,
            'state_energies': state_energies
        }
    
    def free_run_eval(self, dist, print_eval=True):
        '''
        prints rank of env states in the sorted (by energy) states array.
        '''

        # sort dist states by energy
        sorted_energies = np.array(sorted(zip(dist['state_energies'], 
                                              np.arange(len(dist['state_energies']))), 
                                          key=lambda x: x[0])
                                  )
        
        sorted_states = dist['states'][sorted_energies[:, 1].astype(int)]

        # printing the ranks of env states in the sorted states array
        ranks = get_env_states_ranks(self.env_states, sorted_states, self.num_vnodes)
        ranks = sorted(ranks)

        if print_eval:
            print(ranks)
    
        return ranks

    def clamped_run_eval(self, dist, print_eval=True):
        '''
        print the hidden state corresponding to most frequent occuring network state in the distribution
        '''
        max_count_hid_state = dist['states'][dist['state_counts'].argmax()][-1-self.num_hnodes:-1]
    
        if print_eval:
            print(max_count_hid_state, end=', ')
        
        return max_count_hid_state

    