#%%
# import sys
# sys.path.append('../hopfield_network\deciphering-hopfield-network')

import math
import numpy as np
# from src.hopfield_network import BaseHopfieldNetwork
from tqdm.notebook import tqdm
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


#%%
v1_h1 = 1
h1_h2 = 1
h2_h3 = 1
h3_h4 = 1
v2_h4 = 1
W_ = np.array([
    [0, 0, v1_h1, 0,     0,     0],
    [0, 0, 0,     0,     0,     v2_h4],
    [0, 0, 0,     h1_h2, 0,     0],
    [0, 0, 0,     0,     h2_h3, 0],
    [0, 0, 0,     0,     0,     h3_h4],
    [0, 0, 0,     0,     0,     0],
])
W = W_ + W_.T
W


#%%


class OldBoltzmannMachine:
    def __init__(self, thinking_steps):
        self.W = W
        self.OFF_ACT_VAL = 0
        self.thinking_steps = thinking_steps

    def remember(self, x):
        x = x.copy()
        x_prev = x.copy()
        self.progression = []
        # self.local_fields = []

        for t in (range(self.thinking_steps)):
            for i in range(len(x)):
                i_local_field = np.dot(self.W[i][:], x)
                if i_local_field > 0:
                    x[i] = 1
                elif i_local_field < 0:
                    x[i] = self.OFF_ACT_VAL

                # self.local_fields.append(i_local_field)
            self.progression.append(x)
            stop = np.equal(x, x_prev).all()
            if stop:
                # print('stopping at', t)
                break
            else:
                x_prev = x.copy()

        return x
    


#%%
steps = 100
pop_size = 1000
num_vis = 2
num_hidd = 4
num_nodes = num_vis + num_hidd 

rng = np.random.default_rng()
init_states = rng.integers(0, 2, size=(pop_size, num_nodes))

bm = OldBoltzmannMachine(steps)

dist = defaultdict(list)
for state in init_states:
    x_rem = bm.remember(state)
    # dist[tuple(x_rem[:2])] += 1
    dist[tuple(x_rem[:2])].append(x_rem)
dist

#%%
###################### Solving Parity problem with BM #################################

class BoltzmannMachine:
    def __init__(self, env_states, num_hnodes):
        self.rng = np.random.default_rng()
        self.env_states = env_states
        self.num_vnodes = env_states.shape[1]
        self.num_hnodes = num_hnodes
        
        self.num_nodes = self.num_vnodes + self.num_hnodes + 1 # last node is for conversion of bias to weights
        self.W = np.zeros((self.num_nodes, self.num_nodes))
        self.sa_sched = [(2, 20), (2, 15), (2, 12), (4, 10)] # list of tuple of `time` and `temperature`
        self.equil_time_temp = (10, 10)
        self.weight_update_magnitude = 2 # refer paper - pp. 12

        self.free_run_debug = []
        self.energy_change_debug = []

    def get_rand_init_state(self, env_state=None):
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
        one_time_states = []

        # self.rng.shuffle(idxs) # this is exactly not equal to the definition of time unit in the paper.
        # for i in idxs:
        for _ in range(len(idxs)): # this follows the definition of `time`
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
        equi_states = []

        # as mentioned in the paper, network was allowed to reach equilibrium twice (pp. 11).
        for i in range(2):
            #SA
            for times, T in self.sa_sched:
                for t in range(times):
                    self.one_time_run(init_state, idxs, T)

            # now it is assumed that it has reached equilibrium, now getting the stats at equilibrium.
            for t in range(self.equil_time_temp[0]):
                equi_states.extend(self.one_time_run(init_state, idxs, self.equil_time_temp[1]))

        return equi_states


    def free_run(self):
        init_state = self.get_rand_init_state()
        
        #*********
        idxs = np.arange(self.num_vnodes + self.num_hnodes)
        #*********
        
        equi_states = self.run(init_state, idxs)    
        equi_states = np.array(equi_states)
        p_prime = (equi_states.T@equi_states)/equi_states.shape[0]

        self.free_run_debug.append({
            'equi_states': equi_states,
            'energies': np.array([self.calc_energy(state_) for state_ in equi_states]),
            'W': self.W.copy()
        })

        return p_prime

    def clamped_run(self):
        equi_states = []
        for env_state in self.env_states:
            init_state = self.get_rand_init_state(env_state)
            
            #*********
            idxs = np.arange(self.num_vnodes, self.num_vnodes+self.num_hnodes)
            #*********
            
            clamped_equi_states = self.run(init_state, idxs)
            equi_states.extend(clamped_equi_states)

        equi_states = np.array(equi_states)
        p = (equi_states.T@equi_states)/equi_states.shape[0]
        return p
    
    def search(self, query):
        '''
        `-1` in the query represents the nodes which needs to be filled.
        '''
        mask = (query==-1)
        query[mask] = self.rng.uniform(0, 2, size=mask.sum())
        init_state = self.get_rand_init_state(query)
        
        idxs_free_vis = np.nonzero(mask)[0]
        idxs_hid = np.arange(self.num_vnodes, self.num_vnodes+self.num_hnodes)
        
        #*********
        idxs = np.append(idxs_free_vis, idxs_hid)
        #*********

        equi_states = self.run(init_state, idxs)
        equi_states = np.array(equi_states)
        return equi_states
        

    
    def learn(self, learning_cycles):
        W_progression = []
        for _ in tqdm(range(learning_cycles)):
            p_prime = self.free_run()
            p = self.clamped_run()
            
            #DIRECTION OF UPDATE - whether to inc or dec
            direction = np.sign(p-p_prime)
            np.fill_diagonal(direction, 0)
            
            #WEIGHT UPDATE by fixed magnitude
            self.W = self.W + direction*self.weight_update_magnitude
            W_progression.append(self.W.copy())
            
        return W_progression
    
    def calc_energy(self, state):
        energy = -(state@(self.W@state))*0.5
        return energy

#%%
env_states = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
    ])

all_states = np.array([[0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 1],
                    [0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1]])

num_hid_nodes = 1
bm = BoltzmannMachine(env_states, num_hid_nodes)
bm.equil_time_temp = (10000, 3)
bm.weight_update_magnitude = 1

Ws = bm.learn(20)
#%%

with PdfPages('ws_plots.pdf') as pdf:
    for W_ in Ws:
        plt.hist(W_.flatten())
        pdf.savefig()
        plt.close()

with PdfPages('plots.pdf') as pdf:
    for debug_dict in bm.free_run_debug:
        # bm_test = BoltzmannMachine(env_states, 1)
        # bm_test.W = debug_dict['W']
        plt.hist(debug_dict['energies'])
        # plt.show()
        # plt.title('Line Plot')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.legend()
        pdf.savefig()  # Save the current figure into the PDF
        plt.close()  # Close the figure


print('PDFs generated!')
#%%
# testing
def parity_problem_testing(env_states, bm):
    test_res = []
    for query in env_states:
        query = query.copy()
        ans = query[2]
        query[2] = -1
        print(query, end=', ')


        res = bm.search(query)
        uniqs, uniqs_counts = np.unique(res[:, 2], return_counts=True)
        pred = uniqs[np.argmax(uniqs_counts)]
        is_correct = (pred==ans)
        print(uniqs, uniqs_counts, is_correct)
        test_res.append(is_correct)

    print(f'Result: {sum(test_res)}/4')
# %%


for i, W_ in enumerate(Ws[-5:]):
    print(f'#{i}')
    bm_test = BoltzmannMachine(env_states, 1)
    bm_test.equil_time_temp = (10000, 3)
    bm_test.W = W_
    parity_problem_testing(env_states, bm_test)
    print('-'*40)
    

# %%
# gt_W = np.array([
#     [0, 0, 1, 1, 0],
#     [0, 0, 1, 1, 0],
#     [1, 1, 0, -2, -0.5],
#     [1, 1, -2, 0, -1.1],
#     [0, 0, -0.5, -1.1, 0]
# ])

# GT from paper
gt_W = np.array([
    [0, -5, 5, 11, -3],
    [-5, 0, 5, 11, -3],
    [5, 5, 0, -11, -2],
    [11, 11, -11, 0, -11],
    [-3, -3, -2, -11, 0]
])

gt_bm = BoltzmannMachine(env_states, 1)
gt_bm.W = gt_W
gt_bm.equil_time_temp = (10000, 3)

# parity_problem_testing(env_states, gt_bm)


#%%
#------ energy distribution
query = np.ones(3)*-1
equi_states = gt_bm.search(query)

energies = np.array([gt_bm.calc_energy(state) for state in equi_states])
# np.unique(energies, return_counts=True)
plt.hist(energies)
plt.show()
# %%

equi_dist = np.unique(equi_states, axis=0, return_counts=True)
states, state_counts = equi_dist

for state, count in zip(states, state_counts):
    print(state, count, gt_bm.calc_energy(state))
# %%



# %%
