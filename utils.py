import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def stringify_vec(vec):
    s = np.array2string(vec)
    s = s.replace('.', '').replace('[', '').replace(']', '').replace(' ', '')
    return s

def plot_sigmoid(max_x, T):
    '''
    plots sigmoid in the domain: (-max_x, max_x)
    "max_x" should be positve.
    '''
    x = np.arange(-max_x, max_x)
    y = 1 / (1 + np.exp(-x/T))
    plt.plot(x, y)

def get_boltzmann_distribution(energies, T):
    boltzmann_factors = np.exp(-(energies/T))
    Z = boltzmann_factors.sum()
    probs = boltzmann_factors/Z
    return probs

def get_probs_from_counts(counts):
    return counts/sum(counts)

def get_encprob_env_states(grp_size):
    '''
    return env states array for a given visble nodes group size.
    '''
    zero_vec = np.zeros(grp_size)
    env_states = []
    for i in range(grp_size):
        half_part = zero_vec.copy()
        half_part[i] = 1
        env_states.append(np.append(half_part, half_part))

    return np.array(env_states, dtype=int)

def get_encprob_weight_mask(grp_size, num_hnodes, winner_takeAll_stage=False):
    '''
    returns a mask with shape same as weights array with "True"
    representing the connections to avoid.
    '''
    num_vnodes = 2*grp_size 
    num_nodes = num_vnodes + num_hnodes + 1
    mask = np.zeros((num_nodes, num_nodes), dtype=bool)

    mask[:grp_size, grp_size:num_vnodes] = True
    mask[grp_size:num_vnodes, :grp_size] = True
    mask[num_vnodes:num_vnodes+num_hnodes, num_vnodes:num_vnodes+num_hnodes] = True

    if winner_takeAll_stage:
        mask[:num_vnodes, num_vnodes:num_vnodes+num_hnodes] = True
        mask[num_vnodes:num_vnodes+num_hnodes, :num_vnodes] = True
        mask[num_vnodes:num_vnodes+num_hnodes, -1] = True

    return mask

def get_shiftprob_weight_mask(grp_size, num_hnodes):
    num_vnodes = 2*grp_size + 3 
    num_nodes = num_vnodes + num_hnodes + 1
    mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    
    mask[num_vnodes:num_vnodes+num_hnodes, num_vnodes:num_vnodes+num_hnodes] = True

    return mask


def gen_free_state_dist_pdf(bm, pdf_name, maxnum_bars=15):
    with PdfPages(pdf_name) as pdf:
        for debug_dict in bm.free_run_debug:

            num_states = debug_dict['states'].shape[0]
            # if num_states < 16:
            #     new_states = all_states - set(debug_dict['state_labels'])
            #     debug_dict['state_labels'].extend(new_states)
            #     debug_dict['state_counts'].extend

            fig = plt.figure(figsize= (15, 5))    
            dist = np.array(sorted(zip(debug_dict['state_labels'], 
                                    debug_dict['state_counts'], 
                                    debug_dict['state_energies'],
                                    np.arange(num_states)), 
                                key=lambda x: x[2]))
            
            # keeping only top `maxnum_bars` no. of bars.
            top_dist = dist[:maxnum_bars]
            low_dist = dist[maxnum_bars:]
            top_dist_min_count = top_dist[:, 1].astype(int).min() 
            low_dist_counts = low_dist[:, 1].astype(int)
            
            if low_dist.shape[0]!=0 and top_dist_min_count<low_dist_counts.max():
                max_idx = low_dist_counts.argmax()
                last_bar_data = low_dist[max_idx]
                top_dist = np.append(top_dist, [last_bar_data], axis=0)
                print('Extra bar! ', end=', ')

            sorted_states = debug_dict['states'][top_dist[:, 3].astype(int)]
            ranks = get_env_states_ranks(bm.env_states, sorted_states, bm.num_vnodes)
            # print(min(ranks), max(ranks))
            print(sorted(ranks))

            sorted_energies = top_dist[:, 2].astype(float)
            norm = plt.Normalize(min(sorted_energies), max(sorted_energies))
            # Create a colormap
            cmap = plt.cm.viridis
            # Map the real values to colors
            colors = cmap(norm(sorted_energies))

            edgecolors = np.array(['black']*top_dist.shape[0])
            edgecolors[ranks] = 'g'

            plt.bar(top_dist[:, 0], 
                    top_dist[:, 1].astype(int), 
                    width=0.5, 
                    color=colors,
                    edgecolor = edgecolors,
                    linewidth = 3)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([]) # What is this?
            fig.colorbar(sm)

            # plt.show()
            # plt.title('Line Plot')
            # plt.xlabel('X-axis')
            # plt.ylabel('Y-axis')
            # plt.legend()
            pdf.savefig()  # Save the current figure into the PDF
            plt.close()  # Close the figure


def gen_clamped_state_dist_pdf(bm, pdf_name, ):
    with PdfPages(pdf_name) as pdf:
        for debug_list in bm.clamped_run_debug:
            fig = plt.figure(figsize= (15, 5))    

            for i, debug_dict in enumerate(debug_list):
                dist = np.array(sorted(zip(debug_dict['state_labels'], 
                                           debug_dict['state_counts'], 
                                           debug_dict['state_energies']), 
                                      key=lambda x: x[2]))
            
                # print(dist.shape, end=', ')
            
                sorted_energies = dist[:, 2].astype(float)
                norm = plt.Normalize(min(sorted_energies), max(sorted_energies))
                # Create a colormap
                cmap = plt.cm.viridis
                # Map the real values to colors
                colors = cmap(norm(sorted_energies))

                plt.subplot(2, 2, i+1)
                plt.bar(dist[:, 0], dist[:, 1].astype(int), width=0.5, color=colors)
            
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm)

            pdf.savefig()  # Save the current figure into the PDF
            plt.close()  # Close the figure

def gen_learning_plots_pdf(bm, pdf_name='learning_plots.pdf'):
    with PdfPages(pdf_name) as pdf:
        for debug_dict in bm.learning_debug:
            fig = plt.figure(figsize= (15, 5))    

            for i, item in enumerate(debug_dict.items()):
            
                # sorted_energies = dist[:, 2].astype(float)
                # norm = plt.Normalize(min(sorted_energies), max(sorted_energies))
                # Create a colormap
                # cmap = plt.cm.viridis
                # Map the real values to colors
                # colors = cmap(norm(sorted_energies))

                ax = plt.subplot(2, 2, i+1,)
                im = plt.imshow(item[1])
                plt.title(item[0])
                plt.colorbar()
            
                # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                # sm.set_array([])
                # fig.colorbar(sm)

            pdf.savefig()  # Save the current figure into the PDF
            plt.close()  # Close the figure


def plot_series(series_data, figsize=(10, 4)):
    plt.figure(figsize=figsize)
    for i in range(series_data.shape[1]):
        plt.plot(series_data[:, i])
    plt.show()

def generate_binary_vectors(n):
    """Generate all binary vectors of length n."""
    if n <= 0:
        return []
    
    result = []
    def helper(current_vector, length):
        if length == n:
            result.append(current_vector)
            return
        helper(current_vector + [0], length + 1)
        helper(current_vector + [1], length + 1)
    
    helper([], 0)
    return np.array(result)

def get_env_states_ranks(env_states, uniq_sorted_states, num_vnodes):
    ranks = []
    for env_state in env_states:
        mask = (uniq_sorted_states[:, :num_vnodes] == env_state).all(axis=1)
        ranks.extend(np.nonzero(mask)[0])

    return sorted(ranks)

def get_free_run_score(free_run_dist, env_states, num_vnodes):
    num_env_states = env_states.shape[0]

    sorted_energies = np.array(sorted(zip(free_run_dist['state_energies'], 
                                            np.arange(len(free_run_dist['state_energies']))), 
                                        key=lambda x: x[0])
                                )
    sorted_states = free_run_dist['states'][sorted_energies[:, 1].astype(int)]
    
    res = []
    for env_state in env_states:
        mask = (sorted_states[:num_env_states, :num_vnodes] == env_state).all(axis=1)
        res.append(mask.any())

    return np.array(res).sum()

def get_free_run_scores(bm):
    free_run_scores = []
    num_env_states = bm.env_states.shape[0]

    for free_run_dist in bm.free_run_debug:
        sorted_energies = np.array(sorted(zip(free_run_dist['state_energies'], 
                                                np.arange(len(free_run_dist['state_energies']))), 
                                            key=lambda x: x[0])
                                    )


        is_energy_lower = (sorted_energies[num_env_states-1, 0]<sorted_energies[num_env_states, 0])

        if (get_free_run_score(free_run_dist, bm.env_states, bm.num_vnodes)==num_env_states) and is_energy_lower:
            free_run_scores.append(True)
        else:
            free_run_scores.append(False)

    return np.array(free_run_scores)

def get_maxCount_hiddState(clamped_dist, num_hnodes):
    return clamped_dist['states'][clamped_dist['state_counts'].argmax()][-1-num_hnodes:-1]

def get_clamped_run_score(clamped_debug_list, num_hnodes):
    env_hidd_states = []
    for clamped_dist in clamped_debug_list:
        max_count_hidd_state = get_maxCount_hiddState(clamped_dist, num_hnodes)
        env_hidd_states.append(max_count_hidd_state)

    score = len(np.unique(env_hidd_states, axis=0))
    return score

def get_clamped_run_scores(bm):
    '''
    return number of unique hidden states in each cycle.
    '''
    uniqs_len_list = []
    for clamped_debug_list in (bm.clamped_run_debug):
        uniqs_len_list.append(get_clamped_run_score(clamped_debug_list, bm.num_hnodes))

    return np.array(uniqs_len_list)

def get_shifterProb_envStates(grp_size):
    env_states = []
    first_grp = generate_binary_vectors(grp_size)
    
    for shift in [-1, 0, 1]:
        sec_grp = np.roll(first_grp, shift=shift, axis=1)
        
        shift_label_arr = np.zeros((first_grp.shape[0], 3), dtype=int)
        shift_label_arr[:, shift+1] = 1

        env_states.extend(np.concatenate((first_grp, sec_grp, shift_label_arr), axis=1))

    env_states = np.array(env_states)
    return env_states


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