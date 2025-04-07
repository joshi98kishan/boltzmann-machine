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

def get_encprob_weight_mask(grp_size, num_hnodes):
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


def get_env_states_ranks(env_states, uniq_sorted_states, num_vnodes):
    ranks = []
    for env_state in env_states:
        mask = (uniq_sorted_states[:, :num_vnodes] == env_state).all(axis=1)
        ranks.extend(np.nonzero(mask)[0])

    return ranks

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
    return result