import numpy as np
import matplotlib.pyplot as plt

def Calculate_final_weight_HNN(weights_of_blocks,show_weights=False,save=''):
    weight_matrix=np.zeros(np.size(weights_of_blocks,axis=1))
    for i in range(np.shape(weights_of_blocks)[0]):
        weight_matrix=weight_matrix+weights_of_blocks[i]
    
    if show_weights:

        plt.plot(weight_matrix)
        plt.title("Final weight matrix")
        
        if save!='':
            plt.savefig(save,dpi=300) 
        plt.show()
    return weight_matrix