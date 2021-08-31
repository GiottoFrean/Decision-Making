import numpy as np

def value_iteration(rewards_array,action_transition_dict,discount,iterations):
    current_utility = np.zeros(rewards_array.shape[1])
    action_names = list(action_transition_dict.keys())
    
    for i in range(iterations):
        action_utility_matrix = np.zeros((len(action_transition_dict),rewards_array.shape[1]))
        for a,action in enumerate(action_names):
            action_utility_matrix[a]=discount*action_transition_dict[action].T.dot(current_utility)+rewards_array[a]
        current_utility = np.max(action_utility_matrix,axis=0)    

    # now get policy
    expected_returns = [action_transition_dict[action].T.dot(current_utility) for action in action_names]
    best_policy = np.argmax(np.concatenate([r.reshape(-1,1) for r in expected_returns],axis=1),axis=1)
    new_policy = [action_names[d] for d in best_policy]
    return current_utility,new_policy
