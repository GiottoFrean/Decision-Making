import numpy as np
import factors

# Samples by setting the variables in order from top to bottom. Requires a directed factor graph. This is probably not going to work for an undirected graph.
def joint_sample_top_down(all_factors):
    assigned_variable_names = []
    variable_assignments = []
    remaining_factors = all_factors.copy()
    
    while(len(remaining_factors)>0): #  while variables haven't been assigned
        new_assigned_variable_names = assigned_variable_names.copy()
        new_variable_assignments = variable_assignments.copy()
        new_remaining_factors = []
        for f in remaining_factors:
            # grab variables with just themselves in the factor, and every variable which has all parent values set.
            if(len(f.names)==1 or np.prod([i in assigned_variable_names for i in f.names[1:]])==1):
                conditioned_factor = factors.drop_variables(f,assigned_variable_names,variable_assignments)
                new_variable_assignments.append(factors.sample(conditioned_factor,1)[0][0]) # sample a new value
                new_assigned_variable_names.append(f.names[0])
            else:
                new_remaining_factors.append(f)
        assigned_variable_names = new_assigned_variable_names
        variable_assignments = new_variable_assignments
        remaining_factors = new_remaining_factors
    return assigned_variable_names,variable_assignments

# Likelihood weighted sampling. For normalized importance sampling to pgms. Similar to above but sets all observed variables and returns weight.
def likelihood_weighting_top_down(all_factors,known_vars,evidence):
    assigned_variable_names = []
    variable_assignments = []
    weight = 1
    remaining_factors = all_factors.copy()
    
    while(len(remaining_factors)>0):
        new_assigned_variable_names = assigned_variable_names.copy()
        new_variable_assignments = variable_assignments.copy()
        new_remaining_factors = []
        for f in remaining_factors:
            if(len(f.names)==1 or np.prod([i in assigned_variable_names for i in f.names[1:]])==1):
                var_dropped_factor = factors.drop_variables(f,assigned_variable_names,variable_assignments)
                conditioned_factor = factors.condition(var_dropped_factor)
                if(f.names[0] in known_vars):
                    evid = evidence[known_vars.index(f.names[0])]
                    new_variable_assignments.append(evid)
                    weight *= conditioned_factor.get([evid])
                else:
                    sample = factors.sample(conditioned_factor,1)[0][0]
                    new_variable_assignments.append(sample)
                    
                new_assigned_variable_names.append(f.names[0])
            else:
                new_remaining_factors.append(f)
        assigned_variable_names = new_assigned_variable_names
        variable_assignments = new_variable_assignments
        remaining_factors = new_remaining_factors
    return assigned_variable_names,variable_assignments,weight

# Gibbs sampling. Done with a step method and a method to generate lots of samples. 
# Quite complicated, but mainly because of indexing issues. I pass the markov blanket factors so they don't need to be recalculated.
# The markov blankets are just the factor product of all variables connected to a node. The gibbs step updates a sample in place.
def gibbs_step(all_variable_markov_blankets,fixed_variables,all_variable_names,current_state_values):
    for var_name in all_variable_names:
        if(not var_name in fixed_variables):
            joint = all_variable_markov_blankets[all_variable_names.index(var_name)]
            index = all_variable_names.index(var_name)
            other_var_names = list(all_variable_names[:index])+list(all_variable_names[(index+1):])
            other_var_vals = list(current_state_values[:index])+list(current_state_values[(index+1):])
            slc = [slice(None)]*len(joint.names)
            for i in range(len(all_variable_names)):
                if(all_variable_names[i] in joint.names):
                    slc[joint.names.index(all_variable_names[i])]=slice(current_state_values[i],current_state_values[i]+1)
            joint_index = joint.names.index(var_name)
            slc[joint_index]=slice(0,joint.array.shape[joint_index])
            array_slice = np.squeeze(joint.array[tuple(slc)])
            norm_array_slice = array_slice/np.sum(array_slice)
            cond_joint = factors.drop_variables(joint,other_var_names,other_var_vals)
            sample = np.random.choice(np.arange(joint.array.shape[joint_index]),1,p=norm_array_slice)
            #print(sample)
            current_state_values[index]=sample
    return current_state_values

# Gibbs sampling begins by making a random vector of values and then applies the gibbs step repeatedly. 
def gibbs_sampling(all_factors,known_vars,evidence,N):
    all_names = []
    # All this below is just to make a general random vector for a factor (and set the evidence)
    for f in all_factors:
        all_names+=list(f.names)
    all_names = list(np.unique(all_names))
    current_state = np.zeros(len(all_names)).astype(int) # start with 0's
    for i,name in enumerate(all_names):
        for f in all_factors: # find a factor to get the sample
            if(name in f.names):
                shape = f.array.shape[list(f.names).index(name)]
                current_state[i]=np.random.randint(0,shape)
                break
    for i in range(len(known_vars)):
        current_state[all_names.index(known_vars[i])]=evidence[i]
    
    # This is for making the markov blanket factors
    all_variable_markov_blankets = []
    for var_name in all_names:
        markov_blanket = []
        for f in all_factors:
            if(var_name in f.names):
                markov_blanket.append(f)
        all_variable_markov_blankets.append(factors.multiple_factor_product(markov_blanket))
    
    # This is the core loop
    all_visited_states = []
    for n in range(N):
        current_state = gibbs_step(all_variable_markov_blankets,known_vars,all_names,current_state)
        all_visited_states.append(current_state.copy())
    return all_names,np.array(all_visited_states)
    
