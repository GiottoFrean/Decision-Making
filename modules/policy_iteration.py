import numpy as np

# gets the policies given the value function.
def get_finite_best_policies(action_transition_dict,utilities,num_steps):
	policies = []
	action_names = list(action_transition_dict.keys())
	for step in range(num_steps):
		utility = utilities[step+1] # I include the utility of the starting state, so skip here.
		expected_returns = [action_transition_dict[action].T.dot(utility) for action in action_names]
		best_policy = np.argmax(np.concatenate([r.reshape(-1,1) for r in expected_returns],axis=1),axis=1)
		new_policy = [action_names[d] for d in best_policy]
		policies = policies + [new_policy]
	return policies

# gets the utilities given the policies.
def get_finite_utilities(action_transition_dict,policies,reward,num_steps):
	utilities = [reward.copy()] # include the reward for the last policy.
	num_states = len(reward)
	for step in reversed(range(num_steps)): # step backward from last policy to first.
		transition = np.zeros((num_states,num_states))
		for state in range(num_states):
			transition[:,state] = action_transition_dict[policies[step][state]][:,state]
		utility = reward + transition.T.dot(utilities[0]) # as going backward, [0] is the utility of the next states.
		utilities = [utility] + utilities 
	return utilities

# runs policy iteration for the finite case
def run_policy_iteration_finite(action_transition_dict,reward,num_steps,iterations):
	action_names = list(action_transition_dict.keys())
	num_states = len(reward)
	policies = [[action_names[np.random.randint(0,len(action_names))] for i in range(num_states)] for step in range(num_steps)]
	for iteration in range(iterations):
		utilities = get_finite_utilities(action_transition_dict,policies,reward,num_steps)
		policies = get_finite_best_policies(action_transition_dict,utilities,num_steps)
	return utilities,policies


def get_infinite_best_policy(action_transition_dict,utility):
	action_names = list(action_transition_dict.keys())
	expected_returns = [action_transition_dict[action].T.dot(utility) for action in action_names]
	best_policy_index = best_policy = np.argmax(np.concatenate([r.reshape(-1,1) for r in expected_returns],axis=1),axis=1)
	new_policy = [action_names[d] for d in best_policy]
	return new_policy

def get_infinite_utility(action_transition_dict,policy,reward,discount):
	num_states = len(reward)
	transition = np.zeros((num_states,num_states))
	for state in range(num_states):
		transition[:,state] = action_transition_dict[policy[state]][:,state]
	utility = np.linalg.inv(np.eye(len(reward))-discount*transition.T).dot(reward)
	return utility

def run_infinite_policy_iteration(action_transition_dict,reward,discount,iterations):
	action_names = list(action_transition_dict.keys())
	num_states = len(reward)
	policy = [action_names[np.random.randint(0,len(action_names))] for i in range(num_states)]
	for iteration in range(iterations):
		utility = get_infinite_utility(action_transition_dict,policy,reward,discount)
		policy = get_infinite_best_policy(action_transition_dict,utility)
	return utility,policy



