import factors
import numpy as np

def get_log_likelihood(all_factors,known_vars,evidence):
	prob = 0
	for f in all_factors:
		f_known_values = [evidence[known_vars.index(name)] for name in f.names if name in known_vars]
		f_unknown_vars = [name for name in f.names if not name in known_vars]
		if(len(f_unknown_vars)>0):
			f_marg = factors.marginalize(f,f_unknown_vars)
			if(not isinstance(f_marg,(int,float))):
				prob+=np.log(f_marg.get(f_known_values))
		else:
			prob+=np.log(f.get(f_known_values))
	return prob

# SUM PRODUCT
# This runs the sum product algorithm for variable elimination. 
# Every name in known_vars has a piece of evidence associated with it.
# Every variable in unknown_vars is marginalized. Returns a joint distribution over what is left.
# The order of unknown_vars is the order in which marginalization happens.

def sum_product_variable_elimination(all_factors,known_vars,evidence,unknown_vars):
	# Step 1: condition by factor by setting each known variable to each piece of evidence
	new_factors = []
	for f in all_factors:
		deleted_f = factors.drop_variables(f,known_vars,evidence)
		if(deleted_f!=None):
			new_factors.append(deleted_f)
	
	# Step 2: marginalize all unknown variables. This requires merging all factors with the same variable name.
	for unknown_var in unknown_vars:
		factors_to_combine = []
		factors_to_exclude = []
		for f in new_factors:
			if(unknown_var in f.names):
				factors_to_combine.append(f)
			else:
				factors_to_exclude.append(f)
		# left over factors are the ones without the unknown variable and the marginalized product
		new_factors = factors_to_exclude
		if(len(factors_to_combine)>0):
			combined_factor = factors.multiple_factor_product(factors_to_combine)
			combined_factor = factors.marginalize(combined_factor,[unknown_var])
			# If the resulting table is a single number, then all variables were marginalized, which means total independence, so ignore.
			if(not isinstance(combined_factor,(int,float))):
				new_factors.append(combined_factor)
				
	# merge all remaining factors
	final_combined_factor = factors.multiple_factor_product(new_factors)
	final_normalized_factor = factors.condition(final_combined_factor)
	return final_normalized_factor

# Does variable elimination by constructing full factor
def full_joint_elimination(all_factors,known_vars,evidence,unknown_vars):
	full_joint_factor = factors.multiple_factor_product(all_factors)
	set_vars = factors.drop_variables(full_joint_factor,known_vars,evidence)
	marginalized = factors.marginalize(set_vars,unknown_vars)
	if(marginalized!=None):
		normalized = factors.condition(marginalized)
		return normalized
	else:
		normalized = factors.condition(set_vars)
		return normalized

# Learns a directed model MLE parameters, using the EM algorithm.
def learn_directed_PGM_EM(prior_factors,data_variable_names,data,iterations):
	old_factors = prior_factors
	for iteration in range(iterations):
		log_likelihood = 0
		new_factors = [f.copy_zeros() for f in prior_factors]
		for x in data:
			known_names = [data_variable_names[v] for v in range(len(data_variable_names)) if x[v]!=-1]
			known_evidence = x[x!=-1]
			if(len(known_names)<len(data_variable_names)):
				# E step: Infer the distribution over unknowns
				infered_factor = sum_product_variable_elimination(old_factors,known_names,known_evidence,[])
				# go through every possible option
				for unknown_vector in infered_factor.indexes:
					# get the full evidence vector, filling in the missing values.
					infered_factor_to_sample_index_transform = [data_variable_names.index(v) for v in infered_factor.names]
					full_evidence = x.copy()
					full_evidence[infered_factor_to_sample_index_transform]=unknown_vector
					probability = infered_factor.get(unknown_vector)
					# For each factor update the counts at the given index.
					for j in range(len(old_factors)):
						# have to first make sure the indexing is the same as the order of variables can change. 
						factor_to_sample_index_transform = [data_variable_names.index(name) for name in old_factors[j].names]
						rearanged_index = full_evidence[factor_to_sample_index_transform]
						old_value = new_factors[j].get(rearanged_index)
						new_factors[j].set(rearanged_index,old_value+probability)
			else:
				for j in range(len(old_factors)):
					factor_to_sample_index_transform = [data_variable_names.index(name) for name in old_factors[j].names]
					rearanged_index = known_evidence[factor_to_sample_index_transform]
					old_value = new_factors[j].get(rearanged_index)
					new_factors[j].set(rearanged_index,old_value+1)
			log_likelihood += get_log_likelihood(old_factors,known_names,known_evidence)
		print("log likelihood",log_likelihood)
		old_factors = [factors.condition(f,axis=f.names[1:]) for f in new_factors]
	return old_factors
