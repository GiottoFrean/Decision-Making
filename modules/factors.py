import numpy as np

# This is code to run factors in numpy.
# The class is fairly simple, just contains:
# 1. the array which has the values at each possible combination of variables. One dimension for each variable.
# 		e.g [[0.2,0.4]
#			 [0.1,0.3]]
# 2. the index which has the index of each combination of variables. This is a 2d array which lists all possible variable combinations.
# 		e.g [[0,0]
#			 [0,1]
#			 [1,0]
#			 [1,1]]
# 3. the names of each variable e.g ["A","B"]
# The reason for having the index is it makes it easier to edit values and easier to print the factor.

class Factor:
	def __init__(self,names,pos_values):
		array = np.zeros((pos_values))
		indexes = np.zeros((np.prod(array.shape),len(names))).astype(int)
		f = 1
		for i in range(len(names)):
			# fills the given column in the indexes
			repeated = np.arange(array.shape[i]).repeat(np.prod(array.shape)/(f*array.shape[i]))
			indexes[:,i]=np.tile(repeated,f)
			f = f*array.shape[i]
		self.names = names
		self.array = array
		self.indexes = indexes
		
	# show the array in a nice format.
	def __repr__(self):
		array = self.array
		indexes = self.indexes
		names = self.names
		name_lengths = [len(str(n)) for n in names]
		formatter = "".join(["{:<"+str(l+2)+"}" for l in name_lengths])+"{}"
		strings = []
		strings.append(formatter.format(*(names+["Values (10 dp)"])))
		for i in range(indexes.shape[0]):
			val = array[tuple(indexes[i])].round(10)
			strings.append(formatter.format(*(list(indexes[i])+[val])))
		return "".join([s+"\n" for s in strings])
		
	# set the value at a given position e.g myfactor.set([0,1],0.3)
	def set(self,index,value):
		ndim = len(self.names)
		if(len(index)==ndim):
			self.array[tuple(index)]=value
		else:
			raise Exception('length of index is incorrect. Provide {} values'.format(ndim))

	# get the value at a given index e.g myfactor.get([0,1])
	def get(self,index):
		return self.array[tuple(index)]
	
	# just sets all the values in the same order as printed. Useful when testing and doing things in as few lines as possible. 
	def set_all(self,values):
		self.array = np.array(values).reshape(self.array.shape,order="C")
		
# There are four major pieces of code to know:

# 1. FACTOR CONDITIONING
# 		returns the same factor but ensures the sum is 1 along a given set of axes.	
def condition(factor,axis="none"):
	array = factor.array
	indexes = factor.indexes
	names = factor.names
	# if axis is none, then normalize the whole array so the total probability is 1.
	if(axis=="none" or len(axis)==0):
		new_factor = Factor(names,array.shape)
		new_factor.set_all(array/np.sum(array))
		return new_factor
	else:
		cond_var_index = [a for a in range(len(names)) if names[a] in axis]
		if(len(cond_var_index)<1):
			print("Error: couldn't find variable")
			return None
		else:
			# find all variables not in the axis list and get their sum. This is easy in numpy. 
			# Gets the array created by summing out variables not in the axis list.
			not_cond_var_index = [b for b in range(len(names)) if not b in cond_var_index]
			sums = np.sum(array,axis=tuple(not_cond_var_index))
			# make a new factor which is the same but divided by the normalization.
			new_factor = Factor(names,array.shape)
			for i in range(indexes.shape[0]):
				norm = sums[tuple(indexes[i,cond_var_index])]
				new_factor.set(indexes[i],array[tuple(indexes[i])]/norm)
			return new_factor

# 2. FACTOR MARGINALIZATION
# 		returns a smaller factor, taking the expectation over all variables in axis
def marginalize(factor,axis="none"):
	array = factor.array
	indexes = factor.indexes
	names = factor.names
	if(axis=="none" or len(axis)==len(factor.names)):
		return np.sum(array)
	else:
		marg_var_index = [a for a in np.arange(len(names)) if names[a] in axis]
		if(len(marg_var_index)<1):
			return None
		else:
			# fairly simple, just sum out the variables in the axis list.
			not_marg_var_index = [b for b in range(len(names)) if not b in marg_var_index]
			summed_array = np.sum(array,axis=tuple(marg_var_index))
			new_names = [names[n] for n in not_marg_var_index]
			new_factor = Factor(new_names,summed_array.shape)
			new_factor.set_all(summed_array)
			return new_factor

# 3. FACTOR PRODUCT
# 		returns a joined factor. Most complicated piece of code here. Book gives good visualization.
#		for each variable in factor1 which is not also in factor2 it needs to go through every option in factor2.

def product(factor1,factor2):
	names1 = factor1.names
	names2 = factor2.names
	# find variables both in factor 1 and 2
	joint_names = [n for n in names1 if n in names2]
	# get a list of all variables which will be in the new factor
	new_names = names1 + [n for n in names2 if not n in joint_names]
	from_2_to_new_index = [new_names.index(n) for n in names2]
	factor2_not_joint_index = [n for n in range(len(names2)) if not names2[n] in joint_names]
	# get the shape of the new array. Essentially adding new dimensions to factor1
	new_shapes = list(factor1.array.shape)+[factor2.array.shape[n] for n in factor2_not_joint_index]
	new_factor = Factor(new_names,new_shapes)
	# every option in the new factor is the value in both factors with their respective variables multiplied.
	for i in new_factor.indexes:
		f1_part = i[:len(names1)]
		f2_part = i[from_2_to_new_index]
		new_factor.set(i,factor1.get(f1_part)*factor2.get(f2_part))
	return new_factor

# 4. DROP VARIABLES
#		selects a variables at particular values. 
#		E.g if the array is:
#		[[0.2,0.4]
#		 [0.1,0.3]]
# 		and I select only row 1, then the new factor is:
#		[0.1,0.3] (1d now instead of 2d)

def drop_variables(factor,axis,values):
	array = factor.array
	indexes = factor.indexes
	names = factor.names
	var_index = [a for a in np.arange(len(names)) if names[a] in axis]
	if(len(var_index)<1):
		return factor
	elif(len(var_index)==len(names)):
		return None
	else:
		not_var_index = [b for b in range(len(names)) if not b in var_index]
		slc = [slice(None)]*len(names)
		for v_i in range(len(var_index)):
			slc[var_index[v_i]]=slice(values[axis.index(names[var_index[v_i]])],values[axis.index(names[var_index[v_i]])]+1)
		sliced_array = np.squeeze(array[tuple(slc)])
		new_names = [names[n] for n in not_var_index]
		new_factor = Factor(new_names,sliced_array.shape)
		new_factor.set_all(sliced_array) # squeeze removes all axes with 1 dim.
		return new_factor

# simple code to do factor multiplication for a list of factors
def multiple_factor_product(all_factors):
	joint_factor = all_factors[0]
	if(len(all_factors)==1):
		return joint_factor
	for i in range(1,len(all_factors)):
		joint_factor = product(joint_factor,all_factors[i])
	return joint_factor

# simple code to sample variables from the factor.
def sample(factor,number_of_samples):
	array = factor.array
	indexes = factor.indexes
	normalized_array = array/np.sum(array)
	rows = np.random.choice(np.arange(indexes.shape[0]),number_of_samples,p=normalized_array.reshape(-1))
	return indexes[rows]





