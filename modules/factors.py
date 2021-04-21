import numpy as np

class Factor:
	def __init__(self,names,pos_values):
		array = np.zeros((pos_values))
		indexes = np.zeros((np.prod(array.shape),len(names))).astype(int)
		f = 1
		for i in range(len(names)):
			repeated = np.arange(array.shape[i]).repeat(np.prod(array.shape)/(f*array.shape[i]))
			indexes[:,i]=np.tile(repeated,f)
			f = f*array.shape[i]
		self.names = names
		self.array = array
		self.indexes = indexes
		
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
		
	def set(self,index,value):
		ndim = len(self.names)
		if(len(index)==ndim):
			self.array[tuple(index)]=value
		else:
			raise Exception('length of index is incorrect. Provide {} values'.format(ndim))

	def get(self,index):
		return self.array[tuple(index)]
	
	def set_all(self,values):
		self.array = np.array(values).reshape(self.array.shape,order="C")
		
def condition(factor,axis="none"):
	array = factor.array
	indexes = factor.indexes
	names = factor.names
	if(axis=="none"):
		new_factor = Factor(names,array.shape)
		new_factor.set_all(array/np.sum(array))
	else:
		cond_var_index = [a for a in range(len(names)) if names[a] in axis]
		if(len(cond_var_index)<1):
			print("Error: couldn't find variable")
			return None
		else:
			not_cond_var_index = [b for b in range(len(names)) if not b in cond_var_index]
			sums = np.sum(array,axis=tuple(not_cond_var_index))
			new_factor = Factor(names,array.shape)
			for i in range(indexes.shape[0]):
				norm = sums[tuple(indexes[i,cond_var_index])]
				new_factor.set(indexes[i],array[tuple(indexes[i])]/norm)
			return new_factor

def marginalize(factor,axis="none"):
	array = factor.array
	indexes = factor.indexes
	names = factor.names
	if(axis=="none" or len(axis)==len(factor.names)):
		return np.sum(array)
	else:
		marg_var_index = [a for a in np.arange(len(names)) if names[a] in axis]
		if(len(marg_var_index)<1):
			print("Error: couldn't find variable")
			return None
		else:
			not_marg_var_index = [b for b in range(len(names)) if not b in marg_var_index]
			summed_array = np.sum(array,axis=tuple(marg_var_index))
			new_names = [names[n] for n in not_marg_var_index]
			new_factor = Factor(new_names,summed_array.shape)
			new_factor.set_all(summed_array)
			return new_factor
	
def product(factor1,factor2):
	names1 = factor1.names
	names2 = factor2.names
	joint_names = [n for n in names1 if n in names2]
	new_names = names1 + [n for n in names2 if not n in joint_names]
	from_2_to_new_index = [new_names.index(n) for n in names2]
	factor2_not_joint_index = [n for n in range(len(names2)) if not names2[n] in joint_names]
	new_shapes = list(factor1.array.shape)+[factor2.array.shape[n] for n in factor2_not_joint_index]
	new_factor = Factor(new_names,new_shapes)
	for i in new_factor.indexes:
		f1_part = i[:len(names1)]
		f2_part = i[from_2_to_new_index]
		new_factor.set(i,factor1.get(f1_part)*factor2.get(f2_part))
	return new_factor

def cut_variables(factor,axis,values):
	array = factor.array
	indexes = factor.indexes
	names = factor.names
	if(len(axis)==len(factor.names)):
		return None
	else:
		var_index = [a for a in np.arange(len(names)) if names[a] in axis]
		if(len(var_index)<1):
			print("Error: couldn't find variable")
			return None
		else:
			not_var_index = [b for b in range(len(names)) if not b in var_index]
			slc = [slice(None)]*len(names)
			for v_i in range(len(var_index)):
				slc[var_index[v_i]]=slice(values[v_i],values[v_i]+1)
			sliced_array = np.squeeze(array[tuple(slc)])
			new_names = [names[n] for n in not_var_index]
			new_factor = Factor(new_names,sliced_array.shape)
			new_factor.set_all(sliced_array) # squeeze removes all axes with 1 dim.
			return new_factor







