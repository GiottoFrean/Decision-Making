import factors

def sum_product_variable_elimination(all_factors,known_vars,evidence,unknown_vars):
    new_factors = []
    for f in all_factors:
        deleted_f = factors.drop_variables(f,known_vars,evidence)
        if(deleted_f!=None):
            new_factors.append(deleted_f)
    
    for unknown_var in unknown_vars:
        factors_to_combine = []
        factors_to_exclude = []
        for f in new_factors:
            if(unknown_var in f.names):
                factors_to_combine.append(f)
            else:
                factors_to_exclude.append(f)
        new_factors = factors_to_exclude
        if(len(factors_to_combine)>0):
            combined_factor = factors.multiple_factor_product(factors_to_combine)
            combined_factor = factors.marginalize(combined_factor,[unknown_var])
            if(not isinstance(combined_factor,(int,float))):
                new_factors.append(combined_factor)
    if(len(new_factors)>1):
        combined_factor = factors.multiple_factor_product(new_factors)
        new_factors = [combined_factor]
    return factors.condition(new_factors[0])

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
