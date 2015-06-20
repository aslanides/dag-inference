import numpy as np

p_b = np.array([0.97, 0.01, 0.02])
p_a_given_b = np.array([[0.9,0.8,0.3],[0.1,0.2,0.7]])

f_ab = p_a_given_b * p_b
f_bcd = np.array([[[0.9,0.08,0.01,0.01],[0.8,0.17,0.01,0.02],[0.1,0.01,0.87,0.02]],
                  [[0.3,0.05,0.05,0.6],[0.4,0.05,0.15,0.4],[0.01,0.01,0.97,0.01]]])
f_c = np.array([0.7,0.3])
f_de = np.array([[0.99,0.99,0.4,0.9],[0.01,0.01,0.6,0.1]])
f_cf = np.array([[0.99,0.2],[0.01,0.8]])

def leaf_messages_variable(leaf):
    ''' Returns the leaf message from a variable, i.e. 1.

        Variable -> numpy.ndarray
    '''
    if not isinstance(leaf,Variable):
        raise ValueError('This node is not a variable!')
    return np.ones(len(leaf.states))

def leaf_messages_factor(leaf):
    ''' Returns the leaf message from a factor, i.e. the factor itself

        Factor -> numpy.ndarray
    '''
    if not isinstance(leaf,Factor):
        raise ValueError('This node is not a factor!')
    return leaf.probs

def variable_to_factor_message(var,fac):
    '''
        Computes the message that var sends to fac.
        Mutually recursive with factor_to_variable_message, defined below.

       (Variable,Factor) -> numpy.ndarray
    '''
    if not var.children:
        return leaf_messages_variable(var)

    if var in fac.children:
        #variable is a child of factor, and we are passing messages towards the root
        mess = [factor_to_variable_message(f,var) for f in var.children]
        mu = np.ones(len(var.states))
        for m in mess:
            mu *= m # elementwise product of vectors is computed by numpy

        return mu
    else:
        # assumes that we have arranged things so that the variable we're not marginalising is at the root
        raise ValueError('No edge between these nodes!')

def factor_to_variable_message(fac,var):
    '''
        Computes the message that fac sends to var.
        Mutually recursive with variable_to_factor_message, defined above.

       (Factor,Variable) -> numpy.ndarray
    '''
    if not fac.children:
        return leaf_messages_factor(fac)

    indices = st.ascii_lowercase # Einstein summation indices. Careful, this means we only permit up to 26 edges from each node!
                                # This is a current limitation of numpy.einsum.
    if fac in var.children:
        #factor is a child of variable, and we are passing messages towards the root
        mess = [variable_to_factor_message(v,fac) for v in fac.children] # collect the messages from neighbouring variables
        idx = [fac.var_dims[v] for v in fac.children] # figure out what dimensions to sum over
        ein = indices[0:len(fac.probs.shape)]
        for i in idx:
            ein += ',' + indices[i] # build up our summation index string

        return np.einsum(ein,fac.probs,*mess)
    else:
        # assumes that we have arranged things so that the variable we're not marginalising is at the root
        raise ValueError('No edge between these nodes!')

class Variable(object):

    def __init__(self, states, children):
        '''
            Make a new variable with the given child factors,
            with possible states given by states

            ([str],[Factor]) -> None
        '''
        self.states = states
        self.children = children


class Factor(object):

    def __init__(self,probs,children,var_dims):
        '''
            Make a new factor with the given child variables,
            with factor values given by the distribution probs,
            and variable-factor dimension correspondences var_dims

            (numpy.ndarray, [Variable]) -> None
        '''
        self.children = children
        self.probs = probs
        self.var_dims = var_dims

        # make sure all the dimensions line up properly
        e = ValueError('Children cardinality mismatch')
        if children:
            if len(var_dims) != len(children):
                raise e

            for var,dim in var_dims.iteritems():
                if probs.shape[dim] != len(var.states):
                    raise e
