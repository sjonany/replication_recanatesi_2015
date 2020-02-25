import numpy as np

def generate_memories(num_neurons, num_memories, f):
    '''Returns num_neurons by num_memories matrix'''
    return np.random.choice(a=[0, 1], 
                            size=(num_neurons, num_memories),
                            p=[1-f, f])

def generate_populations(memory_pattern):
    '''Returns populations and neurons in each population'''
    return np.unique(memory_pattern, axis=0, return_counts=True)

def generate_proto_conn_matrix(pops, exct_param, f):
    '''Return a prototype connection matrix without inhibition'''
    return exct_param * (pops - f) @ (pops - f).T

def generate_inhibition_seq(phi_min, phi_max, t_period, time_seq, phase=0):
    '''Returns a series of inhibition phi'''
    phi = np.zeros(len(time_seq))
    for j, t in enumerate(time_seq):
        phi[j] = (phi_max - phi_min) / 2 * np.sin(t * (2*np.pi) / t_period + phase) \
               + (phi_max + phi_min) / 2
    return phi

def gain_function(x, thres, gain, param_curr=4.75):
    '''Gain function'''
    return (x*param_curr + thres)**gain if (x + thres) > 0 else 0