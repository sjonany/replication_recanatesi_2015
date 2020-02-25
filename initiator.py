import numpy as np

def generate_memories(num_neurons, num_memories, f):
    # Returns num_neurons by num_memories matrix
    return np.random.choice(a=[0,1], 
                            size=(num_neurons, num_memories),
                            p=[1-f, f])

def generate_populations(memory_pattern):
    # Returns populations and neurons in each population
    return np.unique(memory_pattern, axis=0, return_counts=True)
    