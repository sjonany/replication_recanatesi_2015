import numpy as np
from collections import defaultdict
from tqdm import tqdm

NUM_NEURONS = int(1e5)
NUM_MEMORIES = 16
SPARSITY = 0.01
EXCITATION_PARAM = 13000
PHI_MIN = 0.7
PHI_MAX = 1.06
TIME_STEP = 0.001
DECAY_TIME = 0.01
OSCILLATION_TIME = 1
TOTAL_TIME = 16
NOISE_VARIANCE = 65
INIT_RATE = 1
GAIN_FUNC_THRES = 0
GAIN_FUNC_EXP = 0.4
PARAM_NOISE = 10
PARAM_CURR = 4.75

class RecallModel:

  """
  Model of free recall.
  Each run will produce average firing rates per memory over time, which
  can then be post-processed by helpers.py > mem_activities_to_single_mem_transitions() to get
  the sequence of single memory recalls. 
  Usage:
    model = RecallModel()
    # This creates the passive connectome
    model.init()
    # You can run the inhibition cycles several times
    avg_firing_rates_per_memory_run1 = model.run(init_mem = 3)
    avg_firing_rates_per_memory_run2 = model.run(init_mem = 4)

    # This gives you an array of 16 elements - the single memory recalls.
    mem_activities_to_single_mem_transitions(avg_firing_rates_per_memory_run1)
  """
  def init(self):
    """
    Generate memory patterns and passive connectome
    """
    # Generate memories and populations
    # memory_pattern = The eta matrix. Eta[i][j] = 1 iff ith neuron is recruited by jth memory
    self.memory_pattern = generate_memories(NUM_NEURONS, NUM_MEMORIES, SPARSITY)
    # pops = A boolean matrix of size [num_encoding_patterns] by [number of memories].  
    # Each row is a single memory encoding pattern, which neurons are grouped by.
    
    # num_neurons_per_pop.shape = 1D array of size [num_encoding_patterns], where each
    # element [i] is the number of neurons with the encoding pattern in pops[i]
    self.pops, self.num_neurons_per_pop = generate_populations(self.memory_pattern)

    # Generate prototype connectivity matrix
    # This is the static part of Jij that doesn't include the moving phi term
    self.proto_conn_mat = generate_proto_conn_matrix(self.pops, EXCITATION_PARAM, SPARSITY)

    # Build a hashmap of corresponding populations for each memory
    # Key = memory id, Value = list of integer i's such that pops[i] is
    # encoding pattern related to this memory. 
    self.pops_of_memory = defaultdict(list)
    for j in range(NUM_MEMORIES):
        self.pops_of_memory[j] = list(np.where(self.pops[:,j]==1)[0])

  def run(self, init_mem):
    """
    Run one simulation of cycling inhibition, starting with init_mem

    Return: average firing rates per memory over time.
      This matrix has dimension [NUM_MEMORY] x [TOTAL_TIMESTEP]
    """

    time_seq = np.arange(0, TOTAL_TIME, TIME_STEP)
    phi_seq = generate_inhibition_seq(PHI_MIN, PHI_MAX, OSCILLATION_TIME, time_seq)

    # Number of different neuron populations.
    num_pops = len(self.pops)

    # Initiate synaptic currents, averaging firing rates for each memory and noises
    currs = np.zeros(num_pops)
    avg_firing_rates = np.zeros((NUM_MEMORIES, len(time_seq)))
    firing_rates = np.zeros(num_pops)
    firing_rates[self.pops_of_memory[init_mem]] += INIT_RATE

    # Start Simulation
    for j, t in tqdm(enumerate(time_seq)):
      # Update the connection matrix by including inhibition
      conn_mat = self.proto_conn_mat - EXCITATION_PARAM * phi_seq[j]
      # Compute averaging firing rates for each memory
      for mu in range(NUM_MEMORIES):
        avg_firing_rates[mu, j] = np.average(firing_rates[self.pops_of_memory[mu]], 
                                               weights=self.num_neurons_per_pop[self.pops_of_memory[mu]])
      # Generate Noises
      noises = np.random.randn(len(firing_rates)) * np.sqrt(NOISE_VARIANCE) * PARAM_NOISE
      noises /= np.sqrt(self.num_neurons_per_pop)
      # Update the averaging synaptic currents
      currs += TIME_STEP / DECAY_TIME * (-currs + conn_mat @ \
                                         (firing_rates * self.num_neurons_per_pop / NUM_NEURONS) + noises)
      # Update populational firing rates
      for k, curr in enumerate(currs):
        firing_rates[k] = gain_function(curr, GAIN_FUNC_THRES, GAIN_FUNC_EXP, PARAM_CURR)

    return avg_firing_rates

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