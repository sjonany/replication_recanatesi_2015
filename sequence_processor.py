"""
Methods for processing the sequence files
"""
import model

from collections import defaultdict
import numpy as np
import pdb

DELIMITER = ","

def read_sequence_file(file):
  """
  Usage:
    seqs = read_sequence_file('sequences/seq_test.txt')
  Return:
    A list of sequences, where each sequence is a single memory recall experiment (default is 16).
  """
  seqs = []
  with open(file, "r") as f:
    line = f.readline()
    while line:
      seq = list(map(int, line.strip().split(DELIMITER)))
      seqs.append(seq)
      line = f.readline()
  return seqs

def compute_n_order_markov(n, seqs):
  """
  Compute nth-order markov transition probabilities from given sequences.
  Usage:
    seqs = read_sequence_file('sequences/seq_test.txt')
    markov_table = compute_n_order_markov(n=2, seqs=seqs)
    markov_table[(3,1)][0]
    # Note the order. When we read 3,1 left to right, that is the past sequence in 
    # recall order.
    # The above returns a scalar, akin to Pr(St = 0 | St-2 = 3, St-1 = 1)
  """
  # transition_tally[(3,1)] returns an array of numbers of size num_memories
  # If the returned array at index 0 is 100, this means we see the sequence [3, 1, 0] 100 times
  transition_tally = defaultdict(lambda: np.array([0] * model.NUM_MEMORIES))
  for seq_id in range(len(seqs)):
    seq = seqs[seq_id]
    for i in range(len(seq) - n):
      # substring of length n+1
      sub = seq[i:i+n+1]
      cur_mem = sub[-1]
      past_mems = tuple(sub[:-1])
      transition_tally[past_mems][cur_mem] += 1

  # Convert everything to probabilities
  markov_table = defaultdict(lambda: np.array([0] * model.NUM_MEMORIES, dtype=np.float64))
  for past_mems in transition_tally.keys():
    normalizer = sum(transition_tally[past_mems])
    markov_table[past_mems] = transition_tally[past_mems] / normalizer

  return markov_table
