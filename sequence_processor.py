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
      stripped_line = line.strip()
      if len(stripped_line) == 0:
        line = f.readline()
        continue

      seq = list(map(int, line.strip().split(DELIMITER)))
      seqs.append(seq)
      line = f.readline()
  return seqs

def compute_n_order_markov(n, seqs, pseudocount=0):
  """
  Compute nth-order markov transition probabilities from given sequences.
  Usage:
    seqs = read_sequence_file('sequences/seq_test.txt')
    markov_table = compute_n_order_markov(n=2, seqs=seqs)
    markov_table[(3,1)][0]
    # Note the order. When we read 3,1 left to right, that is the past sequence in 
    # recall order.
    # The above returns a scalar, akin to Pr(St = 0 | St-2 = 3, St-1 = 1)
  We initially assume that we have seen all sequences, "pseudocount" times.
    See "additive smoothing" for more readings.
    This helps us avoid getting P(transition) = 1's for higher markovian orders where
    getting the exact same sequences is very rare.
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
    markov_table[past_mems] = 1.0 * (transition_tally[past_mems] + pseudocount) / \
        (normalizer + pseudocount * model.NUM_MEMORIES)
  return markov_table

def compute_average_markov_probs(markov_n, markov_table, start_i, seq):
  """
  Compute the average transition probabilities over the sequence.
  Not the same as log_likelihood, which multiplies out the probability,
  but can help use see how deterministic a system is
  """
  prob_sum = 0.0
  prob_count = 0
  for cur_i in range(start_i, len(seq)):
    cur_mem = seq[cur_i]
    past_mems = tuple(seq[cur_i-markov_n:cur_i])
    prob = markov_table[past_mems][cur_mem]
    # We shouldn't get prob = 0 because we get the markov table from the sequence themselves
    prob_sum += prob
    prob_count += 1
  return prob_sum / prob_count

def compute_average_markov_probs_all_seqs(markov_n, markov_table, start_i, seqs):
  """
  Just like compute_log_likehood_markov, but average out across all sequences
  """
  total_avg_proba = 0
  for seq in seqs:
    total_avg_proba += compute_average_markov_probs(markov_n, markov_table, start_i, seq)
  return total_avg_proba / len(seqs)

def compute_log_likehood_markov(markov_n, markov_table, start_i, seq):
  """
  Compute the log-likelihood that the given sequence is generated by an n-order markov process
  Usage:
    seqs = read_sequence_file('sequences/seq_test.txt')
    markov_table_1 = compute_n_order_markov(n=1, seqs=seqs)
    markov_table_2 = compute_n_order_markov(n=2, seqs=seqs)
    seq = seqs[0]
    # It's important that you pick a start_i that's the same for ALL the markov orders you are
    # comparing. Otherwise, you are computing probabilities across different sequence lengths.
    # In this case, we pick i=2 because max order is 2, and we have to look back twice.
    start_i = 2

    ll1 = compute_log_likehood_markov(1, markov_table_1, start_i, seq)
    ll2 = compute_log_likehood_markov(2, markov_table_2, start_i, seq)
  """
  ll = 0
  for cur_i in range(start_i, len(seq)):
    cur_mem = seq[cur_i]
    past_mems = tuple(seq[cur_i-markov_n:cur_i])
    prob = markov_table[past_mems][cur_mem]
    # We shouldn't get prob = 0 because we get the markov table from the sequence themselves
    ll += np.log(prob)
  return ll

def compute_avg_log_likehood_markov_all_seqs(markov_n, markov_table, start_i, seqs):
  """
  Just like compute_log_likehood_markov, but average out across all sequences
  """
  total_ll = 0
  for seq in seqs:
    total_ll += compute_log_likehood_markov(markov_n, markov_table, start_i, seq)
  return total_ll / len(seqs)
