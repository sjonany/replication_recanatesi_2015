"""
Methods for processing the sequence files
"""
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
