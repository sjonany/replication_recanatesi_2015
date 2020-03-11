"""
Generates sequences of memory recalls using markov model and save it as a text file.
The markov model consists of just 2 states.
The point of doing this is to show how our current analysis is misleading.
"""

import numpy as np
import os
import pdb
import sys

# How many sequences of length 16 to generate
REP_PER_MEMORY = 5
SEQ_LEN = 16
NUM_MEMORY = 16
# Probability of transitioning to memory + 1.
P_TRANSITION = 0.2
PATH_TO_DUMP = "sequences/seq_markov_80.txt"
ALLOW_APPEND = True
DELIMITER = ","

# Check if sequence file already exists
if os.path.isfile(PATH_TO_DUMP):
  if not ALLOW_APPEND: 
    # If you want to append, set ALLOW_APPEND to True
    sys.exit("%s already exists. Quitting" % PATH_TO_DUMP)
else: 
  # Create empty textfile
  with open(PATH_TO_DUMP, "w") as f:
    f.write("")

for start_mem in range(NUM_MEMORY):
  next_memories = list(range(NUM_MEMORY))
  next_memories.remove(start_mem)
  for rep in range(REP_PER_MEMORY):
    cur_state = start_mem
    seq = [cur_state]
    for i in range(SEQ_LEN - 1):
      # Go to a different random memory, or stay
      if np.random.rand() < P_TRANSITION:
        next_idx = np.random.randint(low=0,high=NUM_MEMORY-1)
        cur_state = next_memories[next_idx]
      seq.append(cur_state)
    # Append sequence to file
    with open(PATH_TO_DUMP, "a") as f:
      f.write(DELIMITER.join(map(str, seq)))
      f.write("\r\n")
