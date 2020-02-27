"""
Generates sequences of memory recalls and save it as a text file.
This is so future analysis can be very fast.
"""

import helpers
import model

import os
import pdb
import sys

# For each memory  used as a starting point, repeat this much
REPS_PER_MEMORY = 20
PATH_TO_DUMP = "sequences/seq_default_2.txt"
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

# Initialize the model
rec_model = model.RecallModel()
# Generate eta matrix with a fixed seed
rec_model.init(seed = 0)

for mem_i in range(model.NUM_MEMORIES):
  for rep in range(REPS_PER_MEMORY):
    print("Rep %d / %d for memory %d / %d" % (rep+1, REPS_PER_MEMORY, mem_i+1, model.NUM_MEMORIES))
    mem_activities = rec_model.run(init_mem=mem_i)
    seq = helpers.mem_activities_to_single_mem_transitions(mem_activities)
    if len(seq) != model.TOTAL_TIME:
      print("WARNING. Sequence length is %d: %s" % (len(seq), seq))
    # Append sequence to file
    with open(PATH_TO_DUMP, "a") as f:
      f.write(DELIMITER.join(map(str, seq)))
      f.write("\r\n")
