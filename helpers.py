import numpy as np

def mem_activities_to_single_mem_transitions(mem_activities):
  """
  Convert the colorful sinuosoidal memory activities to a list of single memory recalls.
  mem_activities is of dimension [num_memories] x [total_timestep]
  Usage:
    model = RecallModel()
    model.init()
    mem_activities = model.run(init_mem = 3)
    # This is now just a list of 16 items (if we have 16 cycles), corresponding to the items
    # That are recalled.
    single_mem_transitions = mem_activities_to_single_mem_transitions(mem_activities)
  """

  # Calculate the max peak at any time.
  max_firing_rate = np.amax(mem_activities)
  # From eyeballing, half the peak height seems like a good threshold
  threshold_firing_rate = 0.5 * max_firing_rate

  num_mem = mem_activities.shape[0]

  # Each item is a tuple of (memory id, timestamp of recall)
  single_mem_recalls = []
  for mem_i in range(num_mem):
    # 1* to do int cast so diff works properly. False-True = True-False = True unfortunately.
    is_above_thresh = 1 * (mem_activities[mem_i] > threshold_firing_rate)
    # Get the indices of upward crossings
    above_thres_indices = np.where(np.diff(is_above_thresh) > 0)[0]
    left_zip = [mem_i] * len(above_thres_indices)
    single_mem_recalls.extend(list(zip(left_zip, above_thres_indices)))
  # Sort in ascending recall time
  single_mem_recalls.sort(key = lambda recall_tuple: recall_tuple[1])
  return [recall[0] for recall in single_mem_recalls]