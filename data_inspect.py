import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display


# choose the dataset path in the dropdown on the right and rerun this cell
# to see multiple samples


builder_dir = "/home/koyo/openvla/data_collection/datasets/test"
# builder_dir = "/home/koyo/openvla/data_collection/datasets/bridge_dataset/1.0.0"

### Dataset Loading ###
b = tfds.builder_from_directory(builder_dir)
print(b.info.splits) #{'train': <SplitInfo num_examples=unknown, num_shards=0>}
breakpoint()
# print("------------------------------------------------------------------------------")
# print("b: ", b) #<tensorflow_datasets.core.read_only_builder.ReadOnlyBuilder object at 0x702d5bdcfb50>


### Dataset Sampling ###
# Loads the dataset's first 10 samples ('train[:10]') and shuffles them.
ds = b.as_dataset(split='train[:2]').shuffle(2) # take only first 10 "episodes"

# print("------------------------------------------------------------------------------")
# print("ds type ", type(ds)) # <class 'tensorflow.python.data.ops.shuffle_op._ShuffleDataset'>
# print("ds: ", ds)

# Takes the first "episode" from the dataset iterator.
episode = next(iter(ds)) # iter converts an iterable (e.g., a list, dictionary) into an iterator
                         # An iterator is an object that allows you to traverse through its elements 
                         # one at a time using next (Retrieves the next element from an iterator.)

# print("------------------------------------------------------------------------------")
# print("episode type ", type(episode)) # <class 'dict'>
# print("episode: ", episode)

print("------------------------------------------------------------------------------")
for key, value in next(iter(episode['steps'])).items():
    print("Key:", key)
    print("Value:", value)
    print("--------------------")




# print("------------------------------------------------------------------------------")
# # other elements of the episode step --> this may vary for each dataset
# for elem in next(iter(episode['steps'])).items():
#   print(elem)