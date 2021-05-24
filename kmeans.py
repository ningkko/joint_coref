import os
import gc
import sys
import time
import math
import json
import spacy
import random
import logging
import argparse
import itertools
import numpy as np
from scorer import *
import _pickle as cPickle

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

out_dir = "model/"
config_path = "train_config.json"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logging.basicConfig(filename=os.path.join(out_dir, "train_log.txt"),
                    level=logging.DEBUG, filemode='w')

# Load json config file
with open(config_path, 'r') as js_file:
    config_dict = json.load(js_file)

with open(os.path.join(out_dir,'train_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

random.seed(config_dict["random_seed"])
np.random.seed(config_dict["random_seed"])

import torch

use_cuda = use_cuda and torch.cuda.is_available()

import torch.nn as nn
from classes import *
from eval_utils import *
from model_utils import *
from model_factory import *
import torch.optim as optim
import torch.nn.functional as F
from spacy.lang.en import English

if config_dict["gpu_num"] != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["gpu_num"])
    use_cuda = True
else:
    use_cuda = False

torch.manual_seed(config_dict["seed"])
if use_cuda:
    torch.cuda.manual_seed(config_dict["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Training with CUDA')


def generate_training_data():
    ## Generate training data
    if config_dict["span_representation"] == "elmo":
        train_path = config_dict["elmo_train_path"]
        dev_path = config_dict["elmo_dev_path"]
    elif config_dict["span_representation"] == "bert":
        train_path = config_dict["elmo_train_path"]
        dev_path = config_dict["elmo_dev_path"]
    else:
        print("Unknown span representation")

    with open(train_path, 'rb') as f:
        train_set = cPickle.load(f)
    with open(dev_path, 'rb') as f:
        dev_set = cPickle.load(f)

    for topic_id, topic in dev_set.topics.items():
        train_set.add_topic(topic.topic_id, topic)

    return train_set


with open("data/feature_ELMO/test_data", 'rb') as f:
    test_set = cPickle.load(f)
