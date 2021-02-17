import os
import gc
import sys
import json

sys.path.append("../")

import pickle
import logging
import argparse
from classes import *
from model_utils import *

parser = argparse.ArgumentParser(description='Run same lemma baseline')

parser.add_argument('--config_path', type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir', type=str,
                    help=' The directory to the output folder')

args = parser.parse_args()

# Loads json configuration file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

# Saves json configuration file in the experiment's folder
with open(os.path.join(args.out_dir,'lemma_baseline_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

from classes import *
from model_utils import *
from eval_utils import *


def get_clusters_by_head_lemma(mentions, is_event):
    '''
    Given a list of mentions, this function clusters mentions that share the same head lemma.
    :param mentions: list of Mention objects (can be event or entity mentions)
    :param is_event: whether the function clusters event or entity mentions.
    :return: list of Cluster objects
    '''
    mentions_by_head_lemma = {}
    clusters = []

    for mention in mentions:
        if mention.mention_head_lemma not in mentions_by_head_lemma:
            mentions_by_head_lemma[mention.mention_head_lemma] = []
        mentions_by_head_lemma[mention.mention_head_lemma].append(mention)

    for head_lemma, mentions in mentions_by_head_lemma.items():
        cluster = Cluster(is_event=is_event)
        for mention in mentions:
            cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters


def run_same_lemmma_baseline(test_set):
    '''
    Runs the head lemma baseline and writes its predicted clusters.
    :param test_set: A Corpus object representing the test set.
    '''
    topics_counter = 0
    topics = load_predicted_topics(test_set,config_dict)
    topics_keys = topics.keys()

    for topic_id in topics_keys:
        topic = topics[topic_id]
        topics_counter += 1

        event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=config_dict["test_use_gold_mentions"])

        event_clusters = get_clusters_by_head_lemma(event_mentions, is_event=True)
        entity_clusters = get_clusters_by_head_lemma(entity_mentions, is_event=False)

        with open(os.path.join(args.out_dir,'entity_clusters.txt'), 'a') as entity_file_obj:
            write_clusters_to_file(entity_clusters, entity_file_obj, topic_id)

        with open(os.path.join(args.out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
            write_clusters_to_file(event_clusters, event_file_obj, topic_id)

        set_coref_chain_to_mentions(event_clusters)
        set_coref_chain_to_mentions(entity_clusters)

    write_event_coref_results(test_set, args.out_dir, config_dict)
    write_entity_coref_results(test_set, args.out_dir, config_dict)


def main():
    '''
    This script loads the test set, runs the head lemma baseline and writes
    its predicted clusters.
    '''
    logger.info('Loading test data...')
    with open(config_dict["test_path"], 'rb') as f:
        test_data = pickle.load(f)

    logger.info('Test data have been loaded.')

    logger.info('Running same lemma baseline...')
    run_same_lemmma_baseline(test_data)
    logger.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main()

