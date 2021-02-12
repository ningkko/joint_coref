import os
import sys
import json
import spacy
import torch
import random
import logging
import itertools
import collections
import numpy as np
from scorer import *
from eval_utils import *
import _pickle as cPickle
from bcubed_scorer import *
import matplotlib.pyplot as plt
from spacy.lang.en import English

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from classes import *

clusters_count = 1

analysis_pair_dict = {}


def get_topic(id):
    '''
    Extracts the topic id from the document ID.
    Note that this function doesn't extract the sub-topic ID (including the ecb/ecbplus notation)
    :param id: document id (string)
    :return: the topic id (string)
    '''
    return id.split('_')[0]


def merge_sub_topics_to_topics(test_set):
    '''
    Merges the test's sub-topics sub-topics to their topics (for experimental use).
    :param test_set: A Corpus object represents the test set
    :return: a dictionary contains the merged topics
    '''
    new_topics = {}
    topics_keys = test_set.topics.keys()
    for topic_id in topics_keys:
        topic = test_set.topics[topic_id]
        if get_topic(topic_id) not in new_topics:
            new_topics[get_topic(topic_id)] = Topic(get_topic(topic_id))
        new_topics[get_topic(topic_id)].docs.update(topic.docs)

    return new_topics


def topic_to_mention_list(topic, is_gold):
    '''
    Gets a Topic object and extracts its event/entity mentions (depends on the is_event flag)
    :param topic: a Topic object
    :param is_event: a flag that denotes whether event mentions will be extracted or
    entity mention will be extracted (True for event extraction, False for entity extraction)
    :param is_gold: a flag that denotes whether to extract gold mention or predicted mentions
    :return: list of the topic's mentions (EventMention or EntityMention objects)
    '''
    event_mentions = []
    entity_mentions = []
    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            if is_gold:
                event_mentions.extend(sent.gold_event_mentions)
                entity_mentions.extend(sent.gold_entity_mentions)
            else:
                event_mentions.extend(sent.pred_event_mentions)
                entity_mentions.extend(sent.pred_entity_mentions)

    return event_mentions, entity_mentions


def load_entity_wd_clusters(config_dict):
    '''
    Loads from a file the within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    model/tool and ordered those clusters in a dictionary according to their documents.
    :param config_dict: a configuration dictionary that contains a path to a file stores the
    within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    system.
    :return: a dictionary contains a mapping of a documents to their predicted entity clusters
    '''
    doc_to_entity_mentions = {}

    with open(config_dict["wd_entity_coref_file"], 'r') as js_file:
        js_mentions = json.load(js_file)

    # load all entity mentions in the json
    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')
        if doc_id not in doc_to_entity_mentions:
            doc_to_entity_mentions[doc_id] = {}
        sent_id = js_mention["sent_id"]
        if sent_id not in doc_to_entity_mentions[doc_id]:
            doc_to_entity_mentions[doc_id][sent_id] = []
        tokens_numbers = js_mention["tokens_numbers"]
        mention_str = js_mention["tokens_str"]

        try:
            coref_chain = js_mention["coref_chain"]
        except:
            continue

        doc_to_entity_mentions[doc_id][sent_id].append((doc_id, sent_id, tokens_numbers,
                                                        mention_str, coref_chain))
    return doc_to_entity_mentions

def get_sub_topics(doc_id):
    '''
    Extracts the sub-topic id from the document ID.
    :param doc_id: document id (string)
    :return: the sub-topic id (string)
    '''
    topic = doc_id.split('_')[0]
    if 'ecbplus' in doc_id:
        category = 'ecbplus'
    else:
        category = 'ecb'
    return '{}_{}'.format(topic, category)


def separate_clusters_to_sub_topics(clusters, is_event):
    '''
    Removes spurious cross sub-topics coreference link (used for experiments in Yang setup).
    :param clusters: a list of Cluster objects
    :param is_event: Clusters' type (event/entity)
    :return: new list of clusters, after spurious cross sub-topics coreference link were removed.
    '''
    new_clusters = []
    for cluster in clusters:
        sub_topics_to_clusters = {}
        for mention in cluster.mentions.values():
            mention_sub_topic = get_sub_topics(mention.doc_id)
            if mention_sub_topic not in sub_topics_to_clusters:
                sub_topics_to_clusters[mention_sub_topic] = []
            sub_topics_to_clusters[mention_sub_topic].append(mention)
        for sub_topic, mention_list in sub_topics_to_clusters.items():
            new_cluster = Cluster(is_event)
            for mention in mention_list:
                new_cluster.mentions[mention.mention_id] = mention
            new_clusters.append(new_cluster)

    return new_clusters


def set_coref_chain_to_mentions(clusters, is_event, is_gold, intersect_with_gold,):
    '''
    Sets the predicted cluster id to all mentions in the cluster
    :param clusters: predicted clusters (a list of Corpus objects)
    :param is_event: True, if clusters are event clusters, False otherwise - currently unused.
    :param is_gold: True, if the function sets gold mentions and false otherwise
     (it sets predicted mentions) - currently unused.
    :param intersect_with_gold: True, if the function sets predicted mentions that were matched
    with gold mentions (used in setting that requires to match predicted mentions with gold
    mentions - as in Yang's setting) , and false otherwise - currently unused.
    :param remove_singletons: True if the function ignores singleton clusters (as in Yang's setting)
    '''
    global clusters_count
    for cluster in clusters:
        cluster.cluster_id = clusters_count
        for mention in cluster.mentions.values():
            mention.cd_coref_chain = clusters_count
        clusters_count += 1


def create_gold_clusters(mentions):
    '''
    Forms within document gold clusters.
    :param mentions: list of mentions
    :return: a dictionary contains the within document gold clusters (list)
    mapped by document id and the gold cluster ID.
    '''
    wd_clusters = {}
    for mention in mentions:
        mention_doc_id = mention.doc_id
        if mention_doc_id not in wd_clusters:
            wd_clusters[mention_doc_id] = {}
        mention_gold_tag = mention.gold_tag
        if mention_gold_tag not in wd_clusters[mention_doc_id]:
            wd_clusters[mention_doc_id][mention_gold_tag] = []
        wd_clusters[mention_doc_id][mention_gold_tag].append(mention)

    return wd_clusters



def write_event_coref_results(corpus, out_dir, config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted event clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
    if not config_dict["test_use_gold_mentions"]:
        out_file = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
        write_span_based_cd_coref_clusters(corpus, out_file, is_event=True, is_gold=False,
                                           use_gold_mentions=config_dict["test_use_gold_mentions"])
    else:
        out_file = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
        write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)

        out_file = os.path.join(out_dir, 'WD_test_event_mention_based.response_conll')
        write_mention_based_wd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)


def write_entity_coref_results(corpus, out_dir,config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted entity clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
    if not config_dict["test_use_gold_mentions"]:
        out_file = os.path.join(out_dir, 'CD_test_entity_span_based.response_conll')
        write_span_based_cd_coref_clusters(corpus, out_file, is_event=False, is_gold=False,
                                           use_gold_mentions=config_dict["test_use_gold_mentions"])
    else:
        out_file = os.path.join(out_dir, 'CD_test_entity_mention_based.response_conll')
        write_mention_based_cd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)

        out_file = os.path.join(out_dir, 'WD_test_entity_mention_based.response_conll')
        write_mention_based_wd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)

