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
import pickle
from bcubed_scorer import *
import matplotlib.pyplot as plt
from spacy.lang.en import English

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from classes import *

clusters_count = 1

analysis_pair_dict = {}


def topic_to_mention_list(topic, is_gold):
    '''
    Gets a Topic object and extracts its event/entity mentions
    :param topic: a Topic object
    :param is_gold: a flag that denotes whether to extract gold mention or predicted mentions
    :return: list of the topic's mentions
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


def merge_sub_topics_to_topics(test_set):
    '''
    Merges the test's sub-topics sub-topics to their topics (for experimental use).
    :param test_set: A Corpus object represents the test set
    :return: a dictionary contains the merged topics
    '''
    def get_topic(id):
        return id.split('_')[0]

    new_topics = {}
    topics_keys = test_set.topics.keys()
    for topic_id in topics_keys:
        topic = test_set.topics[topic_id]
        if get_topic(topic_id) not in new_topics:
            new_topics[get_topic(topic_id)] = Topic(get_topic(topic_id))
        new_topics[get_topic(topic_id)].docs.update(topic.docs)

    return new_topics


def separate_clusters_to_sub_topics(clusters, is_event):
    '''
    Removes spurious cross sub-topics coreference link (used for experiments in Yang setup).
    :param clusters: a list of Cluster objects
    :param is_event: Clusters' type (event/entity)
    :return: new list of clusters, after spurious cross sub-topics coreference link were removed.
    '''

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

def write_clusters_to_file(clusters, file_obj, topic):
    '''
    Write the clusters to a text file (used for analysis)
    :param clusters: list of Cluster objects
    :param file_obj: file to write the clusters
    :param topic - topic name
    '''
    i = 0
    file_obj.write('Topic - ' + topic +'\n')
    for cluster in clusters:
        i += 1
        file_obj.write('cluster #' + str(i) + '\n')
        mentions_list = []
        for mention in cluster.mentions.values():
            mentions_list.append('{}_{}'.format(mention.mention_str,mention.gold_tag))
        file_obj.write(str(mentions_list) + '\n\n')


def set_coref_chain_to_mentions(clusters):
    '''
    Sets the predicted cluster id to all mentions in the cluster
    :param clusters: predicted clusters (a list of Corpus objects)
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


def load_predicted_topics(test_set, path):
    '''
    Loads the document clusters that were predicted by a document clustering algorithm and
    organize the test's documents to topics according to document clusters.
    :param test_set: A Corpus object represents the test set
    :param path: path to the predicted topic file
    stores the results of the document clustering algorithm.
    :return:  a dictionary contains the documents ordered according to the predicted topics
    '''
    new_topics = {}
    with open(config_dict["predicted_topics_path"], 'rb') as f:
        predicted_topics = pickle.load(f)
    all_docs = []
    for topic in test_set.topics.values():
        all_docs.extend(topic.docs.values())

    all_doc_dict = {doc.doc_id:doc for doc in all_docs }

    topic_counter = 1
    for topic in predicted_topics:
        topic_id = str(topic_counter)
        new_topics[topic_id] = Topic(topic_id)

        for doc_name in topic:
            print(topic_id)
            print(doc_name)
            if doc_name in all_doc_dict:
                new_topics[topic_id].docs[doc_name] = all_doc_dict[doc_name]
        topic_counter += 1

    print(len(new_topics))
    return new_topics
