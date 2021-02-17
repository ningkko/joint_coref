import os
import sys
import json
import torch
import argparse
import pickle
import logging

import spacy

sys.path.append("../shared/")

from collections import defaultdict
from classes import Document, Sentence, Token, EventMention, EntityMention
from extraction_utils import *


nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(description='Feature extraction (predicate-argument structures,'
                                             'mention heads, and ELMo embeddings)')

parser.add_argument('--config_path', type=str,
                    help=' The path to the configuration json file')
parser.add_argument('--output_path', type=str,
                    help=' The path to output folder (Where to save the processed data)')

args = parser.parse_args()

out_dir = args.output_path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

with open(os.path.join(args.output_path,'build_features_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)


def load_mentions_from_json(mentions_json_file, docs, is_event, is_gold_mentions):
    '''
    Loading mentions from JSON file and add those to the documents objects
    :param mentions_json_file: the JSON file contains the mentions
    :param docs:  set of document objects
    :param is_event: a boolean indicates whether the function extracts event or entity mentions
    :param is_gold_mentions: a boolean indicates whether the function extracts gold or predicted
    mentions
    '''
    with open(mentions_json_file, 'r') as js_file:
        js_mentions = json.load(js_file)

    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')
        sent_id = js_mention["sent_id"]
        tokens_numbers = js_mention["tokens_number"]
        mention_type = js_mention["mention_type"]
        is_singleton = js_mention["is_singleton"]
        is_continuous = js_mention["is_continuous"]
        mention_str = js_mention["tokens_str"]
        coref_chain = js_mention["coref_chain"]
        if mention_str is None:
            print(js_mention)
        head_text, head_lemma = find_head(mention_str)
        score = js_mention["score"]
        try:
            token_objects = docs[doc_id].get_sentences()[sent_id].find_mention_tokens(tokens_numbers)
        except:
            print('error when looking for mention tokens')
            print('doc id {} sent id {}'.format(doc_id, sent_id))
            print('token numbers - {}'.format(str(tokens_numbers)))
            print('mention string {}'.format(mention_str))
            print('sentence - {}'.format(docs[doc_id].get_sentences()[sent_id].get_raw_sentence()))
            raise

        # Sanity check - check if all mention's tokens can be found
        if not token_objects:
            print('Can not find tokens of a mention - {} {} {}'.format(doc_id, sent_id,tokens_numbers))

        # Mark the mention's gold coref chain in its tokens
        if is_gold_mentions:
            for token in token_objects:
                if is_event:
                    token.gold_event_coref_chain.append(coref_chain)
                else:
                    token.gold_entity_coref_chain.append(coref_chain)

        if is_event:
            mention = EventMention(doc_id, sent_id, tokens_numbers,token_objects,mention_str, head_text, head_lemma, is_singleton, is_continuous, coref_chain)
        else:
            mention = EntityMention(doc_id, sent_id, tokens_numbers,token_objects, mention_str, head_text, head_lemma, is_singleton, is_continuous, coref_chain, mention_type)

        mention.probability = score  # a confidence score for predicted mentions (if used), set gold mentions prob to 1.0
        if is_gold_mentions:
            docs[doc_id].get_sentences()[sent_id].add_gold_mention(mention, is_event)
        else:
            docs[doc_id].get_sentences()[sent_id]. \
                add_predicted_mention(mention, is_event,
                                      relaxed_match=config_dict["relaxed_match_with_gold_mention"])


def load_gold_mentions(docs,events_json, entities_json):
    '''
    A function loads gold event and entity mentions
    :param docs: set of document objects
    :param events_json:  a JSON file contains the gold event mentions (of a specific split - train/dev/test)
    :param entities_json: a JSON file contains the gold entity mentions (of a specific split - train/dev/test)
    '''
    load_mentions_from_json(events_json,docs,is_event=True, is_gold_mentions=True)
    load_mentions_from_json(entities_json,docs,is_event=False, is_gold_mentions=True)


def load_gold_data(split_txt_file, events_json, entities_json):
    logger.info('Loading gold mentions...')
    docs = load_ECB_plus(split_txt_file)
    load_gold_mentions(docs, events_json, entities_json)
    return docs

def find_head(x):
    '''
    This function finds the head and head lemma of a mention x
    :param x: A mention object
    :return: the head word and
    '''

    x_parsed = nlp(x)
    for tok in x_parsed:
        if tok.head == tok:
            if tok.lemma_ == u'-PRON-':
                return tok.text, tok.text.lower()
            return tok.text,tok.lemma_


def find_topic_gold_clusters(topic):
    '''
    Finds the gold clusters of a specific topic
    :param topic: a topic object
    :return: a mapping of coref chain to gold cluster (for a specific topic) and the topic's mentions
    '''
    event_mentions = []
    entity_mentions = []
    # event_gold_tag_to_cluster = defaultdict(list)
    # entity_gold_tag_to_cluster = defaultdict(list)

    event_gold_tag_to_cluster = {}
    entity_gold_tag_to_cluster = {}

    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            event_mentions.extend(sent.gold_event_mentions)
            entity_mentions.extend(sent.gold_entity_mentions)

    for event in event_mentions:
        if event.gold_tag != '-':
            if event.gold_tag not in event_gold_tag_to_cluster:
                event_gold_tag_to_cluster[event.gold_tag] = []
            event_gold_tag_to_cluster[event.gold_tag].append(event)
    for entity in entity_mentions:
        if entity.gold_tag != '-':
            if entity.gold_tag not in entity_gold_tag_to_cluster:
                entity_gold_tag_to_cluster[entity.gold_tag] = []
            entity_gold_tag_to_cluster[entity.gold_tag].append(entity)

    return event_gold_tag_to_cluster, entity_gold_tag_to_cluster, event_mentions, entity_mentions


def write_dataset_statistics(split_name, dataset, check_predicted):
    '''
    Prints the split statistics
    :param split_name: the split name (a string)
    :param dataset: an object represents the split
    :param check_predicted: whether to print statistics of predicted mentions too
    '''
    docs_count = 0
    sent_count = 0
    event_mentions_count = 0
    entity_mentions_count = 0
    event_chains_count = 0
    entity_chains_count = 0
    topics_count = len(dataset.topics.keys())
    predicted_events_count = 0
    predicted_entities_count = 0
    matched_predicted_event_count = 0
    matched_predicted_entity_count = 0


    for topic_id, topic in dataset.topics.items():
        event_gold_tag_to_cluster, entity_gold_tag_to_cluster, \
        event_mentions, entity_mentions = find_topic_gold_clusters(topic)

        docs_count += len(topic.docs.keys())
        sent_count += sum([len(doc.sentences.keys()) for doc_id, doc in topic.docs.items()])
        event_mentions_count += len(event_mentions)
        entity_mentions_count += len(entity_mentions)

        entity_chains = set()
        event_chains = set()

        for mention in entity_mentions:
            entity_chains.add(mention.gold_tag)

        for mention in event_mentions:
            event_chains.add(mention.gold_tag)

        # event_chains_count += len(set(event_gold_tag_to_cluster.keys()))
        # entity_chains_count += len(set(entity_gold_tag_to_cluster.keys()))

        event_chains_count += len(event_chains)
        entity_chains_count += len(entity_chains)

        if check_predicted:
            for doc_id, doc in topic.docs.items():
                for sent_id, sent in doc.sentences.items():
                    pred_events = sent.pred_event_mentions
                    pred_entities = sent.pred_entity_mentions

                    predicted_events_count += len(pred_events)
                    predicted_entities_count += len(pred_entities)

                    for pred_event in pred_events:
                        if pred_event.has_compatible_mention:
                            matched_predicted_event_count += 1

                    for pred_entity in pred_entities:
                        if pred_entity.has_compatible_mention:
                            matched_predicted_entity_count += 1

    with open(os.path.join(args.output_path, '{}_statistics.txt'.format(split_name)), 'w') as f:
        f.write('Number of topics - {}\n'.format(topics_count))
        f.write('Number of documents - {}\n'.format(docs_count))
        f.write('Number of sentences - {}\n'.format(sent_count))
        f.write('Number of event mentions - {}\n'.format(event_mentions_count))
        f.write('Number of entity mentions - {}\n'.format(entity_mentions_count))

        if check_predicted:
            f.write('Number of predicted event mentions  - {}\n'.format(predicted_events_count))
            f.write('Number of predicted entity mentions - {}\n'.format(predicted_entities_count))
            f.write('Number of predicted event mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_event_count,
                                        (matched_predicted_event_count/float(event_mentions_count)) *100 ))
            f.write('Number of predicted entity mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_entity_count,
                                        (matched_predicted_entity_count / float(entity_mentions_count)) * 100))


def main(args):
    """
        This script loads the train, dev and test json files (contain the gold entity and event
        mentions) builds mention objects, extracts predicate-argument structures, mention head
        and ELMo embeddings for each mention.

        Runs data processing scripts to turn intermediate data from (../intermid) into
        processed data ready to use in training and inference(saved in ../processed).
    """
    logger.info('Training data - loading event and entity mentions')
    training_data = load_gold_data(config_dict["train_text_file"],config_dict["train_event_mentions"],
                                   config_dict["train_entity_mentions"])

    logger.info('Dev data - Loading event and entity mentions ')
    dev_data = load_gold_data(config_dict["dev_text_file"],config_dict["dev_event_mentions"],
                              config_dict["dev_entity_mentions"])

    logger.info('Test data - Loading event and entity mentions')
    test_data = load_gold_data(config_dict["test_text_file"], config_dict["test_event_mentions"],
                               config_dict["test_entity_mentions"])

    train_set = order_docs_by_topics(training_data)
    dev_set = order_docs_by_topics(dev_data)
    test_set = order_docs_by_topics(test_data)

    write_dataset_statistics('train', train_set, check_predicted=False)

    write_dataset_statistics('dev', dev_set, check_predicted=False)

    write_dataset_statistics('test', test_set, check_predicted=True)


    logger.info('Storing processed data...')
    with open(os.path.join(args.output_path,'training_data'), 'wb') as f:
        pickle.dump(train_set, f)
    with open(os.path.join(args.output_path,'dev_data'), 'wb') as f:
        pickle.dump(dev_set, f)
    with open(os.path.join(args.output_path, 'test_data'), 'wb') as f:
        print(test_set)
        pickle.dump(test_set, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main(args)
