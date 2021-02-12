import os
import sys
import json
import argparse
import pickle as cPickle
import logging

import spacy
import stanza
# stanza.download('en') # download English model

nlp_tag = "spacy"

sys.path.append("../shared/")

from classes import Document, Sentence, Token, EventMention, EntityMention
from extraction_utils import *

# model used by the paper
if nlp_tag == "stanza":
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
else:
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


def load_mentions_from_json(mentions_json_file, docs, is_event):
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
        for token in token_objects:
            if is_event:
                token.gold_event_coref_chain.append(coref_chain)
            else:
                token.gold_entity_coref_chain.append(coref_chain)

        if is_event:
            mention = EventMention(doc_id, sent_id, tokens_numbers,token_objects,mention_str, head_text,
                                   head_lemma, is_singleton, is_continuous, coref_chain)
        else:
            mention = EntityMention(doc_id, sent_id, tokens_numbers,token_objects, mention_str, head_text,
                                    head_lemma, is_singleton, is_continuous, coref_chain, mention_type)

        mention.probability = score  # a confidence score for predicted mentions (if used), set gold mentions prob to 1.0
        docs[doc_id].get_sentences()[sent_id].add_gold_mention(mention, is_event)


def load_gold_data(split_txt_file, events_json, entities_json):
    '''
    This function loads the texts of each split and its gold mentions, create document objects
    and stored the gold mentions within their suitable document objects
    :param split_txt_file: the text file of each split is written as 5 columns (stored in data/intermid)
    :param events_json: a JSON file contains the gold event mentions
    :param entities_json: a JSON file contains the gold event mentions
    :return:
    '''
    logger.info('Loading gold mentions...')
    docs = load_ECB_plus(split_txt_file)
    load_mentions_from_json(events_json,docs,is_event=True)
    load_mentions_from_json(entities_json,docs,is_event=False)

    return docs



def find_head(x):
    '''
    This function finds the head and head lemma of a mention x
    :param x: A mention object, e.g. "first-degree murder"
    :return: the head word and
    '''

    if nlp_tag == "stanza":
        # ------- using stanza --------
        x_parsed = nlp(x).sentences[0]
        for tok in x_parsed.words:
            if tok.head == 0:
                if tok.upos == "PRON":
                    return tok.text, tok.text.lower()
                return tok.text,tok.lemma
    else:
        # -------  using spacy --------
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


def write_dataset_statistics(split_name, dataset):
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


    with open(os.path.join(args.output_path, '{}_statistics.txt'.format(split_name)), 'w') as f:
        f.write('Number of topics - {}\n'.format(topics_count))
        f.write('Number of documents - {}\n'.format(docs_count))
        f.write('Number of sentences - {}\n'.format(sent_count))
        f.write('Number of event mentions - {}\n'.format(event_mentions_count))
        f.write('Number of entity mentions - {}\n'.format(entity_mentions_count))


def main(args):
    """
        This script loads the train, dev and test json files (contain the gold entity and event
        mentions) builds mention objects, extracts predicate-argument structures, mention head
        and ELMo embeddings for each mention.

        Runs data processing scripts to turn intermediate data from (../intermid) into
        processed data ready to use in training and inference(saved in ../processed).
    """
    logger.info('Training data - loading event and entity mentions')

    print("loading training data")
    training_data = load_gold_data(config_dict["train_text_file"],config_dict["train_event_mentions"],
                                   config_dict["train_entity_mentions"])

    logger.info('Dev data - Loading event and entity mentions ')
    print("loading dev data")
    dev_data = load_gold_data(config_dict["dev_text_file"],config_dict["dev_event_mentions"],
                              config_dict["dev_entity_mentions"])

    logger.info('Test data - Loading event and entity mentions')
    print("loading testing data")
    test_data = load_gold_data(config_dict["test_text_file"], config_dict["test_event_mentions"],
                               config_dict["test_entity_mentions"])

    print("ordering")

    train_set = order_docs_by_topics(training_data)
    dev_set = order_docs_by_topics(dev_data)
    test_set = order_docs_by_topics(test_data)

    print("writing")

    write_dataset_statistics('train', train_set)

    write_dataset_statistics('dev', dev_set)

    # check_predicted = True if config_dict["load_predicted_mentions"] else False
    write_dataset_statistics('test', test_set)

    print("finding args")


    logger.info('Augmenting predicate-arguments structures using dependency parser')
    find_args_by_dependency_parsing(train_set, nlp)
    logger.info('Dev gold mentions - loading predicates and their arguments with dependency parser')
    find_args_by_dependency_parsing(dev_set, nlp)
    logger.info('Test gold mentions - loading predicates and their arguments with dependency parser')
    find_args_by_dependency_parsing(test_set, nlp)


    logger.info('Storing processed data...')
    with open(os.path.join(args.output_path,'training_data'), 'wb') as f:
        cPickle.dump(train_set, f)
    with open(os.path.join(args.output_path,'dev_data'), 'wb') as f:
        cPickle.dump(dev_set, f)
    with open(os.path.join(args.output_path, 'test_data'), 'wb') as f:
        cPickle.dump(test_set, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main(args)
