import os
import sys
import logging
import operator
import collections

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

written_mentions = 0
cd_clusters_count = 10000
wd_clusters_count = 10

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from classes import *
def write_mention_based_cd_clusters(corpus, is_event, is_gold,out_file):
    '''
    This function writes the cross-document (CD) predicted clusters to a file (in a CoNLL format)
    in a mention based manner, means that each token represents a mention and its coreference chain id is marked
    in a parenthesis.
    Used in Cybulska setup, when gold mentions are used during evaluation and there is no need
    to match predicted mention with a gold one.
    :param corpus: A Corpus object, contains the documents of each split, grouped by topics.
    :param out_file: filename of the CoNLL output file
    :param is_event: whether to write event or entity mentions
    :param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
    or to write a system file (response) that contains the predicted clusters.
    '''
    out_coref = open(out_file, 'w')
    cd_coref_chain_to_id = {}
    cd_coref_chain_to_id_counter = 0
    ecb_topics = {}
    ecbplus_topics = {}
    for topic_id, topic in corpus.topics.items():
        if 'plus' in topic_id:
            ecbplus_topics[topic_id] = topic
        else:
            ecb_topics[topic_id] = topic

    generic = 'ECB+/ecbplus_all'
    out_coref.write("#begin document (" + generic + "); part 000" + '\n')
    topic_keys = sorted(ecb_topics.keys()) + sorted(ecbplus_topics.keys())

    for topic_id in topic_keys:
        curr_topic = corpus.topics[topic_id]
        for doc_id in sorted(curr_topic.docs.keys()):
            curr_doc = curr_topic.docs[doc_id]
            for sent_id in sorted(curr_doc.sentences.keys()):
                curr_sent = curr_doc.sentences[sent_id]
                mentions = curr_sent.gold_event_mentions if is_event else curr_sent.gold_entity_mentions
                mentions.sort(key=lambda x: x.start_offset, reverse=True)
                for mention in mentions:
                    # map the gold coref tags to unique ids
                    if is_gold:  # creating the key files
                        if mention.gold_tag not in cd_coref_chain_to_id:
                            cd_coref_chain_to_id_counter += 1
                            cd_coref_chain_to_id[mention.gold_tag] = cd_coref_chain_to_id_counter
                        coref_chain = cd_coref_chain_to_id[mention.gold_tag]
                    else:  # writing the clusters at test time (response files)
                        coref_chain = mention.cd_coref_chain
                    out_coref.write('{}\t({})\n'.format(generic,coref_chain))
    out_coref.write('#end document\n')
    out_coref.close() 
    
def write_mention_based_wd_clusters(corpus, is_event, is_gold, out_file):
    '''
    This function writes the within-document (WD) predicted clusters to a file (in a CoNLL format)
    in a mention based manner, means that each token represents a mention and its coreference chain id is marked
    in a parenthesis.
    Specifically in within document evaluation, we cut all the links across documents, which
    entails evaluating each document separately.
    Used in Cybulska setup, when gold mentions are used during evaluation and there is no need
    to match predicted mention with a gold one.
    :param corpus: A Corpus object, contains the documents of each split, grouped by topics.
    :param out_file: filename of the CoNLL output file
    :param is_event: whether to write event or entity mentions
    :param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
    or to write a system file (response) that contains the predicted clusters.
    '''
    doc_names_to_new_coref_id = {}
    next_doc_increment = 0
    doc_increment = 10000

    out_coref = open(out_file, 'w')
    cd_coref_chain_to_id = {}
    cd_coref_chain_to_id_counter = 0
    ecb_topics = {}
    ecbplus_topics = {}
    for topic_id, topic in corpus.topics.items():
        if 'plus' in topic_id:
            ecbplus_topics[topic_id] = topic
        else:
            ecb_topics[topic_id] = topic

    generic = 'ECB+/ecbplus_all'
    out_coref.write("#begin document (" + generic + "); part 000" + '\n')
    topic_keys = sorted(ecb_topics.keys()) + sorted(ecbplus_topics.keys())

    for topic_id in topic_keys:
        curr_topic = corpus.topics[topic_id]
        for doc_id in sorted(curr_topic.docs.keys()):
            curr_doc = curr_topic.docs[doc_id]
            for sent_id in sorted(curr_doc.sentences.keys()):
                curr_sent = curr_doc.sentences[sent_id]
                mentions = curr_sent.gold_event_mentions if is_event else curr_sent.gold_entity_mentions
                for mention in mentions:
                    # map the gold coref tags to unique ids
                    if is_gold:  # creating the key files
                        if mention.gold_tag not in cd_coref_chain_to_id:
                            cd_coref_chain_to_id_counter += 1
                            cd_coref_chain_to_id[mention.gold_tag] = cd_coref_chain_to_id_counter
                        coref_chain = cd_coref_chain_to_id[mention.gold_tag]
                    else:  # writing the clusters at test time (response files)
                        coref_chain = mention.cd_coref_chain

                    if mention.doc_id not in doc_names_to_new_coref_id:
                        next_doc_increment += doc_increment
                        doc_names_to_new_coref_id[mention.doc_id] = next_doc_increment

                    coref_chain += doc_names_to_new_coref_id[mention.doc_id]

                    out_coref.write('{}\t({})\n'.format(generic,coref_chain))
    out_coref.write('#end document\n')
    out_coref.close()


def write_event_coref_results(corpus, out_dir, config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted event clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
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
    out_file = os.path.join(out_dir, 'CD_test_entity_mention_based.response_conll')
    write_mention_based_cd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)

    out_file = os.path.join(out_dir, 'WD_test_entity_mention_based.response_conll')
    write_mention_based_wd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)

