
import os
import sys

sys.path.append("../shared/")

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

from classes import *

matched_args = 0
matched_args_same_ix = 0

matched_events = 0
matched_events_same_ix = 0


def find_args_by_dependency_parsing(dataset,nlp):
    '''
    Runs dependency parser on the split's sentences and augments the predicate-argument structures
    :param dataset: an object represents the split (Corpus object)
    '''
    global matched_args, matched_args_same_ix, matched_events,matched_events_same_ix
    matched_args = 0
    matched_args_same_ix = 0
    matched_events = 0
    matched_events_same_ix = 0
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                sent_str = sent.get_raw_sentence()
                parsed_sent = nlp(sent_str)
                findSVOs(parsed_sent=parsed_sent, sent=sent, is_gold=is_gold)

    print('matched events : {} '.format(matched_events))
    print('matched args : {} '.format(matched_args))


def order_docs_by_topics(docs):
    '''
    Gets list of document objects and returns a Corpus object.
    The Corpus object contains Document objects, ordered by their gold topics
    :param docs: list of document objects
    :return: Corpus object
    '''
    corpus = Corpus()
    for doc_id, doc in docs.items():
        topic_id, doc_no = doc_id.split('_')
        if 'ecbplus' in doc_no:
            topic_id = topic_id + '_' +'ecbplus'
        else:
            topic_id = topic_id + '_' +'ecb'
        if topic_id not in corpus.topics:
            topic = Topic(topic_id)
            corpus.add_topic(topic_id, topic)
        topic = corpus.topics[topic_id]
        topic.add_doc(doc_id, doc)
    return corpus


def load_ECB_plus(processed_ecb_file):
    '''
    This function gets the intermediate data  (train/test/dev split after it was extracted
    from the XML files and stored as a text file) and load it into objects
    that represent a document structure
    :param processed_ecb_file: the filename of the intermediate representation of the split,
    which is stored as a text file
    :return: dictionary of document objects, represents the documents in the split
    '''
    doc_changed = True
    sent_changed = True
    docs = {}
    last_doc_name = None
    last_sent_id = None

    for line in open(processed_ecb_file, 'r'):
        stripped_line = line.strip()
        try:
            if stripped_line:
                doc_id,sent_id,token_num,word, coref_chain = stripped_line.split('\t')
                doc_id = doc_id.replace('.xml','')
        except:
            row = stripped_line.split('\t')
            clean_row = []
            for item in row:
                if item:
                    clean_row.append(item)
            doc_id, sent_id, token_num, word, coref_chain = clean_row
            doc_id = doc_id.replace('.xml', '')

        if stripped_line:
            sent_id = int(sent_id)

            if last_doc_name is None:
                last_doc_name = doc_id
            elif last_doc_name != doc_id:
                doc_changed = True
                sent_changed = True
            if doc_changed:
                new_doc = Document(doc_id)
                docs[doc_id] = new_doc
                doc_changed = False
                last_doc_name = doc_id

            if last_sent_id is None:
                last_sent_id = sent_id
            elif last_sent_id != sent_id:
                sent_changed = True
            if sent_changed:
                new_sent = Sentence(sent_id)
                sent_changed = False
                new_doc.add_sentence(sent_id,new_sent)
                last_sent_id = sent_id

            new_tok = Token(token_num,word,'-')
            new_sent.add_token(new_tok)

    return docs