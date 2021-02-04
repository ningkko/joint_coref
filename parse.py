# -*- coding: utf-8 -*-
import os
import csv
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from mention_data import MentionData
from _token import Token 

# This file parses ecb data

parser = argparse.ArgumentParser(description='Parsing ECB+ corpus')

parser.add_argument('--ecb_path', type=str,
                    help=' The path to the ECB+ corpus')
parser.add_argument('--output_dir', type=str,
                        help=' The directory of the output files')
# parser.add_argument('--selected_sentences_file', type=str,
#                     help=' The path to a file contains selected sentences from the ECB+ corpus (relevant only for '
#                          'the second evaluation setup (Cybulska setup)')

args = parser.parse_args()
args.selected_sentences_file = 'data/raw/ECBplus_coreference_sentences.csv'

out_dir = args.output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def read_selected_sentences(filename):
    '''
    ## This function selects sentences based on the suggestion of Cy, etc.
    ## The file referred (data/raw/ECBplus_coreference_sentences.csv) is a file manually created by the author

    Topic     File      Sentence Number
    1       19ecbplus        1
    1       19ecbplus        3
    1       19ecbplus        4

    :param filename: the CSV file
    :return: a dictionary
            {"topic_file.xml" : sentence number}
    '''
    xml_to_sent_dict = {}

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        for line in reader:
            xml_filename = '{}_{}.xml'.format(line[0],line[1])
            sent_id = int(line[2])

            if xml_filename not in xml_to_sent_dict:
                xml_to_sent_dict[xml_filename] = []
            xml_to_sent_dict[xml_filename].append(sent_id)

    return xml_to_sent_dict


def find_mention_class(tag):
    '''
    Given a string represents a mention type, this function returns its abbreviation
    :param tag:  a string represents a mention type
    :return: Abbreviation of the mention type (a string)
    '''
    if 'ACTION' in tag:
        return 'ACT'
    if 'LOC' in tag:
        return 'LOC'
    if 'NON' in tag:
        return 'NON'
    if 'HUMAN' in tag:
        return 'HUM'
    if 'TIME' in tag:
        return 'TIM'
    else:
        print(tag)

def coref_chain_id_to_mention_type(coref_chain_id):
    '''
    Given a string represents a mention's coreference chain ID,
    this function returns a string that represents the mention type.
    :param coref_chain_id: a string represents a mention's coreference chain ID
    :return: a string that represents the mention type
    '''
    if 'ACT' in coref_chain_id or 'NEG' in coref_chain_id:
        return 'ACT'
    if 'HUM' in coref_chain_id or 'CON' in coref_chain_id:
        return 'HUM'
    else:
        return coref_chain_id


def read_ecb_plus_doc(docs,file_obj):
    '''
    This function reads an ECB+ XML file (i.e. document), extracts its gold mentions and texts.
    the text file of each split is written as 5 columns -  the first column contains the document
    ID (which has the following format - {topic id}_{docuemnt id}_{ecb/ecbplus type}). Note that the topic id
    and the document type determines the sub-topic id, e.g. a suc-topic with topic_id = 1 and
    document_type = ecbplus is different from a sub-topic with topic_id = 1 and document_type = ecb.
    the second column cotains the sentence id, the third column contains the token id,
    and the fourth column contains the token string. The fifth column should be ignored.
    :param docs: indicators of the ECB+ document: [selected_sent_list, doc_filename, doc_id]
    :param extracted_mentions: a list of split's extracted mentions
    '''
    extracted_mentions = []
    # @TODO: Check if these 2 are the same 
    #ecb_file = open(file, 'r')

    for doc in docs:
        selected_sent_list, doc_filename, doc_id = doc[0], doc[1], doc[2]
        ecb_file = open(doc_filename, 'r')
        tree = ET.parse(ecb_file)
        root = tree.getroot()

        related_events = {}
        within_coref = {}
        mid_to_tid_dict = {}
        mid_to_event_tag_dict = {}
        mid_to_tag = {}
        tokens = {}
        cur_mid = ''

        mid_to_coref_chain = {}

        ## Sample doc 
        # <Document doc_name="5_2ecbplus.xml" doc_id="DOC15646029050754558">
        # <token t_id="988" sentence="44" number="30">play</token>
        # <token t_id="989" sentence="44" number="31">,</token>
        # <token t_id="990" sentence="44" number="32">"</token>
        # <token t_id="991" sentence="44" number="33">King</token>
        # <token t_id="992" sentence="44" number="34">said</token>
        # <token t_id="993" sentence="44" number="35">.</token>
        #
        # <Markables>
        # <ACTION_OCCURRENCE m_id="11"  >
        #   <token_anchor t_id="55"/>
        # </ACTION_OCCURRENCE>
        # <ACTION_OCCURRENCE m_id="13"  >
        #   <token_anchor t_id="23"/>
        #   <token_anchor t_id="24"/>
        # </ACTION_OCCURRENCE>
        # <ACTION_OCCURRENCE m_id="49"  >
        #   <token_anchor t_id="168"/>
        # </Markables>
        #
        # <Relations>
        # <CROSS_DOC_COREF r_id="24019" note="HUM16703856404579669" >
        #   <source m_id="56" />
        #   <source m_id="65" />
        #   <target m_id="73" />
        # </CROSS_DOC_COREF>
        # <CROSS_DOC_COREF r_id="24020" note="HUM16701709483995952" >
        #   <source m_id="30" />
        #   <source m_id="58" />
        #   <source m_id="16" />
        #   <source m_id="17" />
        #   <source m_id="35" />
        #   <target m_id="74" />
        # </CROSS_DOC_COREF>
        # </CROSS_DOC_COREF>
        # </Relations>
        # </Document>

        ## find tag discriptors 
        for action in root.find('Markables').iter():
            if action.tag == 'Markables':
                continue
            elif action.tag == 'token_anchor':
                mid_to_tid_dict[cur_mid].append(action.attrib['t_id'])
            else:
                cur_mid = action.attrib['m_id']

                if 'TAG_DESCRIPTOR' in action.attrib:
                    if 'instance_id' in action.attrib:
                        mid_to_event_tag_dict[cur_mid] = (
                            action.attrib['TAG_DESCRIPTOR'], action.attrib['instance_id'])
                    else:
                        mid_to_event_tag_dict[cur_mid] = (action.attrib['TAG_DESCRIPTOR'], action.tag) #intra doc coref
                else:
                    mid_to_tid_dict[cur_mid] = []
                    mid_to_tag[cur_mid] = action.tag

        ## find cross-doc info
        cur_instance_id = ''
        source_ids = []
        mapped_mid = []

        for within_doc_coref in root.find('Relations').findall('INTRA_DOC_COREF'):
            for child in within_doc_coref.iter():
                if child.tag == 'INTRA_DOC_COREF':

                    mention_coref_class = mid_to_event_tag_dict[within_doc_coref.find('target').get('m_id')][1] # find the mention class of intra doc coref mention
                    if mention_coref_class == 'UNKNOWN_INSTANCE_TAG':
                        cls = 'UNK'
                    else:
                        mention_class = mid_to_tag[within_doc_coref.find('source').get('m_id')]
                        cls = find_mention_class(mention_class)
                    cur_instance_id = 'INTRA_{}_{}_{}'.format(cls,child.attrib['r_id'],doc_id)
                    within_coref[cur_instance_id] = ()
                else:
                    if child.tag == 'source':
                        source_ids += (mid_to_tid_dict[child.attrib['m_id']])
                        mapped_mid.append(child.attrib['m_id'])
                        mid_to_coref_chain[child.attrib['m_id']] = cur_instance_id
                    else:
                        within_coref[cur_instance_id] = (source_ids, mid_to_event_tag_dict[child.attrib['m_id']][0])
                        source_ids = []

        cur_instance_id = ''
        source_ids = []

        for cross_doc_coref in root.find('Relations').findall('CROSS_DOC_COREF'):
            for child in cross_doc_coref.iter():
                if child.tag == 'CROSS_DOC_COREF':
                    related_events[child.attrib['note']] = ()
                    cur_instance_id = child.attrib['note']
                else:
                    if child.tag == 'source':
                        source_ids += (mid_to_tid_dict[child.attrib['m_id']])
                        mapped_mid.append(child.attrib['m_id'])
                        mid_to_coref_chain[child.attrib['m_id']] = cur_instance_id
                    else:
                        related_events[cur_instance_id] = (
                            source_ids, mid_to_event_tag_dict[child.attrib['m_id']][0])
                        source_ids = []

        for token in root.findall('token'):
            tokens[token.attrib['t_id']] = Token(token.text, token.attrib['sentence'], token.attrib['number'])

        for key in related_events:
            for token_id in related_events[key][0]:
                tokens[token_id].rel_id = (key, related_events[key][1])
        for key in within_coref:
            for token_id in within_coref[key][0]:
                tokens[token_id].rel_id = (key, within_coref[key][1])

        for mid in mid_to_tid_dict:
            if mid not in mapped_mid:  # singleton mention
                mention_class = mid_to_tag[mid]
                cls = find_mention_class(mention_class)
                singleton_instance_id = 'Singleton_{}_{}_{}'.format(cls,mid,doc_id )
                mid_to_coref_chain[mid] = singleton_instance_id
                unmapped_tids = mid_to_tid_dict[mid]
                for token_id in unmapped_tids:
                    if tokens[token_id].rel_id is None:
                        tokens[token_id].rel_id = (singleton_instance_id,'padding')

        # creating an instance for each mention
        for mid in mid_to_tid_dict:
            tids = mid_to_tid_dict[mid]
            token_numbers = []  # the ordinal token numbers of each token in its sentence
            tokens_str = []
            sent_id = None

            if mid not in mid_to_coref_chain:
                continue
            coref_chain = mid_to_coref_chain[mid]
            type_tag = mid_to_tag[mid]
            mention_type = find_mention_class(type_tag)

            mention_type_by_coref_chain = coref_chain_id_to_mention_type(coref_chain)
            if mention_type != mention_type_by_coref_chain:
                print('coref chain: {}'.format(coref_chain))
                print('mention type by coref chain: {}'.format(mention_type_by_coref_chain))
                print('mention type: {}'.format(mention_type))

            for token_id in tids:
                token = tokens[token_id]
                if sent_id is None:
                    sent_id = int(token.sent_id)

                if int(token.tok_id) not in token_numbers:
                    token_numbers.append(int(token.tok_id))
                    # tokens_str.append(token.text.encode('ascii', 'ignore'))
                    tokens_str.append(token.text)

            is_continuous = True if token_numbers == range(token_numbers[0], token_numbers[-1]+1) else False
            is_singleton = True if 'Singleton' in coref_chain else False
            if sent_id in selected_sent_list:
                if 'plus' in doc_id:
                    if sent_id > 0:
                        sent_id -= 1
                print(tokens_str)

                mention_obj = MentionData(doc_id, sent_id, token_numbers, ' '.join(tokens_str),
                                           coref_chain, mention_type,is_continuous=is_continuous,
                                           is_singleton=is_singleton, score=float(-1))

                extracted_mentions.append(mention_obj)
        prev_sent_id = None

        for token in root.findall('token'):
            token = tokens[token.attrib['t_id']]
            token_id = int(token.tok_id)
            sent_id = int(token.sent_id)
            text = token.text
            if sent_id not in selected_sent_list:
                continue
            if 'plus' in doc_id:
                if sent_id > 0:
                    sent_id -= 1
                else:
                    continue

            if prev_sent_id is None or prev_sent_id != sent_id:
                file_obj.write('\n')
                prev_sent_id = sent_id
            # text = token.text.encode('ascii', 'ignore')

            if text == '' or text == '\t':
                text = '-'

            if token.rel_id is not None:
                file_obj.write(doc_id + '\t' + str(sent_id) + '\t' + str(token_id) + '\t' + text + '\t' + \
                                token.rel_id[0] + '\n')
            else:
                # print(type(text))
                file_obj.write(doc_id + '\t' + str(sent_id) + '\t' + str(token_id) + '\t' + text + '\t-' + '\n')
    
    return extracted_mentions

def obj_dict(obj):
    return obj.__dict__
    
def save_split_mentions_to_json(split_name, mentions_list):
    '''
    This function gets a mentions list of a specific split and saves its mentions in a JSON files.
    Note that event and entity mentions are saved in separate files.
    :param split_name: the split name
    :param mentions_list: the split's extracted mentions list
    '''
    event_mentions = []
    entity_mentions = []

    for mention_obj in mentions_list:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)

    json_event_filename = os.path.join(args.output_dir, 'ECB_{}_Event_gold_mentions.json'.format(split_name))
    json_entity_filename =  os.path.join(args.output_dir, 'ECB_{}_Entity_gold_mentions.json'.format(split_name))

    with open(json_event_filename, 'w') as f:
        json.dump(event_mentions, f, default=obj_dict, indent=4, sort_keys=True)

    with open(json_entity_filename, 'w') as f:
        json.dump(entity_mentions, f, default=obj_dict, indent=4, sort_keys=True)

def parse_selected_sentences(xml_to_sent_dict):
    '''
    Reads in selected sentences as in Cybulska's setup (the better-performance one)
    :param xml_to_sent_dict: selected sentences dictionary
    '''

    ## split the corpus into train, test, and dev 
    dev_topics = [2, 5, 12, 18, 21, 23, 34, 35]
    train_topics = [i for i in range(1,36) if i not in dev_topics]  # train topics 1-35 , test topics 36-45

    dirs = os.listdir(args.ecb_path)
    dirs_int = [int(dir) for dir in dirs]
    train_ecb_files_sorted = []
    test_ecb_files_sorted = []
    train_ecb_plus_files_sorted = []
    test_ecb_plus_files_sorted = []
    dev_ecb_files_sorted = []
    dev_ecb_plus_files_sorted = []

    for topic in sorted(dirs_int):
        dir = str(topic)

        doc_files = os.listdir(os.path.join(args.ecb_path,dir))
        ecb_files = []
        ecb_plus_files = []
        for doc_file in doc_files:
            if 'plus' in doc_file:
                ecb_plus_files.append(doc_file) # ECB and ECB+ has different contents though with same topic & sentence numbers 
            else:
                ecb_files.append(doc_file)

        ecb_files = sorted(ecb_files)
        ecb_plus_files=sorted(ecb_plus_files)

        for ecb_file in ecb_files:
            if ecb_file in xml_to_sent_dict:
                xml_filename = os.path.join(os.path.join(args.ecb_path,dir),ecb_file)
                selected_sentences = xml_to_sent_dict[ecb_file]
                if topic in train_topics:
                    train_ecb_files_sorted.append((selected_sentences, xml_filename,
                                                   ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_files_sorted.append((selected_sentences, xml_filename,
                                                   ecb_file.replace('.xml', '')))
                else:
                    test_ecb_files_sorted.append((selected_sentences, xml_filename,
                                                  ecb_file.replace('.xml', '')))
        ## do the sam for ECB+
        for ecb_file in ecb_plus_files:
            if ecb_file in xml_to_sent_dict:
                xml_filename = os.path.join(os.path.join(args.ecb_path,dir),ecb_file)
                selected_sentences = xml_to_sent_dict[ecb_file]
                if topic in train_topics:
                    train_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_filename, ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_filename, ecb_file.replace('.xml', '')))
                else:
                    test_ecb_plus_files_sorted.append(
                        (selected_sentences, xml_filename, ecb_file.replace('.xml', '')))

    dev_files = dev_ecb_files_sorted + dev_ecb_plus_files_sorted
    train_files = train_ecb_files_sorted + train_ecb_plus_files_sorted
    test_files = test_ecb_files_sorted + test_ecb_plus_files_sorted

    # # print(dev_files)
    # [([1], 'data/raw/ECB+/35/35_8ecbplus.xml', '35_8ecbplus'), 
    #  ([4], 'data/raw/ECB+/35/35_9ecbplus.xml', '35_9ecbplus'),
    #  ...]
    ## now extract mentions 

    with open(os.path.join(args.output_dir, 'ECB_Dev_corpus.txt'), 'w') as dev_out:
        save_split_mentions_to_json('Dev', read_ecb_plus_doc(dev_files, dev_out))
    dev_out.close()

    with open(os.path.join(args.output_dir, 'ECB_Train_corpus.txt'), 'w') as train_out:
        save_split_mentions_to_json('Train', read_ecb_plus_doc(train_files, train_out))
    dev_out.close()

    with open(os.path.join(args.output_dir, 'ECB_Test_corpus.txt'), 'w') as test_out:
        save_split_mentions_to_json('Test', read_ecb_plus_doc(test_files, test_out))
    dev_out.close()


def main():
    '''
        Parses the ECB+ corpus

    '''
    logger.info('Read ECB+ files')
    xml_to_sent_dict = read_selected_sentences(args.selected_sentences_file)
    parse_selected_sentences(xml_to_sent_dict=xml_to_sent_dict)
    logger.info('ECB+ Reading was done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()