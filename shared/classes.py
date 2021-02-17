from collections import defaultdict


class MentionData(object):
    '''
    An helper class for a mid-representation of a mention when reading the corpus.
    '''
    def __init__(self, doc_id, sent_id, tokens_numbers, tokens_str, coref_chain, mention_type='ACT',
                 is_continuous=True, is_singleton=False, score=float(-1)):
        '''

        :param doc_id: the mention's document ID
        :param sent_id: the mention's sentence ID
        :param tokens_numbers: the token IDs in the mention's text span
        :param tokens_str: the mention's string
        :param coref_chain: the mention's gold coreference chain
        :param mention_type: the mention's type, marked with ACT for event mention and with the
        following types for entity mentions: HUM for human participant, NON for non-human participant,
        LOC for location, and TIM for time.
        :param is_continuous: a variable indicates whether the mention span is continuous or not
        :param is_singleton: a variable indicates whether the mention belongs to a singleton  coreference
        cluster ( a cluster which contains only a single mention)
        :param score: a confidence score of the span for being a real mention (this score relates to
        predicted mentions only, and it sets to -1 dealing with gold mentions)
        '''
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.tokens_number = tokens_numbers
        self.tokens_str = tokens_str
        self.mention_type = mention_type
        self.coref_chain = coref_chain
        self.is_continuous = is_continuous
        self.is_singleton = is_singleton
        self.score = score


class Corpus(object):
    '''
    A class that represents a corpus, containing the documents of each split, grouped by topics
    (in a dictionary of Topic objects).
    '''
    def __init__(self):
        self.topics = {}

    def add_topic(self,topic_id, topic):
        '''
        Gets a topic id and a topic object and add it to the topics dictionary
        :param topic_id: topic id
        :param topic: topic object
        '''
        if topic_id not in self.topics:
            self.topics[topic_id] = topic


class Topic(object):
    '''
    A class that represents a topic in the corpus.
    It contains a dictionary of Document objects.
    '''
    def __init__(self,topic_id):
        self.topic_id = topic_id
        self.docs = {}

        self.event_mentions = []
        self.entity_mentions = []

    def add_doc(self, doc_id, doc):
        '''
        Gets a document id and document object and add it to the documents dictionary
        :param doc_id: document id
        :param doc: document object
        '''
        if doc_id not in self.docs:
            self.docs[doc_id] = doc


class Document(object):
    '''
    A class that represents a document.
    It contains the document ID and a dictionary of sentence objects.
    '''
    def __init__(self,doc_name):
        self.doc_id = doc_name
        self.sentences = {}

    def get_sentences(self):
        return self.sentences

    def add_sentence(self,sent_id,sent):
       if sent_id not in self.sentences:
           self.sentences[sent_id] = sent

    def add_mention(self,sent_id,mention):
        self.sentences[sent_id].add_mention(mention)

    def fetch_mention_string(self,sent_id,start_offset,end_offset):
        if sent_id in self.sentences:
            return self.sentences[sent_id].fetch_mention_string(start_offset,end_offset)
        else:
            return None

    def get_raw_doc(self, add_boundary):
        raw_doc = []
        for sent_id, sent in self.sentences.items():
            raw_doc.append(sent.get_raw_sentence())

        if add_boundary:
            return ' @@@ '.join(raw_doc)
        else:
            return ' '.join(raw_doc)

    def get_all_tokens(self):
        tokens = []
        for sent_id, sent in self.sentences.items():
            tokens.extend(sent.tokens)

        return tokens


class Sentence(object):
    '''
    A class that represents a sentence.
    It contains the sentence ID, a list of token objects, a list of event mention objects
     and a list of entity mention objects.
    '''
    def __init__(self,sent_id):
        '''
        A c'tor for a document object.
        sets the sentence ID, creates empty lists for the token and mention objects (gold mentions and predicted mentions).
        :param sent_id: the sentence ID (its ordinal number in the document)
        '''
        self.sent_id = sent_id # a string
        self.tokens = []
        self.gold_event_mentions = [] # gold event mentions
        self.gold_entity_mentions = [] # gold event mentions
        self.pred_event_mentions = []  # predicted event mentions
        self.pred_entity_mentions = []  # predicted entity mentions

    def add_token(self, token):
        self.tokens.append(token)

    def get_tokens(self):
        return self.tokens

    def get_raw_sentence(self):
        toks = []
        for tok in self.tokens:
            toks.append(tok.get_token())
        return ' '.join(toks)

    def get_tokens_strings(self):
        toks = []
        for tok in self.tokens:
            toks.append(tok.get_token())

        return toks

    def add_gold_mention(self, mention, is_event):
        if is_event:
            self.gold_event_mentions.append(mention)
        else:
            self.gold_entity_mentions.append(mention)

    def add_predicted_mention(self, mention, is_event, relaxed_match):
        if is_event:
            self.pred_event_mentions.append(mention)
        else:
            self.pred_entity_mentions.append(mention)

        return self.match_predicted_to_gold_mention(mention, is_event, relaxed_match)

    def match_predicted_to_gold_mention(self, pred_mention, is_event, relaxed_match):
        '''
        This function gets a predicted mention object and try to match it with a gold mention.
        The match is based on an exact string match, head match or
        a boundary match (one mention contains the other mention).
        Useful in a setting that requires a match
        :param pred_mention: the predicted mention
        :param is_event: a flag that indicates whether the mention is an event mention or an
         entity mention
        :return: True if a match was found
        '''
        gold_mentions = self.gold_event_mentions if is_event else self.gold_entity_mentions
        found = False

        for gold_mention in gold_mentions:
            if pred_mention.mention_str == gold_mention.mention_str and\
                    pred_mention.start_offset == gold_mention.start_offset \
                    and not gold_mention.has_compatible_mention:
                pred_mention.has_compatible_mention = True
                gold_mention.has_compatible_mention = True
                pred_mention.gold_mention_id = gold_mention.mention_id
                pred_mention.gold_tokens = gold_mention.tokens
                pred_mention.gold_start = gold_mention.start_offset
                pred_mention.gold_end = gold_mention.end_offset
                found = True
                break
            elif relaxed_match and self.same_head(pred_mention,gold_mention) and not gold_mention.has_compatible_mention: #not sure about the has_compatible_mention
                pred_mention.has_compatible_mention = True
                gold_mention.has_compatible_mention = True
                pred_mention.gold_mention_id = gold_mention.mention_id
                pred_mention.gold_tokens = gold_mention.tokens
                pred_mention.gold_start = gold_mention.start_offset
                pred_mention.gold_end = gold_mention.end_offset
                found = True
                break
            elif relaxed_match and self.i_within_i(pred_mention, gold_mention) and not gold_mention.has_compatible_mention:
                pred_mention.has_compatible_mention = True
                gold_mention.has_compatible_mention = True
                pred_mention.gold_mention_id = gold_mention.mention_id
                pred_mention.gold_tokens = gold_mention.tokens
                pred_mention.gold_start = gold_mention.start_offset
                pred_mention.gold_end = gold_mention.end_offset
                found = True
                break

        return found

    def i_within_i(self, mention_i, mention_j):
        '''
        Checks whether mention_i contains mention_j (or vice versa)
        :param mention_i: the first Mention object
        :param mention_j: the second Mention object
        :return: True if one mention contains the other, and False otherwise.
        '''
        if mention_i.start_offset >= mention_j.start_offset and mention_i.end_offset <= mention_j.end_offset \
            and len(set(mention_i.tokens_numbers).intersection(set(mention_j.tokens_numbers)) ) > 0:
            return True
        if mention_j.start_offset >= mention_i.start_offset and mention_j.end_offset <= mention_i.end_offset \
            and len(set(mention_i.tokens_numbers).intersection(set(mention_j.tokens_numbers))) > 0:
            return True
        return False

    def same_head(self, mention_i, mention_j):
        '''
        :return: True if they have a head match, and False otherwise.
        '''
        if mention_i.mention_head == mention_j.mention_head and \
                len(set(mention_i.tokens_numbers).intersection(set(mention_j.tokens_numbers))) > 0:
            return True
        return False

    def find_nearest_entity_mention(self, event, is_left, is_gold):
        '''
        :return: the closest entity if it was found, and None otherwise.
        '''
        sent_entities = self.gold_entity_mentions if is_gold else self.pred_entity_mentions
        event_start_idx =  event.start_offset
        event_end_idx = event.end_offset

        nearest_ent = None
        min_diff = float('inf')

        for entity in sent_entities:
            diff = event_start_idx - entity.end_offset if is_left else entity.start_offset - event_end_idx
            if diff < 0:
                continue
            if diff > 0 and diff < min_diff:
                if entity.mention_type != 'LOC' and entity.mention_type != 'TIM':
                    nearest_ent = entity
                    min_diff = diff
        return nearest_ent

    def fetch_mention_string(self, start_offset, end_offset):
        '''
        :return: the mention string and a list of token objects
        '''
        mention_tokens = []
        tokens = []
        for i in range(start_offset, end_offset+1):
            mention_tokens.append(self.tokens[i].get_token())
            tokens.append(self.tokens[i])
        return ' '.join(mention_tokens), tokens

    def find_mention_tokens(self, token_numbers):
        '''
        :return: a list of token objects
        '''
        tokens = []
        for token_number in token_numbers:
            tokens.append(self.tokens[token_number])

        return tokens


class Mention(object):
    '''
     An abstract class which represents a mention in the corpus.
    '''
    def __init__(self, doc_id, sent_id, tokens_numbers,tokens ,mention_str, head_text, head_lemma,
                 is_singleton, is_continuous, coref_chain):
        '''
        A c'tor for a mention object
        :param doc_id: the document ID
        :param sent_id: the sentence ID (its ordinal number in the document)
        :param start_offset: a start index of the mention's span
        :param end_offset: a end index of the mention's span
        :param mention_str: the string of the mention's span
        :param context: a string that represents the mention's context
        :param head_text: a string that represents mention's head
        :param head_lemma:  a string that represents mention's head lemma
        :param is_singleton: a boolean indicates whether the mention belongs to a singleton class
        :param is_continuous: a boolean indicates whether the mention span is continuous or not
        :param coref_chain: the mention's gold coreference chain
        '''
        self.doc_id = doc_id  # a string
        self.sent_id = sent_id  # a string
        self.start_offset = tokens_numbers[0] # an integer
        self.end_offset = tokens_numbers[-1] # an integer
        self.mention_id = '_'.join([doc_id,str(sent_id),str(self.start_offset),str(self.end_offset)])
        self.mention_str = mention_str
        self.mention_head = head_text
        self.mention_head_lemma = head_lemma
        self.is_singleton = is_singleton
        self.is_continuous = is_continuous
        self.tokens_numbers = tokens_numbers
        self.tokens = tokens
        self.gold_tag = coref_chain
        self.probability = 1.0
        self.cd_coref_chain = '-'
        self.wd_coref_chain = '-'
        self.has_compatible_mention = False
        self.gold_mention_id = None
        self.gold_tokens = []
        self.gold_start = None
        self.gold_end = None


    def get_tokens(self):
        return [tok.get_token() for tok in self.tokens]

    def get_head_index(self):
        '''
        Returns the token ID of the mention's head
        :return: the token ID of the mention's head
        '''
        for token in self.tokens:
            if token.get_token() == self.mention_head or self.mention_head in token.get_token():
                return token.token_id

    def __str__(self):
        return '{}_{}'.format(self.mention_str, self.gold_tag)

    @classmethod
    def get_comparator_function(cls):
        return lambda mention: (mention.doc_id, int(mention.sent_id) ,int(mention.start_offset))

class EntityMention(Mention):
    '''
    A class that represents an entity mention.
    '''
    def __init__(self, doc_id, sent_id, tokens_numbers,tokens,mention_str, head_text, head_lemma,
                 is_singleton, is_continuous, coref_chain, mention_type):
        super(EntityMention, self).__init__(doc_id, sent_id, tokens_numbers, tokens,
                                            mention_str, head_text, head_lemma, is_singleton,
                                            is_continuous, coref_chain)
        self.mention_type = mention_type

class EventMention(Mention):
    '''
    A class that represents an event mention.
    '''
    def __init__(self, doc_id, sent_id, tokens_numbers,tokens,mention_str, head_text, head_lemma,
                 is_singleton, is_continuous, coref_chain):
        super(EventMention, self).__init__(doc_id, sent_id, tokens_numbers,tokens,mention_str, head_text, head_lemma,
                 is_singleton, is_continuous, coref_chain)


class Token(object):
    '''
    A class represents a token in a sentence and contains the token ID
     (its ordinal number in the sentence), the token's string, its coreference gold chain and its predicted coreference chains
    '''
    def __init__(self, token_id,token,gold_coref_chain):
        self.token_id = token_id
        self.token = token
        self.gold_event_coref_chain = []
        self.gold_entity_coref_chain = []

    def get_token(self):
        return self.token


class Cluster(object):
    '''
    A class represents a coreference cluster
    '''
    def __init__(self, is_event):
        self.cluster_id = 0
        self.mentions = {}  # mention's dictionary, key is a mention id and value is a Mention object (either event or entity)
        self.is_event = is_event
        self.merged = False

    def __repr__(self):
        mentions_strings = []
        for mention in self.mentions.values():
            mentions_strings.append('{}_{}_{}'.format(mention.mention_str,
                                                      mention.gold_tag, mention.mention_id))
        return str(mentions_strings)

    def __str__(self):
        mentions_strings = []
        for mention in self.mentions.values():
            mentions_strings.append('{}_{}_{}'.format(mention.mention_str,
                                                      mention.gold_tag, mention.mention_id))
        return str(mentions_strings)

    def get_mentions_str_list(self):
        '''
        Returns a list contains the strings of all mentions in the cluster
        :return:
        '''
        mentions_strings = []
        for mention in self.mentions.values():
            mentions_strings.append(mention.mention_str)
        return mentions_strings



