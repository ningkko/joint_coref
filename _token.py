class Token(object):
    '''
    An helper class which represents a single when reading the corpus.
    '''
    def __init__(self, text, sent_id, tok_id, rel_id=None):
        '''

        :param text: The token text
        :param sent_id: The sentence id
        :param tok_id: The token id
        :param rel_id: The relation id (i.e. coreference chain)
        '''

        self.text = text
        self.sent_id = sent_id
        self.tok_id = tok_id
        self.rel_id = rel_id