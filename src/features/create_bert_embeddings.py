import logging
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np

logger = logging.getLogger(__name__)


def find_token_index(tokenized_text):
    '''
    :param tokenized_text: find token indices of each word
    '''
    output = []
    current_word = []

    for i in range(0, len(tokenized_text)):
        if "Ġ" in tokenized_text[i]:
            if current_word:
                output.append(current_word)  # store the previous word
            current_word = [i]  # start documenting the current one
        else:
            current_word.append(i)
    output.append(current_word)  # the last word was not handled in the loop

    return output


def get_mean_embedding(embeddings):
    '''
    :param embeddings: a list of embeddings
    '''
    arrays = [np.array(x) for x in embeddings]
    return [np.mean(k) for k in zip(*arrays)]

class RobertaEmbedding(object):
    '''
    A wrapper class for the ElmoEmbedder of Allen NLP
    '''
    def __init__(self, layer_num):
        logger.info('Loading Roberta module')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base',
                                                  output_hidden_states = True)
        self.layer_num = -layer_num
        self.model.eval()
        logger.info('Roberta Embedding module loaded successfully')

    def get_embedding(self, sentence):
        '''
        This function gets a sentence object and returns and ELMo embeddings of
        each word in the sentences (specifically here, we average over the 3 ELMo layers).
        :param sentence: a sentence object
        :return: the averaged ELMo embeddings of each word in the sentences
        '''

        text = sentence.get_raw_sentence()
        # print(text)

        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(text)
        print("Bert tokenized text: %s"%" ".join(tokenized_text))
        # Map the token strings to their vocabulary indices.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # layer_i = 0
        # batch_i = 0
        # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)  # Sum the vectors from the last four layers.
            token_vecs_sum.append(sum_vec)

        token_indices = find_token_index(tokenized_text)
        # print(token_indices)
        # print(len(token_vecs_sum))
        embeddings = []
        for token_index in token_indices:
            start_index = token_index[0]
            end_index = token_index[-1] + 1
            if len(token_index) != 1:
                embeddings.append(get_mean_embedding(token_vecs_sum[start_index:end_index]))
            else:
                # print(token_index)
                embeddings.append(token_vecs_sum[token_index[0]])

        return embeddings





