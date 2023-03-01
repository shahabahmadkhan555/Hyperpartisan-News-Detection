from defaults import *
import argparse
import os
import json
import math
import numpy as np
from optparse import OptionParser
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

batch_size = 50
maxTokens = 200
maxSentences = 200

bert_model_name = 'bert-base-cased'
bert_tokenizer = None
bert_model = None
maxTokens = -1
outfile = ''

# model = None

def setParser():
    parser = OptionParser()
    parser.add_option("--bertmodel", help="Bert Model", type=str, default='bert-base-cased')
    # parser.add_option("--bsize", help="Batch Size", type=int, default=50)
    parser.add_option("--outfile", help="Name of Pickled model", type=str, default='cased')
    parser.add_option("--maxtokens", help="Max tokens in a sentence", type=int, default=512)
    # parser.add_option("--maxsentences", help="Max sentences in an article", type=int, default=200)
    options, _ = parser.parse_args()
    return options

def openTextConvert():
    global outfile
    
    dataset = []

    counter = 0

    with open(train_text_tsv, 'r') as tsv:
        for ts in tsv:
            print(counter+1)
            counter+=1

            data_elem = {}

            arr = ts.split('\t')

            isHyper = arr[1]
            if isHyper == 'true':
                isHyper = True
            else:
                isHyper = False

            data_elem['is_hyper'] = isHyper

            sent = arr[4]
            arr_sent = sent.split('<splt>')

            data_elem['sentences'] = arr_sent

            finalSent = '[CLS] '
            for i in range(len(arr_sent)):
                finalSent += arr_sent[i].strip() + ' [SEP] '

            tokenized, indexed_tokens, segment_ids = bertSentTokenize(finalSent)

            indexed_tokens_tensor, segment_ids_tensor = get_tensors(indexed_tokens, segment_ids)
            encoded_layers = getEncodingLayers(indexed_tokens_tensor, segment_ids_tensor)
 
            # visualizeEncoding(encoded_layers)
            # checkEncodedLayers(encoded_layers)
            # print(type(encoded_layers))

            token_vecs = encoded_layers[11][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            
            # print(sentence_embedding.size())
            # print(sentence_embedding)

            data_elem['tensor'] = sentence_embedding
            data_elem['tensor_size'] = sentence_embedding.size()

            dataset.append(data_elem)
            # token_embeddings = getTokenEmbeddings(encoded_layers)
            # print(token_embeddings.size())
            # TODO: Remove this break
            # break
    # print(dataset)
    print('Saving model')
    savePickle(dataset, outfile)
    print('Model saved')



def getTokenEmbeddings(encoded_layers):
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)    
    return token_embeddings

def checkEncodedLayers(encoded_layers):
    print('Encoded Layers Type', type(encoded_layers))
    for i in range(len(encoded_layers)):
        print('Layer', i+1, encoded_layers[i].size())

def getEncodingLayers(tokens_tensor, segments_tensors):
    # Predict hidden state features for each layer
    global bert_model
    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)      
    return encoded_layers

def visualizeEncoding(encoded_layers):
    import matplotlib.pyplot as plt
    token_ind = 5
    layer_ind = 5
    batch_ind = 0
    vecs = encoded_layers[layer_ind][batch_ind][token_ind]

    plt.figure(figsize=(10,10))
    plt.hist(vecs, bins=200)
    # plt.show()
    plt.savefig('vis.png')

def get_segments_ids(tokens):
    segments_ids = [1] * len(tokens)
    # print('seg id len', len(segments_ids))
    return segments_ids

def get_tensors(indexed_tokens,segments_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    return tokens_tensor, segments_tensors

def bertSentTokenize(sent):
    global bert_tokenizer, maxTokens

    # TODO: Change length here
    tokenized = bert_tokenizer.tokenize(sent)[:maxTokens]
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized)
    segment_ids = get_segments_ids(tokenized)
    
    return tokenized, indexed_tokens, segment_ids
    # print(segment_ids)
    # print(tokenized)
    # for i in arr:
    #     tokens,indexed_tokens = sentTokenize(i)
    #     segments_ids = get_segments_ids(tokens)
    #     tokens_tensor,segments_tensors = get_tensors(indexed_tokens,segments_ids)
    #     # TODO: Remove break
    #     break

# def get_vectors(model,tokens_tensor,segments_tensors):
#     # Convert the hidden state embeddings into single token vectors

#     # Holds the list of 12 layer embeddings for each token
#     # Will have the shape: [# tokens, # layers, # features]
#     with torch.no_grad():
#         encoded_layers, _ = model(tokens_tensor, segments_tensors)   
#     token_embeddings = [] 
#     # For the 5th token in our sentence, select its feature values from layer 5.

#     # For each token in the sentence...
#     for token_i in range(len(tokenized_text)):
    
#     # Holds 12 layers of hidden states for each token 
#         hidden_layers = [] 
    
#     # For each of the 12 layers...
#         for layer_i in range(len(encoded_layers)):
            
#             # Lookup the vector for `token_i` in `layer_i`
#             vec = encoded_layers[layer_i][batch_i][token_i]
            
#             hidden_layers.append(vec)
            
#         token_embeddings.append(hidden_layers)

#     # Sanity check the dimensions:
#     print ("Number of tokens in sequence:", len(token_embeddings))
#     print ("Number of layers per token:", len(token_embeddings[0])) 
#     sentence_embedding = torch.mean(encoded_layers[11], 1)

# # def sentTokenize(sent):
# #     global bert_tokenizer
# #     tokens = bert_tokenizer.tokenize(sent) 
# #     #index of tokens in BERT
# #     indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokens)

# #     for tup in zip(tokens, indexed_tokens):
# #         print (tup)    
# #     print(tokens)
# #     return tokens, indexed_tokens

def loadBertModel():
    global bert_tokenizer, bert_model, bert_model_name
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=False)   

    bert_model = BertModel.from_pretrained(bert_model_name)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()   
    print('Bert loaded')
    return None

if __name__=='__main__':
    options = setParser()
    # batch_size = options.bsize
    maxTokens = int(options.maxtokens)
    bert_model_name = options.bertmodel
    outfile = options.outfile
    # maxSentences = options.maxsentences

    loadBertModel()
    openTextConvert()