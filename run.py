# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import tensorflow as tf
import json
import pandas as pd

from transformers import BertTokenizer, TFBertModel, TFBertForMaskedLM
from tensorflow.keras.models import load_model
from transformers import pipeline
import fasttext

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def generate_dis(unmasker, ds_model, sent, answer):
    WEIGHT = {"s0": 0.6, "s1": 0.15, "s2": 0.15, "s3": 0.1}
    # Answer relating
    target_sent = sent + " [SEP] " + answer

    # Generate Candidate Set
    cs = list()
    for cand in unmasker(target_sent):
        word = cand["token_str"].replace(" ", "")
        if len(word) > 0:  # Skip empty
            cs.append({"word": word, "s0": cand["score"], "s1": 0.0, "s2": 0.0, "s3": 0.0})

    # Confidence Score s0
    s0s = [c["s0"] for c in cs]
    new_s0s = min_max_y(s0s)

    for i, c in enumerate(cs):
        c["s0"] = new_s0s[i]

    # Word Embedding Similarity s1
    answer_vector = ds_model.get_word_vector(answer)
    word_similarities = list()
    for c in cs:
        c_vector = ds_model.get_word_vector(c["word"])
        word_similarity = similarity(answer_vector, c_vector)   # Cosine similarity between A and Di
        word_similarities.append(word_similarity)

    new_similarities = min_max_y(word_similarities)

    for i, c in enumerate(cs):
        c["s1"] = 1-new_similarities[i]

    # Contextual-Sentence Embedding Similarity s2
    correct_sent = sent.replace('[MASK]', answer)
    correct_sent_vector = ds_model.get_sentence_vector(correct_sent)

    cand_sents = list()
    for c in cs:
        cand_sents.append(sent.replace('[MASK]', c["word"]))

    sent_similarities = list()
    for cand_sent in cand_sents:
        cand_sent_vector = ds_model.get_sentence_vector(cand_sent)
        sent_similarity = similarity(correct_sent_vector, cand_sent_vector) # Cosine similarity between S(A) and S(Di)
        sent_similarities.append(sent_similarity)

    new_similarities = min_max_y(sent_similarities)
    for i, c in enumerate(cs):
        c["s2"] = 1-new_similarities[i]

    # POS match score s3
    origin_token = word_tokenize(sent)
    origin_token.remove("[")
    origin_token.remove("]")

    mask_index = origin_token.index("MASK")
    
    correct_token = word_tokenize(correct_sent)
    correct_pos = nltk.pos_tag(correct_token)
    answer_pos = correct_pos[mask_index]    # POS of A

    for i, c in enumerate(cs):
        cand_sent_token = word_tokenize(cand_sents[i])
        cand_sent_pos = nltk.pos_tag(cand_sent_token)
        cand_pos = cand_sent_pos[mask_index]    # POS of Di

        if cand_pos[1] == answer_pos[1]:
            c["s3"] = 1.0
        else:
            c["s3"] = 0.0

    # Weighted final score
    cs_rank = list()
    for c in cs:
        fs = WEIGHT["s0"]*c["s0"] + WEIGHT["s1"]*c["s1"] + WEIGHT["s2"]*c["s2"] + WEIGHT["s3"]*c["s3"]
        cs_rank.append((c["word"], fs))

    # Rank by final score
    cs_rank.sort(key=lambda x: x[1], reverse=True)

    # Top K
    result = [d[0] for d in cs_rank[:10]]

    return result


# Cosine similarity
def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:  # Denominator can not be zero
        return 1
    else:
        return np.dot(v1, v2) / (n1 * n2)


# Min–max normalization
def min_max_y(raw_data):
    min_max_data = []

    # Min–max normalization
    for d in raw_data:
        min_max_data.append((d - min(raw_data)) / (max(raw_data) - min(raw_data)))

    return min_max_data



clozeTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# clozeBert = TFBertModel.from_pretrained("bert-base-uncased")
# clozeModelv2 = load_model('./J/Models/2.0')
# clozeModelv3 = load_model('./J/Models/3.0')
# clozeModelv3_1 = load_model('./J/Models/3.1')
clozeModelv3_2 = load_model('./Models/3.2')

distractorTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
distractorModel = TFBertForMaskedLM.from_pretrained("./Models/distractors", from_pt = True)

unmasker = pipeline("fill-mask", tokenizer=distractorTokenizer, model=distractorModel, top_k=10)
fasttext.FastText.eprint = lambda x: None
ds_model = fasttext.load_model('./Models/cdgp-ds-fasttext.bin')



DONTPICKTHESE = set(stopwords.words('english')).union(set(['it', ',', '.', '?', '!', '"', "'", '[sep]', '[cls]', '[pad]', '—', '“']))

def SentenceandDistractors(sentence, v='2.0'):

    '''
    if v == '2.0':
      
        clozeTokenized = clozeTokenizer(sentence, return_tensors='tf')
        clozeModel = clozeModelv2
        pot, prob, *_ = clozeModel({'input_token': clozeTokenized.input_ids, 'masked_token': clozeTokenized.attention_mask})
        _, indices = tf.math.top_k(prob[0, 1:-1, 4], k=prob.shape[1]-3)
    elif v == '3.0':
        clozeTokenized = clozeTokenizer(sentence, padding='max_length', max_length=64, truncation=True, return_tensors='tf')
        clozeModel = clozeModelv3
        pot, prob, *_ = clozeModel({k: v for k, v in clozeTokenized.items() if k != 'token_type_ids'})
        _, indices = tf.math.top_k(prob[0, :, 4], k=prob.shape[1])
    elif v == '3.1':
        clozeTokenized = clozeTokenizer(sentence, padding='max_length', max_length=64, truncation=True, return_tensors='tf')
        clozeModel = clozeModelv3_1
        pot, prob, *_ = clozeModel({k: v for k, v in clozeTokenized.items() if k != 'token_type_ids'})
        _, indices = tf.math.top_k(prob[0, :, 4], k=prob.shape[1])
    '''
    if v == '3.2':
        clozeTokenized = clozeTokenizer(sentence, padding='max_length', max_length=64, truncation=True, return_tensors='tf')
        clozeModel = clozeModelv3_2
        pot, prob, *_ = clozeModel({k: v for k, v in clozeTokenized.items() if k != 'token_type_ids'})
        _, indices = tf.math.top_k(prob[0, :, 4], k=prob.shape[1])
      
    for i in indices:
        
        clozeWord = clozeTokenizer.decode(clozeTokenized.input_ids[0, i]).replace(' ', '').lower()
        
        if clozeWord not in DONTPICKTHESE and not clozeWord.startswith('#') and not clozeWord.endswith('#'):
            index = i
            break
      
    distractorSentence = clozeTokenizer.decode(tf.concat([clozeTokenized.input_ids[0][:index], tf.constant([103]), clozeTokenized.input_ids[0][index+1:]], axis=0))
    result = generate_dis(unmasker, ds_model, distractorSentence, clozeWord)

    return {'answer': clozeWord, 'distractor': result, 'sentence': distractorSentence}