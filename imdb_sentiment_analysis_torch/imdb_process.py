import logging
import os
import re
import sys
from itertools import chain

import gensim
import pandas as pd
import torch
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import pickle

embed_size = 300
max_len = 512

# Resolve paths relative to this script, so it works from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, 'corpus', 'imdb')
GLOVE_MODEL = os.path.join(os.path.dirname(BASE_DIR), 'model', 'glove_model.txt')

# Read data from files
train = pd.read_csv(os.path.join(CORPUS_DIR, "labeledTrainData.tsv"), header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(CORPUS_DIR, "testData.tsv"), header=0,
                   delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv(os.path.join(CORPUS_DIR, "unlabeledTrainData.tsv"), header=0,
                              delimiter="\t", quoting=3)


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    # if remove_stopwords:
    #     stops = set(stopwords.words("english"))
    #     words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)


def encode_samples(tokenized_samples):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=max_len, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    clean_train_reviews, train_labels = [], []
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        train_labels.append(train["sentiment"][i])

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))

    vocab = set(chain(*clean_train_reviews)) | set(chain(*clean_test_reviews))
    vocab_size = len(vocab)

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(clean_train_reviews, train_labels,
                                                                            test_size=0.2, random_state=0)

    wvmodel_file = GLOVE_MODEL
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(wvmodel_file, binary=False, encoding='utf-8')

    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'

    train_features = torch.tensor(pad_samples(encode_samples(train_reviews)))
    val_features = torch.tensor(pad_samples(encode_samples(val_reviews)))
    test_features = torch.tensor(pad_samples(encode_samples(clean_test_reviews)))

    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    weight = torch.zeros(vocab_size + 1, embed_size)
    for word in wvmodel.index_to_key:
        index = word_to_idx.get(word)
        if index is None:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(word))

    os.makedirs(os.path.join(BASE_DIR, 'pickle'), exist_ok=True)
    pickle_file = os.path.join(BASE_DIR, 'pickle', 'imdb_glove.pickle3')
    with open(pickle_file, 'wb') as f:
        pickle.dump(
            [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word, vocab],
            f)
    print('data dumped!')
