import jieba
import numpy
import os
import sys
import pickle
import random
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from janome.tokenizer import Tokenizer as JanomeTokenizer

MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 4
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1


def shuffle_kv(d):
    keys = []
    values = []
    for key, value in d.items():
        keys.append(key)
        values.append(value)
    random.shuffle(values)
    return dict(zip(keys, values))


def split_data(data, labels):
    # split data into training and validation
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    # 80% to train, 10% to validation, 10% to test
    nb_validation_test_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data.shape[0])
    nb_test_samples = int((TEST_SPLIT) * data.shape[0])

    x_train = data[:-nb_validation_test_samples]
    y_train = labels[:-nb_validation_test_samples]
    x_val = data[-nb_validation_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_test_samples:-nb_test_samples]
    x_test = data[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_ChnSenti_texts(dirname):
    TEXT_DATA_DIR = dirname
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    # maxlen = 0    # used the max length of sentence in the data. but large number makes cudnn die
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='gbk')
                try:
                    t = f.read()
                except UnicodeDecodeError:
                    continue
                t = t.translate(str.maketrans("", "", "\n"))
                if len(t) > MAX_SENTENCE_LENGTH:
                    t = t[:MAX_SENTENCE_LENGTH]
                texts.append(t)
                f.close()
                labels.append(label_id)
    print('Found %s texts.' % len(texts))
    return texts, labels, labels_index


def get_Rakuten_texts(datasize):
    data_limit_per_class = datasize // 2
    data_size = data_limit_per_class * 2
    with open("../CMWE/rakuten/rakuten_review.pickle", "rb") as f:
        positive, negative = pickle.load(f)
    random.shuffle(positive)
    random.shuffle(negative)
    positive = positive[:data_limit_per_class]
    negative = negative[:data_limit_per_class]
    labels = [1] * data_limit_per_class + [0] * data_limit_per_class
    return positive + negative, labels


def prepare_char(lang, shuffle=None, dict_limit=0):
    # shuffle characters in each word: word -> dorw (shuffle="shuffle") or randomly swap words (shuffle="random")
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    if lang == "CH":
        texts, labels, labels_index = get_ChnSenti_texts("ChnSentiCorp_htl_unba_10000/")
    elif lang == "JP":
        janome_tokenizer = JanomeTokenizer()
        datasize = 10000
        texts, labels = get_Rakuten_texts(datasize)
    data_size = len(texts)
    processed_texts = []
    word_vocabulary = {}
    # build word_vocabulary
    for i, text in enumerate(tqdm(texts)):
        if lang == "CH":
            t_list = list(jieba.cut(text, cut_all=False))
        elif lang == "JP":
            t_list = janome_tokenizer.tokenize(text)
        processed_texts.append(t_list)
        for word in t_list:
            if lang == "JP":
                word = word.surface
            word_vocabulary[word] = word
    if shuffle == "random":
        word_vocabulary = shuffle_kv(word_vocabulary)
    elif shuffle == "shuffle":
        word_vocabulary_new = {}
        for k, v in word_vocabulary.items():
            list_v = list(v)
            random.shuffle(list_v)
            word_vocabulary_new[k] = "".join(list_v)
        word_vocabulary = word_vocabulary_new
    else:
        pass
    # build data
    char_vocab = ["</s>"]
    data_char = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH), dtype=numpy.int32)  # data_char
    if dict_limit > 0:
        char_vocab_freq = {"</s>": 2 ** 30}
        # order by the freq
        for i, text in enumerate(tqdm(processed_texts)):
            for j, word in enumerate(text):
                if lang == "JP":
                    word = word.surface
                for k, char in enumerate(word_vocabulary[word]):
                    if char not in char_vocab:
                        char_vocab.append(char)
                        char_vocab_freq[char] = 1
                    else:
                        char_vocab_freq[char] += 1
        sorted_char_vocab_freq = sorted(char_vocab_freq.items(), key=lambda x: -x[1])
        char_vocab = [k for k, v in sorted_char_vocab_freq]
    for i, text in enumerate(tqdm(processed_texts)):
        for j, word in enumerate(text):
            if lang == "JP":
                word = word.surface
            if j < MAX_SENTENCE_LENGTH:
                for k, char in enumerate(word_vocabulary[word]):
                    if char not in char_vocab:
                        char_vocab.append(char)
                        char_index = len(char_vocab) - 1
                    else:
                        char_index = char_vocab.index(char)
                    if k < MAX_WORD_LENGTH:
                        if dict_limit == 0 or char_index < dict_limit:
                            data_char[i, j, k] = char_index
    labels = to_categorical(numpy.asarray(labels))
    # split data into training and validation
    indices = numpy.arange(data_char.shape[0])
    numpy.random.shuffle(indices)
    data_char = data_char[indices]
    labels = labels[indices]
    # 80% to train, 10% to validation, 10% to test
    nb_validation_test_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data_char.shape[0])
    nb_test_samples = int((TEST_SPLIT) * data_char.shape[0])

    x_train = data_char[:-nb_validation_test_samples]
    y_train = labels[:-nb_validation_test_samples]
    x_val = data_char[-nb_validation_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_test_samples:-nb_test_samples]
    x_test = data_char[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]

    if dict_limit > 0:
        char_vocab_size = dict_limit
    else:
        char_vocab_size = len(char_vocab)

    return x_train, y_train, x_val, y_val, x_test, y_test, char_vocab_size


def prepare_word(lang, dict_limit=0):
    # shuffle characters in each word: word -> dorw (shuffle="shuffle") or randomly swap words (shuffle="random")
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    if lang == "CH":
        texts, labels, labels_index = get_ChnSenti_texts("ChnSentiCorp_htl_unba_10000/")
    elif lang == "JP":
        janome_tokenizer = JanomeTokenizer()
        datasize = 10000
        texts, labels = get_Rakuten_texts(datasize)
    data_size = len(texts)
    processed_texts = []
    word_freq = {}
    # build word_vocabulary
    for i, text in enumerate(tqdm(texts)):
        if lang == "CH":
            t_list = list(jieba.cut(text, cut_all=False))
        elif lang == "JP":
            t_list = janome_tokenizer.tokenize(text)
        processed_texts.append(t_list)
        for word in t_list:
            if lang == "JP":
                word = word.surface
            if word not in list(word_freq.keys()):
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    sorted_vocab_freq = sorted(word_freq.items(), key=lambda x: -x[1])
    word_vocab = ["</s>"] + [k for k, v in sorted_vocab_freq]
    data_char = numpy.zeros((data_size, MAX_SENTENCE_LENGTH), dtype=numpy.int32)  # data_char
    for i, text in enumerate(tqdm(processed_texts)):
        for j, word in enumerate(text):
            if lang == "JP":
                word = word.surface
            word_index = word_vocab.index(word)
            if j < MAX_SENTENCE_LENGTH:
                if dict_limit == 0 or word_index < dict_limit:
                    data_char[i, j] = word_index

    labels = to_categorical(numpy.asarray(labels))
    # split data into training and validation
    indices = numpy.arange(data_char.shape[0])
    numpy.random.shuffle(indices)
    data_char = data_char[indices]
    labels = labels[indices]
    # 80% to train, 10% to validation, 10% to test
    nb_validation_test_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data_char.shape[0])
    nb_test_samples = int((TEST_SPLIT) * data_char.shape[0])

    x_train = data_char[:-nb_validation_test_samples]
    y_train = labels[:-nb_validation_test_samples]
    x_val = data_char[-nb_validation_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_test_samples:-nb_test_samples]
    x_test = data_char[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]

    if dict_limit > 0:
        vocab_size = dict_limit
    else:
        vocab_size = len(word_vocab)

    return x_train, y_train, x_val, y_val, x_test, y_test, vocab_size