import string, pickle, numpy, mojimoji, os, jieba, sys
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten, GRU
from keras.legacy.layers import Highway
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils.np_utils import to_categorical
from attention import AttentionWithContext
from getShapeCode import get_all_word_bukken, get_all_character
from tqdm import tqdm
from plot_results import plot_results, save_curve_data
from dataReader import prepare_char, prepare_word, shuffle_kv
from util import slice_batch, to_multi_gpu, _make_kana_convertor

# MAX_SENTENCE_LENGTH = 739  # large number as 739 makes cudnn die
MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 4
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 100
WORD_DIM = 600
MAX_RUN = 1
VERBOSE = 0
EPOCHS = 50


def load_shape_data(datafile="usc-shape_bukken_data.pickle"):
    with open(datafile) as f:
        data = pickle.load(f)
    return data["words"], data["bukkens"], data["word_bukken"]


def train_and_test_model(model, x_train, y_train, x_val, y_val, x_test, y_test, model_name, early_stop=False):
    # model = to_multi_gpu(model)
    print(model_name)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    if early_stop:
        stopper = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_loss = ModelCheckpoint(filepath="checkpoints/" + model_name + "_bestloss.hdf5", monitor="val_loss",
                                      verbose=VERBOSE, save_best_only=True, mode="min")
    print("complisng...")
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['categorical_crossentropy', "acc"], )
    print("fitting...")
    if early_stop:
        result = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=VERBOSE,
                           epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[reducelr, stopper, checkpoint_loss])
    else:
        result = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=VERBOSE,
                           epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[reducelr, checkpoint_loss])
    model.load_weights("checkpoints/" + model_name + "_bestloss.hdf5")
    print("testing...")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))

    predict_value(model, model_name, x_test)

    return result


def predict_value(model, model_name, x_test):
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['categorical_crossentropy'], )
    model.load_weights("checkpoints/" + model_name + "_bestloss.hdf5")
    predicted = model.predict(x_test, verbose=1)
    numpy.savetxt(model_name + "_predict.data", predicted, fmt='%1.10f')


def visualize_embedding(picklename):
    f = open(picklename, "rb")
    (full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
     x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val,
     x1_test, x2_test, x3_test, y_test) \
        = pickle.load(f)
    f.close()
    print("2")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway=None,
                               nohighway="linear",
                               attention=True, shape_filter=True,
                               char_filter=True)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['categorical_crossentropy', "acc"], )
    model.load_weights("checkpoints/Rakuten10000model4_bestloss.hdf5")
    tensor_name = ["time_distributed_1"]
    # tensor_name += next(filter(lambda x: x.name == 'embedding', model.layers)).W.name
    tb_cb = TensorBoard(log_dir="./tflog/", histogram_freq=1, embeddings_freq=1, embeddings_layer_names=tensor_name) # need to fix to get the exact embeddings
    stopper = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(x1_train, y_train, validation_data=(x1_val, y_val), epochs=30, batch_size=BATCH_SIZE,
              callbacks=[tb_cb, stopper])


def get_vocab(shuffle=False):
    # convert kata to hira
    char_emb_dim = CHAR_EMB_DIM
    use_component = True  # True for component level False for chara level

    _, _, hirakana_list = _make_kana_convertor()
    addition_translate = str.maketrans("ッャュョヮヵヶ?？⁇⁈⁉﹗!‼！″＂“”『』「」‘’´｀:;。、・"
                                       "＼([｛)]｝〔〕【〘〖】〙〗｟｠«»ー－—–‐゠〜～〳〵￥",
                                       "っゃゅょゎゕゖ?????!!!!\"\"\"\"\"\"'''''':;.,･"
                                       "\\((()))()((()))()《》-----=~~/\\$")

    hira_punc_number_latin = "".join(hirakana_list) + string.punctuation + \
                             'ヴゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々〻' \
                             '〟ゝゞ〈《〉》〝…‥･〴' \
                             '1234567890' \
                             'abcdefghijklmnopqrstuvwxyz ' \
                             '○●☆★■♪ヾω*≧∇≦※→←↑↓'
    # note: the space and punctuations in Jp sometimes show emotion

    vocab_chara, vocab_bukken, chara_bukken = get_all_word_bukken()
    # print(hira_punc_number_latin)
    hira_punc_number_latin_number = len(hira_punc_number_latin) + 2
    print("totally {n} kana, punctuation and latin char".format(n=str(hira_punc_number_latin_number)))
    vocab = ["</padblank>", "</s>"] + list(hira_punc_number_latin) + vocab_bukken
    real_vocab_number = len(vocab)  # the part of the vocab that is really used. only basic components
    vocab_chara_strip = [chara for chara in vocab_chara if chara not in vocab_bukken]  # delete 独体字
    print("totally {n} puctuation, kana, latin, and chara components".format(n=str(real_vocab_number)))
    full_vocab = vocab + vocab_chara_strip  # add unk at the head, and complex charas for text encoding at the tail
    chara_bukken_revised = {}
    for i_word, i_bukken in chara_bukken.items():  # update the index
        if vocab_chara[i_word] not in vocab_bukken:  # delete 独体字
            chara_bukken_revised[full_vocab.index(vocab_chara[i_word])] = \
                [k + hira_punc_number_latin_number for k in i_bukken]
    del vocab_chara
    del chara_bukken

    return full_vocab, real_vocab_number, chara_bukken_revised, addition_translate, hira_punc_number_latin


def text_to_char_index(full_vocab, real_vocab_number, chara_bukken_revised, sentence_text, addition_translate,
                       comp_width=COMP_WIDTH, preprocessed_char_number=0,
                       skip_unknown=False, shuffle=None):
    # mode:
    # average: will repeat the original index to #comp_width for the process of the embedding layer
    # padding: will pad the original index to #comp_width with zero for the process of the embedding layer
    # char_emb_dim  char embedding size
    # comp_width  #components used

    if preprocessed_char_number == 0:
        preprocessed_char_number = len(full_vocab)

    # convert digital number and latin to hangaku
    text = mojimoji.zen_to_han(sentence_text, kana=False)
    # convert kana to zengaku
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    # convert kata to hira
    _, katakana2hiragana, _ = _make_kana_convertor()
    text = katakana2hiragana(text)
    text = text.translate(addition_translate)
    # finally, lowercase
    text = text.lower()
    # expanding every character with 3 components
    ch2id = {}
    for i, w in enumerate(full_vocab):
        ch2id[w] = i
    int_text = []
    # print(text)
    for c in text:
        # print(c)
        try:
            i = ch2id[c]
        except KeyError:
            print("Unknown Character: ", c)
            if skip_unknown:
                continue  # skip unknown words
            else:
                i = 1  # assign to unknown words
        # print(i)
        if real_vocab_number < i < preprocessed_char_number:
            comps = chara_bukken_revised[i]
            if shuffle == "flip":
                comps = comps[::-1]
            # print(comps)
            if len(comps) >= comp_width:
                int_text += comps[:comp_width]
            else:
                int_text += comps + [0] * (comp_width - len(comps))
        else:
            if shuffle == "random":
                if i < real_vocab_number:
                    i = (i + 20) % real_vocab_number
            int_text += [i] + [0] * (comp_width - 1)
    return int_text


def build_word_feature_shape(vocab_size=5, char_emb_dim=CHAR_EMB_DIM, comp_width=COMP_WIDTH,
                             mode="padding", cnn_encoder=True,
                             highway="linear", nohighway=None, shape_filter=True, char_filter=True):
    # build the feature computed by cnn for each word in the sentence. used to input to the next rnn.
    # expected input: every #comp_width int express a character.
    # mode:
    # "average": average pool the every #comp_with input embedding, output average of the indexed embeddings of a character
    # "padding": convoluate every #comp_width embedding

    # real vocab_size for ucs is 2481, including paddingblank, unkown, puncutations, kanas
    assert shape_filter or char_filter
    init_width = 0.5 / char_emb_dim
    init_weight = numpy.random.uniform(low=-init_width, high=init_width, size=(vocab_size, char_emb_dim))
    init_weight[0] = 0  # maybe the padding should not be zero
    # print(init_weight)
    # first layer embeds
    #  every components
    word_input = Input(shape=(COMP_WIDTH * MAX_WORD_LENGTH,))
    char_embedding = \
        Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(word_input)
    # print("char_embedding:", char_embedding._keras_shape)
    if cnn_encoder:
        if mode == "padding":
            # print(char_embedding._keras_shape)
            # print(comp_width)
            if shape_filter and char_filter:
                filter_sizes = [50, 100, 150]
            else:
                filter_sizes = [100, 200, 300]
            if shape_filter:
                feature_s1 = Conv1D(filters=filter_sizes[0], kernel_size=1, activation='relu')(
                    char_embedding)
                feature_s1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH * COMP_WIDTH)(feature_s1)
                feature_s2 = Conv1D(filters=filter_sizes[1], kernel_size=2, activation='relu')(
                    char_embedding)
                feature_s2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH * COMP_WIDTH - 1)(feature_s2)
                feature_s3 = Conv1D(filters=filter_sizes[2], kernel_size=3, activation='relu')(
                    char_embedding)
                feature_s3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH * COMP_WIDTH - 2)(feature_s3)
            if char_filter:
                feature1 = Conv1D(filters=filter_sizes[0], kernel_size=1 * comp_width, strides=comp_width,
                                  activation='relu')(
                    char_embedding)
                feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
                feature2 = Conv1D(filters=filter_sizes[1], kernel_size=2 * comp_width, strides=comp_width,
                                  activation='relu')(
                    char_embedding)
                feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
                feature3 = Conv1D(filters=filter_sizes[2], kernel_size=3 * comp_width, strides=comp_width,
                                  activation='relu')(
                    char_embedding)
                feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
            if shape_filter and char_filter:
                feature = concatenate([feature_s1, feature_s2, feature_s3, feature1, feature2, feature3])
            elif shape_filter and not char_filter:
                feature = concatenate([feature_s1, feature_s2, feature_s3])
            elif char_filter and not shape_filter:
                feature = concatenate([feature1, feature2, feature3])
            else:
                feature = None
        feature = Flatten()(feature)
        # print(feature._keras_shape)
        if highway:
            if isinstance(highway, str):
                feature = Highway(activation=highway)(feature)
            else:
                feature = Highway(activation='relu')(feature)
        else:
            if nohighway:
                feature = Dense(units=600, activation=nohighway)(feature)
            else:
                pass
    else:
        feature = Flatten()(char_embedding)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder


def build_word_feature_char(vocab_size=5, char_emb_dim=CHAR_EMB_DIM,
                            mode="padding", cnn_encoder=True, highway=True):
    # build the feature computed by cnn for each word in the sentence. used to input to the next rnn.
    # expected input: every #comp_width int express a character.
    # mode:
    # "average": average pool the every #comp_with input embedding, output average of the indexed embeddings of a character
    # "padding": convoluate every #comp_width embedding

    # real vocab_size for ucs is 2481, including paddingblank, unkown, puncutations, kanas
    init_width = 0.5 / char_emb_dim
    init_weight = numpy.random.uniform(low=-init_width, high=init_width, size=(vocab_size, char_emb_dim))
    init_weight[0] = 0  # maybe the padding should not be zero
    # print(init_weight)
    # first layer embeds
    #  every components
    word_input = Input(shape=(MAX_WORD_LENGTH,))
    char_embedding = \
        Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(word_input)
    # print("char_embedding:", char_embedding._keras_shape)
    if cnn_encoder:
        if mode == "padding":
            # print(char_embedding._keras_shape)
            # conv, filter with [1, 2, 3]*#comp_width, feature maps 50 100 150
            feature1 = Conv1D(filters=100, kernel_size=1, activation='relu')(
                char_embedding)
            feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
            feature2 = Conv1D(filters=200, kernel_size=2, activation='relu')(
                char_embedding)
            feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
            feature3 = Conv1D(filters=300, kernel_size=3, activation='relu')(
                char_embedding)
            feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
            feature = concatenate([feature1, feature2, feature3])
        feature = Flatten()(feature)
        # print(feature._keras_shape)
        if highway:
            feature = Highway(activation="relu")(feature)
    else:
        feature = Flatten()(char_embedding)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder


def build_sentence_rnn(real_vocab_number, word_vocab_size=10, char_vocab_size=10,
                       classes=2, attention=False, dropout=0,
                       word=True, char=False, char_shape=True, model="rnn", cnn_encoder=True,
                       highway=None, nohighway=None, shape_filter=True, char_filter=True):
    # build the rnn of words, use the output of build_word_feature as the feature of each word
    if char_shape:
        word_feature_encoder = build_word_feature_shape(vocab_size=real_vocab_number,
                                                        cnn_encoder=cnn_encoder,
                                                        highway=highway, nohighway=nohighway,
                                                        shape_filter=shape_filter,
                                                        char_filter=char_filter)
        sentence_input = Input(shape=(MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(sentence_input)
        # print(word_feature_sequence._keras_shape)
    if word:
        sentence_word_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        word_embedding_sequence = Embedding(input_dim=word_vocab_size, output_dim=WORD_DIM)(sentence_word_input)
    if char:
        word_feature_encoder = build_word_feature_char(vocab_size=char_vocab_size,
                                                       cnn_encoder=cnn_encoder, highway=highway)
        char_input = Input(shape=(MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(char_input)
    if char_shape and word and not char:
        word_feature_sequence = concatenate([word_feature_sequence, word_embedding_sequence], axis=2)
    if word and not char_shape and not char:
        word_feature_sequence = word_embedding_sequence
    # print(word_feature_sequence._keras_shape)
    if model == "rnn":
        if attention:
            lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=True))(word_feature_sequence)
            if highway:
                lstm_rnn = TimeDistributed(Highway(activation=highway))(lstm_rnn)
            elif nohighway:
                lstm_rnn = TimeDistributed(Dense(units=300, activation=nohighway))(lstm_rnn)
            lstm_rnn = AttentionWithContext()(lstm_rnn)
        else:
            lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=False))(word_feature_sequence)
        x = lstm_rnn
    if classes < 2:
        print("class number cannot less than 2")
        exit(1)
    else:
        preds = Dense(classes, activation='softmax')(x)
    if char_shape and not word and not char:
        sentence_model = Model(sentence_input, preds)
    if word and not char_shape and not char:
        sentence_model = Model(sentence_word_input, preds)
    if word and char_shape and not char:
        sentence_model = Model([sentence_input, sentence_word_input], preds)
    if char and not word and not char_shape:
        sentence_model = Model(char_input, preds)
    sentence_model.summary()
    return sentence_model


def build_hatt(word_vocab_size, classes):
    MAX_SENT_LENGTH = 100
    MAX_SENTS = 5
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(input_dim=word_vocab_size, output_dim=WORD_DIM, input_length=MAX_SENT_LENGTH)(
        sentence_input)
    l_lstm = Bidirectional(GRU(150, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(300))(l_lstm)
    l_att = AttentionWithContext()(l_dense)
    sentEncoder = Model(sentence_input, l_att)
    # print("sentEncoder Shape:", l_att._keras_shape)
    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    # print("RevewEncoder Shape:", review_encoder._keras_shape)
    l_lstm_sent = Bidirectional(GRU(150, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(300))(l_lstm_sent)
    l_att_sent = AttentionWithContext()(l_dense_sent)
    preds = Dense(classes, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)
    model.summary()
    return model


def build_fasttext(word_vocab_size, classes):
    sentence_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer = Embedding(input_dim=word_vocab_size, output_dim=WORD_DIM)(sentence_input)
    avraged = AveragePooling1D(pool_size=MAX_SENTENCE_LENGTH)(embedded_sequences)
    avraged = Flatten()(avraged)
    l_dens = Dense(10, activation="linear")(avraged)
    preds = Dense(classes, activation='softmax')(l_dens)
    model = Model(sentence_input, preds)
    model.summary()
    return model


def split_data(data_shape, data_char, data_word, labels):
    # split data into training and validation
    indices = numpy.arange(data_char.shape[0])
    numpy.random.shuffle(indices)
    data_shape = data_shape[indices]
    data_word = data_word[indices]
    data_char = data_char[indices]
    labels = labels[indices]
    # 80% to train, 10% to validation, 10% to test
    nb_validation_test_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data_char.shape[0])
    nb_test_samples = int((TEST_SPLIT) * data_char.shape[0])

    x1_train = data_shape[:-nb_validation_test_samples]
    x2_train = data_word[:-nb_validation_test_samples]
    x3_train = data_char[:-nb_validation_test_samples]
    y_train = labels[:-nb_validation_test_samples]
    x1_val = data_shape[-nb_validation_test_samples:-nb_test_samples]
    x2_val = data_word[-nb_validation_test_samples:-nb_test_samples]
    x3_val = data_char[-nb_validation_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_test_samples:-nb_test_samples]
    x1_test = data_shape[-nb_test_samples:]
    x2_test = data_word[-nb_test_samples:]
    x3_test = data_char[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]

    return x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val, \
           x1_test, x2_test, x3_test, y_test


def prepare_ChnSenti_classification(filename="ChnSentiCorp_htl_unba_10000/", dev_mode=False, skip_unk=False,
                                    shuffle=None):
    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised, addtional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2

    TEXT_DATA_DIR = filename
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
                t_list = list(jieba.cut(t, cut_all=False))
                # if len(t_list) > maxlen:
                #     maxlen = len(t_list)
                if len(t_list) > MAX_SENTENCE_LENGTH:
                    t_list = t_list[:MAX_SENTENCE_LENGTH]
                texts.append(t_list)
                f.close()
                labels.append(label_id)

    print('Found %s texts.' % len(texts))

    data_size = len(texts)
    preprocessed_char_number = len(full_vocab)

    # global MAX_SENTENCE_LENGTH
    # MAX_SENTENCE_LENGTH = maxlen

    # change the sentence into matrix of word sequence
    data_char = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH),
                            dtype=numpy.int32)  # data_shape
    data_word = numpy.zeros((data_size, MAX_SENTENCE_LENGTH), dtype=numpy.int32)
    data_gram = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH), dtype=numpy.int32)  # data_char
    print("Data shape: {shape}".format(shape=str(data_char.shape)))

    word_vocab = ["</s>"]
    char_vocab = ["</s>"] + get_all_character()

    num_words = 0
    num_chars = 0
    num_ideographs = 0

    if shuffle == "random":
        chara_bukken_revised = shuffle_kv(chara_bukken_revised)

    for i, text in enumerate(tqdm(texts)):
        for j, word in enumerate(text):
            # word level
            num_words += 1
            if word not in word_vocab:
                word_vocab.append(word)
                word_index = len(word_vocab) - 1
            else:
                word_index = word_vocab.index(word)
            data_word[i, j] = word_index
            # single char gram level
            for l, char_g in enumerate(word):
                num_chars += 1
                if char_g not in char_vocab:
                    char_vocab.append(char_g)
                    char_g_index = len(char_vocab) - 1
                else:
                    char_g_index = char_vocab.index(char_g)
                if l < MAX_WORD_LENGTH:
                    data_gram[i, j, l] = char_g_index
                if not skip_unk:
                    if char_g not in full_vocab:
                        full_vocab.append(char_g)
                if char_g in full_vocab:
                    if n_hira_punc_number_latin < full_vocab.index(char_g) < preprocessed_char_number:
                        num_ideographs += 1
            # char shape level
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                            chara_bukken_revised=chara_bukken_revised,
                                            addition_translate=addtional_translate,
                                            sentence_text=word, preprocessed_char_number=preprocessed_char_number,
                                            skip_unknown=skip_unk, shuffle=shuffle)

            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                data_char[i, j, k] = comp

    print("# words: ", num_words)
    print("# chars: ", num_chars)
    print("# ideographas: ", num_ideographs)
    # convert labels to one-hot vectors
    labels = to_categorical(numpy.asarray(labels))
    print('Label Shape:', labels.shape)

    x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val, \
    x1_test, x2_test, x3_test, y_test = split_data(data_shape=data_char,
                                                   data_char=data_gram, data_word=data_word,
                                                   labels=labels)

    print('Number of different reviews for training and test')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    with open(filename[:-1] + ".pickle", "wb") as f:
        pickle.dump((full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
                     x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val,
                     x1_test, x2_test, x3_test, y_test), f)

    return full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab, \
           x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val, \
           x1_test, x2_test, x3_test, y_test


def do_ChnSenti_classification_multimodel(filename, dev_mode=False, cnn_encoder=True, skip_unk=False,
                                          shape_filter=True, char_filter=True, shuffle=None,
                                          highway_options=None, nohighway_options=None, attention_options=None):
    picklename = filename[:-1] + "_shuffle_" + str(shuffle) + ".pickle"
    if not skip_unk:
        if os.path.isfile(picklename):
            f = open(picklename, "rb")
            (full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
             x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val,
             x1_test, x2_test, x3_test, y_test) \
                = pickle.load(f)
            f.close()
        else:
            (full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
             x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val,
             x1_test, x2_test, x3_test, y_test) \
                = prepare_ChnSenti_classification(filename=filename, dev_mode=dev_mode, skip_unk=skip_unk,
                                                  shuffle=shuffle)
            with open(picklename, "wb") as f:
                pickle.dump((full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
                             x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val,
                             x1_test, x2_test, x3_test, y_test), f)
    else:
        (full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
         x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val,
         x1_test, x2_test, x3_test, y_test) \
            = prepare_ChnSenti_classification(filename=filename, dev_mode=dev_mode, skip_unk=skip_unk, shuffle=shuffle)

    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)

    print("word vocab size", word_vocab_size)
    print("char vocab size", char_vocab_size)

    num_class = 2

    model_index = 0

    data_set_name = filename[:-1]
    if highway_options is None:
        # highway_options = [None, "linear", "relu"]
        highway_options = [None]
    if nohighway_options is None:
        # nohighway_options = [None, "linear", "relu"]
        nohighway_options = ["linear"]
    if attention_options is None:
        # attention_options = [False, True]
        attention_options = [True]

    result_shape = None

    for highway_option in highway_options:
        for nohighway_option in nohighway_options:
            if highway_option and nohighway_option:
                continue
            for attention_option in attention_options:
                model_index += 1
                print("MODEL:", str(model_index), " SHAPE Highway:", highway_option, " Dense:", nohighway_option,
                      " Attention:", attention_option)
                model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=num_class,
                                           char_shape=True, word=False, char=False,
                                           cnn_encoder=cnn_encoder, highway=highway_option,
                                           nohighway=nohighway_option,
                                           attention=attention_option, shape_filter=shape_filter,
                                           char_filter=char_filter)
                result_shape = train_and_test_model(model, x1_train, y_train, x1_val, y_val, x1_test, y_test,
                                                    data_set_name + "model" + str(model_index))
    return result_shape


if __name__ == "__main__":
    do_ChnSenti_classification_multimodel("ChnSentiCorp_htl_unba_10000/")
