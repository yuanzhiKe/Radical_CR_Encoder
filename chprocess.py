import os
import sys
from keras.preprocessing.text import Tokenizer

def getTexts(TEXT_DATA_DIR):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
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
                    f = open(fpath, encoding='gbk', errors='replace')
                try:
                    t = f.read()
                except UnicodeDecodeError:
                    print(fpath)
                    import traceback
                    traceback.print_exc()
                    raise
                t = t.replace("\n", "").strip()
                texts.append(t)
                f.close()
                labels.append(label_id)

    print('Found %s texts.' % len(texts))
    return texts, labels_index, labels

if __name__ == "__main__":
    # filename = "data/2000/pos"
    BASE_DIR = "/home/ke/CMWE/"
    TEXT_DATA_DIR = BASE_DIR+"data/2000/"
    texts, labels_index, labels = getTexts(TEXT_DATA_DIR)
    print(texts[:1])
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    print(list(word_index.keys())[:5])