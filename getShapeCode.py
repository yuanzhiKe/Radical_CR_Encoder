# -*- coding: utf-8 -*-
import pickle
import collections


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def strip_ideographic(text):
    # Ideographic_Description_Characters = ["⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"]
    Ideographic_Description_Characters = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"
    translator = str.maketrans("", "", Ideographic_Description_Characters)
    return text.translate(translator)


def get_all_word_bukken(filename="IDS-UCS-Basic.txt"):
    bukkens = []
    words = []  # actually chars
    word_bukken = {}
    for i, line in enumerate(open(filename, "r").readlines()):
        if line[0] != "U":  # not start with U+XXXX means it is not a word
            continue
        line = line.split()
        word = line[1]
        components = line[2]
        components = strip_ideographic(components)
        bukken = []
        while ";" in components:
            bukken.append(components[:components.find(";") + 1])
            components = components[components.find(";") + 1:]
        while len(components) > 1:
            bukken.append(components[0])
            components = components[1:]
        bukken.append(components)
        words.append(word)
        word_bukken[words.index(word)] = bukken
        if len(bukken) == 1 and bukken[0] == word:
            bukkens.append(word)

    def expand_bukken(bukken):
        expanded_bukken = []
        for b in bukken:
            if b in bukkens:
                expanded_bukken.append(bukkens.index(b))
            else:
                if b in words:
                    expanded_bukken.append(expand_bukken(word_bukken[words.index(b)]))
                else:
                    bukkens.append(b)
                    expanded_bukken.append(bukkens.index(b))
        return expanded_bukken

    for i_word, i_bukken in word_bukken.items():
        b_list = expand_bukken(i_bukken)
        b_list = flatten(b_list)
        word_bukken[i_word] = b_list
    return words, bukkens, word_bukken


def get_all_character(filename="IDS-UCS-Basic.txt"):
    chars = []

    for i, line in enumerate(open(filename, "r").readlines()):
        if line[0] != "U":  # not start with U+XXXX means it is not a word
            continue
        line = line.split()
        char = line[1]
        chars.append(char)
    return chars

if __name__ == "__main__":
    # print(strip_ideographic('⿱⿰&CDP-895C;&CDP-895C;一'))
    words, bukkens, word_bukken = get_all_word_bukken("IDS-UCS-test.txt")
    # for word, bukken in word_bukken.items():
    #     print(word + ": " + str(bukken))
    # words, bukkens, word_bukken = get_all_word_bukken("IDS-UCS-Basic.txt")
    shape_bukken_data = {
        "words": words,
        "bukkens": bukkens,
        "word_bukken": word_bukken
    }
    print(len(words), len(bukkens), word_bukken)
    for i_word, i_bukken in word_bukken.items():
        print(words[i_word], str([bukkens[k] for k in i_bukken]))
    # with open("usc-shape_bukken_data.pickle", "wb") as f:
    #     pickle.dump(shape_bukken_data, f)
