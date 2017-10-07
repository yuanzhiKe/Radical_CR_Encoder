# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import pickle


def getJp():
    count = 0
    char_vocab = []
    shape_vocab = []
    char_shape = {}
    for line in open("joyo2010.txt").readlines():
        if line[0] == "#":
            continue
        text = line[0]
        im = Image.new("1", (28,28), 0)
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/takao-mincho/TakaoPMincho.ttf", 28)
        dr.text((0, 0), text, font=font, fill=1)
        im.save("1.jpg")
        img = np.array(im, dtype="int32")
        char_vocab.append(text)
        shape_vocab.append(img)
        char_shape[text] = img
        count += 0
    shape_vocab_data = {
        "chars": char_vocab,
        "shapes": shape_vocab,
        "char_shape": char_shape
    }
    with open("jp_shape_vocab.pickle", "wb") as f:
        pickle._dump(shape_vocab_data, f)


def getCh():
    count = 0
    char_vocab = []
    shape_vocab = []
    char_shape = {}
    def getShapes(filename, char_vocab, shape_vocab, char_shape, count):
        for line in open(filename).readlines():
            line = line.split()
            if len(line)<2:
                continue
            text = line[1][0]
            im = Image.new("1", (28,28), 0)
            dr = ImageDraw.Draw(im)
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc", 28)
            dr.text((0, -7), text, font=font, fill=1)
            im.save("ch_shape_jpg/"+text+".jpg")
            img = np.array(im, dtype="int32")
            char_vocab.append(text)
            shape_vocab.append(img)
            char_shape[text] = img
            count += 0
    getShapes("../cjkvi-tables/zibiao2009-1.txt", char_vocab, shape_vocab, char_shape, count)
    getShapes("../cjkvi-tables/zibiao2009-2.txt", char_vocab, shape_vocab, char_shape, count)
    print("Read {number} character shapes.".format(number=len(char_vocab)))
    shape_vocab_data = {
        "chars": char_vocab,
        "shapes": shape_vocab,
        "char_shape": char_shape
    }
    with open("ch_shape_vocab.pickle", "wb") as f:
        pickle._dump(shape_vocab_data, f)

if __name__ == "__main__":
    getCh()

