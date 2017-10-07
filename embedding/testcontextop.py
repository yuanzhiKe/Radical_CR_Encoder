import os
import tensorflow as tf

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

if __name__ == "__main__":
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            (words, counts, words_per_epoch, _epoch, _words, examples,
             labels, contexts) = word2vec.full_context_skipgram_word2vec(filename="test.corpus",
                                                                    batch_size=5,
                                                                    window_size=3,
                                                                    min_count=1,
                                                                    subsample=1e-4)
            (vocab, a, b, c) = session.run([words, examples, labels, contexts])
            for index, word in enumerate(vocab):
                print(str(index)+": "+word.decode("utf-8"))
            for i in range(5):
                print(vocab[a[i]], vocab[b[i]], [vocab[x] for x in c[i]])
