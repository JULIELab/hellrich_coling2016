import gensim
import sys
import collections
import codecs
import re
import math
from copy import deepcopy


def main():
    if len(sys.argv) < 2:
        raise Exception(
            "Provide 1+ arguments:\n\t1+,model(s)")
    models = [gensim.models.Word2Vec.load_word2vec_format(
        name, binary=True) for name in sys.argv[1:]]
    words = ["romantic", "card", "sleep", "parent", "address", "gay", "mouse", "king", "checked", "check", "actually",
             "supposed", "guess", "cell", "headed", "ass", "mail", "toilet", "cock", "bloody", "nice", "guy"]

    for word in words:
        sim = [w for model in models for w, s in model.most_similar(
            word, topn=1)]
        print(word, " - ".join(sim))


if __name__ == "__main__":
    main()
