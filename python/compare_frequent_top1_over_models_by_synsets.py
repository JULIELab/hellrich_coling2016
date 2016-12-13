import gensim
import sys
import collections
import codecs
import re
import math
from copy import deepcopy
from nltk.corpus import wordnet as wn
bins = 100


def compare(models):
    senses2rel = {}
    senses2count = {}

    for i in range(len(models)):
        models[i] = gensim.models.Word2Vec.load_word2vec_format(
            models[i], binary=True)
    v = models[0].vocab
    reliability = [0 for i in range(bins)]
    inside = [0 for i in range(bins)]

    for word in v:
        rank = models[0].vocab[word].index
        percentile_rank = int((rank * bins) / len(v))
        if percentile_rank < 10 : #inverted scale
            senses = len(wn.synsets(word))
            if senses > 10:
                senses = 10

            if not senses in senses2rel:
                senses2rel[senses] = 0
            if not senses in senses2count:
                senses2count[senses] = 0

            senses2count[senses] += 1
            senses2rel[senses] += len(intersection(
                [[word for word, sim in model.most_similar(word, topn=1)] if word in model.vocab else [] for model in models]))
    return sorted([(senses, senses2rel[senses] / senses2count[senses]) for senses in senses2count])


def intersection(sequences):
    if len(sequences) < 2:
        raise Exception("Need multiple sequences for comparisson")
    inter = set(sequences[0])
    for s in sequences[1:]:
        inter = inter.intersection(set(s))
    return inter


def main():
    if len(sys.argv) < 2:
        raise Exception(
            "Provide 1+ arguments:\n\t1+,model(s)")
    models = sys.argv[1:]

    senses2rel = compare(models)
    print("senses reliability")
    for senses, rel in senses2rel:
        print(str(senses), str(rel))

if __name__ == "__main__":
    main()
