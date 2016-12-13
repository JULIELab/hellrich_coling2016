import gensim
import sys
from scipy import stats
import statistics


def correlation(gold_name, model_name, sep):
    model = gensim.models.Word2Vec.load_word2vec_format(
                model_name, binary=True)
    cosine_similarities = []
    gold_similarities = []
    first = True
    if gold_name.endswith("gur350.csv"):
        encoding="ISO-8859-1"
    else:
        encoding="UTF-8"
    with open(gold_name,"r",encoding=encoding) as gold:
        for line in gold:
            if first:
                first = False
                continue
            parts = line.split(sep)
            word1 = parts[0].lower().strip()
            word2 = parts[1].lower().strip()
            sim = float(parts[2].lower().strip())
            if word1 in model and word2 in model:
                gold_similarities.append(sim)
                cosine_similarities.append(model.similarity(word1,word2))
    rho, p = stats.spearmanr(cosine_similarities, gold_similarities)
    return rho

def main():
    if len(sys.argv) < 4:
        raise Exception(
            "Provide 2+ arguments:\n\t1, Output format \n\t2, gold data file\n\t3 sep (e.g. ',')\n\t4+,model(s)")
    compact_output = sys.argv[1] == "compact"
    gold_name = sys.argv[2]
    sep = sys.argv[3]
    model_names = sys.argv[4:]
    rho = []
    for model_name in model_names:
        rho.append(correlation(gold_name, model_name, sep))
    if not compact_output:
        print(rho)
    mean = statistics.mean(rho)
    stdev = statistics.pstdev(rho,mean)
    if compact_output:
        print("{:.2f}".format(mean))
    else:
        print("$","{:.2f}".format(mean),"\\pm","{:.2f}".format(stdev),"$")
if __name__ == "__main__":
    main()
