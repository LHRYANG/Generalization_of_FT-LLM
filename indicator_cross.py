#from rouge import Rouge
import json
import os

from datasets import load_metric
from sacrebleu.metrics import BLEU
from os import listdir


def get_rouge(hys,gts):
    rouge = load_metric('rouge')
    scores = rouge.compute(predictions=hys, references=gts)#,use_aggregator=False)
    return scores

def get_bleu(hys,gts):
    bleu = BLEU()
    scores = bleu.corpus_score(hys, [gts])
    return scores


def ppp_sentimnet(path):
    with open(path) as f:
        d_list = json.load(f)

    acc = 0
    for d in d_list:
        if d["max_idx"] ==0 and d["output"] =="positive" or d["max_idx"] ==1 and d["output"] =="negative":
            acc+=1
    return acc, len(d_list)

def ppp_inference(path):
    with open(path) as f:
        d_list = json.load(f)

    #entailment,neutral,contradiction
    acc = 0
    for d in d_list:
        if d["max_idx"] ==0 and d["output"] =="entailment" or d["max_idx"] ==1 and d["output"] =="neutral" or d["max_idx"]==2 and d["output"]=="contradiction":
            acc+=1
    return acc, len(d_list)

def ppp_detection(path):
    with open(path) as f:
        d_list = json.load(f)

    #entailment,neutral,contradiction
    acc = 0
    for d in d_list:
        if d["max_idx"] ==0 and d["output"] =="yes" or d["max_idx"] ==1 and d["output"] =="no":
            acc+=1
    return acc, len(d_list)


def ppp(path):
    with open(path) as f:
        d_list = json.load(f)

    refs = []
    gens = []
    for d in d_list:
        refs.append(d["output"])
        gens.append(d["generation"].split("\n\n")[0])

    #print(gens[0:4])
    r = get_rouge(gens,refs)
    b = get_bleu(gens,refs)
    #print(b)
    return r["rougeL"].mid,b


res = []
count = 0
prefix = results_path_here

summary = ["cnn","xsum","xlsum","pdfs"]
sentiment = ["amazon","amazonfood","review","sst2","yelp"]
detection = ["paws","qqp","stsb"]
inference = ["gptnli","multinli","multinli2","rte"]
question = ["sciqa","socialqa","tweetqa"]

with open(prefix+"res.txt",'w') as f:
    for file in os.listdir(prefix):
        if "res" in file:
            continue
        data = file.split("_")[0]
        count += 1

        if data in summary or data in question:
            r, b = ppp(prefix + file)
            f.write(file+"\n")
            r,b = ppp(prefix+file)
            print(file)
            print(r)
            print(b)
            f.write(str(r)+"\n")
            f.write(str(b)+"\n")
            f.write("\n")
            continue
        if data in sentiment:
            acc, total = ppp_sentimnet(prefix + file)
        if data in detection:
            acc, total = ppp_detection(prefix + file)
        if data in inference:
            acc, total = ppp_inference(prefix + file)
        print(file, acc, total, acc / total)
        f.write(file)
        f.write('\n')
        f.write(str(acc) + "/" + str(total) + "/" + str(acc / total) + "\n")
        f.write("\n")
