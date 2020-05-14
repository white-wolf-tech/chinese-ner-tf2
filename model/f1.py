#coding=utf-8
import json
def get_f1_score_label(pre_lines, gold_lines, label="organization"):
    """
    打分函数
    """
    TP = 0
    FP = 0
    FN = 0
    for pre, gold in zip(pre_lines, gold_lines):
        pre = pre["label"].get(label, {}).keys()
        gold = gold["label"].get(label, {}).keys()
        for i in pre:
            if i in gold:
                TP += 1
            else:
                FP += 1
        for i in gold:
            if i not in pre:
                FN += 1
    try:
        p = float(TP) / float(TP + FP)
        r = float(TP) / float(TP + FN)
        f = 2.0 * p * r / (p + r)
    except Exception as e:
        p = 0.0
        r = 0.0
        f = 0.0
    print('****************{}*******************'.format(label))
    print('precision:{}, recall:{}, f1:{}'.format(p, r, f))
    return f


def compare_f1(pre_file="ner_predict.json", gold_file="data/thuctc_valid.json"):
    pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    f_score = {}
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    sum = 0
    for label in labels:
        f = get_f1_score_label(pre_lines, gold_lines, label=label)
        f_score[label] = f
        sum += f
    avg = sum / len(labels)
    return f_score, avg


if __name__ == "__main__":
    f_score, avg = compare_f1(pre_file="ner_predict_large.json", gold_file="data/thuctc_valid.json")
    print(f_score, avg)
