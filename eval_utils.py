# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
import string
import sys
from icecream import ic
sys.path.append('/data/liuyuxuan/SI-T2S/ABSA-QUAD/transformers/src/transformers')

from data_utils import sentword2opinion, sentiment_word_list, laptop_acos_aspect_cate_list, res_acos_aspect_cate_list

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, seq, seq_type, num_id):
    quads = []
    sents = [s.strip() for s in seq.split(' [SSEP] ')]
   
    for s in sents:
        # food quality is bad because pizza is over cooked.
        try:
            ac_sp, at_ot = s.split(' because ')
            ac, sp = ac_sp.split(' is ')
            at, ot = at_ot.split(' is ')

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'NULL'
        except ValueError:
            try:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))
    
    return quads

def extract_spans_extraction(seq, y_type, num_id, use_sent_flag, use_prompt_flag):
    extractions = []
    global at, ac, sp, ot
    punctuations = string.punctuation
    # removed_template_seq = seq.split(' [SEP] ')[-1]
    all_pt = seq.split(' [SSEP] ')
    for pt in all_pt:
        # print("pt is",pt)
        if use_sent_flag:
            if use_prompt_flag:
                try:
                    pattern = re.compile(r"\[C\]:(.*?) \[S\]:(.*?) \[A\]:(.*?) \[O\]:(.*?)\.")
                    matches = re.search(pattern, pt)
                    ac = matches.group(1).strip()
                    ac = ''.join(char for char in ac if char not in punctuations)

                    sp = matches.group(2).strip()
                    sp = ''.join(char for char in sp if char not in punctuations)

                    at = matches.group(3).strip()
                    at = ''.join(char for char in at if char not in punctuations)

                    ot = matches.group(4).strip()
                    ot = ''.join(char for char in ot if char not in punctuations)
                    extractions.append((ac.lower(), at.lower(), sp.lower(), ot.lower()))
                except ValueError:
                    ac, at, sp, ot = '', '', '', ''
        else:
            try:
                ac, at, sp, ot = pt.split(', ')
            except ValueError:
                ac, at, sp, ot = '', '', '', ''
            extractions.append((ac.lower(), at.lower(), sp.lower(), ot.lower()))
        return extractions

def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

# 
def compute_scores(pred_seqs, gold_seqs, sents, use_sent_flag, use_prompt_flag):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_extraction(gold_seqs[i], 'gold', i, use_sent_flag, use_prompt_flag)
        pred_list = extract_spans_extraction(pred_seqs[i], 'pred', i, use_sent_flag, use_prompt_flag)
        # lowercase_gold_list = [[element.lower() for element in inner_list] for inner_list in gold_list]
        # lowercase_pred_list = [[element.lower() for element in inner_list] for inner_list in pred_list]
        # all_labels.append(lowercase_gold_list)
        # all_preds.append(lowercase_pred_list)
        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    print("*-"*40)
    all_labels_cates = []
    all_preds_cates = []
    scores = compute_f1_scores(all_preds, all_labels)
    for i in zip(all_preds, all_labels):
        # print("labels is :", i[1], "predictions is :", i[0])
        all_preds_cates.append(i[0][0][0])
        all_labels_cates.append(i[1][0][0])
    print(scores)
    # compute cate acc
    correct_predictions = sum(1 for pred, label in zip(all_preds_cates, all_labels_cates) if pred == label)
    total_samples = len(all_preds_cates)
    cate_acc = correct_predictions / total_samples


    # case study
    for idx, pre_gold_cate in enumerate(zip(all_preds_cates, all_labels_cates)):
        pre_cate = pre_gold_cate[0]
        gold_cate = pre_gold_cate[1]
        if pre_cate != gold_cate:
            print(f"预测cate: {pre_cate}, 真实cate: {gold_cate} 数据:{sents[idx]}")
    
    print(f"Category Accuracy: {cate_acc * 100:.4f}%")
    return scores, all_labels, all_preds
