# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
import sys
import re
import os
from torch.utils.data import Dataset
import torch
import sys
from tqdm import tqdm
from icecream import ic
from get_cluster import get_cluster, map_categories
sys.path.append('/data/liuyuxuan/SI-T2S/ABSA-QUAD/transformers/src/transformers')
from transformers import AdamW, T5Tokenizer, AutoTokenizer, RobertaModel, T5Config, RobertaConfig
from static_type import sentword2opinion, sentiment_word_list, laptop_acos_aspect_cate_list, res_acos_aspect_cate_list, res_cate_list, laptop_cate_list

def read_list_from_txt(file_path):
    output_list = []
    with open(file_path, 'r') as f:
        for line in f:
            output_list.append(line.strip())
    return output_list

# 原始数据读取 data_path +  -> text:list[str]  label:list[str]
def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels

def read_line_examples_from_acos(data_path, E_I='ALL'):
    text = ''
    lines = []
    items = []
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.split('\n')
    for l in lines:
        items.append(l.split('\t'))
    acos_texts = []
    acos_labels =[]
    for i in items:
        acos_texts.append(i[:1])
        acos_labels.append(i[1:])
    print("sentiment type is: ", E_I)
    # 过滤IA\IO等情况
    def _filter_E_I_(E_I):
        _filtered_text = []
        _filtered_labels = []
        for i in zip(acos_texts, acos_labels):
            _texts = i[0]
            _labels = i[1]
            if E_I == "EAEO":
                _filter_labels = list(filter(lambda x: x[:5]!='-1,-1' and x[-5:]!='-1,-1', _labels))
            if E_I == "IAEO":
                _filter_labels = list(filter(lambda x: x[:5]=='-1,-1' and x[-5:]!='-1,-1', _labels))
            if E_I == "EAIO":
                _filter_labels = list(filter(lambda x: x[-5:]=='-1,-1' and x[:5]!='-1,-1', _labels))
            if E_I == "IAIO":
                _filter_labels = list(filter(lambda x: x[:5]=='-1,-1' and x[-5:]=='-1,-1', _labels))
            if _filter_labels != []:
                _filtered_text.append(_texts)
                _filtered_labels.append(_filter_labels)              
        return _filtered_text, _filtered_labels
    
    # 过滤类别cluster等情况-筛选无数据增强的数据
    # def __filter_cate_type__(cate_type):
    #     _filtered_cate_type_text = []
    #     _filtered_cate_type_labels = []
    #     for i in zip(acos_texts, acos_labels):
    #         _texts = i[0]
    #         _labels = i[1]
    #     # 用于categories mapping
    #     mapping_cate = {}
    

    if E_I == "ALL":
        # if cate_type != -1:
        #     if "rest16" in data_path:
        #         mapping_cate = get_cluster(res_acos_aspect_cate_list, cluster_nums)
        #     else:
        #         mapping_cate = get_cluster(laptop_acos_aspect_cate_list, cluster_nums)
            
        return acos_texts, acos_labels
    else:
        new_texts, new_labels = _filter_E_I_(E_I)
        print(f"the len of {E_I} is", len(new_texts), " and ", len(new_labels))
        
        
        return new_texts, new_labels

def get_transformed_io_acos(acos_texts, acos_labels, use_augmentation, data_dir_all=None, cate_type=None, num_clusters=6):
    # lyx add
    # transfomrmed acos_texts
    # acos_items = read_line_examples_from_acos(data_path)
    # acos_texts = []
    # acos_labels =[]
    # for i in acos_items:
    #     acos_texts.append(i[:1])
    #     acos_labels.append(i[1:])
    
    acos_texts_splited = []
    acos_labels_splited = []
    for i, acos_text in enumerate(acos_texts):
        acos_texts_splited.append(acos_text[0].split(" "))
    
    def _senttag2word_and_get_index(x):
        # get senttag2word
        if len(x) == 1:
            if int(x) == 0:
                x = "negative"
            elif int(x) == 1:
                x = "neutral"
            else:
                x = "positive"
        # get index
        if ',' in x and '-1' not in x.split(","):
            pos = x.index(",")
            x = x[:pos] + "," + str((int(x.split(",")[-1]) - 1))
        else:
            pass
        return x

    acos_labels_splited = []
    for i in acos_labels:
        acos_one_label_splited = []
        for j in i:
            split_one_label = j.split(" ")
            split_one_label = list(map(_senttag2word_and_get_index, split_one_label))
            acos_one_label_splited.append(split_one_label)
        acos_labels_splited.append(acos_one_label_splited)


    acos_texts_and_labels = list(zip(acos_texts_splited, acos_labels_splited))
    # transform all index into word
    index2word_labels = []
    for index, t_l in enumerate(acos_texts_and_labels):
        def _replace_word(x):
                if ',' in x and '-1' not in x.split(","):
                    pos = x.index(",")
                    start = int(x[:pos]) # 1
                    # print("start is", start)
                    end = int(x[pos:].replace(',', ''))  # 4 youwyx
                    # print("end is", end)
                    x_list = t_l[0][start:end+1]
                    x_sent = ' '.join(x_list)
                    # print("x_sent is:", x_sent) 
                    x = x_sent
                if ',' in x and '-1' in x.split(","):
                    x = 'null'
                return x
        index2word_one_label = []
        for l in t_l[1]:
            index2word_one_label.append(list(map(_replace_word, l)))
        index2word_labels.append(index2word_one_label)
    # transformed index2word_labels to string 
    word_labels = []
    for i in index2word_labels:
        temp_labels = []
        for j in i:
            temp_one_label = ', '.join(j)
            temp_one_label = '(' + temp_one_label + ')'
            temp_labels.append(temp_one_label)
        word_labels.append('; '.join(temp_labels))
        
    
    # --------------------------------------------------------------
    def get_augmentate_inputs_labels(data_dir_all):
        if data_dir_all == None and use_augmentation:
            print("error use data_augmentate_dataset checkout dir!")
            sys.exit(0)
        data_augmentat_items = get_data_augmentate_dataset(data_dir_all)
        
        raw_labels = list(
            map(
                lambda x: x.split("#####")[0], data_augmentat_items
            )
        )
        # raw_labels 
        # ASPECT: blond wood decor, premium sake, service. CATEGORY: ambience general, drinks quality, service general. OPINION: soothing, excellent, great. SENTIMENT: positive, positive, positive ->(food, FOOD#QUALITY, negative, too sweet); (food, FOOD#QUALITY, negative, too salty); (portions, FOOD#STYLE_OPTIONS, negative, tiny).

        def __format_raw_lables__(raw_label):
            aspect_match = re.search(r'ASPECT: (.+?)\.', raw_label)
            raw_a = aspect_match.group(1) if aspect_match else ""
            
            category_match = re.search(r'CATEGORY: (.+?)\.', raw_label)
            raw_c = category_match.group(1) if category_match else ""
            
            sentiment_match = re.search(r'SENTIMENT: (.+)', raw_label)
            raw_s = sentiment_match.group(1) if sentiment_match else ""
            
            opinion_match = re.search(r'OPINION: (.+?)\.', raw_label)
            raw_o = opinion_match.group(1) if opinion_match else ""
            # print(raw_a, "#", raw_c, "#", raw_s, "#", raw_o, "#", raw_label)
            assert len(raw_a.split(", ")) == len(raw_s.split(", "))
            one_quad = []
            for item in zip(raw_a.split(", "), raw_c.split(", "), raw_s.split(", "), raw_o.split(", ")):
                # 1 a  2 c  3 s  4 o
                one_quad.append(f"({item[0]}, {item[1]}, {item[2]}, {item[3]})")
                
            return "; ".join(one_quad)
                   
        augmentate_labels = list(
            map(
                __format_raw_lables__, raw_labels
            )
        )
        
        raw_texts = list(
            map(
                lambda x: x.split("#####")[1].replace("  . ", "", 1), data_augmentat_items
            )
        )
        
        augmentate_texts = list(
            map(
                lambda x: x.split(" "), raw_texts
            )
        )
        # print(raw_text[1], "####", augmentate_labels[1])
        return augmentate_texts, augmentate_labels
        # if use_sent_flag:
        #     one_template = f"[C]:{c}. [S]:{s}. [A]:{a}. [O]:{o}."
    # 得到增强数据
    augmentate_texts = []
    augmentate_labels = []
    if data_dir_all != None and use_augmentation:
        augmentate_texts, augmentate_labels = get_augmentate_inputs_labels(data_dir_all)
    # sys.exit(0)
    # TODO 去除 ''
    new_augmentate_texts = [word for word in augmentate_texts if word != '']
    acos_texts_splited.extend(new_augmentate_texts)
    word_labels.extend(augmentate_labels)
    cate_mapping = {} 
    if cate_type != -1:
        if num_clusters <= 0 or num_clusters is None:
            print(f"num_clusters can't be setted {num_clusters}, only > 0")
            sys.exit(0)
        print(f"****** cluster ing ! ****** and total: {num_clusters}. current cate_type: {cate_type}")
        if "laptop14" in data_dir_all:
            cate_mapping = get_cluster(laptop_cate_list, num_clusters)
        else:
            cate_mapping = get_cluster(res_cate_list, num_clusters)
        # log to debug
        # ic(cate_mapping)
        cate_type_acos_texts = []
        cate_type_word_labels = []
        for item in zip(acos_texts_splited, word_labels):
            # filter_cate_type = [0, 2, 4] etc.
            filter_cate_type = map_categories(item[1].lower().replace("#", ' ').replace("_", ' '), cate_mapping)
            if cate_type in filter_cate_type:
                cate_type_acos_texts.append(item[0])
                cate_type_word_labels.append(item[1])
        assert len(cate_type_acos_texts) == len(cate_type_word_labels)
        print(f"****** select cate type {cate_type}******  and length is {len(cate_type_acos_texts)}")
        return cate_type_acos_texts, cate_type_word_labels
    
    else:
        return acos_texts_splited, word_labels

def listtostr(arr):
    return " ".join(arr)

# 
def get_sentence_vectors(sentences, bert_model, bert_tokenizer):
    sentence_vectors = []
    for sentence in tqdm(sentences):
        inputs = bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        sentence_vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        sentence_vectors.append(sentence_vector)
    
    return sentence_vectors

# 生成为模板target 和target句子
def get_para_acos_targets_adma(texts, labels, use_sent_flag, use_prompt_flag):
    # # 是否用句子生成
    # use_sent_flag = False
    # # 是否用prompt
    # use_prompt_flag = False
    # 计算
    count = 0
    EA_EO_count = 0
    EA_IO_count = 0
    IA_EO_count = 0
    IA_IO_count = 0
    #输入带有<aspect>等的input
    targets = []
    template_text = []
    # index为texts索引

    for index, label in enumerate(labels):
        label = label.replace('(','')
        label = label.replace(')','')
        one_sent_label = label.split('; ')
        count += 1
        origin_template = '[C]:[category]. [S]:[sentiment]. [A]:[aspect]. [O]:[opinion].'
        template = []
        querys = []
        # index_label为label里面的每个label的索引,方便<aspect_0>
        for index_label, o_l in enumerate(one_sent_label):
            
            a = o_l.split(", ")[0]
            c = o_l.split(", ")[1].replace("#", " ").lower()
            s = o_l.split(", ")[2]
            o = o_l.split(", ")[3]
            if a == '' and c == '' and s == '' and o == '':
                break
            # TODO 0 #######  ###### , , , 有脏数据
            # print(index_label, "#######",a,c,s, "######", texts[index])
            s = sentword2opinion[s.strip()]
            
            

            if o_l.split(", ")[0] != 'null' and o_l.split(", ")[3] != 'null':
                # EA_EO_count += 1
                if use_sent_flag:
                    one_template = f"[C]:{c}. [S]:{s}. [A]:{a}. [O]:{o}."
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
    
            if o_l.split(", ")[0] == 'null' and o_l.split(", ")[3] != 'null':
                # EA_IO_count += 1
                if use_sent_flag:
                    one_template = f"[C]:{c}. [S]:{s}. [A]:{a}. [O]:null."
                    
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
                  
            if o_l.split(", ")[0] != 'null' and o_l.split(", ")[3] == 'null':
                # IA_EO_count += 1
                if use_sent_flag:
                    one_template = f"[C]:{c}. [S]:{s}. [A]:null. [O]:{o}."
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
                
            if o_l.split(", ")[0] == 'null' and o_l.split(", ")[3] == 'null':
                # IA_IO_count += 1
                if use_sent_flag:
                    one_template = f"[C]:{c}. [S]:{s}. [A]:null. [O]:null."
                else:
                    one_template = f"{a}, {c}, {s}, {o}"

            query = listtostr(texts[index])
            # query = query + ' [SEP] ' + origin_template
            # 获得模板和句子 query就是一条句子的里的所有四元组模板
            querys.append(query)
            template.append(one_template)
            
        template_text.append((' [SSEP] '.join(querys)).split())

        targets.append(' [SSEP] '.join(template))
    texts = list(
        map(lambda x: origin_template + ' [SEP] ' + ' '.join(x), texts)
    )
    # targets = list(map(lambda x: origin_template + ' [SEP] ' + x, targets))
    return texts, targets

# 2024 1月8号 获得增强数据 extend进原始数据
def get_data_augmentate_dataset(data_dir_all):
        #####
        split_token =  "#####"
        
        prefix = "data_augmentation" 
        files = os.listdir(data_dir_all)
        all_files = [data_dir_all + "/"+ file for file in files if file.startswith(prefix)]
        data_items = []
        for f in all_files:
            with open(f, 'r', encoding='utf-8') as data_f:
                data_lines = data_f.readlines()
            data_items.extend(data_lines)
        data_items = list(set(data_items))
        return data_items
                

def get_para_acos_targets(texts, labels, use_sent_flag, use_prompt_flag):
# 计算
    count = 0
    EA_EO_count = 0
    EA_IO_count = 0
    IA_EO_count = 0
    IA_IO_count = 0
    # 输入加入prompt
    targets = []
    template_text = []
    # index为texts索引
    for index, label in enumerate(labels):
        label = label.replace('(','')
        label = label.replace(')','')
        one_sent_label = label.split('; ')
        count += 1
        
        template = []
        querys = []
        # index_label为label里面的每个label的索引
        for index_label, o_l in enumerate(one_sent_label):
            # debug o_l
            # print(o_l)
            
            if o_l.split(", ")[0] != 'NULL' and o_l.split(", ")[3] != 'NULL':
                # EA_EO_count += 1
                a = o_l.split(", ")[0]
                c = o_l.split(", ")[1].replace("#", " ").lower()
                s = o_l.split(", ")[2]
                s = sentword2opinion[s]
                o = o_l.split(", ")[3]

                if use_sent_flag:
                    one_template = f"It is explicit aspect, explicit opinion, {c} is {s} because {a} is {o}."
                    
                else:
                    one_template = f"{a}, {c}, {s}, {o}"

                query = listtostr(texts[index])
                    
            if o_l.split(", ")[0] == 'NULL' and o_l.split(", ")[3] != 'NULL':
                # EA_IO_count += 1
                a = o_l.split(", ")[0]
                c = o_l.split(", ")[1].replace("#", " ").lower()
                s = o_l.split(", ")[2]
                s = sentword2opinion[s]
                o = o_l.split(", ")[3]

                if use_sent_flag:
                    one_template = f"It is explicit aspect, implicit opinion, {c} is {s} because it is {o}."
                    
                else:
                    one_template = f"{a}, {c}, {s}, {o}"

                query = listtostr(texts[index])

class ABSADataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_dir,
                 data_type,
                 max_len=256, 
                 E_I='ALL', 
                 use_sent_flag=True, 
                 use_prompt_flag=True, 
                 use_augmentation=False, 
                 cate_type=-1,
                 num_clusters=-1):
        # './data/rest16/train.txt'
        # /data/liuyuxuan/SI-T2S/ABSA-QUAD/data/acos/rest16
        self.data_type = data_type
        self.data_path = f'data/acos/{data_dir}/{data_type}.tsv'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.inputs = []
        self.targets = []
        self.E_I = E_I
        self.cate_type = cate_type
        self.num_clusters = num_clusters
        self.use_sent_flag = use_sent_flag
        self.use_prompt_flag = use_prompt_flag
        self.use_augmentation = use_augmentation
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):
        if self.data_type != 'train':
            self.use_augmentation = False
        temp_input, temp_labels = read_line_examples_from_acos(self.data_path, self.E_I)
        inputs, targets = get_transformed_io_acos(temp_input, 
                                                  temp_labels, 
                                                  use_augmentation=self.use_augmentation,
                                                  data_dir_all=f'data/acos/{self.data_dir}',
                                                  cate_type=self.cate_type,
                                                  num_clusters=self.num_clusters)
        # demo
        # print(" demo data : ", inputs[9], targets[9])
        # print(inputs[3],"########", targets[3])
        # print(type(targets[3]))
        # sys.exit(0)
        inputs, targets = get_para_acos_targets_adma(inputs, 
                                                targets, 
                                                use_sent_flag=self.use_sent_flag, 
                                                use_prompt_flag=self.use_prompt_flag)
        for i in range(len(inputs)):
            # change input and target to two strings
            input = inputs[i]
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


if __name__ == "__main__":
    from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
    t5_path = "/data/liuyuxuan/SI-T2S/ABSA-QUAD-V2/model_dir/t5_base"
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    # tokenizer = T5Tokenizer.from_pretrained(r'D:\ABSA-QUAD\model_dir\t5_base')
    tokenizer.add_special_tokens({"additional_special_tokens":['[aspect]', '[category]', '[opinion]', '[sentiment]']})
    tokenizer.add_special_tokens({"additional_special_tokens": [" [SSEP] ", "[A]", "[C]", "[S]", "[O]"]})
    data_type = 'EAIO'
    acos_dataset_origin = ABSADataset(tokenizer=tokenizer,
                               data_dir='rest16',
                               data_type='train',
                               E_I=data_type,
                               use_sent_flag=True,
                               use_prompt_flag=False,
                               use_augmentation=False)
    
    acos_dataset_aug = ABSADataset(tokenizer=tokenizer,
                               data_dir='rest16',
                               data_type='train',
                               E_I=data_type,
                               use_sent_flag=True,
                               use_prompt_flag=False,
                               use_augmentation=True)
    
    print(len(acos_dataset_origin))
    print(len(acos_dataset_aug))
    # T-SNE
    # data_origin = read_list_from_txt("data_ori.txt")
    # data_aug = read_list_from_txt("data_aug.txt")
    data_origin = []
    data_aug = []
    for item in tqdm(acos_dataset_origin):
        data_o = tokenizer.decode(item['source_ids'], skip_special_tokens=True)
        data_origin.append(data_o)
    for item in tqdm(acos_dataset_aug):
        data_a = tokenizer.decode(item['source_ids'], skip_special_tokens=True)
        data_aug.append(data_a)
    data_aug = list(
        map(
            lambda x:x.split(" [SEP] ")[-1], data_aug
        )
    )
    data_origin = list(
        map(
            lambda x:x.split(" [SEP] ")[-1], data_origin
        )
    )
    print(data_origin[41])
    
    bert_path = "/data/liuyuxuan/SI-T2S/ABSA-QUAD-V2/model_dir/roberta_base"
    bert_model = RobertaModel.from_pretrained(bert_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    print("-------------###------------")
    vec_origin = get_sentence_vectors(data_origin, bert_model, bert_tokenizer)
    vec_ag = get_sentence_vectors(data_aug, bert_model, bert_tokenizer)
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # kmeans_origin = KMeans(n_clusters=n_clusters, init='k-means++')
    # kmeans_origin.fit(vec_origin)
    kmeans_aug = KMeans(n_clusters=len(vec_origin), init='k-means++')
    kmeans_aug.fit(vec_ag)
    centroids1 = np.stack(vec_origin)
    # centroids2 = np.stack(vec_ag)
    # centroids1 = kmeans_origin.cluster_centers_
    print(centroids1.shape)
    centroids2 = kmeans_aug.cluster_centers_
    print(centroids2.shape)
    # for c_id in centroids:
    #     print(type(c_id))
    #     # print(c_id)
    #     print(c_id.shape)
    # 合并两组聚类中心点的数据
    all_centroids = np.concatenate([centroids1, centroids2], axis=0)

    # 创建标签
    labels = np.concatenate([np.zeros(len(centroids1)), np.ones(len(centroids2))], axis=0)

    # 创建颜色数组，根据标签设置不同的颜色
    colors = ['#82B0D2' if label == 0 else '#FA7F6F' for label in labels]

    # 将高维向量进行 T-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    centroids_2d = tsne.fit_transform(all_centroids)

    # 绘制 T-SNE 可视化图
    plt.figure(figsize=(8, 8), dpi=800)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=colors, marker='o', s=50)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(True)
    plt.savefig(f'pic_tsne_{data_type}.jpg')

    
    
    
    
    
    
    
    
    
    # data_sample = acos_dataset[round(len(acos_dataset)/2)]
    # data_sample = acos_dataset[3]
    # # 0 622
    # print(data_sample['source_ids'].size())
    # print(data_sample['target_ids'].size())
    # print('██████ Input is ██████\n', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    # print('██████ Output is ██████\n', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

    # {'ambience general': 3,
    #                'drinks prices': 1,
    #                'drinks quality': 1,
    #                'drinks style options': 1,
    #                'food prices': 5,
    #                'food quality': 4,
    #                'food style options': 4,
    #                'location general': 2,
    #                'restaurant general': 0,
    #                'restaurant miscellaneous': 0,
    #                'restaurant prices': 0,
    #                'service general': 2}
    # 0 : 622  1222
    # 1 ：80   683
    # 2 ：508  553
    # 3 ：247  396
    # 4 ：823  20
    # 5 ：81   956

    # {'battery design features': 1,
    #                 'battery general': 0,
    #                 'battery operation performance': 5,
    #                 'battery quality': 2,
    #                 'company design features': 1,
    #                 'company general': 3,
    #                 'company operation performance': 5,
    #                 'company price': 2,
    #                 'company quality': 2,
    #                 'cpu design features': 1,
    #                 'cpu general': 3,
    #                 'cpu operation performance': 5,
    #                 'cpu price': 2,
    #                 'cpu quality': 2,
    #                 'display design features': 1,
    #                 'display general': 3,
    #                 'display operation performance': 5,
    #                 'display price': 2,
    #                 'display quality': 2,
    #                 'display usability': 1,
    #                 'fans&cooling design features': 4,
    #                 'fans&cooling general': 4,
    #                 'fans&cooling operation performance': 4,
    #                 'fans&cooling quality': 4,
    #                 'graphics design features': 1,
    #                 'graphics general': 0,
    #                 'graphics operation performance': 5,
    #                 'graphics usability': 1,
    #                 'hard disc design features': 1,
    #                 'hard disc general': 0,
    #                 'hard disc miscellaneous': 0,
    #                 'hard disc operation performance': 5,
    #                 'hard disc quality': 2,
    #                 'hardware design features': 1,
    #                 'hardware general': 0,
    #                 'hardware operation performance': 5,
    #                 'hardware quality': 2,
    #                 'hardware usability': 1,
    #                 'keyboard design features': 1,
    #                 'keyboard general': 0,
    #                 'keyboard operation performance': 5,
    #                 'keyboard portability': 1,
    #                 'keyboard quality': 2,
    #                 'keyboard usability': 1,
    #                 'laptop connectivity': 1,
    #                 'laptop design features': 1,
    #                 'laptop general': 0,
    #                 'laptop miscellaneous': 0,
    #                 'laptop operation performance': 5,
    #                 'laptop portability': 1,
    #                 'laptop price': 2,
    #                 'laptop quality': 2,
    #                 'laptop usability': 1,
    #                 'memory design features': 1,
    #                 'memory general': 3,
    #                 'memory operation performance': 5,
    #                 'memory quality': 2,
    #                 'memory usability': 1,
    #                 'motherboard operation performance': 5,
    #                 'motherboard quality': 2,
    #                 'mouse design features': 1,
    #                 'mouse general': 3,
    #                 'multimedia devices connectivity': 0,
    #                 'multimedia devices design features': 1,
    #                 'multimedia devices general': 0,
    #                 'multimedia devices operation performance': 5,
    #                 'multimedia devices price': 0,
    #                 'multimedia devices quality': 2,
    #                 'optical drives design features': 1,
    #                 'optical drives general': 0,
    #                 'optical drives operation performance': 5,
    #                 'os design features': 1,
    #                 'os general': 3,
    #                 'os operation performance': 5,
    #                 'os price': 2,
    #                 'os quality': 2,
    #                 'os usability': 1,
    #                 'out of scope general': 3,
    #                 'out of scope operation performance': 5,
    #                 'ports connectivity': 2,
    #                 'ports design features': 1,
    #                 'ports general': 3,
    #                 'ports operation performance': 5,
    #                 'ports portability': 2,
    #                 'ports quality': 2,
    #                 'ports usability': 1,
    #                 'power supply connectivity': 2,
    #                 'power supply design features': 1,
    #                 'power supply general': 0,
    #                 'power supply operation performance': 5,
    #                 'power supply quality': 2,
    #                 'shipping general': 0,
    #                 'shipping operation performance': 5,
    #                 'shipping price': 2,
    #                 'shipping quality': 2,
    #                 'software design features': 1,
    #                 'software general': 3,
    #                 'software operation performance': 5,
    #                 'software portability': 1,
    #                 'software price': 2,
    #                 'software quality': 2,
    #                 'software usability': 1,
    #                 'support design features': 1,
    #                 'support general': 3,
    #                 'support operation performance': 5,
    #                 'support price': 2,
    #                 'support quality': 2,
    #                 'warranty general': 0,
    #                 'warranty quality': 2}
