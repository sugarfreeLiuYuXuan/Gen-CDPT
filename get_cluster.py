from sklearn.cluster import KMeans
import torch
from icecream import ic 
# from data_utils import laptop_acos_aspect_cate_list, res_acos_aspect_cate_list
from static_type import sentword2opinion, sentiment_word_list, laptop_acos_aspect_cate_list, res_acos_aspect_cate_list
import sys, re
from transformers import AutoTokenizer, RobertaModel
import numpy as np

# model_path = r"D:\ABSA-QUAD\model_dir\roberta_base"

model_path = "/data/liuyuxuan/SI-T2S/ABSA-QUAD-V2/model_dir/roberta_base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path)

res_cate_list = list(
    map(
        lambda x: x.replace("_", " "), res_acos_aspect_cate_list
    )
)

laptop_cate_list = list(
    map(
        lambda x: x.replace("_", " "), laptop_acos_aspect_cate_list
    )
)

class ClusterDict:
    def __init__(self, word, word_embedding, label):
        self.word_embedding = word_embedding
        self.word = word
        self.label = label

def get_cluster(word_list, num_clusters=8):
    word_embeddings = []

    for word in word_list:
        tokens = tokenizer(word, return_tensors="pt")
        # 获取 RoBERTa 模型的输出
        with torch.no_grad():
            output = model(**tokens)
        hidden_states = output.last_hidden_state
        # 取出 [CLS]
        cls_embedding = hidden_states[:, 0, :].squeeze().numpy()
        word_embeddings.append(cls_embedding)

    word_embeddings = np.array(word_embeddings)
    # k means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(word_embeddings)

    labels = kmeans.labels_
    # print(labels)
    map_cate_dict = {}
    ClusterDict_list = []
    for word, word_embedding, label in zip(word_list, word_embeddings, labels):
        cd = ClusterDict(word, word_embedding, label)
        ClusterDict_list.append(cd)
        # print(word, label)
    for i in ClusterDict_list: 
        map_cate_dict[i.word] = int(i.label)
    ic(map_cate_dict)
    return map_cate_dict
    
def map_categories(input_string, map_cate_dict):
    result_list = []

    # 去除括号和分号，然后按逗号分割字符串
    entries = input_string.replace('(', '').replace(')', '').replace(';', '').split(',')
    # 每四个元素为一组，进行映射
    for i in range(1, len(entries), 3):
        category = entries[i].strip()
        mapped_value = map_cate_dict.get(category.strip(), -1)
        
        # 如果映射存在，则添加到结果列表中
        if mapped_value != -1:
            result_list.append(mapped_value)

    return result_list

if __name__ == "__main__":
    # print(laptop_cate_list)
    input_string = "(null, restaurant prices, positive, well); (null, food quality, positive, well); (null, location general, positive, bad)"

    file_path1 = r"D:\ABSA-QUAD\data\acos\rest16\train.tsv"
    file_path2 = r"D:\ABSA-QUAD\data\acos\rest16\data_augmentation_clear.tsv"
    print(res_cate_list)
    # laptop_avg_list = get_cluster(laptop_cate_list, 16)
    res_cate_mapping = get_cluster(res_cate_list, 6)
    # with open(file_path1, 'r', encoding='utf-8') as file:
    #     all_text = file.read()
    # print(all_text)
    # matches = re.findall(r'\b\w*#\w*\b', all_text)
    result = map_categories(input_string, res_cate_mapping)
    print(result)



# labels is : [['null', 'restaurant general', 'great', 'null']] predictions is : [['rays boathouse', 'restaurant general', 'great', 'taurant general', 'great', 'deserving']]
# {'precision': 0.6288659793814433, 'recall': 0.6288659793814433, 'f1': 0.6288659793814433}