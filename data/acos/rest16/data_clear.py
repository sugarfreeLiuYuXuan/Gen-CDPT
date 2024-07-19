import os
import sys
sys.path.append('/data/liuyuxuan/SI-T2S/ABSA-QUAD')

from data_utils import get_data_augmentate_dataset
# ASPECT: hot dogs. CATEGORY: food quality. OPINION: great. SENTIMENT: positive #####  ##### EAEO

def is_mostly_non_letters(s):
   letter_count = sum(1 for c in s if c.isalpha())
   return letter_count / len(s) >= 0.6

file = open('data_augmentation_clear.tsv', 'w')
data_items = get_data_augmentate_dataset('/data/liuyuxuan/SI-T2S/ABSA-QUAD/data/acos/data_agu1')
data_items_clear = []
for i in data_items:
    s = i.split("#####")[1]
    if len(i.split("#####")[1]) > 4 and is_mostly_non_letters(s):
        data_items_clear.append(i)

for i in data_items_clear:
    file.write(i)
file.close()



