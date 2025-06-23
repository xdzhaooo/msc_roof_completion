import json
from collections import OrderedDict
file_path = "config/roof_completioncopy.json"
json_str = ''
with open(file_path, 'r',encoding="utf-8") as f:
    for line in f:
        line = line.split('//')[0] + '\n'
        json_str += line
print(json_str)
#另存json_str为json文件
with open('roof_completionNLdataTuning.json', 'w') as f:
    f.write(json_str)

opt = json.loads(json_str, object_pairs_hook=OrderedDict)