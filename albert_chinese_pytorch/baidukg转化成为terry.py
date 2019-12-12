
import Terry_toolkit as tkit
from random import sample
# Tjson=tkit.Json(tjson)
import os
from tqdm import tqdm

def build_dataset(train_file,type="all"):
    """
    百度训练集
    train_file 文件路径
    type="all" 或者mini 
    mini
    """
    tjson=tkit.Json(file_path=train_file)
    tjson_save=tkit.Json(file_path="dataset/terry_kg/train.json")
    dev_json_save=tkit.Json(file_path="dataset/terry_kg/dev.json")
    data=[]
    # f = open('data/gpt2kg.txt','a')
    for item in tqdm(tjson.load()):
        
        text= item['text']
        # print(text)
        # print(item['spo_list'])
        predicate={}

        kg_list=[] #定义已经已经标注数据

        for n in item['spo_list']:
            # predicate[n['predicate']]=[]
            # print(n)
            # print(n)
            kg=n['subject']+","+n['predicate']+","+n['object']
            kg_list.append([n['subject'],n['predicate'],n['object']])
            label=1
            one={"label":label,'sentence':kg+"[kg]"+text}
            data.append(one)

        words=[]
        for w in  item['postag']:
            # print(w["word"])
           words.append(w['word'])
        # allkg=[]
        if len(words)>3:
            for i in range(len(kg_list)):
                nokg=sample(words,3)
                # allkg.append(nokg)
                if nokg in kg_list:
                    # print("已标记数据")
                    pass
                else:
                    #产生非数据
                    one={"label":label,'sentence':",".join(nokg)+"[kg]"+text}
                    label=0
                    data.append(one)
            # print(allkg)


        # print("***"*10)
    if type=="all":
        pass
    elif type=="mini":
        data=data[:200]
    f=int(len(data)*0.85)
    tjson_save.save(data=data[:f])
    dev_json_save.save(data=data[f:])


if __name__ == '__main__':
    # fire.Fire()
    train_files=["dataset/terrykg/train.json","dataset/terrykg/dev.json"]
    train_file="data/train.json"
    dev_file="data/train.json"
    if os.path.exists(train_file) or os.path.exists(dev_file):
        print("文件已经存在")
        print("请手动删除")
    else:
        for f in train_files:
            # build_dataset(f,type="all")
            build_dataset(f,type="mini")