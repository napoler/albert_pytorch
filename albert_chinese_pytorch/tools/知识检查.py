#encoding=utf-8
import Terry_toolkit as tkit
from random import sample
# Tjson=tkit.Json(tjson)
import os
from tqdm import tqdm
import csv    #加载csv包便于读取csv文件
from random import shuffle


def load(file):
    with open(file,'r') as f:
        line=f.readline()
        print(line)

def load_kg(file):
    csv_file=open(file)    #打开csv文件
    csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件
    date=[]    #创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        # print(one_line)
        yield one_line
    #     date.append(one_line)    #将读取的csv分行数据按行存入列表‘date’中
    #     renshu = renshu + 1    #统计行数（这里是学生人数）
    # i=0
    # while i < renshu:
    #     print (date[i][3])    #访问列表date中的数据验证读取成功（这里是打印所有学生的姓名）
    #     i = i+1



def build_dataset(file,type="all"):
    """
    构建检查是不是知识
    file 文件路径
    type="all" 或者mini 
    mini
    """
    # tjson=tkit.Json(file_path=train_file)
    tjson_save=tkit.Json(file_path="../dataset/terry_kg_check/train.json")
    dev_json_save=tkit.Json(file_path="../dataset/terry_kg_check/dev.json")
    data=[]
    # f = open('data/gpt2kg.txt','a')
    k=0
    for item in tqdm(load_kg(file)):
        #排除掉大于100的文本
        if len(" #u# ".join(item))>100:
            continue
        elif  k==0:
            k=k+1
            pass
        a=[0,1,2]

        one={"label":1,'sentence':" #u# ".join(item)}
        data.append(one)

        l=[0,1,2]
        shuffle(l)
        # print(l)
        if a==l:
            pass
        else:
            text=''
            i=0
            for n in l:
                if i==2:
                   text=text+str(item[n])
                else:
                    text=text+str(item[n])+" #u# "
                i=i+1
            one={"label":0,'sentence':text}
            data.append(one)
        if k%10000==0:
                # print("***"*10)
            if type=="all":
                pass
            elif type=="mini":
                data=data[:200]
            f=int(len(data)*0.85)
            tjson_save.save(data=data[:f])
            dev_json_save.save(data=data[f:])
            data=[]

        k=k+1

    f=int(len(data)*0.85)
    tjson_save.save(data=data[:f])
    dev_json_save.save(data=data[f:])

if __name__ == '__main__':
    # fire.Fire()
    file="/mnt/data/dev/tdata/7Lore_triple.csv"
    build_dataset(file,type="all")
