# coding=utf-8
# 测试分类效国

from albert_pytorch import classify
import Terry_toolkit as tkit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def bulid_labels():
    file_path="dataset/terry_rank/labels.json"
    tjosn=tkit.Json(file_path=file_path)
    data=[{"label": 0, "sentence": "小于2000"},{"label": 1, "sentence":"2000-1万"},{"label": 2, "sentence": "大于1万"}]
    tjosn.save(data)
def dev():
    file_path="dataset/terry_rank/dev.json"
    tjosn=tkit.Json(file_path=file_path).auto_load()
    n=0
    all=1
    data=[]
    xs=[]
    ys= []
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)
    # 打开交互模式
    plt.ion()


    for item in tqdm(tjosn):

        # text="我把小狗宠坏了，现在的小狗已经长大，一直追着兔子跑！"
        text=item['sentence']
        tclass=classify(model_name_or_path='outputs/terry_rank_output')
        if tclass.pre(text)<3:
            # print(item)
            # print("预测结果",tclass.pre(text))
            if tclass.pre(text)==item["label"]:
                n=n+1
            # print("总共预测",all)
            # print("准确数目",n)
            # print("准确率",n/all)
        # data.append((all,n))
        xs.append(all)
        ys.append(n/all)
        if all%10==0:
                    # 清除原有图像
            plt.cla()

            # 设定标题等
            # plt.title("动态曲线图", fontproperties=myfont)
            plt.grid(True)
            plt.plot(xs, ys)
            # 暂停
            plt.pause(0.1)
            plt.show()
        #     plot_dev(xs,ys)
        all=all+1
    # plot_dev(xs,ys)
    print("####"*30)
    print("总共预测",all)
    print("准确数目",n)
    print("准确率",n/all)
 

    # 关闭交互模式
    plt.ioff()
    # 图形显示
    plt.show()    
def loss_plot(task_name):
    file_path="dataset/"+task_name+".json"
 
    n=0
    all=1
    data=[]

    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)
    # 打开交互模式
    plt.ion()
    for x in range(100000*10000):
        tjosn=tkit.Json(file_path=file_path).auto_load()
        i=0
        xs=[]
        ys= []
        p_xs=[]
        p_ys=[]  
        for item in tqdm(tjosn):

            # text="我把小狗宠坏了，现在的小狗已经长大，一直追着兔子跑！"
            # text=item['sentence']

            xs.append(i)
            ys.append(item['loss'])
            i=i+1
        if ys==p_ys:
            pass
        else:
            # 清除原有图像
            plt.cla()
            # 设定标题等
            # plt.title("动态曲线图", fontproperties=myfont)
            plt.grid(True)
            plt.plot(xs, ys)
            # 暂停
            plt.pause(0.1)
            plt.show()
            p_xs=xs
            p_ys=ys
        time.sleep(1)
    # 关闭交互模式
    plt.ioff()
    # 图形显示
    plt.show()    

def plot_dev(xs,ys):
    
    print("显示图表")
    plt.figure(1) # 创建图表1
    # 打开交互模式
    plt.ion()
    # 清除原有图像
    plt.cla()
    # ax1 = plt.subplot(211) # 在图表2中创建子图1
    # for x,y in data:
    plt.plot(xs, ys)
        # plt.sca(ax1)   #❷ # 选择图表2的子图1
    plt.show()

if __name__ == "__main__":
    dev()
 
    # loss_plot("terry")
     
    # pre(text)

    # for item in tjosn:
    #     print(item)