# coding=utf-8
from albert_pytorch import *

tclass = classify(model_name_or_path='outputs/terry_r_rank/',num_labels=1,device='cuda')
text="威尔士柯基犬为1107年由法兰德斯工人携带过来的犬种，根据其近似狐狸的头部，有人认为本犬与尖嘴犬祖先关系密切"
text_b="威尔士柯基犬名字来自威而斯语“corrci”娇小之犬的意思。"
# ppl=tclass.ppl(text)
# print(ppl)
p=tclass.pre(text)
print(p)