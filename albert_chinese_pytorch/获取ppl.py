from albert_pytorch import *


tclass = classify(model_name_or_path='prev_trained_model/petclass/',num_labels=2,device='cuda')

text="看到别人的 个句子的ppl竟然需要1秒钟。"
ppl=tclass.ppl(text)
print(ppl)