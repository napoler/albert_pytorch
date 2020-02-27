# from albert_pytorch import *
# import Terry_toolkit as tkit

# text="""


from random import sample
input_ids=[1,3,4,54,61,3,4,54]
print("int(0.15*len(input_ids))",int(0.15*len(input_ids)))
print("range(1,len(input_ids))",range(1,len(input_ids)))
for num in sample(range(1,len(input_ids)),int(0.15*len(input_ids))):
    print('num',num)
    input_ids[num] =222
print('input_ids',input_ids)


# 奇闻：印度女孩整天与蛇为伴，饿了就吃蛇肉，但是蛇从不攻击她 
# 2019-12-27 02:14
# 天下之大，无奇不有，据了解在印度有一个女孩，整天与蛇在一起，饿了的时候就吃蛇肉，但是蛇从来不攻击她！反而还有几条巨型蛇整天保护着她！

# 蛇在印度有着极高的地位，因为当地人将蛇成为守护神，然而突然有一天一个女孩的出生打破了蛇是保护神的地位，而这个女孩成了蛇的神，并且每天与蛇在一起！



# 据了解在女孩母亲怀孕的时候，由于家里极其简陋时不时的就能窜出几条蛇出来，这让家里人非常的担心，所以花了很多钱将家里修整了一下，但是还是抵挡不住身形小巧的蛇窜进家里来！但是这些蛇从来不攻击人，而是看见有人就悄悄的离开了！



# 经过十月怀胎，女孩终于临盆了，由于家境贫寒，于是就没有前去医院生产，而是选择在了家里！就在女孩出生的那是瞬间，家里突然窜出很多蛇出来，看上去极其兴奋的样子，这让女孩的奶奶感到非常的奇怪！



# 由于女孩的出生给原本就不富裕的家庭增加了更重的经济压力，女孩的父亲看见家里经常出现有蛇，便想着抓些蛇卖掉来补贴家用，在第一次卖掉蛇以后给女孩买了很多的食物，到了后来家里就经常出现一堆蛇像是集体排好让女孩父亲抓走一样！可以说女孩的成长是用卖蛇换来的钱长大的！



# 女孩慢慢的长大后，每天都会有三条体型巨大剧毒无比的眼镜王蛇伴随，就算是女孩睡觉的时候也会默默的守护在女孩身边，从来不会攻击女孩！这一消息很快就传开了，很多专家也无法解释蛇的这一行为！



# 大家知道这是怎么回事吗？如果有一堆蛇整天围着你，你会怎么对待它们呢？欢迎在评论区留言讨论！返回搜狐，查看更多







# """


# tt=tkit.Text()


# # seq_list=tt.sentence_segmentation_v1(text)
# # # print(seq_list)
# # tclass=classify(model_name_or_path='prev_trained_model/terry_rank_output')

# # for seq in seq_list:
# #     if tclass.pre(seq)>1:
# #         print(seq)
# #         print("预测结果",tclass.pre(seq))




# def _read_data( input_file):
#     """Reads a BIO data."""
#     with open(input_file) as f:
#         lines = []
#         words = []
#         labels = []
#         for line in f:
#             contends = line.strip()
#             word = line.strip().split(' ')[0]
#             label = line.strip().split(' ')[-1]
#             if contends.startswith("-DOCSTART-"):
#                 words.append('')
#                 continue
#             # if len(contends) == 0 and words[-1] == '。':
#             if len(contends) == 0:
#                 l = ' '.join([label for label in labels if len(label) > 0])
#                 w = ' '.join([word for word in words if len(word) > 0])
#                 lines.append([l, w])
#                 words = []
#                 labels = []
#                 continue
#             words.append(word)
#             labels.append(label)
#         return lines

# input_file="/mnt/data/dev/github/albert_pytorch/albert_pytorch/albert_chinese_pytorch/dataset/terryner/test.txt"
# data=_read_data(input_file)
# print(data)