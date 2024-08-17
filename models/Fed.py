import copy
import torch

# FedAvg算法
# 输入：w，模型参数列表
# 输出：w_avg，全局模型参数
# 算法描述：对于每一个参数，计算所有客户端的参数的平均值

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.mean(torch.stack([w[i][k] for i in range(len(w))]), dim=0)
    return w_avg

#等效为
'''
def FedAvg(w):
    w_avg = copy.deepcopy(w[0]) #深拷贝，不会因为w_avg的改变而改变w[0]
    for k in w_avg.keys(): #遍历字典的key，即模型的每一个参数
        for i in range(1, len(w)): #遍历每一个客户端的模型参数，从1开始，因为w_avg已经赋值为w[0]
            w_avg[k] += w[i][k] #权重累加
        w_avg[k] = torch.div(w_avg[k], len(w)) #权重平均
    return w_avg
'''