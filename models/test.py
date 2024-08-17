
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval() #评估模式
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs) #加载测试集
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda() #数据加载到GPU
        log_probs = net_g(data) #前向传播
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()  #计算并累加当前批次的交叉熵损失
        y_pred = log_probs.data.max(1, keepdim=True)[1] #找到概率最大的类别
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum() #计算正确率

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

