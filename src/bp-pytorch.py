import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(4, 6)
        self.hidden_layer1 = nn.Linear(6, 6)
        # self.hidden_layer2 = nn.Linear(6, 6)
        # self.hidden_layer3 = nn.Linear(5, 4)
        self.output_layer = nn.Linear(6, 3)

    def forward(self, x):
        # relu激活函数
        # activation function
        # lr = 0.02xxxx
        # af = F.relu, step 3000, 0.950
        # af = F.relu, step 4000, 0.967
        # af = torch.sigmoid, step3000, 0.769
        # lr = 0.05
        # af = F.relu, step 3000, 0.941
        # af = F.relu, step 4000, 0.967
        # af = torch.sigmoid, step3000, 0.991
        # af = torch.sigmoid, step4000, 1.000
        # lr = 0.1
        # af = F.relu, step 3000, 0.971
        # af = F.relu, step 4000, 0.989
        # af = torch.sigmoid, step3000, 1.000
        # af = torch.sigmoid, step4000, 
        
        af = F.relu
        # af = torch.sigmoid

        x = af(self.input_layer(x))
        x = af(self.hidden_layer1(x))
        # x = af(self.hidden_layer2(x))
        # x = af(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x

def main():
    # 读入训练数据 trainingData 
    trainingData = np.loadtxt('../data/iris-training-data.txt')
    # 读入测试数据 testData
    testData = np.loadtxt('../data/iris-test-data.txt')

    # 样本特征
    trainingSamples = torch.FloatTensor(trainingData[:,1:])
    testSamples = torch.FloatTensor(testData[:,1:])

    # 样本标签
    target_trainingSamples = torch.LongTensor(trainingData[:, :1].reshape(1, -1)[0]) - 1
    target_testSamples = torch.LongTensor(testData[:, :1].reshape(1, -1)[0]) - 1
    
    avg_accuracy = []

    for n in range(1):
        net = Net()
        # 随机梯度下降
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 训练
        for i in range(3000):
            # 梯度清零
            optimizer.zero_grad()
            # 将样本 trainingSamples 传入网络得到结果 out
            out = net(trainingSamples)
            # 输出 out 与 target 对比
            loss = criterion(out, target_trainingSamples)
            # 前馈操作
            loss.backward()
            # 使用梯度优化器
            optimizer.step()

        out = net(testSamples)
        # 判断结果
        prediction = (torch.max(out, 1)[1])
        print('判断结果: ', prediction + 1)
        # 准确率
        accuracy = (prediction == target_testSamples).count_nonzero().item() / prediction.numel()
        print("准确率: ", accuracy)
        avg_accuracy.append(accuracy)

    print(np.average(np.array(avg_accuracy)))


main()