# d导入包
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
from data_create import DataCreater


class FitANN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """定义初始化"""
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        """初始化权重，取均值为1，方差为下一层节点数的根"""
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        """初始化偏置，全为0"""
        self.hidden_bias = np.zeros((self.hnodes, 1))
        self.final_bias = np.zeros((self.onodes, 1))
        """学习率"""
        self.lr = learningrate
        """激活函数"""
        self.activation_function = lambda x: ss.expit(x)
        """误差列表"""
        self.losses = []

    def train(self, input, target, i=1):
        """正向传播过程"""
        # 输入层输入
        # inputs = np.array(input,ndmin=2).T # 1 * 1标量
        # targets = np.array(targetlist,ndmin=2).T # 1* 1 标量
        # 隐藏层输入
        hidden_inputs = self.wih * input # 10,1
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs + self.hidden_bias) # 10,1
        # 输出层输入
        final_inputs = self.who @ hidden_outputs # 1,1
        # 输出层输出
        final_outputs = final_inputs + self.final_bias
        # 获得误差
        final_error = target - final_outputs
        # 添加误差
        self.losses.append(final_error)

        """反向传播"""
        # 更新who
        who_grid = final_error * hidden_outputs.T #1,10
        self.who += i * self.lr * who_grid
        # 更新fb
        fb_grid = final_error
        self.final_bias += i * self.lr * fb_grid
        # 更新wih
        wih_grid = final_error * self.who.T * hidden_outputs * (1 - hidden_outputs) * input# 10,1
        self.wih += i * self.lr * wih_grid
        # 跟新hb
        hb_grid = final_error * self.who.T * hidden_outputs * (1 - hidden_outputs)# 10,1
        self.hidden_bias += i * self.lr * hb_grid

    def query(self, input):
        """正向传播过程"""
        # 输入层输入
        # =
        # 隐藏层输入
        hidden_inputs = self.wih * input
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs + self.hidden_bias)
        # 输出层输入
        final_inputs = self.who @ hidden_outputs
        # 输出层输出
        final_outputs = final_inputs + self.final_bias
        return final_outputs

    def get_sumlosses(self):
        sum_losses = sum(np.array(self.losses) ** 2) / (2 * len(self.losses))
        return sum_losses


if __name__ == '__main__':
    """主逻辑"""
    np.random.seed(1)
    """初始化数据，x归一化，y正常"""
    num = 100 # 训练数据数量
    data = DataCreater(num)  # 获取一百个数据
    # 添加随机值
    #data.make_random()
    #归一化
    data.normliza_x()
    x, y = data.get_data()
    """初始化神经网络"""
    inputnodes = 1
    hiddennodes = 10
    outputnodes = 1
    learningrate = 0.003
    n = FitANN(inputnodes, hiddennodes, outputnodes, learningrate)
    """训练"""
    # 设置衰减学习率
    decay = 0.001
    epoch = 10000  # 训练次数
    """动画"""
    plt.ion()
    for j in range(epoch):
        for i in range(num):
            n.train(x[i], y[i])
        # lr = learningrate / (1 + learningrate * decay)
        # n.set_lr(lr)

        if j % 100 == 0:
            """查询"""
            num_query = 1000
            data_query = DataCreater(num_query)
            data_query.normliza_x()
            once_x_query, _ = data_query.get_data()
            once_y_query = [float(n.query(j)) for j in once_x_query]

            plt.cla()
            data.plot_data()
            plt.plot(once_x_query,once_y_query,color='r')
            if len(n.losses) != 0:
                plt.text(0.5,0,"LOSS=%.4f" % n.get_sumlosses(), fontdict={'size': 20, 'color':  'red'})
                # print("第{}次均方误差为{}".format(j, n.get_sumlosses()))
            plt.show()
            plt.pause(0.1)


        # 误差归零
        n.losses = []
    plt.ioff()
