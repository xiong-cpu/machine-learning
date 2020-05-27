#编程实现用前向后概率计算给定的p(o,)

# @author：  熊荐辕   学号：201700171089
# encoding=utf8

import numpy as np
import csv


class HMM(object):
    def __init__(self, A, B,Pi,O):
        self.A = A # 状态转移概率矩阵
        self.B = B  # 观测概率矩阵
        self.Pi = Pi  # 初始状态概率矩阵

        self.N = np.shape(A)[0]  # 可能的状态数
        self.O = O
        self.T = len(O)


    def forward(self):
        """
        前向算法
        """
        self.alpha = np.zeros((self.T, self.N))

        # 公式 10.15
        for i in range(self.N):
            self.alpha[0][i] = self.Pi[i] * self.B[i][self.O[0]]

        # 公式10.16
        for t in range(1, self.T):
            for i in range(self.N):
                sum = 0
                for j in range(self.N):
                    sum += self.alpha[t - 1][j] * self.A[j][i]
                self.alpha[t][i] = sum * self.B[i][self.O[t]]

    def backward(self):
        """
        后向算法
        """
        self.beta = np.zeros((self.T, self.N))

        # 公式10.19
        for i in range(self.N):
            self.beta[self.T - 1][i] = 1

        # 公式10.20
        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j] * self.B[j][self.O[t + 1]] * self.beta[t + 1][j]


    def shuchu(self,t,i):
        self.forward()
        self.backward()
        t=t-1          #矩阵是从0开始计算，少一个
        i=i-1
        m= self.alpha[t][i] * self.beta[t][i] #前向后向乘积
        sub=0
        for j in range(self.N):
            sub = sub+self.alpha[t][j] * self.beta[t][j]
        return m/sub

if __name__ == '__main__':
    A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]  # 定义A，B，Pi，和O
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    Pi = ([0.2, 0.3, 0.5])
    O = [1,0,1,1,0,1,0,0]
    hmm=HMM(A,B,Pi,O)
    print(hmm.shuchu(4,3))
