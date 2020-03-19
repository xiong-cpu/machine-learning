# encoding=utf-8
# @Author: WenDesi
# @Date:   09-08-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

# @author：  熊荐辕
# @description:  此代码为本人参考上面原文链接修改训练算法为对偶形式而来

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001 #学习提低
        self.max_iteration = 5000 #最大迭代次数

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])#w的长度从0到最大
        return int(wx > 0)  #wx大于0返回1，小于0返回0

    def train(self, features, labels):
        self.a= [0.0] * (len(features[0]) + 1)  #a的赋值
        self.w= [0.0] * (len(features[0]) + 1)  #w的赋值
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)  #在0到len（labels）-1中取一个随机数
            x = list(features[index])
            x.append(1.0)           #1.0插入到x链表
            y = 2 * labels[index] - 1  #x从0到1变成（-1,1）
            wx = sum([self.w[j] * x[j]  for j in range(len(self.a))])  #wx是个0到最大长度的链表

            if wx * y > 0:       #大于零表示没有误分类点，这一段的意思是正确分类达到最大分类的次数超过最大迭代次数后就停止循环
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.a[i] += self.learning_step      #对a的梯度更新
                self.w[i]=self.a[i] * x [i] *y    #由a算出w

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)     #把1插入到x链表
            labels.append(self.predict_(x))  #把预测值插入到标签两边
        return labels  #把标签返回


if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()  #当前读取时间1

    raw_data = pd.read_csv('../../lihang_book/data/train_binary.csv', header=0)
    data = raw_data.values

    imgs= data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time() #当前读取时间2
    print ('read data cost ', time_2 - time_1, ' second', '\n')  #输出数据读取所花费的时间

    print ('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' second', '\n') #输出训练数据所花费的时间

    print ('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n') #输出预测花费的时间

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score) #输出分数
