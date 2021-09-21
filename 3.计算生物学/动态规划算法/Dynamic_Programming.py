
from numpy import *
import copy
from matplotlib import pyplot as plt

# 创建全局比对得分矩阵
def GlobalScoreMatrix(m,n,w,replace,s,path,senquence1,senquence2,gap):
    for i in range(m):
        for j in range(n):
            # 判断s（0,0）这一特殊情况
            if i == 0 and j == 0:
                s[i][j] = 0
            elif i-1 < 0:  # 判断第一行的特殊情况
                s[i][j] = s[i][j - 1] + gap
                path[i, j, 0] = 1
            elif j-1 < 0:  # 判断第一列的特殊情况
                s[i][j] = s[i - 1][j] + gap
                path[i,j,1] = 1
            else:
                # 获取最大值
                s[i][j] = max(s[i - 1][j - 1] + w[replace[senquence2[i - 1]]][replace[senquence1[j - 1]]],
                              s[i - 1][j] + gap, s[i][j - 1] + gap)
                # 记录最大值来的方向
                if s[i - 1][j - 1] + w[replace[senquence2[i - 1]]][replace[senquence1[j - 1]]] == s[i][j]:
                    path[i,j,2] = 1
                if s[i - 1][j] + gap == s[i][j]:
                    path[i,j,1] = 1
                if s[i][j - 1] + gap == s[i][j]:
                    path[i,j,0] = 1


# 寻找全局序列比对路径
def FindGlobalPath(i,j,path,OnePath,LastGlobalPath):
    # 递归终止条件：回到原点（0，0）
    if i == 0 and j == 0:
        OnePath.append((i, j))
        # 将OnePath进行深拷贝再加入至最终路径列表LastGlobalPath中
        LastGlobalPath.append(copy.deepcopy(OnePath))
        # 将该点出栈
        OnePath.pop()
    else:
        for k in range(3):
            # 判断每个点来的方向
            if path[i,j,k] == 1:
                # 下标0处记录从左来的方向
                if k == 0:
                    # 将该点入栈
                    OnePath.append((i,j))
                    # 进行递归记录下一个点
                    FindGlobalPath(i,j - 1,path,OnePath,LastGlobalPath)
                    # 递归返回后将该点出栈，记录另一方向
                    OnePath.pop()
                # 下标1处记录从上来的方向
                elif k == 1:
                    OnePath.append((i, j))
                    FindGlobalPath(i - 1, j, path,OnePath,LastGlobalPath)
                    OnePath.pop()
                # 下标2处记录从左上来的方向
                else:
                    OnePath.append((i, j))
                    FindGlobalPath(i - 1, j - 1, path,OnePath,LastGlobalPath)
                    OnePath.pop()


# 输出比对后的序列
def ShowContrastResult(LastPath,senquence1,senquence2):
    # 依次输出每条路径
    for k, aPath in enumerate(LastPath):
        rowS = ''
        colS = ''
        # 每条路径倒序遍历
        for i in range(len(aPath) -1,0,-1):
            # 方向从左边来
            if aPath[i][0] == aPath[i - 1][0]:
                rowS += senquence1[aPath[i - 1][1] - 1]
                colS += '-'
            # 方向从上面来
            elif aPath[i][1] == aPath[i - 1][1]:
                colS += senquence2[aPath[i - 1][0] -1]
                rowS += '-'
            # 方向从左上来
            else:
                rowS += senquence1[aPath[i - 1][1] - 1]
                colS += senquence2[aPath[i - 1][0] - 1]
        # 依次输出每条路的序列比对结果
        print("======比对结果",k+1,"======")
        print("序列1:",rowS)
        print("序列2:",colS)


# 判断是否为最终路径中的点
def judgePath(point, LastPath):
    for aPath in LastPath:
        if point in aPath:
            return True
    return False


# 画出路径图
def ShowPaths(senquence1, senquence2, LastPath):
    s1 = "0" + senquence1
    s2 = "0" + senquence2
    # 列索引
    col = list(s1)
    # 行索引
    row = list(s2)
    # 设置画布大小
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[], )
    the_table = plt.table(cellText=s, rowLabels=row, colLabels=col, rowLoc='right',
                          loc='center', cellLoc='bottom right', bbox=[0, 0, 1, 1])
    # 设置表格文本字体大小
    the_table.set_fontsize(8)
    # 画出每个点的路径图
    for i in range(m):
        for j in range(n):
            for k in range(3):
                if path[i, j, k] == 1:  # 画出记录的方向
                    # 下标0处记录从左来的方向
                    if k == 0:
                        if judgePath((i, j), LastPath):  # 若某点在在最终路径中
                            # 画出红色箭头
                            plt.annotate('', xy=(j / n, (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         xytext=((2 * j + 1) / (2 * n), (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         arrowprops=dict(arrowstyle="->", color='r', connectionstyle="arc3"))
                        else:
                            # 未在最终路径中则画出黑色箭头
                            plt.annotate('', xy=(j / n, (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         xytext=((2 * j + 1) / (2 * n), (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
                    # 下标1处记录从上来的方向
                    elif k == 1:
                        if judgePath((i, j), LastPath):
                            plt.annotate('', xy=((2 * j + 1) / (2 * n), (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         xytext=((2 * j + 1) / (2 * n), (m - i) / (m + 1)),
                                         arrowprops=dict(arrowstyle="<-", color='r', connectionstyle="arc3"))
                        else:
                            plt.annotate('', xy=((2 * j + 1) / (2 * n), (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         xytext=((2 * j + 1) / (2 * n), (m - i) / (m + 1)),
                                         arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
                    # 下标1处记录从上来的方向
                    elif k == 2:
                        if judgePath((i, j), LastPath):
                            plt.annotate('', xy=((2 * j + 1) / (2 * n), (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         xytext=(j / n, (m - i) / (m + 1)),
                                         arrowprops=dict(arrowstyle="<-", color='r', connectionstyle="arc3"))
                        else:
                            plt.annotate('', xy=((2 * j + 1) / (2 * n), (2 * m - 2 * i - 1) / (2 * (m + 1))),
                                         xytext=(j / n, (m - i) / (m + 1)),
                                         arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
    plt.show()


# 定义需要比对的序列
senquence1 = input("请输入序列1：").upper()
senquence2 = input("请输入序列2：").upper()
# 定义打分规则
ma = int(input("请输入match："))
mi = int(input("请输入mismatch："))
gap = int(input("请输入insertion/deletion：："))
# 将碱基转换为集合下标
replace = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
# 构造替换计分矩阵
w = [[ma, mi, mi, mi], [mi, ma, mi, mi], [mi, mi, ma, mi], [mi, mi, mi, ma]]
# 获取序列的长度
m = len(senquence2) + 1
n = len(senquence1) + 1
# 构建m*n全0矩阵
s = zeros((m, n))
# 记录每个点的方向，下标0处存储从左来的方向，下标1处存储从上来的方向，下标2处存储从左上来的方向
# 初始值均为0，若存在从某方向上来则将其对应下标的值置为1
path = zeros((m, n, 3))
# 记录每条路径
OnePath = []
# 记录所有全局序列比对路径
LastGlobalPath = []
# 构建得分矩阵
GlobalScoreMatrix(m,n,w,replace,s,path,senquence1,senquence2,gap)
FindGlobalPath(m-1,n-1,path,OnePath,LastGlobalPath)
ShowContrastResult(LastGlobalPath,senquence1,senquence2)
ShowPaths(senquence1, senquence2, LastGlobalPath)
