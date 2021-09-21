from numpy import *

# 状态的样本空间,0表示ZZ、1表示MM、2表示MZ
states = ['0','1','2']
# 初始概率,取对数是为了将后面乘算变为加算，从而简化计算
S = [log(0.4), log(0.4), log(0.2)]
# 状态转移概率
T = [
  [log(0.99*0.99), log(0.01*0.01), log(2*0.99*0.01)],
  [log(0.01*0.01), log(0.99*0.99), log(2*0.01*0.99)],
  [log(0.01*0.99), log(0.01*0.99), log(0.01*0.01+0.99*0.99)]
]
# 输出概率
E = [
  [log(0.97), log(0.03)],
  [log(0.03), log(0.97)],
  [log(0.25), log(0.25)]
]
#输入观测值
print("请输入观测值：")
obs = input()
# 将输入的字符串转换为整数
replace = {'0': 0, '1': 1}
# 获取序列的长度
n = len(obs) 
# 构建3*m全0矩阵
s = zeros((3, n))
# 记录每个点的方向，下标0处存储从第一行ZZ来的方向，下标1处存储从第二行MM来的方向，下标2处存储从第三行MZ来的方向
# 初始值均为0，若存在从某方向上来则将其对应下标的值置为1
path = zeros((3, n, 3))
# 记录最终打分路径
LastPath = []

#创建得分矩阵
for j in range(n):
  for i in range(3):
    # 判断第一列的特殊情况
    if j == 0:   
      s[i][j] = S[i] + E[i][replace[obs[0]]]
    else:
      # 获取最大值
      s[i][j] = max(s[0][j-1] + T[0][i] + E[i][replace[obs[j]]],
      s[1][j-1] + T[1][i] + E[i][replace[obs[j]]],
      s[2][j-1] + T[2][i] + E[i][replace[obs[j]]])   
      # 记录最大值来的方向0表示从第一行ZZ来，1表示从第二行MM来，2表示从第三行MZ来
      if s[0][j-1] + T[0][i] + E[i][replace[obs[j]]] == s[i][j]:
          path[i,j,0] = 1
      if s[1][j-1] + T[1][i] + E[i][replace[obs[j]]] == s[i][j]:
          path[i,j,1] = 1
      if s[2][j-1] + T[2][i] + E[i][replace[obs[j]]] == s[i][j]:
          path[i,j,2] = 1

# 寻找最佳路径
for j in range(n-1, 0, -1): 
  if j == n-1:
    for i in range(3):
      if s[i][j] == max(s[0][j], s[1][j], s[2][j]):
        x = i
        LastPath.append(x)
    for k in range(3):
      if path[x,j,k] == 1:
        x = k
        LastPath.append(x)
  else:     
    for k in range(3):
      if path[x,j,k] == 1:
        x = k
        LastPath.append(x)
LastPath.reverse()
print("真实序列结果:")
for i in range(n):
  print(LastPath[i],end='')
print()
#print(s)

        


