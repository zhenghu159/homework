
import numpy as np
#矩阵运算
import math
#数学公式
import struct
#用到文件解码
from pathlib import Path
#用到路径导入
import matplotlib.pyplot as plt
#用到函数绘图
import copy
#用到deepcopy
from PIL import Image,ImageFilter
#用到图片预处理

'''
#通过截图得到图片需要预处理
def ImagePrepare(Img):
    img=Image.open('./image/'+str(Img)+'.png')
    plt.imshow(img)
    data=list(img.getdata())
    result=[(255-x)*1.0/255.0 for x in data]
    return np.array(result)
'''

#导入数据集
dataset_path = Path('./data')
train_img_path = dataset_path/'train-images.idx3-ubyte'
train_labels_path = dataset_path/'train-labels.idx1-ubyte'
test_img_path = dataset_path/'t10k-images.idx3-ubyte'
test_labels_path = dataset_path/'t10k-labels.idx1-ubyte'

#with as 语句用于读取文件数据
#struct.unpack()读入每个数据集开头的信息 (不需要用到)
#np.fromfile()函数中dtype 数据类型为无符号8位int
#由于不知道读入数据具体维度有多少个28*28的图片
#reshape(-1,28*28)表示自动转化成28*28行的矩阵每个矩阵代表一个图片

train_num=50000
valid_num=10000
test_num=10000

with open(train_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    temp_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255
    train_img=temp_img[:train_num]
    valid_img=temp_img[train_num:]

with open(test_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    test_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255

with open(train_labels_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    temp_label=np.fromfile(f,dtype=np.uint8)
    train_label=temp_label[:train_num]
    valid_label=temp_label[train_num:]

with open(test_labels_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    test_label=np.fromfile(f,dtype=np.uint8)

#img=train_img[0].reshape(28,28)
#plt.imshow(img,cmap='gray')
#plt.show()

#激活函数

def Relu(x):
    return np.where(x>0,x,0.0001)

#激活函数
def tanh(x):
    return np.tanh(x)
#激活函数
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()

#激活函数的调用
activation=[Relu,tanh,softmax]


#预设参数 
dimensions=[28*28,100,10] #输入图片的大小为1*(28*28),经过第一层后变为1*10的矩阵
learn_rate=1
epoch_num=1 #反复训练次数
distribution=[
    {},
    {'b':[0,0],'w':[-math.sqrt( 6/(dimensions[0]+dimensions[1]) ),math.sqrt( 6/(dimensions[0]+dimensions[1]) )]},
    {'b':[0,0],'w':[-math.sqrt( 6/(dimensions[1]+dimensions[2]) ),math.sqrt( 6/(dimensions[1]+dimensions[2]) )]},
]



#激活函数的导数

def d_Relu(data):
    return np.where(x>0,1,0.0001)

def d_softmax(data):
    temp=softmax(data)
    return np.diag(temp)-np.outer(temp,temp)

def d_tanh(data):
    #return np.diag(1/(np.cosh(data))**2)
    return 1/(np.cosh(data)**2)
    #加速运算优化不用对角矩阵

#激活函数的导数的调用
differential={Relu:d_Relu,softmax:d_softmax,tanh:d_tanh}
diff_type={Relu:'times',softmax:'dot',tanh:'times'}

#初始化
def init_parameters_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]

def init_parameters_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]

def init_parameters():
    parameter=[]
    for i in range(len(distribution)):
        layer_parameter={}
        for j in distribution[i].keys():
            if j=='b':
                layer_parameter['b']=init_parameters_b(i)
                continue
            if j=='w':
                layer_parameter['w']=init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    #print(parameter)
    return parameter

#parameters=init_parameters()


#处理图片数据并输出预测
def predict(img,parameters):
    layer_in=img
    layer_out=activation[0](layer_in)
    layer_in_list=[layer_in]
    layer_out_list=[layer_out]
    for layer in range(1,len(dimensions)):
        layer_in=np.dot(layer_out_list[layer-1],parameters[layer]['w'])+parameters[layer]['b']
        layer_out=activation[layer](layer_in)
        layer_in_list.append(layer_in)
        layer_out_list.append(layer_out)
    
    #return {'layer0_in':layer0_in,'layer0_out':layer0_out,'layer1_in':layer1_in,'layer1_out':layer1_out}
    return {'layer_out':layer_out_list,'layer_in':layer_in_list}

#获得预测答案
def get_answer(img,parameters):
    return predict(img,parameters)['layer_out'][-1].argmax()


#onehot生成正确结果矩阵y
onehot=np.identity(dimensions[-1])

#定义损失函数 loss founction=(y-ypred)^2 其中y和ypred都是1*10的矩阵向量 会导致局部最优解 
def sqr_loss(img,label,parameters):
    y_pred=predict(img,parameters)['layer_out'][-1]
    y=onehot[label]
    diff=y-y_pred
    return np.dot(diff,diff)



#定义计算损失函数对参数b,w求导得到每一层的dJ/dw and dJ/db
#grad_parameters(train_img[2],train_label[2],init_parameters())
def grad_parameters(img,label,parameters):

    InOut=predict(img,parameters)

    #diff_layer 这里是“da[2]”=dJ/da[2]
    diff_layer= -2*(onehot[label]-InOut['layer_out'][-1])
    
    grad_result=[None]*len(dimensions)

    #注意每一层的w和dJ/dw,b和dJ/db的维度都是相同的
    
    for layer in range(len(dimensions)-1,0,-1):
        if diff_type[activation[layer]]=='times':
            diff_layer=differential[activation[layer]](InOut['layer_in'][layer])*diff_layer
            #dz[layer]=da[layer]*g[layer]'(z[layer])
            #这里用*本来是应该变成对角矩阵用点乘 优化加速后用同维度矩阵逐个元素相乘

        if diff_type[activation[layer]]=='dot':
            diff_layer=np.dot(differential[activation[layer]](InOut['layer_in'][layer]),diff_layer)
            #dz[layer]=g[layer]'(z[layer]) dot da[layer]
            #这里用点乘是因为dsoftmax求导后产生对角矩阵变为[10,10] dot [10,1]

        grad_result[layer]={}

        grad_result[layer]['b']=diff_layer
        #db[layer]=dz[layer]

        grad_result[layer]['w']=np.outer(InOut['layer_out'][layer-1],diff_layer)
        #dw[layer]=dz[layer] dot a[layer-1].T = a[layer-1] outer dz[layer]
        #(784,1) dot (1,100) = (784,1) outer (100,1) =(784,100) 两种计算含义相同
        
        diff_layer=np.dot(parameters[layer]['w'],diff_layer)
        #da[layer-1]=w[layer].T dot dz[layer]
    
    return grad_result
    

#test

#未经训练输出全为0正常
'''
det=0.00001
for i in range(784):
    for j in range(10):
        img_i=np.random.randint(train_num)
        #img_i取出一个图片的下标
        test_parameters=init_parameters()
        #随机生成参数
        derivative=grad_parameters(train_img[img_i],train_label[img_i],test_parameters)['dw1']
        #生成导数
        value1=sqr_loss(train_img[img_i],train_label[img_i],test_parameters)
        test_parameters[1]['w'][i][j]+=det
        value2=sqr_loss(train_img[img_i],train_label[img_i],test_parameters)
        print(derivative[i][j]-(value2-value1)/det)
'''

#计算训练模型精确度
def valid_loss(parameters):
    loss_accu=0
    for img_i in range(valid_num):
        loss_accu+=sqr_loss(valid_img[img_i],valid_label[img_i],parameters)
    return loss_accu/(valid_num/10000)

def valid_accuracy(parameters):
    correct=[predict(valid_img[img_i],parameters)['layer_out'][-1].argmax()==valid_label[img_i] for img_i in range(valid_num)]
    print('validation accuracy : ',correct.count(True)/len(correct))
    return correct.count(True)/len(correct)

def train_loss(parameters):
    loss_accu=0
    for img_i in range(train_num):
        loss_accu+=sqr_loss(train_img[img_i],train_label[img_i],parameters)
    return loss_accu/(train_num/10000)

def train_accuracy(parameters):
    correct=[predict(train_img[img_i],parameters)['layer_out'][-1].argmax()==train_label[img_i] for img_i in range(train_num)]
    #print('validation accuracy : ',correct.count(True)/len(correct))
    return correct.count(True)/len(correct)



#训练神经网络模型
#batch_size 将所有训练样本划分成多个batch

#grad_add和grad_divide用于train_batch函数
def grad_add(grad1,grad2):
    for layer in range(1,len(grad1)):
        for pname in grad1[layer].keys():
            grad1[layer][pname]+=grad2[layer][pname]
    return grad1

def grad_divide(grad,denominator):
    for layer in range(1,len(grad)):
        for pname in grad[layer].keys():
            grad[layer][pname]/=denominator
    return grad

#训练一个batch大小的train集合
#逐个样本进行训练 
batch_size=100
def train_batch(current_batch,parameters):
    grad_accu=grad_parameters(train_img[current_batch*batch_size+0],train_label[current_batch*batch_size+0],parameters)
    for img_i in range(1,batch_size):
        grad_temp=grad_parameters(train_img[current_batch*batch_size+img_i],train_label[current_batch*batch_size+img_i],parameters)
        grad_add(grad_accu,grad_temp)
        #for key in grad_accu.keys():
        #    grad_accu[key]+=grad_temp[key]
    grad_divide(grad_accu,batch_size)
    #for key in grad_accu.keys():
    #    grad_accu[key]/=batch_size
    return grad_accu


'''
我们寻常意义的复制就是深复制，即将被复制对象完全再复制一遍作为独立的新个体单独存在。
所以改变原有被复制对象不会对已经复制出来的新对象产生影响。
而浅复制并不会产生一个独立的对象单独存在，他只是将原有的数据块打上一个新标签，
所以当其中一个标签被改变的时候，数据块就会发生变化，另一个标签也会随之改变。
这就和我们寻常意义上的复制有所不同了。
'''
#更新parameters
def combine_parameters(parameters,grad,learn_rate):
    #深拷贝是建立完全新的对象将被拷贝对象复制一份
    #因为物理地址不同其中一个被修改另一个不变
    #赋值则是同一个对象同一个物理地址被不同变量名引用
    #修改其中一个另一个也会被改变
    parameter_temp=copy.deepcopy(parameters)
    #不用深拷贝也可以正常运行
    #parameter_temp=parameters
    
    for layer in range(len(parameter_temp)):
        for pname in parameter_temp[layer].keys():
            parameter_temp[layer][pname]-=learn_rate*grad[layer][pname]
    
    return parameter_temp
    

#训练模型更新参数parameters

#缓存损失和正确率用于可视化训练效果
train_loss_list=[]
train_accu_list=[]
valid_loss_list=[]
valid_accu_list=[]

def train():
    for i in range(train_num//batch_size):
        if i%100==99:
            print('running batch {}/{}'.format(i+1,train_num//batch_size))
        global parameters
        #此处需要声明函数内使用的parameters是全局变量
        #因为函数内既有声明局部同名变量又有调用全局变量作为传参
        #声明在调用之后parameters被判断为局部变量后作为传参的parameters将会被视为未定义
        grad_temp=train_batch(i,parameters)
        parameters=combine_parameters(parameters,grad_temp,learn_rate)

def train_epoch():
    for epoch in range(epoch_num):
        print('running epoch {}/{}'.format(epoch+1,epoch_num))
        train();
        #train_loss_list.append(train_loss(parameters))
        #train_accu_list.append(train_accuracy(parameters))
        #valid_loss_list.append(valid_loss(parameters))
        #valid_accu_list.append(valid_accuracy(parameters))

#可视化训练效果
#对比验证集和训练集的正确率和损失
'''
lower=0
plt.plot(valid_loss_list[lower:],color='black',label='ValidLoss')
plt.plot(train_loss_list[lower:],color='red',label='TrainLoss')

plt.plot(valid_accu_list[lower:],color='black',label='ValidAccu')
plt.plot(train_accu_list[lower:],color='red',label='TrainAccu')

plt.show
'''
#观察学习率与正确率的关系
def LearnRate():
    rand_batch=np.random.randint(train_num//batch_size)
    grad_parameter=train_batch(rand_batch,parameters)

    LearnRate_list=[]

    lower=-1
    upper=2
    step=0.1
    Power=[lower+step*i for i in range(int((upper-lower)//step)) ]
    for power in Power:
        learnrate=10**power
        parameters_temp=combine_parameters(parameters,grad_parameter,learnrate)
        train_loss_temp=train_loss(parameters_temp)
        LearnRate_list.append([power,train_loss_temp])

    plt.plot(np.array(LearnRate_list)[:,0],np.array(LearnRate_list)[:,1],color='red',label='LearnRate-Loss')
    plt.show()



#print(softmax(np.array([1,2,3,4])))

#定义图片展示函数
def show_train(index):
    plt.imshow(train_img[index].reshape(28,28),cmap='gray')
    plt.show()
    print('label:',train_label[index])

def show_test(index):
    plt.imshow(test_img[index].reshape(28,28),cmap='gray')
    plt.show()
    print('label:',test_label[index])

def show_valid(index):
    plt.imshow(valid_img[index].reshape(28,28),cmap='gray')
    plt.show()
    print('label:',valid_label[index])

#初始化+计算正确率+训练+计算正确率
parameters = init_parameters()
valid_accuracy(parameters)
train_epoch()
valid_accuracy(parameters)


