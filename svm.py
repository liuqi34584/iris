from sklearn import datasets    # 数据集的库
from sklearn import model_selection    
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib



iris = datasets.load_iris()     #从datasets导入鸢尾花数据集

x = iris.data
y = iris.target 

# 将原始数据集划分成训练集和测试集
x1=x[:,0:4] #在 X中取前两列作为特征（为了后期的可视化画图更加直观，故只取前两列特征值向量进行训练
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)
# 搭建模型，训练SVM分类器
classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
#开始训练
classifier.fit(x_train,y_train.ravel())

result = classifier.predict(x[:])

class0 = x[y==0, :]#把y=0,即Iris-setosa的样本取出来，山鸢尾
class1 = x[y==1, :]#把y=1，即Iris-versicolo的样本取出来，变色鸢尾
class2 = x[y==2, :]#把y=2，即Iris-virginica的样本取出来，维吉尼亚鸢尾

p_class0 = x[result==0, :]#把result=0,即预测为Iris-setosa的样本取出来，山鸢尾
p_class1 = x[result==1, :]#把result=1，即预测为Iris-versicolo的样本取出来，变色鸢尾
p_class2 = x[result==2, :]#把result=2，即预测为Iris-virginica的样本取出来，维吉尼亚鸢尾

# 为了显示中文，加载中文字体
zhfont = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf") 
#散点图可视化
#第四个维度的数据用点的大小表示
#定义坐标轴
fig = plt.figure(figsize=(20, 10))
ax = fig.gca(projection='3d')

# x轴:花瓣宽度x[:,3] y轴:花萼宽度x[:,1] z轴:花萼长度x[:,0] 点直径：花瓣长度x[:,2]

# 展示原数据
ax.scatter3D(class0[:,3],class0[:,2],class0[:,0],s=class0[:,2]*30,color='r',label='Iris-setosa') 
ax.scatter3D(class1[:,3],class1[:,2],class1[:,0],s=class1[:,1]*30,color='green',label='Iris-versicolo') 
ax.scatter3D(class2[:,3],class2[:,2],class2[:,0],s=class2[:,1]*30,color='black',label='Iris-virginica') 

# 将预测数据与原数据重合
ax.scatter3D(p_class0[:,3],p_class0[:,2],p_class0[:,0],s=p_class0[:,2]*5,color='cyan',label='Iris-setosa-predict') 
ax.scatter3D(p_class1[:,3],p_class1[:,2],p_class1[:,0],s=p_class1[:,2]*5,color='violet',label='Iris-versicolo-predict') 
ax.scatter3D(p_class2[:,3],p_class2[:,2],p_class2[:,0],s=p_class2[:,2]*5,color= 'pink',label='Iris-virginica-predict') 

# 设置坐标系名称
ax.set_xlabel('花瓣宽度', fontproperties=zhfont)
ax.set_ylabel('花萼宽度', fontproperties=zhfont)
ax.set_zlabel('花萼长度', fontproperties=zhfont)

# 设置坐标系范围
ax.set_xlim(0, 2.5) # 花瓣宽度
ax.set_ylim(1.0, 7.0) # 花萼宽度
ax.set_zlim(4.0, 8.0) # 花萼长度

# 设置坐标系刻度
ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]) # 花瓣宽度
ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) # 花萼宽度
ax.set_zticks([4.0, 5.0, 6.0, 7.0]) # 花萼长度

# 设置图例位置
plt.legend(loc='best', fontsize=10)

plt.show()

print('SVM-输出训练集的准确率为： %.2f' % classifier.score(x_train, y_train))
print('SVM-输出测试集的准确率为： %.2f' % classifier.score(x_test, y_test))