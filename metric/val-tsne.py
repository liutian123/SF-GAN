import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
 
size = 256 #resize图片的大小，运行时如果爆显存的话把这里调小即可
 
# get_data(Input_path,Label)
# 作用：读取Input_path里的图片，并给每张图打上自定义标签Label
def get_data(Input_path,Label): 
    Image_names=os.listdir(Input_path) #获取目录下所有图片名称列表
    data=np.zeros((len(Image_names),size*size*3)) #初始化一个np.array数组用于存数据,自己图片是n维的就把3改成n即可
    label=np.zeros((len(Image_names),1)) #初始化一个np.array数组用于存数据
 
    #为当前文件下所有图片分配自定义标签Label
    for k in range(len(Image_names)):
        label[k][0]=Label
        
    for i in range(len(Image_names)):
        image_path=os.path.join(Input_path,Image_names[i])
        img=cv2.imread(image_path)
        img=cv2.resize(img,(size,size)) #(size,size,3)
        img = img.flatten() #(3*size*size,)
        data[i]=img
    return data, label
 
#重点来了，这里是根据自己想查看的数据来自定义修改代码,得到自己的x_train和y_train
#x_train是待分析的数据
#y_train是待分析的自定义标签
#比如，我想分析训练集中5个类的分布情况
#先读取每一个类的数据，然后给他们自定义标签1-5
#然后把data拼在一起,label拼在一起，前者叫x_train,后者叫y_train
data1, label1 = get_data('../result-1/input',1) #根据自己的路径合理更改
data2, label2 = get_data('../result-1/ground-truth',2) 
data3, label3 = get_data('../result-1/pix-out',3) 
data4, label4 = get_data('../result-1/tev-out',4) 
#得出数据后把他们拼起来
data = np.vstack((data1,data2,data3,data4))
label = np.vstack((label1,label2,label3,label4))
 
(x_train,y_train) = (data,label)
print(y_train.shape) #(n_samples,1)
print(x_train.shape) #(n_samples,size*size*3)
 
#t-SNE，输出结果是(n_samples,2)
#TSNE的参数和sklearn的T-SNE一样，不懂的自行查看即可
# tsne = TSNE(n_iter=1000, verbose=1,num_neighbors=32,device=0)
tsne = TSNE(n_components=2, init='pca', perplexity=50,learning_rate=1000,n_iter=1000,random_state=42)
tsne_results = tsne.fit_transform(x_train)
 
print(tsne_results.shape) #(n_samples,2)
 
# 画图
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title='TSNE-radar' )
 
# Create the scatter
#ax.scatter()的用法自行百度
scatter  = ax.scatter(
    x=tsne_results[:,0],
    y=tsne_results[:,1],
    c=y_train,
    # cmap=plt.cm.get_cmap('Paired'),
    # alpha=0.4,
    s=10)
 
#ax.legend添加类标签
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
ax.add_artist(legend1)
 
#显示图片
plt.show()
#保存图片
plt.savefig('./tSNE_radar.jpg')
 
 
 