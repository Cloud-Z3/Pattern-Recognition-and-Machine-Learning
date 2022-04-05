import numpy as np
from numpy import linalg
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import *
sns.set_style("dark")

# plt.style.use('fivethirtyeight')
# plt.rcParams['text.usetex'] = True#默认为false，此处设置为TRUE
plt.rc('font',family='Times New Roman')

# Try numpy.

class PCA():

    def __init__(self):
        self.images=[]

    def readdata(self):
        path = './ORL'
        self.size = set()
        for file in os.listdir(path):
            for item in os.listdir(path + '/' + file):
                im = Image.open(path + '/' + file + '/' + item)
                self.images.append(list(im.getdata()))
                self.size.add(im.size)
            #     if len(self.images)==10:break
            # if len(self.images) == 10: break
        if len(self.size)==1:
            self.size=list(self.size)[0]
            self.images=np.array(self.images,dtype=float)
            print('image size:',self.size)
        else:raise Exception("size not match")
        N, D = len(self.images), len(self.images[0])  # 样本数，特征数
        self.D=D;self.N=N

    def pca(self):
        X=np.array(self.train.copy())
        D=self.D;N=self.N
        print(f'sample number:{N},feature number:{D}')
        X = np.array(X)
        u,sigma,eigvec=linalg.svd(X)
        self.eigval=[i**0.5 for i in sigma]
        self.eigvec=eigvec
        print(f'eigvec number:{len(eigvec)},eigvec dim:{len(eigvec[0])}' )
        Zall=np.dot(eigvec,X.transpose())
        self.Zall=Zall

    def embedding(self,k):
        self.Z=self.Zall[:k]
        self.U=self.eigvec[:k].transpose()

    def recompose(self,idx):
        X=np.array(self.images).transpose()
        fig=plt.figure(figsize=(8,7))
        ax=fig.add_subplot(5,6,1)
        ori=self.shape(X[:,idx])
        ax.imshow(ori,cmap = plt.cm.gray)
        plt.title('Original', y=-0.25)
        plt.xticks([]);plt.yticks([])
        # ax.set_title(r'Original')
        i=1
        for k in [1,2,3,4,5,7,10,15,20,50,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10304]:
            print(k)
            i+=1
            ax = fig.add_subplot(5, 6, i)
            self.embedding(k)
            Z = self.Z;U = self.U
            X_ = np.dot(U,Z)
            img = self.shape(X_[:, idx])
            ax.imshow(img,cmap = plt.cm.gray)
            # ax.set_title(r'n={}'.format(k))
            plt.title(r'k={}'.format(k),y=-0.25)
            plt.xticks([]);plt.yticks([])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.3)
        # plt.tight_layout()
        # plt.show()
        plt.savefig('./images/'+str(idx)+'.png',dpi=2000)

        # self.show(img)

    def featureface(self):
        fig=plt.figure(figsize=(8,7))
        features=self.eigvec
        for i in range(1,49+1):
            ax=fig.add_subplot(7,7,i)
            ff=self.shape(features[i])
            ax.imshow(ff, cmap=plt.cm.gray)
            plt.title('{}'.format(i), y=-0.4)
            plt.xticks([]);plt.yticks([])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.5)
        plt.savefig('./images/featureface' + '.png', dpi=2000)

    def shape(self,image):
        return image.reshape(self.size[1],self.size[0])

    def show(self,image):
        Image._show(image)

    def loss(self,A,B):
        absloss=sum([(A[i]-B[i])**2 for i in range(len(A))])
        s=sum([Bi**2 for Bi in B])
        relaloss=absloss/s
        return absloss,relaloss

    def lossplot(self):
        X=self.images
        absloss=[]
        relaloss=[]
        n=10304
        self.embedding(n)
        Zall = self.Zall
        eigvec = self.eigvec.transpose()
        for k in range(1,n):
            print(k)
            X_ = np.dot(eigvec[:,:k], Zall[:k,0:1])
            m=1
            al=0;rl=0
            for i in range(1):
                dal,drl=self.loss(X[i],[i[0] for i in X_[:]])
                al+=dal;rl+=drl
            absloss.append([al/m,k]);relaloss.append([rl/m,k])
        absloss = pd.DataFrame(absloss, columns=['Absolute error ', 'k'])
        relaloss = pd.DataFrame(relaloss, columns=['Relative error ', 'k'])
        fig=plt.figure(figsize=(8,3.5))
        ax=fig.add_subplot(1,2,1)
        sns.lineplot(data=absloss,y='Absolute error ',x='k')
        ax=fig.add_subplot(1,2,2)
        sns.lineplot(data=relaloss,y='Relative error ',x='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
        plt.savefig('./images/loss' + '.png', dpi=2000)

    def findmostsimilar(self,train,test,t):
        n=len(train)
        print('len train:',n)
        dis=[]
        for i in range(n):
            traini=train[i]
            dis.append([sum([(traini[j]-test[j])**2 for j in range(400)]),i])
        dis.sort()
        dis=dis[:3]
        idx=[d[1] for d in dis]
        fig=plt.figure(figsize=(8,4))
        ax=fig.add_subplot(2,4,1)
        ax.imshow(self.shape(self.images[self.testsid[t]]), cmap=plt.cm.gray)
        plt.title('original', y=-0.25)
        plt.xticks([]);plt.yticks([])
        sim=['/most similar','/second similar','/third similar']
        for i in range(1,4):
            ax = fig.add_subplot(2, 4, i+1)
            ax.imshow(self.shape(self.images[self.trainid[idx[i-1]]]), cmap=plt.cm.gray)
            plt.title('original'+sim[i-1], y=-0.25)
            plt.xticks([]);plt.yticks([])
        ax=fig.add_subplot(2,4,5)
        ax.imshow(self.shape(self.recompose2(test)), cmap=plt.cm.gray)
        plt.title('k=400', y=-0.25)
        plt.xticks([]);plt.yticks([])
        for i in range(5,8):
            ax = fig.add_subplot(2, 4, i+1)
            ax.imshow(self.shape(self.recompose2(train[idx[i-5]])), cmap=plt.cm.gray)
            plt.title('k=400'+sim[i-5], y=-0.25)
            plt.xticks([]);plt.yticks([])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
        plt.savefig('./images/similar'+str(t) + '.png', dpi=2000)

    def recompose2(self,cor):
        k=len(cor)
        eigvec=self.eigvec
        return np.dot(eigvec[:k].transpose(),cor)

    def averageface(self):
        n=len(self.images)
        ave=np.array([sum([self.images[j][i] for j in range(n)])/n for i in range(len(self.images[0]))])
        plt.imshow(self.shape(ave), cmap=plt.cm.gray)
        plt.savefig('./images/average')

    # 考察降维效果，展示特征脸
    def test(self):
        self.readdata()
        self.train=self.images
        self.pca()
        # self.lossplot()
        # self.featureface()
        # for i in [1,50,100,200,380]:
        #     self.recompose(i)

    # 利用降维后的数据做人脸识别
    def test2(self):
        self.readdata()
        n=len(self.images)
        items=[i for i in range(n)]
        shuffle(items)
        trainnum=n//10*9
        print(f'train number:{trainnum},test number:{n-trainnum}')
        train=items[:trainnum]
        tests=items[trainnum:]
        self.train=self.images
        self.trainid=train;self.testsid=tests
        testsitem=np.array([self.images[i] for i in tests])
        # self.images=[self.images[i] for i in train]
        self.pca()
        eigvec=self.eigvec
        Ztrain=np.dot(eigvec[:400],np.array([self.images[i] for i in train]).transpose()).transpose()
        print(len(Ztrain),len(Ztrain[0]))
        Ztest=np.dot(eigvec[:400],testsitem.transpose()).transpose()
        print(len(Ztest),len(Ztest[0]))
        for i in range(len(Ztest)):
            print(i)
            self.findmostsimilar(Ztrain,Ztest[i],i)

    # 平均脸
    def test3(self):
        self.readdata()
        self.averageface()

def main():
    test=PCA()
    # test.test()
    # test.test2()
    test.test3()

if __name__=='__main__':
    main()


