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

class SVD():

    def __init__(self):
        self.images=[]

    def readdata(self,show=False):
        path = './images'
        file='/butterfly.bmp'
        im=Image.open(path+file)
        self.images=list(im.getdata())
        self.size=im.size
        # print(self.images)
        self.R,self.G,self.B=map(np.array,list(zip(*self.images)))
        if show==True:self.showdata(self.R,self.G,self.B,'./images/figshow2.png')

    def showdata(self,R,G,B,path,flag=0):
        # print('R G B')
        R,G,B=map(self.regular,[R,G,B])
        # print(R);print(G);print(B)
        if flag==0:
            fig=plt.figure(figsize=(7,5))
            ax=fig.add_subplot(2,2,1)
            plt.imshow(self.shape2(list(zip(R,G,B))))
            plt.title('Original',y=-0.4)
            ax=fig.add_subplot(2,2,2)
            plt.imshow(self.shape(R), cmap='Reds')
            plt.title('R',y=-0.4)
            ax=fig.add_subplot(2,2,3)
            plt.imshow(self.shape(G), cmap='Greens')
            plt.title('G', y=-0.4)
            ax=fig.add_subplot(2,2,4)
            plt.imshow(self.shape(B), cmap='Blues')
            plt.title('B', y=-0.4)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
            # plt.show()
            plt.savefig(path,dpi=2000, bbox_inches='tight')
        else:
            fig=plt.figure(figsize=(7,5))
            ax=fig.add_subplot(2,2,1)
            plt.imshow([[(int(R[j][i]),int(G[j][i]),int(B[j][i])) for j in range(len(R))] for i in range(len(R[0]))])
            plt.title('RGB',y=-0.4)
            ax=fig.add_subplot(2,2,2)
            plt.imshow(R.transpose(), cmap='Reds')
            plt.title('R',y=-0.4)
            ax=fig.add_subplot(2,2,3)
            plt.imshow(G.transpose(), cmap='Greens')
            plt.title('G', y=-0.4)
            ax=fig.add_subplot(2,2,4)
            plt.imshow(B.transpose(), cmap='Blues')
            plt.title('B', y=-0.4)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
            # plt.show()
            plt.savefig(path,dpi=2000, bbox_inches='tight')

    def regular(self,RGB):
        if type(RGB[0])!=np.ndarray and type(RGB[0])!=list:
            m=min(RGB);M=max(RGB)
            return [int((rgb-m)/(M-m)*255) for rgb in RGB]
        else:
            m,M=1e100,-1e100
            for i in RGB:
                for j in i:m=min(m,j);M=max(M,j)
            for i in range(len(RGB)):
                for j in range(len(RGB[0])):RGB[i][j]=int((RGB[i][j]-m)/(M-m)*255)
            return RGB

    def svd(self):
        self.R, self.G, self.B = map(np.array, list(zip(*self.images)))
        D=self.size[1];N=self.size[0]
        print(f'sample number:{N},feature number:{D}')
        R, G, B = map(self.svd_1, map(self.shape2, [self.R, self.G, self.B]))
        self.UR, self.sigmaR, self.VR = R
        self.UG, self.sigmaG, self.VG = G
        self.UB, self.sigmaB, self.VB = B

    def svd_1(self,X):
        X=np.array(X)
        D = self.size[1];N = self.size[0]
        # print('cov',len(cov))
        # if len(cov)!=437:raise Exception('error')
        U, sigma_, V = linalg.svd(X)

        print(f'eigvec number:{len(V)},eigvec dim:{len(V[0])}')
        n=max(len(U),len(V))
        sigma=[[sigma_[i] if i==j and i<len(sigma_) else 0 for i in range(n)] for j in range(n)]
        return U,sigma,V

    def recompose(self,eigvec,cor):
        k=len(cor)
        return np.dot(eigvec[:k].transpose(),cor)

    def featureface(self):
        fig=plt.figure(figsize=(8,7))
        Us = [self.UR, self.UG, self.UB]
        sigmas = [self.sigmaR, self.sigmaG, self.sigmaB]
        Vs = [self.VR, self.VG, self.VB]
        for k in range(1,49+1):
            for i in range(3):
                if i == 0: R = np.dot(np.array([j*Us[i][:, k] for j in sigmas[i][k]]).transpose(), Vs[i][:])
                elif i == 1:G = np.dot(np.array([j*Us[i][:, k] for j in sigmas[i][k]]).transpose(), Vs[i][:])
                else:B = np.dot(np.array([j*Us[i][:, k] for j in sigmas[i][k]]).transpose(), Vs[i][:])
                ax=fig.add_subplot(7,7,k)
            R, G, B = map(self.regular, [R, G, B])
            plt.imshow([[(int(R[i][j]),int(G[i][j]),int(B[i][j])) for j in range(len(R[0]))] for i in range(len(R))])
            plt.title('{}'.format(k), y=-0.45)
            plt.xticks([]);plt.yticks([])
        print('over')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.3)
        plt.savefig('./images/featureface' + '.png',dpi=2000, bbox_inches='tight')

    # 将向量变为矩阵
    def shape(self,image):
        return image.reshape(self.size[1],self.size[0])

    # 将元素为元组的向量转换为矩阵
    def shape2(self,image):
        imnew = []
        line = []
        s=self.size
        for i in range(len(image) + 1):
            if i % s[0] == 0 and i != 0:
                imnew.append(line)
                line = []
            if i != len(image): line.append(image[i])
        return imnew


    def loss(self,A,B):
        l=lambda x,y:sum([(x[i]-y[i])**2 for i in range(len(x))])
        abl=0
        normb=0
        for i in range(len(A)):
            for j in range(len(A[0])):
                abl+=l(A[i][j],B[i][j])
                normb+=sum([s**2 for s in B[i][j]])
        if normb==0:raise Exception('b is zero')
        return abl,abl/normb

    def lossplot(self):
        X=self.images
        absloss=[]
        relaloss=[]
        X=self.shape2(X)
        # n=len(self.VB)
        n=len(self.UR)
        Us= [self.UR, self.UG, self.UB]
        sigmas = [self.sigmaR, self.sigmaG, self.sigmaB]
        Vs = [self.VR, self.VG, self.VB]
        for k in range(1,n):
            print(k)
            for i in range(3):
                if i==0:R=np.dot(np.dot(Us[i][:,:k],sigmas[i][:k]),Vs[i][:])
                elif i==1:G=np.dot(np.dot(Us[i][:,:k],sigmas[i][:k]),Vs[i][:])
                else:B=np.dot(np.dot(Us[i][:,:k],sigmas[i][:k]),Vs[i][:])
            X_ = [[(R[i][j],G[i][j],B[i][j]) for j in range(len(R[0]))] for i in range(len(R))]
            abl,rel=self.loss(X_,X)
            absloss.append([abl,k]);relaloss.append([rel,k])
        absloss = pd.DataFrame(absloss, columns=['Absolute error ', 'k'])
        relaloss = pd.DataFrame(relaloss, columns=['Relative error ', 'k'])
        fig=plt.figure(figsize=(8,3.5))
        ax=fig.add_subplot(1,2,1)
        sns.lineplot(data=absloss,y='Absolute error ',x='k')
        ax=fig.add_subplot(1,2,2)
        sns.lineplot(data=relaloss,y='Relative error ',x='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
        plt.savefig('./images/loss' + '.png', dpi=2000, bbox_inches='tight')
        # plt.show()

    # 考察不同数量主成分的效果
    def test(self):
        self.readdata()
        ks=[1,2,3,4,5,7,10,20,30,50,100,150,200,250,300,350,437]
        # ks=[243]
        self.svd()
        # # raise Exception('stop')
        Us= [self.UR, self.UG, self.UB]
        sigmas = [self.sigmaR, self.sigmaG, self.sigmaB]
        Vs = [self.VR, self.VG, self.VB]
        for k in ks:
            print(k)
            for i in range(3):
                if i==0:R=np.dot(np.dot(Us[i][:,:k],sigmas[i][:k]),Vs[i][:])
                elif i==1:G=np.dot(np.dot(Us[i][:,:k],sigmas[i][:k]),Vs[i][:])
                else:B=np.dot(np.dot(Us[i][:,:k],sigmas[i][:k]),Vs[i][:])
            print(len(R),len(R[0]))
            self.showdata(R.transpose(),G.transpose(),B.transpose(),'./images/axis1_'+str(k)+'.png',flag=1)

        # self.lossplot()
        # self.featureface()
        # for i in [1,50,100,200,380]:
        #     self.recompose(i)
    # 特征
    def test2(self):
        self.readdata()
        self.svd()
        self.featureface()

    # loss
    def test3(self):
        self.readdata()
        self.svd()
        self.lossplot()

def main():
    test=SVD()
    # test.test()
    # test.test2()
    test.test3()

if __name__=='__main__':
    main()


