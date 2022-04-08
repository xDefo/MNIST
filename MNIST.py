import load_database
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import math
import cv2

trainx,trainy,testx,testy=load_database.load_mnist()

def baricentro(img,dx,dy):
    xg=0 
    yg=0
    peso=0
    tmp=0
    for y in range(dy):
        for x in range(dx):
            tmp=img[x,y]
            peso+=tmp
            xg+=tmp*x
            yg+=tmp*y
    return xg/peso,yg/peso

def normalizza(img,x,y,par):
    for i in range(x):
        for j in range(y):
            if(img[i,j]!=0):
                 img[i,j]=par
    return img

def genRette(xg,yg,ndiv):
    alpha=0
    m=[]
    for i in range(int(ndiv)):
       
        if (alpha!=(math.pi/2)) and (alpha!=(3/2)*math.pi) :
            m.append(math.tan(alpha))
        else :
            m.append(float('inf'))
        alpha+=((2*math.pi)/ndiv)
    return m

def findIntersection(img,m,dx,dy,xg,yg): #riscrivere in quanto il riferimento è sbagliato, non puoi usare x e y dei for come coordinate
    idx=[]
    for i in range(dx):
        for j in range(dy):
            print(img[i,j],end='')
        print("\n",end='')
    """
    for x in range(dx):
        for y in range(dy):
            if m!=0:
                #controllo sotto
                tmp=(y-0.5-yg)/m+xg
                if (tmp>(x-0.5))and(tmp<(x+0.5)):
                    idx.append((x,y))
                elif m>0:
                    #contralla destra
                    tmp=m*(x-0.5-xg)+yg
                    if (tmp>(y-0.5))and(tmp<(y+0.5)):
                        idx.append((x,y))
                elif m<0:
                    tmp=m*(x+0.5-xg)+yg
                    if (tmp>(y-0.5))and(tmp<(y+0.5)):
                        idx.append((x,y))
            else:
                tmp=m*(x-0.5-xg)+yg
                if (tmp>(y-0.5))and(tmp<(y+0.5)):
                    idx.append((x,y))
    """
    for x in range(dx):
        for y in range(dy):
           # print(x,y)
            if m!=0:
                #controllo sotto
                tmp=((27-y)-0.5-yg)/m+xg
                if (tmp>(x-0.5))and(tmp<(x+0.5))and(img[x,y]):
                    print(img[x,y],(x,y))
                    idx.append((x,y))
                elif m>0:
                    #contralla destra
                    tmp=m*(x-0.5-xg)+yg
                    
                    if (tmp>((27-y)-0.5))and(tmp<((27-y)+0.5))and(img[x,y]):
                        #print(img[x,y],(x,y))
                        idx.append((x,y))
                elif m<0:
                    tmp=m*(x+0.5-xg)+yg
                   
                    if (tmp>((27-y)-0.5))and(tmp<((27-y)+0.5))and(img[x,y]):
                        #print(img[x,y],(x,y))
                        idx.append((x,y))
            else:
                tmp=m*(x-0.5-xg)+yg
                
                if (tmp>((27-y)-0.5))and(tmp<((27-y)+0.5))and(img[x,y]):
                    #print(img[x,y],(x,y))
                    idx.append((x,y))         
    return idx

def scacchiera(img):
    
    idx=0
    c=1
    for i in range(28):
        for j in range(28):
            img[i][j]=c
            if(j!=27):
                c=(not (c))
    #print(img)         
    return img
"""
for i in range(9):  
   
   plt.imshow(trainx[i], cmap=plt.get_cmap('gray'))
   #print(trainy[i])

plt.subplot(330+1)
plt.imshow(trainx[0],cmap=plt.get_cmap('gray'))
prova=filtra(trainx[0])
plt.subplot(330+2)
plt.imshow(prova,cmap=plt.get_cmap('gray'))

xg,yg=baricentro(trainx[0])

prova[14][13]=198
plt.subplot(330+3)
plt.imshow(prova,cmap=plt.get_cmap('gray'))
"""

"""
x=np.where(trainy==1)

x=x[0]
idx=0
for i in x:
    plt.subplot(330+1+idx)
    plt.imshow(trainx[i],cmap=plt.get_cmap('gray'))
    idx+=1
"""
"""
trainx=trainx.reshape(60000,784)
test=  testx.reshape(10000,784)

trainx=trainx/255
testx=testx/255


"""

x=np.arange(-0.5,28,0.5)

"""
for i in range(1):
    fig,ax=plt.subplots()


    #trainx[i]=cv2.rotate(trainx[i],cv2.ROTATE_180)
    trainx[i]=cv2.flip(trainx[i],0)
  
    ciao=normalizza(trainx[i],28,28,1)
    ciao=skeletonize(ciao,method="lee")
    #print(ciao[4,6])
    ax.imshow(ciao,cmap=plt.get_cmap('gray'))
    xg,yg=baricentro(ciao,28,28)
    m=genRette(round(xg), round(yg), 5)
    
   
    for j in m:
        if j!=float('inf'):
            idx=np.array(findIntersection(ciao,j,28,28,round(xg), round(yg)))
            ax.plot(x,j*(x-round(xg))+round(yg),label="{coef}".format(coef=j))
            ax.scatter(idx[:,1],idx[:,0],label="{coef}".format(coef=j))
        else:
            print("ciao")
            ax.plot(np.ones(30)*xg,np.arange(0,30,1))
        #prova=scacchiera(trainx[0])
        #ax.imshow(prova,cmap=plt.get_cmap('gray'),origin='lower')
        ax.legend(loc='best')
    #fig.savefig("img{indice}.jpg".format(indice=i))

"""
#prova=scacchiera(trainx[0])
#plt.imshow(prova,cmap=plt.get_cmap('gray'),origin='lower')



ciao=normalizza(trainx[0],28,28,1)
ciao=skeletonize(ciao,method="lee")

plt.imshow(ciao,cmap=plt.get_cmap('gray'))
xg,yg=baricentro(ciao,28,28)
plt.scatter(xg,yg)
m=genRette(round(xg), round(yg), 5)
idx=np.array(findIntersection(ciao,m[1],28,28,round(xg), round(yg)))
plt.plot(x,-m[1]*(x-round(xg))+round(yg),label="{coef}".format(coef=m[1]))
plt.scatter(idx[:,0],idx[:,1],label="{coef}".format(coef=m[1]))
"""

plt.xlim(-1, 29)
plt.ylim(-1,29)
plt.show()        
        """
