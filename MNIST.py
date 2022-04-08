from cProfile import label
from tkinter import Y
import load_database
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import math
import cv2


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

def normalizza(img,dx,dy,par):
    for i in range(dx):
        for j in range(dy):
            if(img[i,j]!=0):
                 img[i,j]=par
    return img

def genRette(xg,yg,nr):
    alpha=0
    step=(2*math.pi)/nr
    m=[]
    for i in range(nr):
        if (alpha!=(math.pi/2)) and (alpha!=(3/2)*math.pi) :
            m.append(math.tan(alpha))
        else :
            m.append(float('inf'))
        alpha+=step
    return m

def findIntersection(img,m,dx,dy,xg,yg): #riscrivere in quanto il riferimento è sbagliato, non puoi usare x e y dei for come coordinate
    #xs=[]
    #ys=[]
    conteggio=0
    for i in range(dy):
        for x in range(dx):
            y=dy-i
            if img[i,x]:
                if m!=0:
                    #check sotto
                    xr=((y-0.5)-yg)/m+xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                    if (xr<(x+0.5)) and (xr>(x-0.5)): # se la x è compresa nel lato allora la retta interseca la casella
                        #xs.append(x)
                        #ys.append(i)
                        conteggio+=1

                    elif m<0:
                        #check destra
                        yr=m*((x+0.5)-xg)+yg
                        if((yr<(y+0.5)))and(yr>(y-0.5)):
                        # xs.append(x)
                        # ys.append(i)
                            conteggio+=1

                    elif m>0:
                        #check sinistra
                        yr=m*((x-0.5)-xg)+yg
                        if((yr<(y+0.5)))and(yr>(y-0.5)):
                        # xs.append(x)
                            #ys.append(i)
                            conteggio+=1

                else:
                    #m=0 cerco solo a sinistra
                    yr=m*((x-0.5)-xg)+yg
                    if((yr<(y+0.5)))and(yr>(y-0.5)):
                        #xs.append(x)
                    # ys.append(i)
                        conteggio+=1

    return conteggio #xs,#ys,conteggio
            
def scacchiera(img):
    
    idx=0
    c=1
    for i in range(28):
        for j in range(28):
            img[i,j]=c
            if(j!=27):
                c=(not (c))
    #print(img)         
    return img

trainx,trainy,testx,testy=load_database.load_mnist()

f=open("test5rette.txt","w")

idx1=0
x=np.arange(-0.5,28.5,0.1)
for img in testx:
    #fig,ax=plt.subplots()

    img=normalizza(img,28,28,1)
    #img=cv2.medianBlur(img,5)
    img=skeletonize(img)
    xg,yg=baricentro(img,28,28)
    xg=round(xg)
    yg=round(yg)
    m=genRette(xg,yg,5)

    #ax.imshow(img)


    f.write("{n}\n".format(n=testy[idx1]))
    for m in m:
        conta=findIntersection(img,m,28,28,xg,yg)
        f.write("{n}\n".format(n=conta))
    idx1+=1
    #print(idx1/6000*100)
        #ax.plot(x,-m*(x-xg)+yg,label='m={:10.2f}'.format(m))
        #ax.scatter(px,py)
    #ax.scatter(xg,yg)
    #plt.legend()
    #plt.ylim(27.5,-0.5)
   # plt.xlim(-0.5,27.5)
    #plt.savefig("img{n}".format(n=i))

