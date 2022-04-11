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
    return round(xg/peso),round(yg/peso)

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
    distanza=float("inf")
    conteggio=0
    for i in range(dy):
        for x in range(dx):
            y=dy-i
            if img[i,x]:
                if m!=0:
                    #check sotto
                    xr=((y-0.5)-yg)/m+xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                    if (xr<(x+0.5)) and (xr>(x-0.5)): # se la x è compresa nel lato allora la retta interseca la casella
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                        conteggio+=1

                    elif m<0:
                        #check destra
                        yr=m*((x+0.5)-xg)+yg
                        if((yr<(y+0.5)))and(yr>(y-0.5)):
                            tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                            if(tmp<distanza): distanza=tmp
                            conteggio+=1

                    elif m>0:
                        #check sinistra
                        yr=m*((x-0.5)-xg)+yg
                        if((yr<(y+0.5)))and(yr>(y-0.5)):
                            tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                            if(tmp<distanza): distanza=tmp
                            conteggio+=1

                else:
                    #m=0 cerco solo a sinistra
                    yr=m*((x-0.5)-xg)+yg
                    if((yr<(y+0.5)))and(yr>(y-0.5)):
                        #xs.append(x)
                    # ys.append(i)
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                        conteggio+=1

    return conteggio,distanza #xs,#ys,conteggio
            
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

def create_file(insieme_train,insieme_test,n,file_name):  #funziona ancora solo con immagini 28x28
    f=open(file_name,"w")
    idx1=0
    x=np.arange(-0.5,28.5,0.1)
    for img in insieme_train:
        img=normalizza(img,28,28,1) #normalizzo immagine
        img=skeletonize(img) #trovo lo skeletro
        xg,yg=baricentro(img,28,28) #calcolo il baricentro
        m=genRette(xg,yg,n) #genero coefficienti angolari rette
        f.write("{n}\n".format(n=insieme_test[idx1]))
        for m in m:
            conta,dist=findIntersection(img,m,28,28,xg,yg)
            f.write("{n} {d}\n".format(n=conta,d=dist))
        idx1+=1
    f.close()



trainx,trainy,testx,testy=load_database.load_mnist()

create_file(trainx,trainy,9,"file_dataset/9rettetrain.txt")
create_file(testx,testy,9,"file_dataset/9rettetest.txt")


   

    

