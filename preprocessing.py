from cProfile import label
from gettext import find
from re import I
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
    if ((nr%2)!=0):
        for i in range(nr):
            if (alpha!=(math.pi/2)) and (alpha!=(3/2)*math.pi) :
                m.append(math.tan(alpha))
            else :
                m.append(float('inf'))
            alpha+=step
    else:
        for i in range(int(nr/2)):
            if (alpha!=(math.pi/2)) and (alpha!=(3/2)*math.pi) :
                m.append(math.tan(alpha))
            else :
                m.append(float('inf'))
            alpha+=step
    return m

def findIntersection(img,m,dx,dy,xg,yg): #riscrivere in quanto il riferimento è sbagliato, non puoi usare x e y dei for come coordinate
    xs=[]
    ys=[]
    distanza=float("inf")
    conteggio=0
    for i in range(dy):
        for x in range(dx):
            y=dy-i
            if img[i,x]:
                if ((m!=0) and (m!=float("inf"))):
                    #check sotto
                    xr=((y-0.5)-yg)/m+xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                    if ((xr<=(x+0.5)) and (xr>=(x-0.5))): # se la x è compresa nel lato allora la retta interseca la casella
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                       # xs.append(x)
                       # ys.append(i)
                        conteggio+=1

                    elif m<0:
                        #check destra
                        yr=m*((x+0.5)-xg)+yg
                        if((yr<=(y+0.5))and(yr>=(y-0.5))):
                            tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                            if(tmp<distanza): distanza=tmp
                            #xs.append(x)
                            #ys.append(i)
                            conteggio+=1

                    elif m>0:
                        #check sinistra
                        yr=m*((x-0.5)-xg)+yg
                        if((yr<=(y+0.5))and(yr>=(y-0.5))):
                            tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                            if(tmp<distanza): distanza=tmp
                           # xs.append(x)
                           # ys.append(i)
                            conteggio+=1

                elif (m==0):
                    #m=0 cerco solo a sinistra
                    yr=yg
                    if((yr<=(y+0.5))and(yr>=(y-0.5))):
                       # xs.append(x)
                       # ys.append(i)
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                        conteggio+=1
                elif (m==float("inf")):
                    xr=xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                    if ((xr<=(x+0.5)) and (xr>=(x-0.5))): # se la x è compresa nel lato allora la retta interseca la casella
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                       # xs.append(x)
                       # ys.append(i)
                        conteggio+=1
    return conteggio,distanza #,xs,ys
""" 
def findIntersectionwithpoint(img,m,dx,dy,xg,yg): #riscrivere in quanto il riferimento è sbagliato, non puoi usare x e y dei for come coordinate
    xs=[]
    ys=[]
    distanza=float("inf")
    conteggio=0
    for i in range(dy):
        for x in range(dx):
            y=dy-i
            if ((m!=0) and (m!=float("inf"))):
                #check sotto
                xr=((y-0.5)-yg)/m+xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                if ((xr<=(x+0.5)) and (xr>=(x-0.5))): # se la x è compresa nel lato allora la retta interseca la casella
                    tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                    if(tmp<distanza): distanza=tmp
                    xs.append(x)
                    ys.append(i)
                    conteggio+=1

                elif m<0:
                    #check destra
                    yr=m*((x+0.5)-xg)+yg
                    if((yr<=(y+0.5))and(yr>=(y-0.5))):
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                        xs.append(x)
                        ys.append(i)
                        conteggio+=1

                elif m>0:
                    #check sinistra
                    yr=m*((x-0.5)-xg)+yg
                    if((yr<=(y+0.5))and(yr>=(y-0.5))):
                        tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                        if(tmp<distanza): distanza=tmp
                        xs.append(x)
                        ys.append(i)
                        conteggio+=1

            elif (m==0):
                #m=0 cerco solo a sinistra
                yr=yg
                if((yr<=(y+0.5))and(yr>=(y-0.5))):
                    xs.append(x)
                    ys.append(i)
                    tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                    if(tmp<distanza): distanza=tmp
                    conteggio+=1
            elif (m==float("inf")):
                xr=xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                if ((xr<=(x+0.5)) and (xr>=(x-0.5))): # se la x è compresa nel lato allora la retta interseca la casella
                    tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                    if(tmp<distanza): distanza=tmp
                    xs.append(x)
                    ys.append(i)
                    conteggio+=1
    return conteggio,distanza ,xs,ys      
"""
def findIntersectionwithpoint(img,m,dx,dy,xg,yg): #riscrivere in quanto il riferimento è sbagliato, non puoi usare x e y dei for come coordinate
    xs=[]
    ys=[]
    distanza=float("inf")
    conteggio=0
    a=m
    b=-1
    c=-m*xg+yg
    for i in range(dy):
        for x in range(dx):
            y=dy-i
            if ((m!=0) and (m!=float("inf"))):
                d=abs(a*x+b*y+c)/math.sqrt(a**2+b**2)
                if ((d>=0)and(d<=math.sqrt(2)/2)):
                    xs.append(x)
                    ys.append(i)
            elif (m==0):
                #m=0 cerco solo a sinistra
                yr=yg
                if((yr<=(y+0.5))and(yr>=(y-0.5))):
                   
                    tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                    if(tmp<distanza): distanza=tmp
                    conteggio+=1
            elif (m==float("inf")):
                xr=xg #fisso y del centro del lato in basso del quadrato con (y-0.5) e calcolo la x della retta in quel punto
                if ((xr<=(x+0.5)) and (xr>=(x-0.5))): # se la x è compresa nel lato allora la retta interseca la casella
                    tmp=math.sqrt((x-xg)**2+(y-yg)**2)
                    if(tmp<distanza): distanza=tmp
                    xs.append(x)
                    ys.append(i)
                    conteggio+=1
    return conteggio,distanza ,xs,ys     

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
    for img in insieme_train:
        img=normalizza(img,28,28,1) #normalizzo immagine
        #img=skeletonize(img) #trovo lo skeletro
        xg,yg=baricentro(img,28,28) #calcolo il baricentro
        m=genRette(xg,yg,n) #genero coefficienti angolari rette
        f.write("{n}\n".format(n=insieme_test[idx1]))
        for m in m:
            conta,dist=findIntersection(img,m,28,28,xg,yg)
            f.write("{n}\n".format(n=conta))
            f.write("{n}\n".format(n=dist))
        idx1+=1
    f.close()

def retta(x,m,xg,yg):
        return m*(x-xg)+yg


def prova(immagini_prova):
    
    fig,ax=plt.subplots()
    ax.set_ylim(27.5,-0.5)
    ax.set_xlim(-0.5,27.5)
    x=np.arange(0,28,0.1)

    img=immagini_prova[0]
    img=normalizza(img,28,28,1)
    xg,yg=baricentro(img,28,28)
    ax.imshow(img)
    m=genRette(xg,yg,9)
    print(m)
    for m in m[2:3]:
        if m!=float("inf"):
            ax.plot(x,retta(x,-m,xg,yg),label="{:.2f}".format(-m))
        else:
            ax.plot(np.ones(np.size(x))*xg,x,label="{:.2f}".format(float("inf")))
        cont,dist,xp,yp=findIntersectionwithpoint(img,m,28,28,xg,yg)
        ax.scatter(xp,yp)
    plt.legend(loc='upper left')
    plt.show()

def main():
    trainx,trainy,testx,testy=load_database.load_mnist()
    
    prova(trainx)
    print("Creazione feature train")
    #create_file(trainx,trainy,7,"file_dataset/7rettetrain.txt")
    print("Creazione feature test")
    create_file(testx,testy,7,"file_dataset/7rettetest.txt")
    
    #prova(trainx[0:9])




   


    

if __name__ == "__main__":
    main()
