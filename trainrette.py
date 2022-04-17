from cProfile import label
from itertools import count
from tkinter import Y
from turtle import color
from cycler import cycler
from cv2 import ellipse
from  tensorflow.keras.models import Sequential
from keras.models import Model   
from keras.layers import * 
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from    keras.models import load_model
import  tensorflow.keras as keras
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

train=open("9rette.txt","r")
y_train=[]
x_train=[]
Ntr=60000
Nts=10000
n=9

for i in range(Ntr):
    tmp=train.readline()
    y_train.append(int(tmp[0:len(tmp)-1]))
    a=[]
    for j in range(n):
        tmp=train.readline()
        a.append(float(tmp[0:len(tmp)-1]))
    x_train.append(a)
train.close()

test=open("9rettetest.txt","r")
y_test=[]
x_test=[]
for i in range(Nts):
    tmp=test.readline()
    y_test.append(int(tmp[0:(len(tmp)-1)]))
    a=[]
    for j in range(n):
        tmp=test.readline()
        a.append(float(tmp[0:len(tmp)-1]))
    x_test.append(a)
test.close()

x_train=np.array(x_train)
y_train=np.array(y_train)

x_test=np.array(x_test)
y_test=np.array(y_test)

print(y_test[0])
print(x_test[0])
"""
idx=np.where(y_train==2)
num=x_train[idx]
print(num)

fig,ax=plt.subplots(subplot_kw={'aspect': 'equal'})
for i in range(0,2*n,2):
    rm=np.mean(num[:,i])
    srm=np.mean(num[:,i])
    dm=10*np.mean(num[:,i+1])
    sdm=10*np.mean(num[:,i+1])
    print(rm, srm, dm,sdm)
    e=Ellipse((rm,dm),width=srm, height=sdm)
    ax.add_artist(e) 
    e.set_clip_box(ax.bbox) 
    e.set_fill(False)
    e.set_label("retta{:.0f}".format(i/2))
    e.set_edgecolor(np.random.rand(3))
    e.set_linestyle('--')

ax.set_xlim(0,35)
ax.set_ylim(0,15)
ax.set_xlabel("Intersezioni")
ax.set_ylabel("Distanza*10")
ax.set_title("due")
plt.savefig("Immagini/due.png")
plt.legend()
plt.show()

#for num in range(10):


for j in range(10):
    idx=np.argwhere(y_train==j)
    num=x_train[idx]
    for i in range(9):
        plt.subplots()
        plt.hist(num[:,0,i])
        plt.savefig("hist/immaginenumero{n}feature{m}.png".format(n=j,m=i))

#plt.show()
""" 


#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='upper left')
#plt.show()


num_categories=10 

y_train=keras.utils.to_categorical(y_train,num_categories)
y_test =keras.utils.to_categorical(y_test, num_categories)

#creo il modello
model=Sequential()

#aggiungo layer + input
model.add(Dense(units=100,activation='relu',input_shape=(9,)))

#aggiungo uno strato nascosto
model.add(Dense(units=50,activation='relu'))

#output layer
model.add(Dense(units=10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10,verbose=1,validation_data=(x_test,y_test))
print(history.history.keys())

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Salvataggio del modello
model.save('modelli/prova_con_9_rettec++.h5')
plt.show()

predictions =model.predict(x_test)
print(predictions)
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
disp = ConfusionMatrixDisplay(matrix)
disp.plot()
plt.show()
"""
f=open("sbagliati.txt","w")
model=load_model("modelli/prova_con_7_rette_e_dist.h5")
model.summary()
idx=0
count=0
for pat in x_test:
    vp=model.predict(pat.reshape(1,14)).reshape(10)
    val=y_test[idx]
    if(y_test[idx]!=np.argmax(vp)):
        #f.write("{m} {n} {k}\n".format(m=idx,n=val, k=np.argmax(vp)))
        count+=1
    idx+=1
print(count)
"""