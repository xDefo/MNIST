from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


train=open("file_dataset/13rettetrain.txt","r")
y_train=[]
x_train=[]
Ntr=60000
Nts=10000
for i in range(Ntr):
    tmp=train.readline()
    y_train.append(int(tmp[0:len(tmp)-1]))
    a=[]
    for j in range(13):
        tmp=train.readline()
        a.append(float(tmp[0:len(tmp)-1]))
    x_train.append(a)

train.close()
test=open("file_dataset/13rettetest.txt","r")
y_test=[]
x_test=[]
for i in range(Nts):
    tmp=test.readline()
    y_test.append(int(tmp[0:(len(tmp)-1)]))
    a=[]
    for j in range(13):
        tmp=test.readline()
        a.append(float(tmp[0:len(tmp)-1]))
    x_test.append(a)

x_train=np.array(x_train)
y_train=np.array(y_train)

x_test=np.array(x_test)
y_test=np.array(y_test)

print(x_train[0])
print(y_train[0])


wait = input("Press Enter to continue.")
num_categories=10

y_train=keras.utils.to_categorical(y_train,num_categories)
y_test =keras.utils.to_categorical(y_test, num_categories)


model=Sequential()

#aggiungo layer + input
model.add(Dense(units=50,activation='relu',input_shape=(13
,)))

#aggiungo uno strato nascosto
model.add(Dense(units=50,activation='relu'))

#output layer
model.add(Dense(units=10,activation='softmax'))

model.summary()
wait = input("Press Enter to continue.")
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=30,verbose=1,validation_data=(x_test,y_test));
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
model.save('modelli/prova_con_9_rette.h5')
plt.show()

