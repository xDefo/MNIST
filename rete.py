import numpy as np
import  tensorflow.keras as keras
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def create_database():
  train=np.loadtxt("train.txt")
  print(np.shape(train))
  x_train=np.empty([60000,9])
  y_train=np.empty([60000])
  print(np.shape(x_train))
  idx=0
  for i in range(60000):
    y_train[i]=train[idx]
    idx+=1
    for j in range(9):
      x_train[i,j]=train[idx]
      idx+=1

  train=np.loadtxt("test.txt")
  x_test=np.empty([10000,9])
  y_test=np.empty([10000])
  idx=0
  for i in range(10000):
    y_test[i]=train[idx]
    idx+=1
    for j in range(9):
      x_test[i,j]=train[idx]
      idx+=1

  return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test=create_database()


num_categories=10 

y_train=keras.utils.to_categorical(y_train,num_categories)
y_test=keras.utils.to_categorical(y_test,num_categories)


model=keras.models.Sequential()
model.add(keras.layers.Dense(units=100,activation='relu',input_shape=(9,)))
model.add(keras.layers.Dense(units=50,activation='relu'))
model.add(keras.layers.Dense(units=10,activation='softmax'))
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