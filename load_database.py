from keras.datasets import mnist

def load_mnist():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    return train_X,train_y,test_X,test_y

if __name__ == "__main__": load_mnist()
