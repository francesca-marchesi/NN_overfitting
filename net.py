import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense


def simple_net(X_train):
    train_dim = len(X_train.columns)
    model = Sequential()
    model.add(Dense(train_dim, activation='relu', input_shape=(train_dim,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
