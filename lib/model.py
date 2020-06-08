from tensorflow import keras

class Model(keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.dense1 = keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)