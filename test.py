import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test,  y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.load_model("model.h5")
# model = tf.keras.models.load_model(
#     "models"
# )
values = model.predict(X_test[67].reshape(1, 28, 28))
value = np.argmax(values)
print(value)