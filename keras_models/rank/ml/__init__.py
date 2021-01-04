import tensorflow as tf

layer1 = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=100)
layer2 = tf.keras.layers.experimental.preprocessing.CategoryCrossing()
layer3 = tf.keras.layers.Embedding(input_dim=100, output_dim=10)
a = tf.constant(value=[["1", "2"], ["1", "3"]])
b = tf.constant(value=[["2", "3"], ["2", "2"]])

print(layer1([tf.sparse.from_dense(a), b]))
print(layer1([a, b]))
print(layer3(layer1([a, b])))
print(layer3(layer1([tf.sparse.from_dense(a), b])))

tf.keras.experimental.LinearModel

tf.io.RaggedFeature
tf.io.FixedLenFeature
tf.io.VarLenFeature

