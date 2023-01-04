import tensorflow as tf
from tensorflow import keras
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the image data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to be of shape (n_samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the encoder model
encoder_input = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
x = keras.layers.MaxPooling2D(padding='same')(x)
x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D(padding='same')(x)
x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
encoder_output = keras.layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output)

# Generate embeddings for the training data
train_embeddings = encoder.predict(x_train)

# Use t-SNE to reduce the dimensionality of the embeddings to 2 dimensions
tsne = TSNE(n_components=2)
train_embeddings_2d = tsne.fit_transform(train_embeddings)

# Visualize the 2D embeddings
plt.figure(figsize=(10, 10))
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=y_train, cmap='tab10')
plt.colorbar()
plt.show()
