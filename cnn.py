import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

def fashion_mnist_data():
    # Load the Fashion MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the pixel values of the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape the images to add a channel dimension
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=1)

    return train_images, val_images, train_labels, val_labels, test_images, test_labels

def original_mnist_data():
    # Load the original MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Downsample the images from 28x28 to 7x7 using the "area" method
    train_images = tf.image.resize(train_images[..., np.newaxis], (7, 7), method='area').numpy()
    test_images = tf.image.resize(test_images[..., np.newaxis], (7, 7), method='area').numpy()
    
    # Normalize the pixel values of the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=1)

    return train_images, val_images, train_labels, val_labels, test_images, test_labels



def fashion_mnist_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.10),  
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.10),  
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),  
        
        layers.Dense(10, activation='softmax')
    ])
    return model


def original_mnist_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(7, 7, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.05),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.15),

        layers.Dense(10, activation='softmax')
    ])
    
    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, train_images, train_labels, val_images, val_labels):
    myCNN = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
    return myCNN


def evaluate_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy}")
    return test_accuracy


def main():
    ####################################################################################
    ######################## FASHION MNIST DATASET #####################################
    # Load and preprocess the data
    train_images_fashion, val_images_fashion, train_labels_fashion, val_labels_fashion, test_images_fashion, test_labels_fashion = fashion_mnist_data()

    fashion_cnn = fashion_mnist_model()
    compile_model(fashion_cnn)
    cnn_data = train_model(fashion_cnn, train_images_fashion, train_labels_fashion, val_images_fashion, val_labels_fashion)
    test_accuracy_fashion = evaluate_model(fashion_cnn, test_images_fashion, test_labels_fashion)

    print(f"Test accuracy: {test_accuracy_fashion}")




    ###############################################################################################
    ######################## ORIGINAL MNIST DATASET ###############################################
    # Load and preprocess the data
    train_images_original, val_images_original, train_labels_original, val_labels_original, test_images_original, test_labels_original = original_mnist_data()

    original_cnn = original_mnist_model()
    compile_model(original_cnn)
    cnn_data = train_model(original_cnn, train_images_original, train_labels_original, val_images_original, val_labels_original)
    test_accuracy_original = evaluate_model(original_cnn, test_images_original, test_labels_original)

    print(f"Test accuracy: {test_accuracy_original}")
    ###############################################################################################


if __name__ == '__main__':
    main()
