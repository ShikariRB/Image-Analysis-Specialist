
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import dump
from skimage.transform import resize

def original_mnist_data():
    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'], mnist['target']

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def downsample_and_flatten_images(images, new_shape):
    images = images.reshape(-1, 28, 28)
    images_lowres = np.array([resize(image, new_shape, anti_aliasing=True) for image in images])
    return images_lowres.reshape((-1, new_shape[0] * new_shape[1]))


def original_mnist_data_lowres():
    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'], mnist['target']

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Downsample and flatten images
    X_train = downsample_and_flatten_images(X_train, (7,7))
    X_test = downsample_and_flatten_images(X_test, (7,7))

    return X_train, X_test, y_train, y_test

def svm_model(X_train, y_train):
    model = svm.SVC()
    model.fit(X_train, y_train)

    dump(model, 'svm_model.joblib')
    
    return model


def evaluate_model(model,X_train, y_train, X_val, y_val, X_test, y_test, save_path='accuracies.npy'):

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    accuracies = np.array([train_accuracy, val_accuracy, test_accuracy])
    np.save(save_path, accuracies)

    return train_accuracy, val_accuracy, test_accuracy



def main():
    
    X_train, X_test, y_train, y_test = original_mnist_data()
    # Split off a validation set from the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    model = svm_model(X_train, y_train)

    train_accuracy, val_accuracy, test_accuracy = evaluate_model(model, X_train, y_train,X_val, y_val, X_test, y_test)

    print(f'Accuracy - Training: {train_accuracy}, Validation: {val_accuracy}, Test: {test_accuracy}')

    X_train, X_test, y_train, y_test = original_mnist_data_lowres()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  

    model = svm_model(X_train, y_train)
    train_accuracy, val_accuracy, test_accuracy = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    print(f'Accuracy - Training_lowres: {train_accuracy}, Validation_lowres: {val_accuracy}, Test_lowres: {test_accuracy}')  

if __name__ == '__main__':
    main()
