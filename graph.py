import numpy as np
import matplotlib.pyplot as plt
from cnn import fashion_mnist_data, original_mnist_data, fashion_mnist_model, original_mnist_model, compile_model, train_model, evaluate_model


def permutation_test(model, images, labels, n_permutations):
    accuracies = []
    for i in range(n_permutations): 
        print(f"Running permutation {i + 1} / {n_permutations}...")
        # Shuffle the labels
        shuffled_labels = np.random.permutation(labels)
        
        # Evaluate the model on the shuffled labels
        _, accuracy = model.evaluate(images, shuffled_labels)
        
        accuracies.append(accuracy)
        
    return accuracies

def plot_performance_fashion(cnn_data, test_accuracy, confidence_interval):
    epochs = range(1, len(cnn_data.history['accuracy']) + 1)
    
    plt.figure(figsize=(8, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, cnn_data.history['accuracy'], 'bo-', label='Training acc')
    plt.plot(epochs, cnn_data.history['val_accuracy'], 'ro-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('D:/Studies/CMPUT466/ASSIGNMENT3/training_validation_accuracy_fashion.png')
    plt.show()

    # Plot confidence interval and test accuracy
    plt.figure(figsize=(6, 4))
    plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test Accuracy')  # Test accuracy line
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='grey', alpha=0.2, label='95% Confidence Interval')  # Confidence interval
    plt.xlabel('CNN Model')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy and 95% Confidence Interval')
    plt.legend()
    plt.xticks([])  
    plt.tight_layout()
    plt.savefig('D:/Studies/CMPUT466/ASSIGNMENT3/test_accuracy_confidence_interval_fashion.png')
    plt.show()

def plot_performance_original(cnn_data, test_accuracy, confidence_interval):
    epochs = range(1, len(cnn_data.history['accuracy']) + 1)
    
    plt.figure(figsize=(8, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, cnn_data.history['accuracy'], 'bo-', label='Training acc')
    plt.plot(epochs, cnn_data.history['val_accuracy'], 'ro-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('D:/Studies/CMPUT466/ASSIGNMENT3/training_validation_accuracy_original.png')
    plt.show()

    # Plot confidence interval and test accuracy
    plt.figure(figsize=(6, 4))
    plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test Accuracy')  # Test accuracy line
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='grey', alpha=0.2, label='95% Confidence Interval')  # Confidence interval
    plt.xlabel('CNN Model')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy and 95% Confidence Interval')
    plt.legend()
    plt.xticks([])  
    plt.tight_layout()
    plt.savefig('D:/Studies/CMPUT466/ASSIGNMENT3/test_accuracy_confidence_interval_original.png')
    plt.show()


def main():
    train_images_fashion, val_images_fashion, train_labels_fashion, val_labels_fashion, test_images_fashion, test_labels_fashion = fashion_mnist_data()
    train_images_original, val_images_original, train_labels_original, val_labels_original, test_images_original, test_labels_original = original_mnist_data()
    
    # Compile and train the Fashion MNIST model
    fashion_cnn = fashion_mnist_model()
    compile_model(fashion_cnn)
    cnn_data_fashion = train_model(fashion_cnn, train_images_fashion, train_labels_fashion, val_images_fashion, val_labels_fashion)
    test_accuracy_fashion = evaluate_model(fashion_cnn, test_images_fashion, test_labels_fashion)
    
    # Compile and train the Original MNIST model
    original_cnn = original_mnist_model()
    compile_model(original_cnn)
    cnn_data_original = train_model(original_cnn, train_images_original, train_labels_original, val_images_original, val_labels_original)
    test_accuracy_original = evaluate_model(original_cnn, test_images_original, test_labels_original)

    # Run the permutation test for Fashion MNIST
    permutation_accuracies_fashion = permutation_test(fashion_cnn, test_images_fashion, test_labels_fashion, n_permutations=100)

    # Calculate confidence intervals from the permutation accuracies for Fashion MNIST
    confidence_interval_fashion = (np.percentile(permutation_accuracies_fashion, 2.5), np.percentile(permutation_accuracies_fashion, 97.5))

    # Run the permutation test for Original MNIST
    permutation_accuracies_original = permutation_test(original_cnn, test_images_original, test_labels_original, n_permutations=100)

    # Calculate confidence intervals from the permutation accuracies for Original MNIST
    confidence_interval_original = (np.percentile(permutation_accuracies_original, 2.5), np.percentile(permutation_accuracies_original, 97.5))

    # Plot the performance for both models
    plot_performance_fashion(cnn_data_fashion, test_accuracy_fashion, confidence_interval_fashion)
    plot_performance_original(cnn_data_original, test_accuracy_original, confidence_interval_original)
    
    # Print accuracies and p-values if needed
    print(f"Permutation test confidence interval for Fashion MNIST: {confidence_interval_fashion}")
    print(f"Permutation test confidence interval for Original MNIST: {confidence_interval_original}")

if __name__ == '__main__':
    main()