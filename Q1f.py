import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

# define what we want to minimize (the thing that we take the derivative of to get the w
def my_loss(y_true, y_predict):
    return (y_true-y_predict)**2


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0 # rescale the images to be between 0 and 1
x_test = x_test / 255.0 # rescale the images to be between 0 and 1

y_train_target = np.eye(10)[y_train]
y_test_target = np.eye(10)[y_test]

training_accuracies = []
testing_accuraies = []
iters = np.arange(0, 10, 1)

N_neurons = [2, 4, 8, 16, 32, 64, 128, 256]
N_epochs = 10

for layer_size in N_neurons:
    print("Number of neurons: ", layer_size)

    training_accuracy_for_N_neurons = np.zeros(N_epochs)
    testing_accuracy_for_N_neurons = np.zeros(N_epochs)

    for i in iters:
        print("Iteration: ", i)
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)), # input is a 28x28 image
            tf.keras.layers.Dense(layer_size, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax') # 10 outputs (one for each category)
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'] # in addition to the loss, also compute the categor
        )
        
        model.fit(x_train, y_train, epochs = N_epochs, validation_data=(x_test, y_test))
    
        
        training_accuracy_for_N_neurons += np.array(model.history.history['accuracy']) # training accuracy
        testing_accuracy_for_N_neurons += np.array(model.history.history['val_accuracy'])
        
    training_accuracy_for_N_neurons /= max(iters)
    testing_accuracy_for_N_neurons /= max(iters)
    
    training_accuracies.append(training_accuracy_for_N_neurons)
    testing_accuraies.append(testing_accuracy_for_N_neurons)
        
        
        
training_accuracies_new = np.array(training_accuracies) * 9 / 10
        
plt.figure()
plt.title("Average training accuracies against number of neurons, Sequential model")
plt.xlabel("Neurons")
plt.ylabel("Accuracy")

end_of_run_accuracies = []

for run in training_accuracies_new:
    end_of_run_accuracies.append(run[-1])

plt.scatter(N_neurons, end_of_run_accuracies)
    
plt.grid()
plt.ylim([0.6, 1])
plt.show()



plt.figure()
plt.title("Average validation accuracies against number of neurons, Sequential model")
plt.xlabel("Neurons")
plt.ylabel("Accuracy")

testing_accuraies_new = np.array(testing_accuraies) * 9 / 10
end_of_run_accuracies = []

for run in testing_accuraies_new:
    end_of_run_accuracies.append(run[-1])

plt.scatter(N_neurons, end_of_run_accuracies)
    
plt.ylim([0.6, 1])
plt.grid()
plt.legend()
plt.show()










