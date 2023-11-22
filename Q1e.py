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

training_losses = []
testing_losses = []
iters = np.arange(0, 10, 1)
iters = np.arange(0, 2, 1)

for i in iters:
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # input is a 28x28 image
        tf.keras.layers.Dense(32, activation='relu'), # 32 neurons in the middle "hidden"
        tf.keras.layers.Dense(10, activation='softmax') # 10 outputs (one for each category)
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] # in addition to the loss, also compute the categor
    )
    
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    
    training_losses.append(model.history.history['loss'])
    testing_losses.append(model.history.history['val_loss'])
    
    

    
plt.figure()
plt.title("Training losses against epochs")
plt.xlabel("Epcohs")
plt.ylabel("Loss")

for i in iters:    
    plt.plot(np.arange(1,21, 1), training_losses[i], label="Trial: " + str(i))
    
plt.grid()
plt.legend()
plt.show()


plt.figure()
plt.title("Validation losses against epochs, Sequential model")
plt.xlabel("Epcohs")
plt.ylabel("Loss")

for i in iters:    
    plt.plot(np.arange(1,21, 1), testing_losses[i], label="Trial: " + str(i))
    
plt.grid()
plt.legend()
plt.show()


