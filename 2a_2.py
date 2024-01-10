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


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Scale pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
y_test = y_test[:,0]

plt.figure(figsize=(14,6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(names[int(y_train[i])])
plt.show()


training_accuracy = []
testing_accuracy = []
iters = np.arange(0, 10, 1)
N_epochs = 10

for i in iters:
    print("Iteration:", i)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(64, activation='relu'), # 32 neurons in the middle "hidden"
        tf.keras.layers.Dense(10, activation='softmax') # 10 outputs (one for each category)
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] # in addition to the loss, also compute the categor
    )
    
    model.fit(x_train, y_train, epochs=N_epochs, validation_data=(x_test, y_test))

    
    training_accuracy.append(model.history.history['accuracy'])
    testing_accuracy.append(model.history.history['val_accuracy'])
    
    
plt.figure()
plt.title("Training Accuracy against epochs")
plt.xlabel("Epcohs")
plt.ylabel("Loss")

for i in iters:    
    plt.plot(np.arange(1,N_epochs + 1, 1), training_accuracy[i], label="Trial: " + str(i))
    
plt.grid()
plt.legend()
plt.show()


plt.figure()
plt.title("Validation Accuracy against epochs, Sequential model")
plt.xlabel("Epcohs")
plt.ylabel("Loss")

for i in iters:    
    plt.plot(np.arange(1,N_epochs + 1, 1), testing_accuracy[i], label="Trial: " + str(i))
    
plt.grid()
plt.legend()
plt.show()
    
    
    
    
    
    
