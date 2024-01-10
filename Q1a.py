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

accuracies = []
iters = np.arange(0, 10, 1)

for i in iters:
    
    y_train_target = np.eye(10)[y_train]
    y_test_target = np.eye(10)[y_test]
        
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # input is a 28x28 image
        tf.keras.layers.Dense(32, activation='relu'), # 32 neurons in the middle "hidden"
        tf.keras.layers.Dense(10, activation='relu') # 10 outputs (one for each category)
    ])
    
    
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1), # use stochastic gra
        loss=my_loss,
        metrics=['accuracy'] # in addition to the loss, also compute the categor
    )
    
    model.fit(
        x_train, 
        y_train_target, 
        epochs = 5, 
        validation_data = (x_test, y_test_target)
    );
    
    accuracies.append(model.history.history['accuracy']) # Training accuracy
       
plt.figure()

plt.xlabel("Epcohs")
plt.ylabel("% Accuracy")


for i in iters:    
    plt.plot(np.arange(1,6, 1), accuracies[i], label="Trial: " + str(i))
    
plt.title('Training Accuracy vs Epochs')
plt.legend()
plt.grid()
plt.show()


