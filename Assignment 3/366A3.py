from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def plot_digit(image):
    """
    This function receives an image and plots the digit. 
    """
    plt.imshow(image, cmap='gray')
    plt.savefig('digit2')
    plt.show()


class LinearRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(labels):
            self._Y[value][index] = 1
            
    def return_weights_of_digit(self, digit):
        """
        Returns the weights of the model for a given digit
        """
        return self._W[digit, :]
            
    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the linear regression rule
        """
        
        # Write the computation of the hypothesis for a given matrix X
        return self._W.dot(X) + self._B
    
    def train(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W = self._W - (self._alpha / self._m) * np.matmul(pure_error, self._x_train.T)
            self._B = np.subtract(self._B, (self._alpha / self._m) * pure_error.sum(axis=1).reshape((10, 1)))

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == np.argmax(self._Y, axis=0))
                percentage_classified_correctly = (classified_correctly / self._m) * 100
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                Y_hat_test = self.h_theta(images_test)
                test_correct = np.count_nonzero(np.argmax(Y_hat_test, axis=0) == self._y_test)
                classified_correctly_test_list.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)
        return classified_correctly_train_list, classified_correctly_test_list    


class LogisticRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(labels):
            self._Y[value][index] = 1
            
    def sigmoid(self, Z):
        """
        Computes the sigmoid value for all values in vector Z
        """
        # Write the computation of the sigmoid function for a given matrix Z
        return 1 / (1 + np.exp((-1) * Z))

    def derivative_sigmoid(self, A):
        """
        Computes the derivative of the sigmoid for all values in vector A
        """
        # Write the derivative of the sigmoid function for a given value A produced by the sigmoid
        return np.multiply(A, 1 - A)

    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the logistic regression rule
        """
        # Write the computation of the hypothesis for a given matrix X
        return self.sigmoid(self._W.dot(X) + self._B)
    
    def return_weights_of_digit(self, digit):
        """
        Returns the weights of the model for a given digit
        """
        return self._W[digit, :]
    
    def train(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W = self._W - (self._alpha / self._m) * np.matmul((np.multiply(pure_error, self.derivative_sigmoid(A))), self._x_train.T)
            self._B = np.subtract(self._B, (self._alpha / self._m) * np.multiply(pure_error, self.derivative_sigmoid(A)).sum(axis=1).reshape((10, 1)))

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == np.argmax(self._Y, axis=0))
                percentage_classified_correctly = (classified_correctly / self._m) * 100
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                Y_hat_test = self.h_theta(images_test)
                test_correct = np.count_nonzero(np.argmax(Y_hat_test, axis=0) == self._y_test)
                classified_correctly_test_list.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)
        return classified_correctly_train_list, classified_correctly_test_list   


class LogisticRegressionCrossEntropy:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(labels):
            self._Y[value][index] = 1
            
    def sigmoid(self, Z):
        """
        Computes the sigmoid value for all values in vector Z
        """
        # Write the computation of the sigmoid function for a given matrix Z
        return 1 / (1 + np.exp((-1) * Z))

    def derivative_sigmoid(self, A):
        """
        Computes the derivative of the sigmoid for all values in vector A
        """
        # Write the derivative of the sigmoid function for a given value A produced by the sigmoid
        return np.multiply(A, 1 - A)

    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the logistic regression rule
        """
        # Write the computation of the hypothesis for a given matrix X
        return self.sigmoid(self._W.dot(X) + self._B)
    
    def return_weights_of_digit(self, digit):
        """
        Returns the weights of the model for a given digit
        """
        return self._W[digit, :]
    
    def train(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W = self._W - (self._alpha / self._m) * np.matmul(pure_error, self._x_train.T)
            self._B = np.subtract(self._B, (self._alpha / self._m) * pure_error.sum(axis=1).reshape((10, 1)))

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == np.argmax(self._Y, axis=0))
                percentage_classified_correctly = (classified_correctly / self._m) * 100
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                Y_hat_test = self.h_theta(images_test)
                test_correct = np.count_nonzero(np.argmax(Y_hat_test, axis=0) == self._y_test)
                classified_correctly_test_list.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)
        return classified_correctly_train_list, classified_correctly_test_list   



# The x variables contain the images of handwritten digits the y variables contain their labels indicating 
# which digit is in the image. We will see an example of image and label shortly. We have two data sets here:
# training and test sets. The idea is that we use the training set to learn the function and then we evaluate 
# the system on images the model did not see during training. This is to simulate the scenario in which we build 
# a system and we use it in the wild, where people write new digits and we would like our system to accurately 
# recognize them.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Each image is of size 28x28 and the training data set has 60,000 images
# the shape of x_train should then be (60000, 28, 28).
print(x_train.shape)

# Let's take a look at the first training instance. The image shows the digit 5. 
# Feel free to change the index of x_train to see other images. 
plot_digit(x_train[0])


# The y_train structure has shape 60,0000, with one entry for each image. The value of the first
# entry of y_train should be a five, indicating that the first image is of a 5.
print(y_train.shape)
print('Label: ', y_train[0])

# Instead of using all 60,000 images, we will use 20,000 in our experiments to speed up training.
training_size = 20000

# We will flatten the images to make our implementation easier. Instead of providing 
# an image of size 28x28 as input, we will provide a vector of size 784, with one entry 
# for each pixel of the image. We will also normalize the values in the images. The pixels
# values in the images vary from 0 to 255. We normalize them to avoid overflow during training. 
# Normalization also helps training because the optimization landscape is friendiler to gradient 
# descent once the input values are normalized. Studying the effects of the input values in the
# optimization landscape is outside the scope of this course. 
images, labels = (x_train[0:training_size].reshape(training_size, 28*28) / 255, y_train[0:training_size])

# The flattened images will be kept as column vectors in the matrix images
images = images.T
print('Shape of flattned images: ', images.shape)

# Here we apply the same transformations described above on the test data. 
images_test = x_test.reshape(x_test.shape[0], 28*28) / 255
images_test = images_test.T



#lin_reg = LinearRegression(images, labels, images_test, y_test)
#print('Training Linear Regression')
#percentage_lin_reg_train, percentage_lin_reg_test = lin_reg.train(3000)


log_reg = LogisticRegression(images, labels, images_test, y_test)
print('Training Logistic Regression (MSE)')
percentage_log_reg_train, percentage_log_reg_test = log_reg.train(3000)


log_reg_cross = LogisticRegressionCrossEntropy(images, labels, images_test, y_test)
print('Training Logistic Regression (Cross Entropy)')
percentage_log_reg_cross_train, percentage_log_reg_cross_test = log_reg_cross.train(3000)

plt.xlabel('Iterations')
plt.ylabel('Percentage Correctly Classified')
plt.title('Training Accuracy')
plt.ylim([60, 100])
plt.plot(percentage_log_reg_train, 'r-.', label='Logistic (MSE)')
plt.plot(percentage_log_reg_cross_train, 'g-', label='Logistic (Cross-Entropy)')
plt.legend(loc='best')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Percentage Correctly Classified')
plt.title('Test Accuracy')
plt.ylim([60, 100])
plt.plot(percentage_log_reg_test, 'r-.', label='Logistic (MSE)')
plt.plot(percentage_log_reg_cross_test, 'g-', label='Logistic (Cross-Entropy)')
plt.legend(loc='best')
plt.show()

#for i in range(0, 10):
    #plot_digit(log_reg_cross.return_weights_of_digit(i).reshape(28, 28))

