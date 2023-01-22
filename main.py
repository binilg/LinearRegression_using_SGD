import pandas as pd
from simple_linear_regr_utils import generate_data, evaluate
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def generate_data():
    """
    Generates a random dataset from a normal distribution.

    Returns:
        diabetes_X_train: the training dataset
        diabetes_y_train: The output corresponding to the training set
        diabetes_X_test: the test dataset
        diabetes_y_test: The output corresponding to the test set

    """
    # Load the diabetes dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20].reshape(-1,1)
    diabetes_y_test = diabetes_y[-20:].reshape(-1,1)

    print(f"# Training Samples: {len(diabetes_X_train)}; # Test samples: {len(diabetes_X_test)};")
    return diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test


def evaluate(model, X, y, y_predicted):
    """ Calculates and prints evaluation metrics. """
    # The coefficients
    print(f"Slope: {model.W}; Intercept: {model.b}")
    # The mean squared error
    mse = mean_squared_error(y, y_predicted)
    print(f"Mean squared error: {mse:.2f}")
    # The coefficient of determination: 1 is perfect prediction
    r2 = r2_score(y, y_predicted)
    print(f"Coefficient of determination: {r2:.2f}")

    # Plot outputs
    plt.scatter(X, y, color="black")
    plt.plot(X, y_predicted, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    if r2 >= 0.4:
        print("****** Success ******")
    else:
        print("****** Failed ******")
        
class SimpleLinearRegression:
    def __init__(self, iterations=15000, lr=0.1):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        self.W, self.b = None, None # the slope and the intercept of the model

    def __loss(self, y, y_hat):
        """

        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error

        """
        #ToDO calculate the loss. use the sum of squared error formula for simplicity
        loss = np.sum((y - y_hat)**2)

        self.losses.append(loss)
        return loss

    def __init_weights(self, X):
        """

        :param X: The training set
        """
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]]#.reshape(-1, X.shape[1])
        self.b = weights[-1]

    def __sgd(self, X, y, y_hat):
        """

        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return:
            sets updated W and b to the instance Object (self)
        """
        # ToDo calculate dW & db.
        dW = (2/len(y)) * np.sum(X*(y_hat-y))
        db = (2/len(y)) * np.sum((y_hat-y))
        print('n = ',len(y))
        print('dW ',dW)
        print('db ',db)
        #  ToDO update the self.W and self.b using the learning rate and the values for dW and db
        self.W = self.W - (self.lr * dW)
        self.b = self.b - (self.lr * db)
        #print(pd.DataFrame({'W':self.W, 'b':self.b}))
        


    def fit(self, X, y):
        """

        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        self.__init_weights(X)
        y_hat = self.predict(X)
        #print('ndim =',y_hat)
        #print(pd.DataFrame({'Actual':y.flatten(), 'predicted':y_hat.flatten()}))
        loss = self.__loss(y, y_hat)
        #print(f"Initial Loss: {loss}")
        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            #print('iteration = ',i)
            #print(pd.DataFrame({'Actual':y.flatten(), 'predicted':y_hat.flatten()}))
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        """

        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        #ToDO calculate the predicted output y_hat. remember the function of a line is defined as y = WX + b
        y_hat = (self.W * X) + self.b
        return y_hat
    
if __name__ == "__main__":
    trained = False
    if not trained:
        X_train, y_train, X_test, y_test = generate_data()
        model = SimpleLinearRegression()
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        W = model.W
        b = model.b
        print('W = ',W,'b = ',b)
        evaluate(model, X_test, y_test, predicted)
        trained = True