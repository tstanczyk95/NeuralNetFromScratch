import numpy as np
import matplotlib.pyplot as plt
import time

INPUT_LAYER_SIZE = 8
HIDDEN_LAYER_SIZE = 3
OUTPUT_LAYER_SIZE = 8


##### Neural Network and backpropagation algorithm implementation #####

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def create_weights():
	theta1 = np.random.normal(0, 0.01, (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1))
	theta2 = np.random.normal(0, 0.01, (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1))

	return theta1, theta2


def forward_propagation_vectorized(X, theta1, theta2):
	x_ones = np.ones((1, X.shape[1]))
	a1 = np.concatenate((x_ones, X))

	z1 = np.matmul(theta1, a1)
	a2 = sigmoid(z1)
	a2_ones = np.ones((1, a2.shape[1]))
	a2 = np.concatenate((a2_ones, a2))

	z2 = np.matmul(theta2, a2)
	a3 = sigmoid(z2) # output

	return a1, a2, a3


def backward_propagation_vectorized(Y, a1, a2, a3, theta1, theta2, m, lambd):
	cumulative_delta1 = np.zeros((theta1.shape[0], theta1.shape[1])) 
	cumulative_delta2 = np.zeros((theta2.shape[0], theta2.shape[1]))

	delta3 = a3 - Y
	delta2 = np.matmul(np.transpose(theta2), delta3)
	delta2 = np.multiply(delta2, a2)
	delta2 = np.multiply(delta2, (1 - a2))
	delta2 = delta2[1:,:]

	cumulative_delta1 += np.matmul(delta2, np.transpose(a1))
	cumulative_delta2 += np.matmul(delta3, np.transpose(a2))

	D1 = cumulative_delta1
	regularization_componentheta1 = np.ones((D1.shape[0], D1.shape[1])) * lambd
	regularization_componentheta1 = np.multiply(regularization_componentheta1, theta1)
	regularization_componentheta1[:,0] = 0
	D1 += regularization_componentheta1
	D1 /= m

	D2 = cumulative_delta2
	regularization_componentheta2 = np.ones((D2.shape[0], D2.shape[1])) * lambd
	regularization_componentheta2 = np.multiply(regularization_componentheta2, theta2)
	regularization_componentheta2[:,0] = 0
	D2 += regularization_componentheta2
	D2 /= m

	return D1, D2


def backpropagation_algorithm(X, Y, theta1, theta2, lr, m, num_iter, lambd):
	rmse_list = []

	for i in range(num_iter):
		a1, a2, a3 = forward_propagation_vectorized(X, theta1, theta2)
		D1, D2 = backward_propagation_vectorized(Y, a1, a2, a3, theta1, theta2, m, lambd)

		theta1 -= lr * D1
		theta2 -= lr * D2

		if i % 50 == 0:
			rmse = np.sqrt(np.mean(np.power(a3 - Y, 2)))
			rmse_list.append(rmse)

	return theta1, theta2, rmse_list


##### Data, hyper-parameters, backpropagation algorithm inputs #####

theta1, theta2 = create_weights()
X = np.identity(8)
Y = np.identity(8)
lr = 1 # learning rate
m = 8
num_iter = 10000
lambd = 0.0001
learning_rates = [0.001,0.01,0.1,1,10,100]
lambdas = [0,0.0001,0.001,0.01,0.1,1,10,100]
iterations = [10,100,1000,10000,100000]


##### Plotting functions #####

def test_learning_rates():    
    for i in range(len(learning_rates)):
        theta1_updated, theta2_updated, rmse = backpropagation_algorithm(X, Y, theta1, theta2, learning_rates[i], m, num_iter,lambd)
        fig = plt.plot(rmse)
        plt.title('RMSE with learning rate = '+str(learning_rates[i]) + " and Lambda = 0.0001", fontsize=14)
        plt.xlabel('Iterations (every 50)')
        plt.savefig("inside_backpropagate_learning_rate_"+str(learning_rates[i])+".jpg")
        plt.show()
        

def test_lambdas():
    for i in range(len(lambdas)):
        theta1_updated, theta2_updated, rmse = backpropagation_algorithm(X, Y, theta1, theta2, lr, m, num_iter,lambdas[i])
        fig = plt.plot(rmse)
        plt.title('RMSE with Lambda = '+str(lambdas[i]) + " and learning rate = 1", fontsize=14)
        plt.xlabel('Iterations (every 50)')
        plt.savefig("inside_backpropagate_lambda_"+str(lambdas[i])+".jpg")
        plt.show()
    

def test_final_error_learning_rates():
    errors_learning_rates = []
    for i in range(len(learning_rates)):
        theta1_updated, theta2_updated, rmse = backpropagation_algorithm(X, Y, theta1, theta2, learning_rates[i], m, num_iter,lambd)
        _, a2, a3 = forward_propagation_vectorized(X, theta1_updated, theta2_updated)
        error = np.sqrt(np.mean(np.power(a3 - Y, 2)))
        errors_learning_rates.append(error)
        
    fig = plt.plot(learning_rates,errors_learning_rates,'x')
    plt.title('RMSE with different learning rates\n(lambda = 0.0001, 10000 iterations each)', fontsize=16)
    plt.xscale('log')
    plt.xlabel('Learning rate, logarithmic scale')
    plt.savefig("end_algorithm_learning_rates.jpg")
    plt.show()
    

def test_final_error_lambdas():
    errors_lambdas = []
    for i in range(len(lambdas)):
        theta1_updated, theta2_updated, rmse = backpropagation_algorithm(X, Y, theta1, theta2, lr, m, num_iter,lambdas[i])
        _, a2, a3 = forward_propagation_vectorized(X, theta1_updated, theta2_updated)
        error = np.sqrt(np.mean(np.power(a3 - Y, 2)))
        errors_lambdas.append(error)

    fig = plt.plot(lambdas,errors_lambdas,'x')
    plt.title('RMSE with different lambdas\n(learning rate = 1, 10000 iterations)', fontsize=16)
    plt.xscale('log')
    plt.xlabel('Lambda, logarithmic scale')
    plt.savefig("end_algorithm_lambdas.jpg")
    plt.show()
    

def test_final_error_iterations():
    errors_number_iterations = []
    for i in range(len(iterations)):
        theta1_updated, theta2_updated, rmse = backpropagation_algorithm(X, Y, theta1, theta2, lr, m, iterations[i],lambd)
        _, a2, a3 = forward_propagation_vectorized(X, theta1_updated, theta2_updated)
        error = np.sqrt(np.mean(np.power(a3 - Y, 2)))
        errors_number_iterations.append(error)

    fig = plt.plot(iterations,errors_number_iterations,'x')
    plt.title('RMSE with different iterations\n(learning rate = 1, lambda = 0.0001)', fontsize=16)
    plt.xlabel('Number of iterations, logarithmic scale')
    plt.xscale('log')
    plt.xlabel('Number of iterations, logarithmic scale')
    plt.savefig("end_algorithm_iterations.jpg")
    plt.show()


##### Running the backpropagation algorithm #####

start_time=time.time()
theta1_updated, theta2_updated, _ = backpropagation_algorithm(X, Y, theta1, theta2, lr, m, num_iter, lambdas[2])
end_time=time.time()
time_elapsed = end_time - start_time

print("\nTime elapsed for the run of backpropagation algorithm: {}s".format(time_elapsed))
print()

a1, a2, a3 = forward_propagation_vectorized(X, theta1_updated, theta2_updated)

error = np.sqrt(np.mean(np.power(a3 - Y, 2)))

print("RMSE:")
print(error)
print()
print("Theta1:")
print(theta1_updated)
print("Theta2:")
print(theta2_updated)
print()
print("a1:")
print(a1)
print()
print("a2:")
print(a2)

a2_binary = np.zeros(a2.shape, dtype=int)
for i in range(a2.shape[0]):
	for j in range(a2.shape[1]):
		if a2[i,j] >= 0.5:
			a2_binary[i,j] = 1

print()
print("a2 binary: (for a2[i,j] >= 0.5)")
print(a2_binary)
print()
print("a3:")
print(a3)
print()


#test_learning_rates()
#test_lambdas()
#test_final_error_learning_rates()
#test_final_error_lambdas()
#test_final_error_iterations()