import math
import numpy as np

class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
    
    # Gaussisan univariate distribution
    def real_likelihood(self, x, mean, variance):
        return 1 / math.sqrt( 2 * math.pi * variance) * math.exp(-1/2 * (x - mean)**2/variance)
   
    # Bernoulli distribution
    def binary_likelihood(self, x, p):
        if(x == 1): 
            return p
        else: 
            return 1 - p


    def fit(self, Xtrain, ytrain):
        self.classes = np.unique(ytrain)
        self.pi = {}

        # compute pi_c
        for c in self.classes:
            self.pi[c] = np.mean(ytrain == c)
        
        # estimate all parameters of NBC
        self.theta = {}
        N, D = Xtrain.shape
        
        # Initialize theta dictionary
        for j in range(D):
            self.theta[j] = {}
            for c in self.classes:
                # Get data for class c and feature j
                class_data = Xtrain[ytrain == c, j]
                
                if self.feature_types[j] == 'r':  # Real-valued feature
                    # Gaussian distribution
                    mean = np.mean(class_data)
                    variance = np.var(class_data)
                    # Add small epsilon to avoid division by zero
                    variance = max(variance, 1e-6)
                    self.theta[j][c] = {'mean': mean, 'variance': variance}
                    
                elif self.feature_types[j] == 'b':  # Binary feature
                    # Bernoulli distribution
                    p = np.mean(class_data)
                    self.theta[j][c] = {'p': p}


    
    def predict(self, Xtest):
        predictions = []
        
        for x in Xtest:
            # Compute probability for each class
            class_probabilities = {}
            
            for c in self.classes:
                # Premultiply by pi_c (?)
                log_prob = np.log(self.pi[c])
                
                # Multiply by likelihood for each feature
                for j in range(len(x)):
                    if self.feature_types[j] == 'r':  # Real-valued feature
                        mean = self.theta[j][c]['mean']
                        variance = self.theta[j][c]['variance']
                        likelihood = self.real_likelihood(x[j], mean, variance)
                        log_prob += np.log(likelihood + 1e-10)  # avoid log(0)
                        
                    elif self.feature_types[j] == 'b':  # Binary feature
                        p = self.theta[j][c]['p']
                        likelihood = self.binary_likelihood(x[j], p)
                        log_prob += np.log(likelihood + 1e-10)  # avoid log(0)
                
                class_probabilities[c] = log_prob
            
            # Predict class with highest probability
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        
        return np.array(predictions) 


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Handin 1: Use l2 penalty with C = 5 to achieve regularization lambda || w || ^ 2    

def compare_classifiers_iris(num_runs=1000):
    """
    Compare NBC and Logistic Regression on Iris dataset with increasing training set sizes
    """
    iris = load_iris()
    X, y = iris['data'], iris['target']
    
    # Store results for averaging
    nbc_errors = np.zeros((num_runs, 9))  # 9 different training sizes (10%, 20%, ..., 90%)
    lr_errors = np.zeros((num_runs, 9))
    
    print(f"Running {num_runs} experiments...")
    
    for n in range(num_runs):
        if (n + 1) % 100 == 0:
            print(f"Completed {n + 1}/{num_runs} runs")
            
        # Shuffle data, put 20% aside for testing
        N, D = X.shape
        Ntrain = int(0.8 * N)
        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        Xtest = X[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]

        # Train 10 classifiers of both types with different training sizes
        for k in range(1, 10):  # k = 1, 2, ..., 9 (10%, 20%, ..., 90% of training data)
            # Calculate number of training samples to use
            n_samples = int(k * 0.1 * Ntrain)
            
            # NBC on Iris dataset
            nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
            nbc.fit(Xtrain[:n_samples], ytrain[:n_samples])
            nbc_pred = nbc.predict(Xtest)
            nbc_error = 1 - np.mean(nbc_pred == ytest)  # Classification error = 1 - accuracy
            nbc_errors[n, k-1] = nbc_error

            # Logistic regression on Iris dataset
            clf = LogisticRegression(penalty='l2', C=5, random_state=42, max_iter=1000)
            clf.fit(Xtrain[:n_samples], ytrain[:n_samples])
            lr_pred = clf.predict(Xtest)
            lr_error = 1 - np.mean(lr_pred == ytest)  # Classification error = 1 - accuracy
            lr_errors[n, k-1] = lr_error
    
    # Calculate average errors across all runs
    avg_nbc_errors = np.mean(nbc_errors, axis=0)
    avg_lr_errors = np.mean(lr_errors, axis=0)
    
    # Training set sizes as percentages
    training_sizes = np.arange(10, 100, 10)  # 10%, 20%, ..., 90%
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(training_sizes, avg_nbc_errors, marker='o', label='Naive Bayes Classifier', capsize=5)
    plt.errorbar(training_sizes, avg_lr_errors, marker='s', label='Logistic Regression (C=5)', capsize=5)
    
    plt.xlabel('Training Set Size (% of available training data)')
    plt.ylabel('Average Classification Error')
    plt.title(f'Learning Curves: NBC vs LR on Iris Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(5, 95)
    plt.ylim(0, max(max(avg_nbc_errors), max(avg_lr_errors)) * 1.1)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('learning_curves_iris.pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'learning_curves_iris.pdf'")
    plt.close()  # Close the figure to free memory
    
    # Print results
    print("\nResults Summary:")
    print("Training Size | NBC Error | LR Error")
    print("-" * 50)
    for i, size in enumerate(training_sizes):
        print(f"{size:11}% | {avg_nbc_errors[i]:.4f} | {avg_lr_errors[i]:.4f}")
    
    return

compare_classifiers_iris()
