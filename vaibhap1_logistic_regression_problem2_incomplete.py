import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

### Helper Functions ###

def logistic_fn(z):
    return 1 / (1 + np.exp(-z))

def load_data_pairs(type_str):
    return pd.read_csv("mnist_2s_and_6s/"+type_str+"_x.csv").values, pd.read_csv("mnist_2s_and_6s/"+type_str+"_y.csv").values

def run_log_reg_model(x, beta):
    theta_hats = logistic_fn(x.dot(beta))
    # To guard agains log(0) due to rounding error
    theta_hats[theta_hats == 0] = 0.001
    theta_hats[1-theta_hats == 0] = .999
    return theta_hats

def calc_log_likelihood(x, y, beta):
    theta_hats = run_log_reg_model(x, beta)
    
    ### Return an average, not a sum!
    ### TO DO ###
    m = theta_hats.shape[0]
    neg_log_loss = (y*np.log(theta_hats) + (1-y)*np.log(1.0-theta_hats)).mean()
    return neg_log_loss

def calc_accuracy(x, y, beta):
    theta_hats = run_log_reg_model(x, beta)
    ### TO DO ###
    preds = theta_hats >= 0.5
    m = y.shape[0]
    return (preds == y).mean()

# ==================== Additional Code =======================
def get_batch(x, y, sz, num):
    if num != x.shape[0]/sz:
        x_batch = x[sz*num : sz*(num+1),:]
        y_batch = y[sz*num : sz*(num+1),:]
    else:
        # final batch
        x_batch = x[sz*num :, :]
        y_batch = y[sz*num : ]
    return x_batch, y_batch




if __name__ == "__main__":

    ### Load the data
    train_x, train_y = load_data_pairs("train")
    valid_x, valid_y = load_data_pairs("valid")
    test_x, test_y = load_data_pairs("test")

    # add a one for the bias term                                                                                                                                                 
    train_x = np.hstack([train_x, np.ones((train_x.shape[0],1))])
    valid_x = np.hstack([valid_x, np.ones((valid_x.shape[0],1))])
    test_x = np.hstack([test_x, np.ones((test_x.shape[0],1))])

    ### Initialize model parameters
    beta = np.random.normal(scale=.001, size=(train_x.shape[1],1))
    
    ### Set training parameters
    learning_rates = [1e-3, 1e-2, 1e-1]
    batch_sizes = [train_x.shape[0]]
    max_epochs = 250
    
    ### Iterate over training parameters, testing all combinations
    valid_ll = []
    valid_acc = []
    all_params = []
    all_train_logs = []

    for lr in learning_rates:
        for bs in batch_sizes:
            ### train model
            final_params, train_progress = train_logistic_regression_model(train_x, train_y, beta, lr, bs, max_epochs)
            all_params.append(final_params)
            all_train_logs.append((train_progress, "Learning rate: %f, Batch size: %d" %(lr, bs)))
    
            ### evaluate model on validation data
            valid_ll.append( calc_log_likelihood(valid_x, valid_y, final_params) )
            valid_acc.append( calc_accuracy(valid_x, valid_y, final_params) )

    ### Get best model
    best_model_idx = np.argmax(valid_acc)
    best_params = all_params[best_model_idx]
    test_ll = calc_log_likelihood(test_x, test_y, best_params) 
    test_acc = calc_accuracy(test_x, test_y, best_params) 
    print "Validation Accuracies: "+str(valid_acc)
    print "Test Accuracy: %f" %(test_acc)

    ### Plot 
    plt.figure()
    epochs = range(max_epochs)
    for idx, log in enumerate(all_train_logs):
        plt.plot(epochs, log[0], '--', linewidth=3, label="Training, "+log[1])
        plt.plot(epochs, max_epochs*[valid_ll[idx]], '-', linewidth=5, label="Validation, "+log[1])
    plt.plot(epochs,  max_epochs*[test_ll], '*', ms=8, label="Testing, "+all_train_logs[best_model_idx][1])

    plt.xlabel(r"Epoch ($t$)")
    plt.ylabel("Log Likelihood")
    plt.ylim([-.8, 0.])
    plt.title("MNIST Results for Various Logistic Regression Models")
    plt.legend(loc=4)
    plt.show()
