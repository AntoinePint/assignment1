import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import os
print(os.getcwd())
os.chdir('c:/Users/antoi/Documents/assignment1')

def load_and_normalize_data():
    # load the numpy arrays inputs and labels from the data folder
    y = np.array(pd.read_csv('data/labels.txt', sep=" ", header=None))
    X = np.array(pd.read_csv('data/inputs.txt', sep=" ", header=None))

    # normalize the target y
    y=y/y.std()
    y=y.reshape(y.shape[0],)
    # print(X.shape,y.shape)
    return X, y


def test_mean():
    A=np.array([[1,2,3,4],[1,1,1,1],[1,1,1,1]])
    print(A.mean(axis=0))

def data_summary(X, y):

    # return several statistics of the data
    # TODO
    X_mean = X.mean() #what is the physical meaning of these matrix norms?
    X_std = X.std()
    y_mean = y.mean()
    y_std = y.std()
    X_min = X.min()
    X_max = X.max()

    return {'X_mean': X_mean,
            'X_std': X_std, 
            'X_min': X_min, 
            'X_max': X_max, 
            'y_mean': y_mean, 
            'y_std': y_std}


def data_split(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=4)

    return X_train, X_test, y_train, y_test, X_validation, y_validation


def fit_linear_regression(X, y, lmbda=0.0, regularization=None):
    """
    Fit a ridge regression model to the data, with regularization parameter lmbda and a given
    regularization method.
    If the selected regularization method is None, fit a linear regression model without a regularizer.

    !! Do not fit the intersept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    # TODO: use the sklearn linear_model module
    if regularization==None:
        linear_regression = LinearRegression(fit_intercept = False)
        linear_regression.fit(X, y)
        w = linear_regression.coef_ # coefficients
        c = linear_regression.intercept_ # intercept
    elif regularization=='ridge':
        linear_regression=Ridge(alpha=lmbda,fit_intercept = False)
        linear_regression.fit(X, y)
        w = linear_regression.coef_ # coefficients
        c = linear_regression.intercept_ # intercept
    elif regularization=='lasso':
        linear_regression=Lasso(alpha=lmbda,fit_intercept = False)
        linear_regression.fit(X, y)
        w = linear_regression.coef_ # coefficients
        c = linear_regression.intercept_ # intercept


    return w, c


def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features) data
    w: 1D numpy array of shape (n_features,) coefficients
    c: float intercept

    Returns: 1D numpy array of shape (n_samples,)
    """
    # TODO
    y_pred = X@w+c
    return y_pred


def mse(y_pred, y):
    """
    Return the mean squared error between the predictions and the true labels.

    y_pred: 1D numpy array of shape (n_samples,)
    y: 1D numpy array of shape (n_samples,)

    Returns: float
    """
    # TODO
    err2=(y-y_pred)**2
    MSE = err2.mean()

    return MSE



def fit_predict_test(X_train, y_train, X_test, y_test, lmbda=0.0, regularization=None):
    """
    Fit a linear regression model, possibly with L2 regularization, to the training data.
    Record the training and testing MSEs.
    Use methods you wrote before

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbda: float, regularization parameter

    Returns: The coefficients and intercept of the fitted model, the training and testing MSEs in a dictionary.
    """

    w, c = fit_linear_regression(X_train,y_train,lmbda=lmbda,regularization=regularization)

    y_pred_train=predict(X_train,w,c)  
    mse_train=mse(y_train,y_pred_train)

    y_pred_test=predict(X_test,w,c)  
    mse_test=mse(y_test,y_pred_test)


    results = {
        'mse_train': mse_train,
        'mse_test': mse_test,
        'lmbda': lmbda,
        'w': w,
        'c': c,
    }

    return results


def plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization=None, filename=None):
    """
    Plot the training and testing MSEs against the regularization parameter alpha.
    Use the functions you just wrote.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    alphas: list of values, the dataset percentage to be checked (alpha=n/d)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """
    
    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here

    # plot mse as a function of size of X_train
    n,d=X_train.shape
    
    
    # TODO alpha liste de valeures en entrée 
    Results=pd.DataFrame(columns=['n_cut','mse_train','mse_test'])
    for n_cut in np.arange(n)[int(alphas[0]*d)+1:int(alphas[-1]*d)]:
        X_train_cut=X_train[:n_cut].copy()
        y_train_cut=y_train[:n_cut].copy()
        X_test_cut=X_test[:n_cut].copy()
        y_test_cut=y_test[:n_cut].copy()
        res=fit_predict_test(X_train_cut, y_train_cut, X_test_cut, y_test_cut, lmbda=lmbda, regularization=regularization)
        Results=pd.concat([Results,pd.DataFrame(columns=['n_cut','mse_train','mse_test'],data=[[n_cut/d,res['mse_train'],res['mse_test']]])],ignore_index=True)
    plt.plot(Results['n_cut'].values,Results['mse_train'].values,label='train')
    plt.plot(Results['n_cut'].values,Results['mse_test'].values,label='test')
    plt.xlabel('alpha=n/d')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    # plt.show()
    plt.clf()



def plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='ridge', filename=None):
    """
    Plot the coefficients of the fitted model against the regularization parameter alpha.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbdas: list of values, the regularization parameter
    plot_coefs: list of integers, the coefficients of w to be plotted
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """

    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here
    Results=pd.DataFrame(columns=['lmbda'])
    for i in plot_coefs:
        Results['w'+str(i)]=None
    for lmbda in lmbdas:
        # print(lmbda)
        res=fit_predict_test(X_train, y_train, X_test, y_test, lmbda=lmbda, regularization=regularization)
        New_line=pd.DataFrame(columns=['lmbda'],data=[res['lmbda']])
        for i in plot_coefs:
            #print('w'+str(i))
            New_line['w'+str(i)]=res['w'][i]
        # print(res['w'][i])
        Results=pd.concat([Results,New_line],ignore_index=True)
    # print(Results)
    for i in plot_coefs:
        plt.plot(Results['lmbda'].values,Results['w'+str(i)].values,label='w'+str(i))
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()

def add_poly_features(X):
    """
    Add squared features to the data X and return a new vector X_poly that contains
    X_poly[i,j]   = X[i,j]
    X_poly[i,2*j] = X[i,j]^2

    X: 2D numpy array of shape (n_samples, n_features)

    Returns: 2D numpy array of shape (n_samples, 2 * n_features ) with the normal and squared features
    """

    # TODO
    X2=np.square(X)
    X_poly = np.concatenate((X, X2), axis=1)

    return X_poly

def optimize_lambda(X_train,X_test,X_validation,y_train,y_test,y_validation, lmbdas,filename):
    """
    Optimize the regularization parameter lambda for the training data for ridge over the validation error and plot the validation error against the parameter lambda
    Show the best parameter lambda and the corresponding test error (not validation error!).

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    X_validation: 2D numpy array of shape (n_validation_samples, n_features)
    y_validation: 1D numpy array of shape (n_validation_samples,)
    lmbdas: list of values, the regularization parameter, the bounds of the optimization range

    Returns: The best regularization parameter and the corresponding test error (!= validation error).
    """

    regularization='ridge'

    # TODO: Experiment code goes here
    # for each lamdba do a fit and calculate the mse on the validation samples
    # return the best one
    err_val=[]
    for lmbda in lmbdas:
         res=fit_predict_test(X_train, y_train, X_validation, y_validation, lmbda=lmbda, regularization=regularization)
         err_val.append(res['mse_test']) # nb this is the validation mse since we pass X_validation as an argument

    best_val_mse = np.array(err_val).min()
    best_lmbda = lmbdas[err_val.index(best_val_mse)]

    best_test_mse=fit_predict_test(X_train, y_train, X_test, y_test, lmbda=best_lmbda, regularization=regularization)['mse_test']

    # TODO: Plotting code goes here
    plt.plot(lmbdas,err_val)
    plt.axvline(best_lmbda, color='red', label='Best Lambda')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()

    return best_test_mse, best_lmbda


if __name__ == "__main__":

    """ 
    !!!!! DO NOT CHANGE THE NAME OF THE PLOT FILES !!!!!
    They need to show in the Readme.md when you submit your code.

    It is executed when you run the script from the command line.
    'conda activate ml4phys-a1'
    'python main.py'

    This already includes the code for generating all the relevant plots.
    You need to fill in the ...
    """

    ## Exercise 1.
    # Load the data
    X, y = load_and_normalize_data()
    print("Successfully loaded and normalized data.")
    data_summary(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = data_split(X, y)
    n, d = X_train.shape

    w_ridge,c_ridge=fit_linear_regression(X_train,y_train,lmbda=1,regularization='ridge')
    y_pred=predict(X_test,w_ridge,c_ridge)  
    mse(y_test,y_pred)

    ## Exercise 4.
    # Plot the learning curves
    print('Plotting dataset size vs. mse curve ...')
    alphas = [2,6] 
    lmbda = 0
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=0')

    print('Plotting dataset size vs. mse curve for L2 ...')
    alphas = [0,2.5]
    lmbda = 0.001
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=001')
    alphas = [0,2.5]
    lmbda = 10.0
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=10')

    ## Exercise 5.
    print('Plotting regularizer vs. coefficient curve...')
    step = 1e1
    lmbdas = np.arange(0,1000+step,step)
    plot_coefs = [0,3,7,8]
    # plot_coefs = np.arange(10)
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='ridge', filename='regularizer_vs_coefficients_Ridge')


    ## Exercise 6.
    print('Find the optimal parameters for the Ridge regression...')
    step=1e-2
    lmbdas = np.arange(0,1e2,step)
    n_ = 80
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='optimal_lambda_ridge_n50')
    print(n_, lmbda, gen_error)
    n_ = 150
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='optimal_lambda_ridge_n150')
    print(n_, lmbda, gen_error)
    #we found a better lambda with a smaller testing error when taking a larger training set

    ## Exercise 7.
    X_train_poly = add_poly_features(X_train) 
    X_test_poly = add_poly_features(X_test) 
    print(X_train_poly.shape,X_train.shape)
    step=1e-2
    lmbdas = np.arange(1e-4+step,1.2,step)
    plot_coefs = [0,2,6,7]

    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='lasso', filename='regularizer_vs_coefficients_LASSO')
    
    plot_coefs = [0,2,6,7,129,154]
    plot_regularizer_vs_coefficients(X_train_poly, y_train, X_test_poly, y_test, lmbdas,  plot_coefs, regularization='lasso', filename='regularizer_vs_coefficients_LASSO_polyfeat')

    # print('Done. All results saved in the results folder.')

