

# #############################################################################
    
# Table of Content

# #############################################################################

# Libraries
# General Functions
# Portfolio optimization related functions
# Garlappi Uppal Wang Functions
# Risk Parity
# Hierarchical Risk Parity
# Risk Metrics
# Plotting Functions


# #############################################################################

'''Library of Functions for Portfolio Calculations'''



# numpy
import numpy as np
from numpy.polynomial import Polynomial as poly
import numpy.polynomial.polynomial as poly

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

# scipy
from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean, winsorize
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet

# others
import getpass as gp
import sys
import os
import pandas as pd
import random
import math
import seaborn as sns
from collections import OrderedDict
import sklearn.covariance
from sklearn.covariance import ledoit_wolf as LW, oas,  shrunk_covariance
import cvxopt as opt
from cvxopt import blas, solvers
import empyrical as ep
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# General Attributes

name = gp.getuser()

np.random.seed(123)

solvers.options['show_progress'] = False


# #############################################################################
    
# General Functions

# #############################################################################


def get_Data(freq, years):
    #    globals() [returns, rf_rate, estLength, nAssets]
    #change directory
    os.chdir("/Users/%s/OneDrive/Master Thesis/Data" %name)
    #import from excel file with worksheet name Return_Data
    file = 'MSCI_8_Indices.xlsx'
    xl = pd.ExcelFile(file)
    returns = xl.parse('Return_Data_%s' %freq,index_col = 'Code')#, parse_dates = True)
    xl = pd.ExcelFile('TBill_Data.xlsx')
    rf_rate = xl.parse('3M_US_%s' %freq, index_col = 'Code')
    if freq == 'M':
        multiplier = 12
    elif freq == 'W':
        multiplier = 52
    elif freq == 'D':
        multiplier = 250
    else:
        print('No correct input for frequency given!!!')
    estLength = years * multiplier
    nAssets = len(returns.columns)
    
    market_name = 'MSCI_World.xlsx'
    xl_market = pd.ExcelFile(market_name)
    market = xl_market.parse('returns_%s' %freq, index_col = 'Code')
    
    return returns, rf_rate, market, estLength, nAssets


#define function for a vector of n random weights that sum up to 1
def rand_weights(n):
    k = np.random.rand(n)
    return k / sum(k)

#define function to calculate volatility based on weights

def PF_variance(w, S):
    if len(w.shape)==1:
        w = np.asmatrix(w).T
    pf_var = np.dot(w.T, np.dot(S, w))
    return float(pf_var)

def PF_volatility(w, S):
    S_PF = np.sqrt(PF_variance(w,S))
    return S_PF

#define function to calculate mean portfolio return
def PF_return(w, r):
    if len(w.shape)==1:
        w = np.asmatrix(w).T
    if len(r.shape) == 1:
        r = np.asmatrix(r).T
    R_PF = np.dot(w.T, r)
    return float(R_PF)

def muRet(returns):
    '''Calculate meean return of return data'''
    muRet = np.array(returns.mean(axis = 0))
    return muRet

def varCovar(returns):
    '''Calculate variance covariance matrix'''
    varCov = np.asmatrix(np.cov(returns.T, ddof=1))
    return varCov


def st_dev_MV_pf(meanRet, varCovar):
    e_ret_pf = np.array([0.00001 * x for x in range(4000)])
    varA = np.float(var_A(meanRet, varCovar))
    varB = np.float(var_B(meanRet, varCovar))
    varC = np.float(var_C(varCovar))
    varD = varA * varC - varB ** 2
    return np.sqrt((varC * e_ret_pf ** 2 - 2 * varB * e_ret_pf + varA) / varD)

def portfolioReturnOutOfSample(weights, returns):
    if not isinstance(weights, np.ndarray):
        weights = np.asmatrix(weights).T
    '''calculates the performance of the portfolio, given the returns provided'''
    retAssets = np.multiply(weights.T, np.exp(returns) - 1)
    return retAssets.sum()


def convertCSV_toDataframe(name, sep = ',', index_col = 0):
    
    df = pd.read_csv(name, sep = ',', index_col = 0)
    dates = [datetime.strptime(df.index.values[i], "%Y-%m-%d") for i in range(len(df.index.values))]
    indices = df.columns.values.tolist()
    
    return pd.DataFrame(df.values, index = dates, columns = indices)

 
#define function to calculate matrix inverse
def mat_inv(a):
    inv = np.linalg.inv(a)
    return inv


#define a boxing function to calculate x'Sx

def box(x, S, y):
    box = np.dot(x.T, np.dot(S, y))
    return box

#define function to create random positive definite correlation matrix

def randCorrelationMat(nAssets, a): 
    '''a determines around which value correlation will be centered'''
    randomCorrelation = np.matrix([np.random.randn(nAssets) + np.random.randn(1)*a for i in range(nAssets)])
    A = randomCorrelation * randomCorrelation.T
    D_half = np.diag(np.diag(A)**(-0.5))
    randomCorrelation = D_half*A*D_half
    return randomCorrelation

def correlation_matrix(returns):
    return np.corrcoef(returns.T)

def cov2cor(X):
    D = np.zeros_like(X)
    d = np.sqrt(np.diag(X))
    np.fill_diagonal(D, d)
    DInv = np.linalg.inv(D)
    R = np.dot(np.dot(DInv, X), DInv)
    return R

def cov_robust(X):
    oas = sklearn.covariance.OAS()
    oas.fit(X)
    return pd.DataFrame(oas.covariance_, index=X.columns, columns=X.columns).values
    
def corr_robust(X):
    cov = cov_robust(X).values
    shrunk_corr = cov2cor(cov)
    return pd.DataFrame(shrunk_corr, index=X.columns, columns=X.columns)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)





# #############################################################################
    
# Portfolio optimization related functions

# #############################################################################



#define function to calculate A,B,C,D (Munk)
def var_all(mu,Sigma):
    varall = np.array([var_A(mu, Sigma),
                      var_B(mu, Sigma),
                      var_C(Sigma),
                      var_D(mu, Sigma)])
    return varall

def var_A(mu, Sigma):
    if len(mu.shape)==1:
        mu = np.asmatrix(mu).T
    varA = np.dot(mu.T, np.dot(mat_inv(Sigma), mu))
    return float(varA)

def var_B(mu, Sigma):
    if len(mu.shape)==1:
        mu = np.asmatrix(mu).T
    ones = np.asmatrix(np.ones(Sigma.shape[0])).T
    varB = np.dot(ones.T, np.dot(mat_inv(Sigma), mu))
    return float(varB)

def var_C(Sigma):
    ones = np.asmatrix(np.ones(Sigma.shape[0])).T
    varC = np.dot(ones.T, np.dot(mat_inv(Sigma), ones))
    return float(varC)

def var_D(mu, Sigma):
    varA = var_A(mu, Sigma)
    varB = var_B(mu, Sigma)
    varC = var_C(Sigma)
    varD = varA * varC - varB ** 2
    return float(varD)

def maxSlopePF(returns):
    meanRet = returns.mean(axis=0)
    varCovar = np.cov(returns.T, ddof=1)
    maxSlopeWeights = (1 / var_B(meanRet, varCovar)
                             * np.dot(mat_inv(varCovar), meanRet))
    return maxSlopeWeights 

def maxSlopePF1(meanRet, varCovar):
    maxSlopeWeights = np.array(np.multiply(1 / var_B(meanRet, varCovar),
                              np.dot(mat_inv(varCovar), meanRet)))
    return maxSlopeWeights 

def minVarPF(returns):
    estSigma = np.cov(returns.T, ddof=1)
    oneVector = np.ones(len(returns.index))
    weights = 1 / (var_C(estSigma)) * np.dot(mat_inv(estSigma), oneVector)
    return weights

def minVarPF1(varCovar):
    oneVector = np.ones(varCovar.shape[0])
    weights = np.array(np.multiply(1 / (var_C(varCovar)), np.dot(mat_inv(varCovar), oneVector)))
    return weights.T

def maxSRPF(returns, rf):
    oneVector = np.asmatrix(np.ones(returns.shape[0])).T
    meanRet = returns.mean(axis = 0)
    varCovar = np.cov(returns.T, ddof = 1)
    varB = var_B(meanRet, varCovar)
    varC = var_C(varCovar)
    weights = np.array(np.float(1 / (varB - varC * rf))* 
                       np.dot(mat_inv(varCovar), 
                       (meanRet - rf)))
    return weights

def maxSRPF1(meanRet, varCovar, rf):
    rf = np.float(rf)
    varB = np.float(var_B(meanRet, varCovar))
    varC = np.float(var_C(varCovar))
    weights = np.array(np.float(1 / (varB - varC * rf))* 
                       np.dot(mat_inv(varCovar), 
                       np.array(meanRet - (rf))).T)
    return weights.T

def tangWeights(meanExcRet, varCovar, rf, gamma):
    varB = var_B(meanExcRet, varCovar)
    varC = var_C(varCovar)
    weights = 1 / (gamma) * np.dot(mat_inv(varCovar), meanExcRet)
    return weights

def maxSlopePortfolio(meanRet, varCovarMatrix):
    return 1 / var_B(meanRet, varCovarMatrix) * np.dot(mat_inv(varCovarMatrix), meanRet)  

def meanVarPF(meanRet, varCovar, m_bar):
    A = var_A(mu, Sigma)
    B = var_B(mu, Sigma)
    C = var_C(Sigma)
    D = varA * varC - varB ** 2
    inv = mat_inv(varCovar)
    ones = np.asmatrix(np.ones(len(meanRet))).T
    pi = (C * m_bar - B) / D * np.dot(inv, meanRet) + (A - B * m_bar) / D * np.dot(inv, ones)
    return pi



# #############################################################################
    
# Garlappi, Uppal, Wang Portfolio Optimization

# #############################################################################
    
def optSigma(returns, epsilon, gamma):
    '''needs to be called with from numpy.polynomial import Polynomial as poly '''
    '''calculate the root of the polynomial so that an optimal portfolio variance exists'''
    '''Input parameters: array of return data, ambiguity aversion = epsilon, risk aversion = gamma'''
    mu = np.mean(returns, axis = 0)
    Sigma = np.cov(returns.T, ddof= 1)
    T = returns.shape[0]
    N = returns.shape[1]
    varepsilon = epsilon * ((T - 1) * N)/(T * (T - N))
    C = var_C(Sigma)
    B = var_B(mu, Sigma)
    A = var_A(mu, Sigma)
    
    equation = np.array([- varepsilon, - 2 * gamma * np.sqrt(varepsilon), 
          (C * varepsilon - A * C + B ** 2 - gamma ** 2), 
          + 2 * C * gamma * np.sqrt(varepsilon), 
          C * gamma ** 2])    
    roots = poly.polyroots(equation)
    optsigma = np.real(np.extract(roots > 0 , roots)[0])
    check = sum(x > 0 for x in roots)
    if check > 1:
        print("Polynomial yields more than one Solution")
    elif check == 1:
        return optsigma
    else: 
        print("Polynomial does not yield any positive solution")


    
    
def GWweights(returns, epsilon, gamma):
    '''calculates the optimal portfolio weights under Garlappi Wang assumptions'''
    sigmap = optSigma(returns, epsilon, gamma)
    mu = muRet(returns)
    Sigma = varCovar(returns)
    invSigma = mat_inv(Sigma)
    T, N = returns.shape
    varepsilon = epsilon * ((T - 1) * N)/(T * (T - N))
    B = var_B(mu, Sigma)
    C = var_C(Sigma)
    ones = np.ones(np.sqrt(Sigma.size).astype(np.int))
    pi =    (np.dot(((1. / gamma) * invSigma) ,((1. / (1 + np.sqrt(varepsilon)/(gamma * sigmap)))
            * (mu - (B - gamma * (1 + np.sqrt(varepsilon)/(gamma * sigmap))) / C * ones))))
    return pi

def GWweights1(returns, muRet, varcovar, epsilon, gamma):
    '''calculates the optimal portfolio weights under Garlappi Wang assumptions'''
    sigmap = optSigma1(returns, muRet, varcovar, epsilon, gamma)
    invSigma = mat_inv(varcovar)
    T, N = returns.shape
    varepsilon = epsilon * ((T - 1) * N)/(T * (T - N))
    B = var_B(muRet, varcovar)
    C = var_C(varcovar)
    ones = np.ones(varcovar.shape[0])
    pi =    (np.dot(((1. / gamma) * invSigma) ,
                    ((1. / (1 + np.sqrt(varepsilon)/(gamma * sigmap)))
            * np.subtract(muRet, (B - gamma * (1 + np.sqrt(varepsilon)/(gamma * sigmap))) / C * ones)[:,0])))
    return pi
    
    
def optSigma1(returns, muRet, varcovar, epsilon, gamma):
    '''needs to be called with from numpy.polynomial import Polynomial as poly '''
    '''calculate the root of the polynomial so that an optimal portfolio variance exists'''
    '''Input parameters: array of return data, ambiguity aversion = epsilon, risk aversion = gamma'''

    T = returns.shape[0]
    N = returns.shape[1]
    varepsilon = epsilon * ((T - 1) * N)/(T * (T - N))
    
    C = var_C(varcovar)
    B = var_B(muRet, varcovar)
    A = var_A(muRet, varcovar)
    
    equation = np.array([- varepsilon, - 2 * gamma * np.sqrt(varepsilon), 
          (C * varepsilon - A * C + B ** 2 - gamma ** 2), 
          + 2 * C * gamma * np.sqrt(varepsilon), 
          C * gamma ** 2])    
    roots = poly.polyroots(equation)
    optsigma = np.real(np.extract(roots > 0 , roots)[0])
    check = sum(x > 0 for x in roots)
    if check > 1:
        print("Polynomial yields more than one Solution")
    elif check == 1:
        return optsigma
    else: 
        print("Polynomial does not yield any positive solution")





# #############################################################################
    
# Risk Parity Portfolio

# #############################################################################

from scipy.optimize import minimize

def riskParity(varCov):
    #calculates the risk parity portfolio weights
    '''set constraints for optimization'''
    cons = ({'type' : 'eq', 'fun' : weight_constraint },
            {'type': 'ineq', 'fun': long_only_constraint})
    '''set input parameters for optimization'''
    nAssets = varCov.shape[0]
    x_t = np.ones(nAssets)/nAssets #equal risk contribution target vector
    w_0 = rand_weights(nAssets) #initial weights from which to start opimization
    #variance covariance matrix
    res = minimize(risk_objective,
                   w_0,
                   args=[varCov, x_t],
                   method = 'SLSQP',
                   constraints = cons,
                   options={'ftol': 1e-12, 'maxiter' : 400, 'disp' : False})    
    w_RP = np.asmatrix(res.x).T
    return w_RP


def risk_contribution(weights, varCov):
    '''function that calculates the asset contribution to total risk'''
    #transform the weights list into a column vector, if the weights are a simple 1D list (otherwise transpose not allowed)
    if not isinstance(weights, np.ndarray):
        weights = np.asmatrix(weights).T        

    sigmapf = np.sqrt(np.dot(weights.T ,np.dot(varCov, weights)))
    '''marginal risk contribution of assets'''
    MRC = np.dot(varCov, weights)
    '''Risk Contribution'''
    RC = np.multiply(weights, MRC) / (sigmapf)
    return RC


def risk_objective(x, args):
    '''objective function to be minimized'''
    Variance = args[0]
    x_t = np.asmatrix(args[1]).T
    #problem is sigmapfd
    x = np.asmatrix(x).T
    sigmapf = float(np.sqrt(np.dot(x.T , np.dot(Variance, x))))
    risk_target = np.multiply(sigmapf, x_t)
    #problem is this
    diff = risk_contribution(x, Variance) - risk_target
    opt_pi = [i**2 for i in diff]
    J = np.multiply(100000 , sum(opt_pi))
    return float(J)

def weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x


# #############################################################################
    
# Asset allocation in a downside risk setting (Harlow, 1991)
# LPM portfolio optimization
# #############################################################################

def lpm_port(estMu, corrReturns):
    global exp_ret_chosen_LPM
    global exprets_LPM
    exprets_LPM = estMu
    exp_ret_chosen_LPM = 0.02 / 12    
    nAssets = len(estMu)
    def expected_return_constraint_LPM(x):
        if len(x.shape)==1:
            x = np.asmatrix(x).T
            constraint = np.dot(x.T, exprets_LPM) - exp_ret_chosen_LPM
        return float(constraint)
    def LPM_PF_optimization(x, args):
        returns = args[1]
        nAssets = returns.shape[1]
        L_matrix = np.zeros((nAssets, nAssets),float)
        '''create the LPM for comovements in several assets to add to the rest of the matrix'''
        for i in range(nAssets):
            for j in range(nAssets):
                if isinstance(returns, pd.DataFrame):
                    L_matrix[i,j] = CO_LowerPartialMoments(returns[indices[i]],returns[indices[j]])
                elif isinstance(returns, np.ndarray):
                    L_matrix[i,j] = CO_LowerPartialMoments(returns[:,i],returns[:,j])
        return PF_variance(x, L_matrix)
    w0 = np.random.rand(nAssets)
    cons = ({'type' : 'eq', 'fun' : weight_constraint},
                    {'type' : 'eq', 'fun' : expected_return_constraint_LPM})
    optimization = minimize(LPM_PF_optimization, 
                            w0, 
                            args = [exp_ret_chosen_LPM, corrReturns],
                            method = 'SLSQP', 
                            constraints = cons,
                            options={'ftol': 1e-12, 'maxiter' : 45, 'disp' : False})
    PF_weights_LPM = np.asmatrix(optimization.x).T
    return PF_weights_LPM
    
#
##function to minimize
#def LPM_PF_optimization(x, args):
#
#    returns = args[1]
#    nAssets = returns.shape[1]
#    L_matrix = np.zeros((nAssets, nAssets),float)
#    '''create the LPM for comovements in several assets to add to the rest of the matrix'''
#    for i in range(nAssets):
#        for j in range(nAssets):
#            if isinstance(returns, pd.DataFrame):
#                L_matrix[i,j] = CO_LowerPartialMoments(returns[indices[i]],returns[indices[j]])
#            elif isinstance(returns, np.ndarray):
#                L_matrix[i,j] = CO_LowerPartialMoments(returns[:,i],returns[:,j])
#    
#    return PF_variance(x, L_matrix)
#
##constraint1
#def expected_return_constraint_LPM(x):
#    if len(x.shape)==1:
#        x = np.asmatrix(x).T
#    constraint = np.dot(x.T, exprets_LPM) - exp_ret_chosen_LPM
#    return float(constraint)
#constraint2
# see method weight_constraint in Risk Parity Portfolio

# #############################################################################
    
# Hierarchical Risk Parity Portfolio (Lopez de Prado)

# #############################################################################



#to do 1
def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

#to do 2
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov = pd.DataFrame(cov)
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index

    return sortIx.tolist()


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5  # distance matrix
    where_are_NaNs = np.isnan(dist)
    dist[where_are_NaNs] = 0;
    return dist


def generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    # 1) generate random uncorrelated data
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    # each row is a variable
    # 2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in xrange(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
    # 4) add specific random shock
    point = np.random.randint(sLength, nObs - 1, size=2)

    x[point, cols[-1]] = np.array([-.5, 2])

    return x, cols


def plotCorrMatrix(filename,corr,labels=None):
# Heatmap of the correlation matrix
    if labels is None:labels=[]
    plt.pcolor(corr)
    plt.colorbar()
    plt.yticks(np.arange(.5,corr.shape[0]+.5),labels) 
    plt.xticks(np.arange(.5,corr.shape[0]+.5),labels) 
    plt.savefig(filename)
    plt.clf();plt.close() # reset pylab
    return


def getCLA(cov, **kargs):
    # Compute CLA's minimum variance portfolio
    mean = np.arange(cov.shape[0]).reshape(-1, 1)
    # Not used by C portf
    lB = np.zeros(mean.shape)
    uB = np.ones(mean.shape)
    cla = CLA(mean, cov, lB, uB)
    cla.solve()
    return cla.w[-1].flatten()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[int(j):int(k)] for i in cItems for j, k in ((0, len(i) / 2),
                                                      (len(i) / 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def getHRP(cov):
    corr = cov2cor(cov)
    # Construct a hierarchical portfolio: returns the weights of the portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    # recover labels
    hrp = getRecBipart(cov, sortIx)
    weights = np.asmatrix(hrp.sort_index()).T
    return weights



#MONTE CARLO SIMULATION 
    
def hrpMC(numIters=10000, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2,
          sigma1F=.25, sLength=260, rebal=22):
    # Monte Carlo experiment on HRP
    methods = {'getHRP': getHRP, 'getIVP': getIVP, 'getCLA': getCLA}
    stats = {k: pd.Series() for k in methods.keys()}
    
    pointers = range(sLength, nObs, rebal)
    for numIter in xrange(int(numIters)):
        # print numIter
        # 1) Prepare data for one experiment
        x, cols = generateData(nObs, sLength, size0,
                               size1, mu0, sigma0, sigma1F)
        r = pd.DataFrame(columns=[methods.keys()],
                         index=range(sLength, nObs))#{i.__name__: pd.Series() for i in methods}
        #print r
        # 2) Compute portfolios in-sample
        for pointer in pointers:
            x_ = x[pointer - sLength:pointer]
            cov_ = np.cov(x_, rowvar=0, ddof=1)
            corr_ = np.corrcoef(x_, rowvar=0)
            # 3) Compute performance out-of-sample
            x_ = x[pointer:pointer + rebal]
            for name, func in methods.iteritems():
                w_ = func(cov=cov_, corr=corr_)
                # callback
                #r_ = pd.Series(np.dot(x_, w_))
                #print r[name].append(r_)
                #print pointer
                r.loc[pointer:pointer + rebal - 1, name] = np.dot(x_, w_)

        # 4) Evaluate and store results
        for name, func in methods.iteritems():
            r_ = r[name].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[name].loc[numIter] = p_.iloc[-1] - 1  # terminal return

    # 5) Report results
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    # stats.to_csv('stats.csv')
    df0, df1 = stats.std(), stats.var()
    print(pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1))
    return stats




# #############################################################################

# Risk Metrics

# #############################################################################


def SharpeRatio(returns, rf):
    d = returns - rf
    mean_d = np.mean(d)
    stdev_d = np.std(d, ddof=1)
    SR = mean_d / stdev_d
    return SR

def BetaRegression(returns, marketReturns, rf):
    X = marketReturns - rf
    X1 = sm.add_constant(X)
    y = returns - rf
    model = sm.OLS(y, X1)
    results = model.fit()
    print(results.summary())
    return 


def TreynorRatio(returns, marketReturns):
    X = marketReturns
    y = returns
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    return intercept / slope

def TreynorRatio_Diego(retPF, marketReturns, rf_vect):
    retPF_cumulative_end_period = np.prod( 1 + np.array(retPF) ) - 1
    rf_period = np.prod( 1 + np.array(rf_vect) ) - 1
    result = (retPF_cumulative_end_period - rf_period) / beta(retPF, marketReturns)
    return result[0]
 

def beta(returns, market):
    # Create a matrix of [returns, market]
    if len(returns.shape)==1:
        returns = np.asmatrix(returns).T
    if len(market.shape)==1:
        market = np.asmatrix(market).T
    m = np.concatenate((returns, market), axis = 1)
    # Return the covariance of m divided by the standard deviation of the market returns
    return np.cov(m, ddof=1)[0][1] / np.std(market)

def InformationRatio(returns, marketReturns):
    X = marketReturns
    y = returns
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    return intercept / slope

def LowerPartialMoments(returns, target = 0, order = 2):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return target
    # order = 1 for target shortfall
    # order = 2 for target semivariance
    target_array = np.empty(len(returns))
    target_array.fill(target)
    # Calculate the difference between the threshold and the returns
    diff = target_array - returns
    # Set the minimum of each to 0
    diff = np.clip(diff, a_min = 0, a_max = None)
    # Return the mean of the n-th powered difference
    return np.sum(diff ** order) / len(returns)


def lpm_test_equivalence_diego(returns, target = 0, order = 2):
    
    # This method is to check that the previous method LowerPartialMoments, yields the same result also by having
    # the np.mean() in the return statement
    # Create an array he same length as returns containing the minimum return target
    # order = 1 for target shortfall
    # order = 2 for target semivariance
    target_array = np.empty(len(returns))
    target_array.fill(target)
    # Calculate the difference between the threshold and the returns
    diff = target_array - returns
    # Set the minimum of each to 0
    diff = np.clip(diff, a_min = 0, a_max = None)
    aux = diff ** order
    # Return the sum of the different to the power of order
    return np.mean(aux)


def CO_LowerPartialMoments(returnA, returnB, wrt = None, target = 0, order = 2):
    '''wrt means WITH RESPECT TO, and if somethig is assigned to wrt, is stated, than the comovement calculated
    will be with respect to returnB, otherwise, the comovements will be always calculated with respect to returnA'''
    target_array = np.empty(len(returnA))
    target_array.fill(target)    
    # Calculate the difference between the threshold and the returns
    diff_A = target_array - returnA
    diff_B = target_array - returnB

    if wrt is not None:
        # Set the minimum of each to 0
        diff_clipped = np.clip(diff_B, a_min = 0, a_max = None)
        deviation_from_target = diff_A
    else:
        diff_clipped = np.clip(diff_A, a_min = 0, a_max = None)
        deviation_from_target = diff_B
    
    product_of_deviations = np.multiply(diff_clipped, deviation_from_target)
    return np.mean(product_of_deviations)

def CO_LowerPartialMoments_approximation(returns, target = 0, order = 2):
    corr_mat = correlation_matrix(returns)
    nAssets = len(returns.columns)
    result_m = np.zeros((nAssets, nAssets))

    for i in range(nAssets):
        for j in range(nAssets):
            prod = np.multiply(LowerPartialMoments(returns[returns.columns[i]], target = target), LowerPartialMoments(returns[returns.columns[j]], target = target)) ** (1 / order)
            result_m[i,j] = np.multiply(prod, corr_mat[i,j])
    return result_m

def LPM_matrix(returns, target=0):
    # returns a matrix of the CO-lower partial moments  
    nAssets = len(returns.columns)
    L_matrix = np.zeros((nAssets, nAssets))
    for i in range(nAssets):
        for j in range(nAssets):
            L_matrix[i,j] = CO_LowerPartialMoments(returns[returns.columns[i]], returns[returns.columns[j]], target = target)
    return L_matrix   

    
def LowerPartialMoment_of_portfolio(weights, returns, target = 0, order = 2):
    '''create the LPM for comovements in several assets to add to the rest of the matrix'''
    L_matrix = LPM_matrix(returns)
    return PF_variance(weights, L_matrix)      

def DrawDown(returns, time_span = 60):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - time_span
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)

 
def MaximumDrawDown(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = DrawDown(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)
    

def AverageDrawDown(returns, periods = 60):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = DrawDown(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


#def AverageDrawDown_Squared(returns, periods = 60):
#    # Returns the average maximum drawdown squared over n periods
#    drawdowns = []
#    for i in range(0, len(returns)):
#        drawdown_i = math.pow(dd(returns, i), 2.0)
#        drawdowns.append(drawdown_i)
#    drawdowns = sorted(drawdowns)
#    total_dd = abs(drawdowns[0])
#    for i in range(1, periods):
#        total_dd += abs(drawdowns[i])
#    return total_dd / periods


def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (np.exp(returns[i])))
    return np.array(s)

def SortinoRatio(portfolioReturn, returns, rf, target=0):
    #the denominator is the square root of the mean of the squared deviations 
    #of the returns from the target return provided, standard is 0%
    return (portfolioReturn - rf) / math.sqrt(LowerPartialMoments(returns, target, 2))

def OmegaRatio(portfolioReturn, returns, rf, target=0):
    #standard target = 0, assumes that the target is a the 0% return. 
    # note: LowerPartialMoments with an "order"=1, like in this case, calculates the target shortfall, hence
    # the average deviations below the target provided.
    return (portfolioReturn - rf) / LowerPartialMoments(returns, target, 1)

def ValueAtRisk(returns, alpha = 0.05):
    # This method calculates the historical simulation var of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])
 
    '''Integer must be rounded down always or not?'''
    
    
def EconomicCapital_N(portfolio_st_dev, alpha, initial_capital = 100):
    ''' Calculates the Economic Capital, based on normal distribution assumption '''
    return initial_capital * portfolio_st_dev * abs(stats.norm.ppf(alpha))
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])

def ExpectedShortfall(returns, alpha):
    # i am not very sure about the way this is calculated.
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)

def ValueAtRisk_N(returns, alpha):
    '''Function calculates VaR based on normal distribution assumption'''
    returns = np.array(returns)
    means = returns.mean(axis = 0)
    stdevs = returns.std(axis = 0, ddof = 1)
    return means - stdevs * ndtri(1 - alpha)

def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).
    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    Returns
    -------
    float
        tail ratio
    """

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))
        

def utility_MV(mu, sigma, gamma):
    return mu - 0.5 * gamma * sigma ** 2

def utility(estimatedWeights, gamma):
    '''give estimated weights as column vector'''
    '''meanRet and varCovar have to be calculated before already'''
    estMu = np.dot(estimatedWeights.T, meanRet)
    estSigma = np.sqrt(np.dot(estimatedWeights.T, np.dot(varCovar, estimatedWeights)))
    estUtility = np.float(utility_MV(estMu,estSigma,gamma))
    return estUtility




# #############################################################################
    
# Plotting Functioncs

# #############################################################################

# Visualization of weights over time
    
def plot_weights(x,y, name):
    plt.figure(figsize = (10, 10))
    plt.plot(x,y)
    plt.show
    plt.savefig('Weights%s.svg' %name)
    return

def plot_returns(x, name):
    plt.figure(figsize = (10, 10))
    plt.plot(x)
    plt.show
    plt.savefig('Weights%s.svg' %name)
    return


def plot_histogram(x, filename):
    '''Plots a histogram and saves it as svg'''
    plt.figure(figsize = (15, 10))
    plt.hist(x, 100)
    plt.savefig('histogram_%s.svg' %filename)
    plt.show()
    return

def plot_emphasize_periods(data_to_plot, period_to_highlight, name):
    '''data_to_plot: pd.Dataframe to plot
       name: str , the name of the file and title of the graph 
       period_to_highlight: list of lists. Each of these lists contain datetime.date elements
       
       EXAMPLE: plot_emphasize_periods(returns, "exampleName", stressed_period)'''
    plt.figure(figsize = (10, 5))
    plt.plot(data_to_plot)
    plt.title(str(name))
    for i in range(len(period_to_highlight)):
        plt.axvline(x = period_to_highlight[i][-1], linewidth=0.4, color = '#00C78C')
        plt.axvspan(period_to_highlight[i][0], period_to_highlight[i][-1], alpha=0.5, color='#FFE4B5')
    plt.savefig(str(name)+'.svg')
    plt.show()

    
    
# https://github.com/quantopian/pyfolio/blob/master/pyfolio/plotting.py
def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)
    
    fig, ax = plt.subplots(figsize = (6,9))
    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=colormap.RdYlGn,
        ax=ax, **kwargs)
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Monthly returns (%)")
    return ax


def plot_cones(name, bounds, oos_returns, num_samples=1000, ax=None,
               cone_std=(1., 1.5, 2.), random_seed=None, num_strikes=3):
    """
    Plots the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Redraws a new cone when
    cumulative returns fall outside of last cone drawn.

    Parameters
    ----------
    name : str
        Account name to be used as figure title.
    bounds : pandas.core.frame.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    oos_returns : pandas.core.frame.DataFrame
        Non-cumulative out-of-sample returns.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    num_strikes : int
        Upper limit for number of cones drawn. Can be anything from 0 to 3.

    Returns
    -------
    Returns are either an ax or fig option, but not both. If a
    matplotlib.Axes instance is passed in as ax, then it will be modified
    and returned. This allows for users to plot interactively in jupyter
    notebook. When no ax object is passed in, a matplotlib.figure instance
    is generated and returned. This figure can then be used to save
    the plot as an image without viewing it.

    ax : matplotlib.Axes
        The axes that were plotted on.
    fig : matplotlib.figure
        The figure instance which contains all the plot elements.
    """

    if ax is None:
        fig = figure.Figure(figsize=(10, 8))
        FigureCanvasAgg(fig)
        axes = fig.add_subplot(111)
    else:
        axes = ax

    returns = ep.cum_returns(oos_returns, starting_value=1.)
    bounds_tmp = bounds.copy()
    returns_tmp = returns.copy()
    cone_start = returns.index[0]
    colors = ["green", "orange", "orangered", "darkred"]

    for c in range(num_strikes + 1):
        if c > 0:
            tmp = returns.loc[cone_start:]
            bounds_tmp = bounds_tmp.iloc[0:len(tmp)]
            bounds_tmp = bounds_tmp.set_index(tmp.index)
            crossing = (tmp < bounds_tmp[float(-2.)].iloc[:len(tmp)])
            if crossing.sum() <= 0:
                break
            cone_start = crossing.loc[crossing].index[0]
            returns_tmp = returns.loc[cone_start:]
            bounds_tmp = (bounds - (1 - returns.loc[cone_start]))
        for std in cone_std:
            x = returns_tmp.index
            y1 = bounds_tmp[float(std)].iloc[:len(returns_tmp)]
            y2 = bounds_tmp[float(-std)].iloc[:len(returns_tmp)]
            axes.fill_between(x, y1, y2, color=colors[c], alpha=0.5)

    # Plot returns line graph
    label = 'Cumulative returns = {:.2f}%'.format((returns.iloc[-1] - 1) * 100)
    axes.plot(returns.index, returns.values, color='black', lw=3.,
              label=label)

    if name is not None:
        axes.set_title(name)
    axes.axhline(1, color='black', alpha=0.2)
    axes.legend(frameon=True, framealpha=0.5)

    if ax is None:
        return fig
    else:
        return axes

