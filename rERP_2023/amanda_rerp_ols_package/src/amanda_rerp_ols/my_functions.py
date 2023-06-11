from collections import namedtuple
from collections import OrderedDict
import numpy as np
import mne
import copy

def OLS_solution(X):
    X_t = np.transpose(X)
    XX_t = np.matmul(X_t, X)
    inv_XX_t = np.linalg.inv(XX_t)
    X_t = np.transpose(X)
    XX_t = np.matmul(X_t, X)
    inv_XX_t = np.linalg.inv(XX_t)
    sol = np.matmul(inv_XX_t, X_t)
    return sol

def rerp_ols(mne_epoch, X, pred_names=None):
    num_coef = X.shape[1]
    if pred_names == None:
        pred_names = ['x'+str(i) for i in range(num_coef)]
            
    rERP_OLS_Wrapper = namedtuple('rERP_OLS_Wrapper', ['Coeffs', 'Fitted', 'Resids'])
    Coeffs_Info = namedtuple('Coeffs_Info', ['beta', 'tval', 'pval'])
    
    data = mne_epoch.copy().get_data()
    num_ep, num_ch, num_t = np.shape(data)
    
    coeffs = np.empty((num_coef, num_ch, num_t))
    fitted, resids = np.empty((num_ep, num_ch, num_t)), np.empty((num_ep, num_ch, num_t))
    
    sol = OLS_solution(X)
    for ch in range(num_ch):
        y = data[:,ch,:]
        coeffs[:,ch,:] = np.matmul(sol, y)
        fitted[:,ch,:] = np.matmul(X, coeffs[:,ch,:])
        resids[:,ch,:] = np.subtract(y, fitted[:,ch,:])
    
    beta, tval, pval = np.empty((num_coef, num_ch, num_t)), np.empty((num_coef, num_ch, num_t)), np.empty((num_coef, num_ch, num_t))

    rerp = mne.stats.linear_regression(mne_epoch, X, names=pred_names)
    for i in range(len(pred_names)):
        beta[i], tval[i], pval[i] = rerp[pred_names[i]].beta.data, rerp[pred_names[i]].t_val.data, rerp[pred_names[i]].p_val.data
        
    coeffs_dict = {'coeffs_arr': coeffs}
    for i in range(len(pred_names)):
        coeffs_dict[pred_names[i]] = Coeffs_Info(beta[i], tval[i], pval[i])
        
    rerp_ols_wrapper = rERP_OLS_Wrapper(coeffs_dict, fitted, resids)
    return rerp_ols_wrapper

def mV(arr):
    arr = np.multiply(arr, 10**6)
    return arr