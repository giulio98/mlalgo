########################################
##Dataset Statistics &                 # 
##Data Exploration                     # 
########################################

###FUNCTIONS AND LIBRARIES###

# Libraries
import numpy
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from scipy.stats import norm 
import scipy.linalg
# Useful Functions:

# Functions to reshape one-dimension vectors
def vrow(vector):
    return (vector.reshape(1, vector.shape[0]))

            
def vcol(vector):
    return (vector.reshape(vector.shape[0], 1))

# Function to extract dataset and labels array
def dataset_extraction(path, features) :
    # Retrieving dataset and labels from file
    file = open(path , 'r')
    dataset = []
    labels = []  
    for line in file:
        if (line[0] == '\n'):
            break
        else :
            parsed = numpy.array(line.replace("\n", "").split(","), dtype = numpy.float64)
            dataset.append(parsed[:len(features)-1])
            labels.append(parsed[len(features)-1:])
    dataset_array = numpy.vstack(tuple(dataset)) 
    dataset_array = dataset_array.T 
    labels_array = numpy.vstack(tuple(labels)).reshape(dataset_array.shape[1],)
    
    return (dataset_array, labels_array)

# Results Folder Creation

dateTime = datetime.now()
dirPath = '1_Results/'+dateTime.strftime("%d%m%y%H%M%S")
if os.path.isdir('1_Results') == False :
    os.mkdir('1_Results')
os.mkdir(dirPath)

###DATASET STATISTICS###


# Function to print a table with dataset statistics
# (it also returns the statistics numpy matrix)
def print_statistics(path, features) :
    # Retrieving dataset and labels from file
    file = open(path , 'r')
    dataset = []
    labels = []  
    for line in file:
        if (line[0] == '\n'):
            break
        else :
            parsed = numpy.array(line.replace("\n", "").split(","), dtype = numpy.float64)
            dataset.append(parsed[:len(features)-1])
            labels.append(parsed[len(features)-1:])
    dataset_array = numpy.vstack(tuple(dataset)) 
    dataset_array = dataset_array.T 
    labels_array = numpy.vstack(tuple(labels))
    
    # Extracting statistics
    dataset_mu = dataset_array.mean(1)
    dataset_max= dataset_array.max(1)
    dataset_min= dataset_array.min(1)
    dataset_count = numpy.array([ dataset_array.shape[1] for i in range (0, len(features)-1) ])
    dataset_null_count = numpy.array((dataset_array.shape[1] - numpy.count_nonzero(dataset_array, axis=1)).tolist())
    dataset_dev = numpy.std(dataset_array,axis=1)
    
    stat_matrix = numpy.vstack([dataset_count, dataset_null_count, dataset_min, dataset_max, dataset_mu, dataset_dev]).T
    
    # Printing the formatted table
    table = []
    i = 0
    for elem in features[:-1]:
        table.append([elem] + stat_matrix[i].tolist())
        i = i+1

    print(tabulate(table, ['Features', 'Count', 'Null_Count', 'Min', 'Max', 'Mean', 'Std_Dev'], stralign="right", numalign="right"))
    
    return stat_matrix
### GAUSSIANIZATION ###

## Z-normalization

# Function to z-normalize the dataset
def normalization(dataset):
    return dataset-vcol(dataset.mean(1))/numpy.std(dataset)
# Function to compute the rank of a sample
# (feature is the feature 'index')
def rank(dataset, x, feature):
    count = 0
    for xi in dataset:
        if( xi[feature] > x ):
            count = count + 1
    # We add 1 and 2 to avoid bad results (0)
    return (count + 1)/(dataset.shape[0] + 2)

# Function to compute percent point function and 'Gaussianize' a dataset.
#* features_list is an array that contains all the features (also the labels), 
#* normalized_dataset contains the Z-normalized dataset 
#* (it works also without normalization)
def ppf_gaussianization(features_list, normalized_dataset):
    dataset_gaussianized = numpy.zeros(normalized_dataset.shape)
    for feature in range(len(features_list) - 1):
        for i in range(len(normalized_dataset[feature])):
            dataset_gaussianized[feature,i] = norm.ppf(rank(normalized_dataset.T,normalized_dataset[feature,i],feature))
    return dataset_gaussianized
# Function to compute covariance
def covariance_numpy(dataset):
    mu = dataset.mean(1)
    
    dataset_centered = dataset - vcol(mu)
    
    C = numpy.dot(dataset_centered, dataset_centered.T) / float(dataset.shape[1])
    
    return C
# Function to get PCA matrix (P) with numpy.linalg.eigh
def PCA_eigh(covariance_matrix, m):
    s, U = numpy.linalg.eigh(covariance_matrix) #U contiene autovettori, s autovalori
    P = U[:, ::-1][:, 0:m] #per ordinarli dal piu grande al piu piccolo
    return P
# Function to get PCA matrix (P) with numpy.linalg.svd
def PCA_svd(covariance_matrix, m):
    U, s, Vh = numpy.linalg.svd(covariance_matrix)
    P = U[:, 0:3]
    return P
########################################
##4.                                   #
##minDCF computations                  # 
#                                      # 
########################################

## Classifiers Functions

# Function to compute log likelihood ratio
def Compute_llr(dataset_ll):
    return(dataset_ll[1]-dataset_ll[0])

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, det = numpy.linalg.slogdet(C)
    det = numpy.log(numpy.linalg.det(C))
    inv = numpy.linalg.inv(C)
    
    res = []
    x_centered = x - mu
    for x_col in x_centered.T:
        res.append(numpy.dot(x_col.T, numpy.dot(inv, x_col)))

    return -M/2*numpy.log(2*numpy.pi) - 1/2*det - 1/2*numpy.hstack(res).flatten()

# Functions to compute min DCF
def Confusion_matrix(Confusion_array, labels_number):
    Confusion_Matrix_ = numpy.zeros((labels_number, labels_number))
    for elem in Confusion_array:
        Confusion_Matrix_[elem[0],int(elem[1])] += 1 
    return Confusion_Matrix_

def Compute_Cost_threshold(t,wine_llr):
    Predictions = []
    for elem in wine_llr:
        if elem> t :
            Predictions.append(1)
        else:
            Predictions.append(0)
    return Predictions

def BayesEmpiricalRisk_threshold(pi,Cfn,Cfp, wine_llr, wine_llr_labels):
    DCF_array = []
    for elem in wine_llr:
        Predictions=Compute_Cost_threshold(elem,wine_llr)
        Confusion_array=[]
        for i in range(len(Predictions)):
            Confusion_array.append([Predictions[i], wine_llr_labels[i]])
        CM = Confusion_matrix(Confusion_array,2)
        FNR=CM[0,1]/(CM[0,1]+CM[1,1])
        FPR=CM[1,0]/(CM[0,0]+CM[1,0])
        Bayes_risk=pi*Cfn*FNR+(1-pi)*Cfp*FPR
        Bayes_risk_dummy=min(pi*Cfn,(1-pi)*Cfp)
        DCF_array.append(Bayes_risk/Bayes_risk_dummy)
    return min(DCF_array)
# Functions to compute score matrices:


def score_matrix_logdensity_mvg(DTE, means, covariances):
    S = numpy.zeros((2, DTE.shape[1]))
    for i in range(2):
        for j, sample in enumerate(DTE.T):
            sample = vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, vcol(means[i]), covariances[i])
    return S
def split_db_3to1(D, L):
    nTrain = int(D.shape[1]*0.8)
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)
#Function to compute the log-likelihood for naive-bayes classifier
def loglikelihood_naive_bayes(x, mu, v):
    x_final = x - mu
    return_array = numpy.zeros((len(v), x.shape[0]))
    for i in range(len(v)) :
        return_array[i] =  (-(numpy.log(2*numpy.pi))/2 - (numpy.log(v[i][v[i]>0]))/2 - (x_final[i])**2/(2*v[i][v[i]>0]))
    return return_array.sum()
def score_matrix_logdensity_naive_bayes(test_dataset, means, covariances):
    S = numpy.zeros((len(means), test_dataset.shape[1]))
    for i in range(len(means)):
        for j in range(test_dataset.shape[1]):
            S[i, j]=loglikelihood_naive_bayes(vcol(test_dataset[:, j]), means[i], covariances[i])
    return S
def score_matrix_logdensity_tied(DTR,LTR,DTE, means, covariances):
    DTR_c = [DTR[:, LTR == i] for i in range(2)]
    Sstar = 0
    for i in range(2):
        Sstar += DTR_c[i].shape[1]*covariances[i]
    Sstar /= DTR.shape[1]

    S = numpy.zeros((2, DTE.shape[1]))

    for i in range(2):
        for j, sample in enumerate(DTE.T):
            sample = vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, vcol(means[i]), Sstar)
    return S
########################################
##5.                                   #
##MVG KFOLD                            # 
#                                      # 
########################################
#NEW VERSION
def kfold(dataset,labels,K,seed=0):
    N = int(dataset.shape[1]/K)
    numpy.random.seed(seed)
    indexes = numpy.random.permutation(dataset.shape[1])

    def combine_all_except_i(i):
        first= int(max(0, i*N))
        last = int(min(dataset.shape[1],(i+1)*N))
        indexes_val = indexes[first:last]
        indexes_train = numpy.r_[indexes[:first],indexes[last:]]
        train_set = dataset[:,indexes_train]
        validation_set = dataset[:,indexes_val]
        label_train_set = labels[indexes_train]
        label_validation_set = labels[indexes_val]
        return train_set,label_train_set,validation_set,label_validation_set
    
    return combine_all_except_i
def covariance_mean_numpy(D):
    mu = D.mean(1)
    
    D_centered = D - vcol(mu)
    
    C = numpy.dot(D_centered, D_centered.T) / float(D.shape[1])
    
    return mu,C
def calculate_params(DTR,LTR):
    LTR_0=(LTR==0)
    DTR_0=DTR[:,LTR_0]
    LTR_1=(LTR==1)
    DTR_1=DTR[:,LTR_1]
    mu_0,cov_0=covariance_mean_numpy(DTR_0)
    mu_1,cov_1=covariance_mean_numpy(DTR_1)
    m_c=[]
    s_c=[]
    m_c.append(mu_0)
    m_c.append(mu_1)
    s_c.append(cov_0)
    s_c.append(cov_1)
    return m_c,s_c
    
def apply_MVG_Classifier(k,func,pi,Cfn,Cfp):
    wine_llr_labels=[]
    wine_llr_mvg=[]
    for j in range(k):
        train_set, train_set_label, validation_set, validation_set_label=func(j)
        m_c,s_c= calculate_params(train_set,train_set_label)
        wine_llr_labels.append(validation_set_label)
        wine_llr_mvg_i=Compute_llr(score_matrix_logdensity_mvg(validation_set,m_c,s_c))
        wine_llr_mvg.append(wine_llr_mvg_i)
    wine_llr_labels=numpy.concatenate(wine_llr_labels)
    wine_llr_mvg=numpy.concatenate(wine_llr_mvg)
    print(BayesEmpiricalRisk_threshold(pi,Cfn,Cfp,wine_llr_mvg,wine_llr_labels))

def apply_Naive_Bayes_Classifier(k,func,pi,Cfn,Cfp):
    wine_llr_labels=[]
    wine_llr_mvg=[]
    bayes_covariances=[]
    for j in range(k):
        train_set, train_set_label, validation_set, validation_set_label=func(j)
        m_c,s_c= calculate_params(train_set,train_set_label)
        wine_llr_labels.append(validation_set_label)
        bayes_covariances=[element * numpy.identity(element.shape[0]) for element in s_c] ## bayes diag covariances
        wine_llr_mvg_i=Compute_llr(score_matrix_logdensity_mvg(validation_set,m_c,bayes_covariances))
        wine_llr_mvg.append(wine_llr_mvg_i)
    wine_llr_labels=numpy.concatenate(wine_llr_labels)
    wine_llr_mvg=numpy.concatenate(wine_llr_mvg)
    print(BayesEmpiricalRisk_threshold(pi,Cfn,Cfp,wine_llr_mvg,wine_llr_labels))

def apply_Tied_Full_Classifier(k,func,pi,Cfn,Cfp):
    wine_llr_labels=[]
    wine_llr_mvg=[]
    for j in range(k):
        train_set, train_set_label, validation_set, validation_set_label=func(j)
        m_c,s_c= calculate_params(train_set,train_set_label)
        wine_llr_labels.append(validation_set_label)
        wine_llr_mvg_i=Compute_llr(score_matrix_logdensity_tied(train_set,train_set_label,validation_set,m_c,s_c))
        wine_llr_mvg.append(wine_llr_mvg_i)
    wine_llr_labels=numpy.concatenate(wine_llr_labels)
    wine_llr_mvg=numpy.concatenate(wine_llr_mvg)
    print(BayesEmpiricalRisk_threshold(pi,Cfn,Cfp,wine_llr_mvg,wine_llr_labels))

def apply_Tied_Diag_Classifier(k,func,pi,Cfn,Cfp):
    wine_llr_labels=[]
    wine_llr_mvg=[]
    bayes_covariances=[]
    for j in range(k):
        train_set, train_set_label, validation_set, validation_set_label=func(j)
        m_c,s_c= calculate_params(train_set,train_set_label)
        wine_llr_labels.append(validation_set_label)
        bayes_covariances=[element * numpy.identity(element.shape[0]) for element in s_c] ## bayes diag covariances
        wine_llr_mvg_i=Compute_llr(score_matrix_logdensity_tied(train_set,train_set_label,validation_set,m_c,bayes_covariances))
        wine_llr_mvg.append(wine_llr_mvg_i)
    wine_llr_labels=numpy.concatenate(wine_llr_labels)
    wine_llr_mvg=numpy.concatenate(wine_llr_mvg)
    print(BayesEmpiricalRisk_threshold(pi,Cfn,Cfp,wine_llr_mvg,wine_llr_labels))
########################################
##6.                                   #
##LOGISTIC REGRESSION                  # 
#                                      # 
########################################   
# Functions for Logistic Regression
class logRegClass():

    def __init__(self, DTR, LTR, l, pi):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pi = pi

    def logRegObj(self, v):
        # Function 2 implementation
        w, b = v[0:-1], v[-1]

        # Remap
        LTR_0 = ( self.LTR == 0 )
        LTR_1 = ( self.LTR == 1 )
        z_0 = 2 * LTR_0 - 1
        z_1 = 2 * LTR_1 - 1
        n_true= (self.LTR == 1).sum()
        n_false = (self.LTR == 0).sum()
        expo_0 = -z_0 * (w.T.dot(self.DTR) + b)
        expo_1 = -z_1 * (w.T.dot(self.DTR) + b)
        normalizer = self.l * (w * w).sum() / 2
        return normalizer + self.pi*((numpy.log1p(numpy.exp(expo_1))).sum())/n_true + (1-self.pi)*((numpy.log1p(numpy.exp(expo_0))).sum())/n_false
    

    def setLambda(self, lamb):
        self.l = lamb

def apply_score_matrix_logReg(k,func,pi):
    lambdaVector = numpy.array([1.E-6, 1.E-5,1.E-4,1.E-3,1.E-2,1.E-1,1,10,100,1000,10000,100000])
    dcf_along_lambda_05 = []
    dcf_along_lambda_01 = []
    dcf_along_lambda_09 = []
    for l in lambdaVector:
        print("----------lambda:",l)
        scores=[]
        wine_llr_labels=[]
        for j in range(k):
            DTR, LTR, DTE, LTE = func(j)
            wine_llr_labels.append(LTE)
            logReg = logRegClass(DTR, LTR.astype('int'), 1.E-3, pi)
            logReg.setLambda(l)

            x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logReg.logRegObj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True, factr=5000, maxfun=20000)
            S = (numpy.dot(x[0:-1].T, DTE) + x[-1])
            #print(l)
            scores.append(S)
        wine_llr_labels=numpy.concatenate(wine_llr_labels)
        wine_llr_lr = numpy.concatenate(scores) 
        dcf_05=BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_lr,wine_llr_labels)
        print("0.5:",dcf_05)
        dcf_01=BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_lr,wine_llr_labels)
        print("0.1:",dcf_01)
        dcf_09=BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_lr,wine_llr_labels)
        print("0.9:",dcf_09)
        dcf_along_lambda_05.append(dcf_05)
        dcf_along_lambda_01.append(dcf_01)
        dcf_along_lambda_09.append(dcf_09)
    return dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09

def expand_d(D):
    D_phi = numpy.empty([D.shape[0]*(D.shape[0]+1),D.shape[1]])
    for col in range(D.shape[1]):
        xi = mCol(D[:,col])
        tmp= mCol(numpy.dot(xi,xi.T).flatten('F'))
        D_phi[:,col] = numpy.vstack([tmp,xi]).flatten()
    return D_phi

def apply_score_matrix_logReg_quadratic(k,func,pi):
    lambdaVector = numpy.array([1.E-6, 1.E-5,1.E-4,1.E-3,1.E-2,1.E-1,1,10,100,1000,10000,100000])
    dcf_along_lambda_05 = []
    dcf_along_lambda_01 = []
    dcf_along_lambda_09 = []
    for l in lambdaVector:
        print("----------lambda:",l)
        scores=[]
        wine_llr_labels=[]
        for j in range(k):
            DTR, LTR, DTE, LTE = func(j)
            wine_llr_labels.append(LTE)
            
            DTR=expand_d(DTR)
            DTE=expand_d(DTE)
            logReg = logRegClass(DTR, LTR.astype('int'), 1.E-3, pi)
            logReg.setLambda(l)
            
            x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logReg.logRegObj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True, factr=5000, maxfun=20000)
            S = (numpy.dot(x[0:-1].T, DTE) + x[-1])
            #print(l)
            scores.append(S)
        wine_llr_labels=numpy.concatenate(wine_llr_labels)
        wine_llr_lr = numpy.concatenate(scores) 
        dcf_05=BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_lr,wine_llr_labels)
        print("0.5:",dcf_05)
        dcf_01=BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_lr,wine_llr_labels)
        print("0.1:",dcf_01)
        dcf_09=BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_lr,wine_llr_labels)
        print("0.9:",dcf_09)
        dcf_along_lambda_05.append(dcf_05)
        dcf_along_lambda_01.append(dcf_01)
        dcf_along_lambda_09.append(dcf_09)
    return dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09


########################################
##7.                                   #
##SUPPORT VECTOR MACHINES              # 
#                                      # 
######################################## 
def mRow(v):
    return v.reshape((1,v.size)) #sizes gives the number of elements in the matrix/array
def mCol(v):
    return v.reshape((v.size, 1))
def compute_lagrangian_wrapper(H):

    def compute_lagrangian(alpha):

        elle = numpy.ones(alpha.size) # 66,
        L_hat_D=0.5*( numpy.linalg.multi_dot([alpha.T, H, alpha]) ) - numpy.dot(alpha.T , mCol(elle))# 1x1
        L_hat_D_gradient= numpy.dot(H, alpha)-elle # 66x1
        
        return L_hat_D, L_hat_D_gradient.flatten() # 66, 
   
    
    return compute_lagrangian
def compute_primal_objective(w_hat_star, C, Z, D_hat):

    w_hat_star = mCol(w_hat_star)
    Z = mRow(Z)
    fun1= 0.5 * (w_hat_star*w_hat_star).sum()   
    fun2 = Z* numpy.dot(w_hat_star.T, D_hat)
    fun3 = 1- fun2
    zeros = numpy.zeros(fun3.shape)
    sommatoria = numpy.maximum(zeros, fun3)
    fun4= numpy.sum(sommatoria)
    fun5= C*fun4
    ris = fun1 +fun5
    return ris

def apply_score_matrix_SVM(k,func):
    cVector = numpy.array([1.E-3,1.E-2,2.E-2,4.E-2,6.E-2,8.E-2,1.E-1,2.E-1,4.E-1,6.E-1,8.E-1,1,10,100,1000])
    dcf_along_c_05 = []
    dcf_along_c_01 = []
    dcf_along_c_09 = []
    for c in cVector:
        print("-------------------")
        print("c=",c)
        Mscores=[]
        wine_llr_labels = []
        for j in range(k):
            DTR, LTR, DTE, LTE=func(j)
            wine_llr_labels.append(LTE)
            K=1
            k_values= numpy.ones([1,DTR.shape[1]]) *K
            #Creating D_hat= [xi, k] with k=1
            D_hat = numpy.vstack((DTR, k_values))
            #Creating H_hat
            # 1) creating G_hat through numpy.dot and broadcasting
            G_hat= numpy.dot(D_hat.T, D_hat)
            # 2)vector of the classes labels (-1/+1)
            Z = numpy.copy(LTR)
            Z[Z == 0] = -1
            Z= mCol(Z)
            # 3) multiply G_hat for ZiZj operating broadcasting
            H_hat= Z * Z.T * G_hat
            # Calculate L_hat_D and its gradient DUAL SOLUTION
            compute_lagr= compute_lagrangian_wrapper(H_hat)
            # Use scipy.optimize.fmin_l_bfgs_b
            C=c
            x0=numpy.zeros(LTR.size) #alpha
            bounds_list = [(0,C)] * LTR.size
            (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)

            sommatoria = mCol(x) * mCol(Z) * D_hat.T
            w_hat_star = numpy.sum( sommatoria,  axis=0 ) 
            w_star = w_hat_star[0:-1] 
            b_star = w_hat_star[-1] 
            scores = numpy.dot(mCol(w_star).T, DTE) + b_star
            Mscores.append(scores.flatten())
            #print(c)
        wine_llr_labels=numpy.concatenate(wine_llr_labels)
        wine_llr_svm = numpy.concatenate(Mscores) #4-fold
        dcf05=BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels)
        dcf_along_c_05.append(dcf05)
        print("0.5:",dcf05)
        dcf01=BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_svm,wine_llr_labels)
        dcf_along_c_01.append(dcf01)
        print("0.1:",dcf01)
        dcf09=BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_svm,wine_llr_labels)
        dcf_along_c_09.append(dcf09)
        print("0.9:",dcf09)
    return dcf_along_c_05, dcf_along_c_01, dcf_along_c_09
    
def apply_score_matrix_SVM_quadratic(k,func):
    cVector = numpy.array([1.E-3,1.E-2,2.E-2,4.E-2,6.E-2,8.E-2,1.E-1,2.E-1,4.E-1,6.E-1,8.E-1,1,10,100,1000])
    CVector = numpy.array([0.01,0.05,0.1,1,10])
    dcf_along_c_05 = []
    dcf_along_c_01 = []
    dcf_along_c_09 = []
    for c in cVector:
        print("----------------------c:",c)
        
        for C in CVector:
            print("-----------C:",C)
            Mscores=[]
            wine_llr_labels = []
            for j in range(k):
                DTR, LTR, DTE, LTE=func(j)
                wine_llr_labels.append(LTE)
                K=1
                k_values= numpy.ones([1,DTR.shape[1]]) *K
                #Creating D_hat= [xi, k] with k=1
                D_hat = numpy.vstack((DTR, k_values))
                #Creating H_hat
                # 1) creating G_hat through numpy.dot and broadcasting
                G_hat= (numpy.dot(DTR.T, DTR)+c)**2
                # 2)vector of the classes labels (-1/+1)
                Z = numpy.copy(LTR)
                Z[Z == 0] = -1
                Z= mCol(Z)
                # 3) multiply G_hat for ZiZj operating broadcasting
                H_hat= Z * Z.T * G_hat
                # Calculate L_hat_D and its gradient DUAL SOLUTION
                compute_lagr= compute_lagrangian_wrapper(H_hat)
                # Use scipy.optimize.fmin_l_bfgs_b
                x0=numpy.zeros(LTR.size) #alpha
                bounds_list = [(0,C)] * LTR.size
                (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)

                sommatoria = mCol(x) * mCol(Z) *(numpy.dot(DTR.T, DTE)+c)**2 
                scores = numpy.sum( sommatoria,  axis=0 ) 
                Mscores.append(scores.flatten())
                #print(c)
            wine_llr_labels=numpy.concatenate(wine_llr_labels)
            wine_llr_svm = numpy.concatenate(Mscores) #4-fold
            dcf05=BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels)
            dcf_along_c_05.append(dcf05)
            print(dcf05)
            dcf01=BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_svm,wine_llr_labels)
            dcf_along_c_01.append(dcf01)
            print(dcf01)
            dcf09=BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_svm,wine_llr_labels)
            dcf_along_c_09.append(dcf09)
            print(dcf09)
    return dcf_along_c_05, dcf_along_c_01, dcf_along_c_09

from itertools import combinations
def apply_score_matrix_SVM_RBF(k,func):
    cVector = numpy.array([1.E-3,1.E-2,5.E-2,1.E-1,5.E-1,8.E-1,1,10,100,1000])
    gammaVector=numpy.array([numpy.exp(-10),numpy.exp(-9),numpy.exp(-8),numpy.exp(-7),numpy.exp(-6),numpy.exp(-5),numpy.exp(-4),numpy.exp(-3),numpy.exp(-2),numpy.exp(-1)])
    dcf_along_gamma_10 = []
    dcf_along_gamma_9 = []
    dcf_along_gamma_8 = []
    dcf_along_gamma_7 = []
    dcf_along_gamma_6 = []
    dcf_along_gamma_5 = []
    dcf_along_gamma_4 = []
    dcf_along_gamma_3 = []
    dcf_along_gamma_2 = []
    dcf_along_gamma_1 = []
    for gamma in gammaVector:
        print("gamma")
        print(numpy.log(gamma))
        print("-----")
        for c in cVector:
            print("c")
            print(c)
            print("------")
            Mscores=[]
            wine_llr_labels = []
            for j in range(k):
                DTR, LTR, DTE, LTE=func(j)
                wine_llr_labels.append(LTE)
                K=1
                k_values= numpy.ones([1,DTR.shape[1]]) *K
                #Creating D_hat= [xi, k] with k=1
                D_hat = numpy.vstack((DTR, k_values))
                #Creating H_hat
                D_dist=scipy.spatial.distance.cdist(DTR.T,DTR.T, 'sqeuclidean')
                # 1) creating G_hat through numpy.dot and broadcasting
                G_hat= numpy.exp(-gamma*D_dist)
                
                # 2)vector of the classes labels (-1/+1)
                Z = numpy.copy(LTR)
                Z[Z == 0] = -1
                Z= mCol(Z)
                # 3) multiply G_hat for ZiZj operating broadcasting
                H_hat= Z * Z.T * G_hat
                # Calculate L_hat_D and its gradient DUAL SOLUTION
                compute_lagr= compute_lagrangian_wrapper(H_hat)
                # Use scipy.optimize.fmin_l_bfgs_b
                C=c
                x0=numpy.zeros(LTR.size) #alpha
                bounds_list = [(0,C)] * LTR.size
                (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)

                DE_dist=scipy.spatial.distance.cdist(DTR.T,DTE.T, 'sqeuclidean')

                sommatoria = mCol(x) * mCol(Z) * numpy.exp(-gamma*DE_dist)
                scores = numpy.sum( sommatoria,  axis=0 ) 
                Mscores.append(scores.flatten())
                #print(c)
            wine_llr_labels=numpy.concatenate(wine_llr_labels)
            wine_llr_svm = numpy.concatenate(Mscores) #4-fold
            print("----------------------------MIN DCF------------------------------------")
            if(gamma==numpy.exp(-10)):
                dcf_along_gamma_10.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-9)):
                dcf_along_gamma_9.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-8)):
                dcf_along_gamma_8.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-7)):
                dcf_along_gamma_7.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-6)):
                dcf_along_gamma_6.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-5)):
                dcf_along_gamma_5.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-4)):
                dcf_along_gamma_4.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-3)):
                dcf_along_gamma_3.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-2)):
                dcf_along_gamma_2.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            if(gamma==numpy.exp(-1)):
                dcf_along_gamma_1.append(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_svm,wine_llr_labels))
            print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_svm,wine_llr_labels))
            print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_svm,wine_llr_labels))
            print("-----------------------------------------------------------------------")
    return dcf_along_gamma_10, dcf_along_gamma_9, dcf_along_gamma_8, dcf_along_gamma_7, dcf_along_gamma_6, dcf_along_gamma_5, dcf_along_gamma_4, dcf_along_gamma_3, dcf_along_gamma_2, dcf_along_gamma_1

########################################
##8.                                   #
##GAUSSIAN MIXTURE MODELS              # 
#                                      # 
######################################## 
def logpdf_GAU_ND2(x, mu2, c):
    z=[-0.5*(numpy.dot(numpy.dot((x1.reshape((x1.size,1))-mu2).T,numpy.linalg.inv(c)),(x1.reshape((x1.size,1))-mu2))[0,0]+
            numpy.linalg.slogdet(c)[1])
    -(x1.size*numpy.log(2*numpy.pi)/2) for x1 in x.T]

    return numpy.array(z)
def logpdf_GMM(X, gmm):
    S= numpy.array([logpdf_GAU_ND2(X,i[1],i[2])+numpy.log(i[0]) for i in gmm])
    logdens = scipy.special.logsumexp(S, axis=0) #calculate log density
   
    return S,logdens
def lbg(x,initial,alpha=0.1,t=0):
    
    
    new=[]
    for i in initial :
        U, n, Vh = numpy.linalg.svd(i[2])

        d = U[:, 0:1] * (n[0]**0.5) * alpha #displacement

        new.append((i[0]/2,i[1]+d,i[2]))
        new.append((i[0]/2,i[1]-d,i[2]))
    new=compute_param(x,new,t)
    return new
def compute_param(X,gmm,diag_tied=0,psi=0.01):
    param=gmm
    z=0
    print('---------------------------')
    while(1):
        
        s,su=logpdf_GMM(X,param)
       
        if(-10**-3<su.mean()-z<10**-3): #convergence
            return param
        post=numpy.exp(s-su)
        zg=post.sum(axis=1)
        fg=numpy.array([(X*u).sum(axis=1) for u in post])
        sg=numpy.array([numpy.dot(x.reshape((x.size,1)),x.reshape((1,x.size))) for x in X.T])
        sg=numpy.array([(sg*u.reshape((u.size,1,1))).sum(axis=0) for u in post])
        param=[]
        z=su.mean()
        tied=0
        for idx, i in enumerate(sg):
            m=fg[idx]/zg[idx]
            var=i/zg[idx]-numpy.dot(m.reshape((m.size,1)),m.reshape((1,m.size)))
            w=zg[idx]/zg.sum()
            
            if(diag_tied==2):
                tied+=var*zg[idx]
                
           
            if diag_tied==1:
                var=var * numpy.eye(var.shape[0])
            if(diag_tied !=2):
                U, n, _ = numpy.linalg.svd(var)
                n[n<psi] = psi
                var = numpy.dot(U, n.reshape((n.size,1))*U.T)
            
            param.append((w,m.reshape(m.size,1),var))
        if(diag_tied==2):
            tied=tied/X.shape[1]
            U, n, _ = numpy.linalg.svd(tied)
            n[n<psi] = psi
            tied = numpy.dot(U, n.reshape((n.size,1))*U.T)
            for i in range(len(param)):
                param[i]=(param[i][0],param[i][1],tied)
def initialize(data,label):
    param=[]
    for i in set(label):
        m=data[:,label==i].mean(axis=1).reshape((data.shape[0],1)) #empirical mean
        var=numpy.dot(data[:,label==i]-m,(data[:,label==i]-m).T)/data[:,label==i].shape[1] #(x-m)*(x-m).T/N
        U, n, _ = numpy.linalg.svd(var) #n containg vector singular values
        n[n<0.01] = 0.01
        var = numpy.dot(U, n.reshape((n.size,1))*U.T)
        param.append([(1,m,var)]) #weight, mean, var
    return param
def apply_GMM(k,func,nb,tied=0):
    score=[]
    wine_labels=[]
    for j in range(nb):
        score.append([])
        wine_labels.append([])
    for j in range(k):
        
        #retrieve sets
        train_set,train_set_label, validation_set, validation_set_label=func(j)
        
        #inizialize params training
        param=initialize(train_set,train_set_label) #starting point
        #build 2 component gmm, use EM to estimate a ML solution for the 2-component model
        param[0]=compute_param(train_set[:,train_set_label==0],param[0],tied)
        param[1]=compute_param(train_set[:,train_set_label==1],param[1],tied)
        
        
        
        ####### split the two component to obtain a 4-components model
        for j in range(nb):
            param[0]=lbg(train_set[:,train_set_label==0],param[0],t=tied)
            param[1]=lbg(train_set[:,train_set_label==1],param[1],t=tied)
            
            scores=[]
            _,z=logpdf_GMM(validation_set, param[0])
            scores.append(z)
            _,z=logpdf_GMM(validation_set, param[1])
            
            scores.append(z)
            
            ratio=scores[1]-scores[0]
           
            score[j].append(ratio)
            wine_labels[j].append(validation_set_label)
    
    for j in range(len(score)):
        score[j]=numpy.concatenate(score[j])
        wine_labels[j]=numpy.concatenate(wine_labels[j])
    return score,wine_labels
########################################
##9.                                   #
##ACTUAL DCF                           # 
#                                      # 
######################################## 
def apply_score_matrix_SVM_RBF_actual(k,func):
    c= 100
    gamma= numpy.exp(-1)
    wine_llr_labels=[]
    Mscores=[]
    for j in range(k):
        DTR, LTR, DTE, LTE=func(j)
        wine_llr_labels.append(LTE)
        K=1
        k_values= numpy.ones([1,DTR.shape[1]]) *K
        #Creating D_hat= [xi, k] with k=1
        D_hat = numpy.vstack((DTR, k_values))
        #Creating H_hat
        D_dist=scipy.spatial.distance.cdist(DTR.T,DTR.T, 'sqeuclidean')
        # 1) creating G_hat through numpy.dot and broadcasting
        G_hat= numpy.exp(-gamma*D_dist)

        # 2)vector of the classes labels (-1/+1)
        Z = numpy.copy(LTR)
        Z[Z == 0] = -1
        Z= mCol(Z)
        # 3) multiply G_hat for ZiZj operating broadcasting
        H_hat= Z * Z.T * G_hat
        # Calculate L_hat_D and its gradient DUAL SOLUTION
        compute_lagr= compute_lagrangian_wrapper(H_hat)
        # Use scipy.optimize.fmin_l_bfgs_b
        C=c
        x0=numpy.zeros(LTR.size) #alpha
        bounds_list = [(0,C)] * LTR.size
        (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)

        DE_dist=scipy.spatial.distance.cdist(DTR.T,DTE.T, 'sqeuclidean')

        sommatoria = mCol(x) * mCol(Z) * numpy.exp(-gamma*DE_dist) 
        scores = numpy.sum( sommatoria,  axis=0 ) 
        Mscores.append(scores.flatten())
        #print(c)
    wine_llr_labels=numpy.concatenate(wine_llr_labels)
    wine_llr_svm = numpy.concatenate(Mscores) #4-fold
                
    return wine_llr_labels, wine_llr_svm
def Compute_Cost_threshold_actual(wine_llr,pi):
    Predictions = []
    for elem in wine_llr:
        if elem> -numpy.log(pi/(1-pi)) :
            Predictions.append(1)
        else:
            Predictions.append(0)
    return Predictions

def BayesEmpiricalRisk_threshold_actual(pi,Cfn,Cfp, wine_llr,wine_llr_labels):
    DCF_array = []
    Predictions=Compute_Cost_threshold_actual(wine_llr,pi)
    Confusion_array=[]
    for i in range(len(Predictions)):
        Confusion_array.append([Predictions[i], wine_llr_labels[i]])
    CM = Confusion_matrix(Confusion_array,2)
    FNR=CM[0,1]/(CM[0,1]+CM[1,1])
    FPR=CM[1,0]/(CM[0,0]+CM[1,0])
    Bayes_risk=pi*Cfn*FNR+(1-pi)*Cfp*FPR
    Bayes_risk_dummy=min(pi*Cfn,(1-pi)*Cfp)
    DCF_array.append(Bayes_risk/Bayes_risk_dummy)
    return min(DCF_array)
def draw_bayes_error(score,label,z=0,c1='r',c2='b'):
    points=[]
    dcf_min=[]
    axis=numpy.linspace(-4,4,11)
    x_axis=[]
    for i in axis:
       
        p=1/(1+numpy.exp(-i))
        x=(p*1)/((1-p)*1)
        threshold=-numpy.log(x)
        x_axis.append(threshold)
        d=BayesEmpiricalRisk_threshold_actual(p,1,1,score,label)
        c=BayesEmpiricalRisk_threshold(p,1,1,score,label)
        dcf_min.append(c)
        points.append(d)
    #sort based on first col arsort return the indexes
    #of the given arrray sorted using the indexes to indexthe array will sort it
    if(z==0):
        plt.figure()
    plt.plot(x_axis,points, label='act DCF', color=c1)
    plt.plot(x_axis,dcf_min, label='mindcf', color=c2)
    plt.legend(loc='upper right')
    plt.ylim([0,1.1])
    plt.xlim([-4,4])
    plt.show()
def score_matrix_logReg_Calibration(k,func,pi):
    l = 10E-2 #insert different value of lambda
    scores=[]
    for j in range(k):
        DTR, LTR, DTE, LTE=func(j)
        logReg = logRegClass(DTR, LTR.astype('int'), 1.E-3, pi)
        logReg.setLambda(l)

        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logReg.logRegObj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True, factr=5000, maxfun=20000)
        S = (numpy.dot(x[0:-1].T, DTE) + x[-1])
            #print(l)
        scores.append(S)
    scores=numpy.concatenate(scores)
    return scores
def apply_score_matrix_logReg_quadratic_actual(k,func,pi=0.9,l=0.001):
    lambdaVector = numpy.array([l])
    for l in lambdaVector:
        print("----------lambda:",l)
        scores=[]
        wine_llr_labels=[]
        for j in range(k):
            DTR, LTR, DTE, LTE = func(j)
            wine_llr_labels.append(LTE)
            
            DTR=expand_d(DTR)
            DTE=expand_d(DTE)
            logReg = logRegClass(DTR, LTR.astype('int'), 1.E-3, pi)
            logReg.setLambda(l)
            
            x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logReg.logRegObj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True, factr=5000, maxfun=20000)
            S = (numpy.dot(x[0:-1].T, DTE) + x[-1])
            #print(l)
            scores.append(S)
        wine_llr_labels=numpy.concatenate(wine_llr_labels)
        wine_llr_lr = numpy.concatenate(scores) 
    return wine_llr_labels, wine_llr_lr

########################################
##10.                                  #
##EVALUATION                           # 
#                                      # 
######################################## 
def apply_RBF_SVM_test(data,label,test,c,gamma):
    K=1
    k_values= numpy.ones([1,data.shape[1]]) *K
    #Creating D_hat= [xi, k] with k=1
    D_hat = numpy.vstack((data, k_values))
    #Creating H_hat
    D_dist=scipy.spatial.distance.cdist(data.T,data.T, 'sqeuclidean')
    # 1) creating G_hat through numpy.dot and broadcasting
    G_hat= numpy.exp(-gamma*D_dist)
                
    # 2)vector of the classes labels (-1/+1)
    Z = numpy.copy(label)
    Z[Z == 0] = -1
    Z= mCol(Z)
    # 3) multiply G_hat for ZiZj operating broadcasting
    H_hat= Z * Z.T * G_hat
    # Calculate L_hat_D and its gradient DUAL SOLUTION
    compute_lagr= compute_lagrangian_wrapper(H_hat)
    # Use scipy.optimize.fmin_l_bfgs_b
    C=c
    x0=numpy.zeros(label.size) #alpha
    bounds_list = [(0,C)] * label.size
    (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)

    DE_dist=scipy.spatial.distance.cdist(data.T,test.T, 'sqeuclidean')

    sommatoria = mCol(x) * mCol(Z) * numpy.exp(-gamma*DE_dist)
    scores = numpy.sum( sommatoria,  axis=0 ) 
    return f,scores

def test_RBF_svm(train,label,test,c,gamma):
    f, score=apply_RBF_SVM_test(train,label,test,c,gamma)
    return f, score
def compute_accuracy_error(predicted_labels, LTE):
    good_predictions = (predicted_labels == LTE) #array with True when predicted_labels[i] == LTE[i]    
    num_corrected_predictions =(good_predictions==True).sum()
    tot_predictions = predicted_labels.size
    accuracy= num_corrected_predictions /tot_predictions
    error = (tot_predictions - num_corrected_predictions ) /tot_predictions

    return (accuracy, error)
def test_GMM_std_2(train,label,test,n):
    score=apply_GMM_std_2_test(train,label,test,n)
    return score
def apply_GMM_std_2_test(train,label,test,n):
    score=[]
    for j in range(n):
        score.append([])
        
    #inizialize params training
    param=initialize(train,label) #starting point
    #build 2 component gmm, use EM to estimate a ML solution for the 2-component model
    param[0]=compute_param(train[:,label==0],param[0])
    param[1]=compute_param(train[:,label==1],param[1])
        
        
        
        ####### split the two component to obtain a 4-components model
    for j in range(n):
        param[0]=lbg(train[:,label==0],param[0])
        param[1]=lbg(train[:,label==1],param[1])
            
        scores=[]
        _,z=logpdf_GMM(test, param[0])
        scores.append(z)
        _,z=logpdf_GMM(test, param[1])
            
        scores.append(z)
            
        ratio=scores[1]-scores[0]
           
        score[j].append(ratio)
    
    for j in range(len(score)):
        score[j]=numpy.concatenate(score[j])
    return score[1] #n=2
def test_logReg_quadratic(train,label,test,l,pi=0.9):
    score= apply_logReg_quadratic_test(train,label,test,l,pi)
    return score
def apply_logReg_quadratic_test(data,label,test,l,pi=0.9):
    data=expand_d(data)
    test=expand_d(test)
    logReg = logRegClass(data, label.astype('int'), 1.E-3, pi)
    logReg.setLambda(l)
            
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logReg.logRegObj, x0=numpy.zeros(data.shape[0] + 1), approx_grad=True, factr=5000, maxfun=20000)
    S = (numpy.dot(x[0:-1].T, test) + x[-1])
    return S
########################################
##11.                                  #
##DET                                  # 
#                                      # 
######################################## 
def Compute_Cost_threshold_det(wine_llr,pi):
    Predictions = []
    for elem in wine_llr:
        if elem> -numpy.log(pi/(1-pi)) :
            Predictions.append(1)
        else:
            Predictions.append(0)
    return Predictions

def BayesEmpiricalRisk_threshold_det(pi,Cfn,Cfp, wine_llr,wine_llr_labels):
    DCF_array = []
    Predictions=Compute_Cost_threshold_det(wine_llr,pi)
    Confusion_array=[]
    for i in range(len(Predictions)):
        Confusion_array.append([Predictions[i], wine_llr_labels[i]])
    CM = Confusion_matrix(Confusion_array,2)
    FNR=CM[0,1]/(CM[0,1]+CM[1,1])
    FPR=CM[1,0]/(CM[0,0]+CM[1,0])
    return FNR, FPR
def draw_det_curve(score_gmm,label_gmm,score_lr,label_lr,score_rbf,label_rbf,z=0,c1='r',c2='b',c3='green'):
    FPR_gmm=[]
    FNR_gmm=[]
    FPR_lr=[]
    FNR_lr=[]
    FPR_rbf=[]
    FNR_rbf=[]
    axis=numpy.linspace(-4,4,100)
    for i in axis:
       
        p=1/(1+numpy.exp(-i))
        x=(p*1)/((1-p)*1)
        threshold=-numpy.log(x)
        fnr,fpr=BayesEmpiricalRisk_threshold_det(p,1,1,score_gmm,label_gmm)
        FPR_gmm.append(fpr*100)
        FNR_gmm.append(fnr*100)
        fnr,fpr=BayesEmpiricalRisk_threshold_det(p,1,1,score_lr,label_lr)
        FPR_lr.append(fpr*100)
        FNR_lr.append(fnr*100)
        fnr,fpr=BayesEmpiricalRisk_threshold_det(p,1,1,score_rbf,label_rbf)
        FPR_rbf.append(fpr*100)
        FNR_rbf.append(fnr*100)
    #sort based on first col arsort return the indexes
    #of the given arrray sorted using the indexes to indexthe array will sort it
    if(z==0):
        plt.figure()
    plt.plot(FPR_gmm,FNR_gmm, label='DET GMM', color=c1)
    plt.plot(FPR_lr,FNR_lr, label='DET LR', color=c2)
    plt.plot(FPR_rbf,FNR_rbf, label='DET RBF', color=c3)
    plt.legend(loc='upper right')
    plt.show()











# Execution
features_list = [ 'Fixed_Acidity', 'Volatile_Acidity', 'Citric_Acid', 'Residual_sugar', 'Chlorides', 'Free_Sulfur_Dioxide', 'Total_Sulfur_Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol', 'Quality']
print_statistics("Datasets/Train.txt", features_list)

###DATA EXPLORATION###

# Extracting dataset and labels
dataset_array = dataset_extraction("Datasets/Train.txt", features_list)[0]
labels_array = dataset_extraction("Datasets/Train.txt", features_list)[1]

# Isolated Features
fixed_acidity= dataset_array[0]
volatile_acidity= dataset_array[1]
citric_acid= dataset_array[2]
residual_sugar= dataset_array[3]
chlorides= dataset_array[4]
free_sulfur_dioxide= dataset_array[5]
total_sulfur_dioxide= dataset_array[6]
density= dataset_array[7]
pH= dataset_array[8]
sulphates= dataset_array[9]
alcohol= dataset_array[10]
l0 = (labels_array == 0)
badQuality = dataset_array[:, l0]
l1 = (labels_array == 1)
goodQuality = dataset_array[:, l1]

## Histrograms
# All together
plt.figure(figsize=(30, 20))

# Fixed Acidity
plt.subplot(4,4,1)
fixed_acidity_bad= badQuality[0]
plt.hist(fixed_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
fixed_acidity_good = goodQuality[0]
plt.hist(fixed_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Fixed Acidity')
plt.legend(loc = 'upper right')

# Volatile Acidity
plt.subplot(4,4,2)
volatile_acidity_bad= badQuality[1]
plt.hist(volatile_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
volatile_acidity_good = goodQuality[1]
plt.hist(volatile_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Volatile Acidity')
plt.legend(loc = 'upper right')

# Citric Acid
plt.subplot(4,4,3)
citric_acid_bad= badQuality[2]
plt.hist(citric_acid_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
citric_acid_good = goodQuality[2]
plt.hist(citric_acid_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Citric Acid')
plt.legend(loc = 'upper right')


# Residual Sugar
plt.subplot(4,4,4)
residual_sugar_bad= badQuality[3]
plt.hist(residual_sugar_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
residual_sugar_good = goodQuality[3]
plt.hist(residual_sugar_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Residual Sugar')
plt.legend(loc = 'upper right')

# Chlorides
plt.subplot(4,4,5)
chlorides_bad= badQuality[4]
plt.hist(chlorides_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
chlorides_good = goodQuality[4]
plt.hist(chlorides_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Chlorides')
plt.legend(loc = 'upper right')

# Free Sulfur Dioxide
plt.subplot(4,4,6)
free_sulfur_dioxide_bad= badQuality[5]
plt.hist(free_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
free_sulfur_dioxide_good = goodQuality[5]
plt.hist(free_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Free Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Total Sulfur Dioxide
plt.subplot(4,4,7)
total_sulfur_dioxide_bad= badQuality[6]
plt.hist(total_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
total_sulfur_dioxide_good = goodQuality[6]
plt.hist(total_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Total Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Density
plt.subplot(4,4,8)
density_bad= badQuality[7]
plt.hist(density_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
density_good = goodQuality[7]
plt.hist(density_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Density')
plt.legend(loc = 'upper right')

# pH
plt.subplot(4,4,9)
pH_bad= badQuality[8]
plt.hist(pH_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
pH_good = goodQuality[8]
plt.hist(pH_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('pH')
plt.legend(loc = 'upper right')

# Sulphates
plt.subplot(4,4,10)
sulphates_bad= badQuality[9]
plt.hist(sulphates_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
sulphates_good = goodQuality[9]
plt.hist(sulphates_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Sulphates')
plt.legend(loc = 'upper right')

# Alcohol
plt.subplot(4,4,11)
alcohol_bad= badQuality[10]
plt.hist(alcohol_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
alcohol_good = goodQuality[10]
plt.hist(alcohol_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Alcohol')
plt.legend(loc = 'upper right')

plt.suptitle("Features Histrograms")
plt.savefig(dirPath+'/features_histrograms.png')

# One by One

plt.figure(figsize=(20,15))
#fixed_acidity
fixed_acidity_bad= badQuality[0]
plt.hist(fixed_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
fixed_acidity_good = goodQuality[0]
plt.hist(fixed_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Fixed Acidity')
plt.legend(loc = 'upper right')

plt.title("Fixed Acidity")
plt.savefig(dirPath+'/fixed_acidity_histrogram.png')

plt.figure(figsize=(20,15))
#volatile_acidity
volatile_acidity_bad= badQuality[1]
plt.hist(volatile_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
volatile_acidity_good = goodQuality[1]
plt.hist(volatile_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Volatile Acidity')
plt.legend(loc = 'upper right')

plt.title("Volatile Acidity")
plt.savefig(dirPath+'/volatile_acidity_histrogram.png')

plt.figure(figsize=(20,15))
#citric_acid
citric_acid_bad= badQuality[2]
plt.hist(citric_acid_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
citric_acid_good = goodQuality[2]
plt.hist(citric_acid_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Citric Acid')
plt.legend(loc = 'upper right')

plt.title("Citric Acid")
plt.savefig(dirPath+'/citric_acid_histrogram.png')

plt.figure(figsize=(20,15))
#residual_sugar
residual_sugar_bad= badQuality[3]
plt.hist(residual_sugar_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
residual_sugar_good = goodQuality[3]
plt.hist(residual_sugar_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Residual Sugar')
plt.legend(loc = 'upper right')

plt.title("Residual Sugar")
plt.savefig(dirPath+'/residual_sugar_histrogram.png')


plt.figure(figsize=(20,15))
#chlorides
chlorides_bad= badQuality[4]
plt.hist(chlorides_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
chlorides_good = goodQuality[4]
plt.hist(chlorides_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Chlorides')
plt.legend(loc = 'upper right')

plt.title("Chlorides")
plt.savefig(dirPath+'/chlorides_histrogram.png')

plt.figure(figsize=(20,15))
#free_sulfur_dioxide
free_sulfur_dioxide_bad= badQuality[5]
plt.hist(free_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
free_sulfur_dioxide_good = goodQuality[5]
plt.hist(free_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Free Sulfur Dioxide')
plt.legend(loc = 'upper right')

plt.title("Free Sulfure Dioxide")
plt.savefig(dirPath+'/free_sulfure_dioxide_histrogram.png')

plt.figure(figsize=(20,15))
#total_sulfur_dioxide
total_sulfur_dioxide_bad= badQuality[6]
plt.hist(total_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
total_sulfur_dioxide_good = goodQuality[6]
plt.hist(total_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Total Sulfur Dioxide')
plt.legend(loc = 'upper right')

plt.title("Total Sulfure Dioxide")
plt.savefig(dirPath+'/total_sulfure_dioxide_histrogram.png')

plt.figure(figsize=(20,15))
#density
density_bad= badQuality[7]
plt.hist(density_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
density_good = goodQuality[7]
plt.hist(density_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Density')
plt.legend(loc = 'upper right')

plt.title("Density")
plt.savefig(dirPath+'/density_histrogram.png')

plt.figure(figsize=(20,15))
#pH
pH_bad= badQuality[8]
plt.hist(pH_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
pH_good = goodQuality[8]
plt.hist(pH_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('pH')
plt.legend(loc = 'upper right')

plt.title("pH")
plt.savefig(dirPath+'/pH_histrogram.png')

plt.figure(figsize=(20,15))
#sulphates
sulphates_bad= badQuality[9]
plt.hist(sulphates_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
sulphates_good = goodQuality[9]
plt.hist(sulphates_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Sulphates')
plt.legend(loc = 'upper right')

plt.title("Sulphates")
plt.savefig(dirPath+'/sulphates_histrogram.png')

plt.figure(figsize=(20,15))
#alcohol
alcohol_bad= badQuality[10]
plt.hist(alcohol_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
alcohol_good = goodQuality[10]
plt.hist(alcohol_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Alcohol')
plt.legend(loc = 'upper right')

plt.title("Alcohol")
plt.savefig(dirPath+'/alcohol_histrogram.png')

## Scatterplots of the 'most separable' features
#TODO: sistema stampa scatterplot

density_bad= badQuality[7]
density_good = goodQuality[7]
alcohol_bad= badQuality[10]
alcohol_good = goodQuality[10]
plt.scatter(density_bad, alcohol_bad, color='red',alpha= 0.5, label = 'Bad Quality')
plt.scatter(density_good, alcohol_good, color='green',alpha= 0.5, label = 'Good Quality')
plt.xlabel('Density')
plt.ylabel('Alcohol')
plt.legend(loc = 'lower right')

plt.title("Density/Alcohol")
plt.savefig(dirPath+'/density_alcohol_scatterplot.png')

residual_sugar_bad= badQuality[3]
residual_sugar_good = goodQuality[3]
alcohol_bad= badQuality[10]
alcohol_good = goodQuality[10]
plt.scatter(residual_sugar_bad, alcohol_bad, color='red',alpha= 0.5, label = 'Bad Quality')
plt.scatter(residual_sugar_good, alcohol_good, color='green',alpha= 0.5, label = 'Good Quality')
plt.xlabel('Residual_sugar')
plt.ylabel('Alcohol')
plt.legend(loc = 'lower right')

plt.title("Residual Sugar/Alcohol")
plt.savefig(dirPath+'/residualSugar_alcohol_scatterplot.png')

## Histograms of the centered dataset

# Centered dataset
dataset_array_centered = dataset_array - dataset_array.mean(1).reshape((dataset_array.shape[0], 1))

l0 = (labels_array == 0)
badQuality_centered = dataset_array_centered[:, l0]
l1 = (labels_array == 1)
goodQuality_centered = dataset_array_centered[:, l1]

# All together (centered)
plt.figure(figsize=(30, 20))

# Fixed Acidity
plt.subplot(4,4,1)
fixed_acidity_bad= badQuality_centered[0]
plt.hist(fixed_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
fixed_acidity_good = goodQuality_centered[0]
plt.hist(fixed_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Fixed Acidity')
plt.legend(loc = 'upper right')

# Volatile Acidity
plt.subplot(4,4,2)
volatile_acidity_bad= badQuality_centered[1]
plt.hist(volatile_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
volatile_acidity_good = goodQuality_centered[1]
plt.hist(volatile_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Volatile Acidity')
plt.legend(loc = 'upper right')

# Citric Acid
plt.subplot(4,4,3)
citric_acid_bad= badQuality_centered[2]
plt.hist(citric_acid_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
citric_acid_good = goodQuality_centered[2]
plt.hist(citric_acid_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Citric Acid')
plt.legend(loc = 'upper right')


# Residual Sugar
plt.subplot(4,4,4)
residual_sugar_bad= badQuality_centered[3]
plt.hist(residual_sugar_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
residual_sugar_good = goodQuality_centered[3]
plt.hist(residual_sugar_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Residual Sugar')
plt.legend(loc = 'upper right')

# Chlorides
plt.subplot(4,4,5)
chlorides_bad= badQuality_centered[4]
plt.hist(chlorides_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
chlorides_good = goodQuality_centered[4]
plt.hist(chlorides_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Chlorides')
plt.legend(loc = 'upper right')

# Free Sulfur Dioxide
plt.subplot(4,4,6)
free_sulfur_dioxide_bad= badQuality_centered[5]
plt.hist(free_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
free_sulfur_dioxide_good = goodQuality_centered[5]
plt.hist(free_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Free Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Total Sulfur Dioxide
plt.subplot(4,4,7)
total_sulfur_dioxide_bad= badQuality_centered[6]
plt.hist(total_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
total_sulfur_dioxide_good = goodQuality_centered[6]
plt.hist(total_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Total Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Density
plt.subplot(4,4,8)
density_bad= badQuality_centered[7]
plt.hist(density_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
density_good = goodQuality_centered[7]
plt.hist(density_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Density')
plt.legend(loc = 'upper right')

# pH
plt.subplot(4,4,9)
pH_bad= badQuality_centered[8]
plt.hist(pH_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
pH_good = goodQuality_centered[8]
plt.hist(pH_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('pH')
plt.legend(loc = 'upper right')

# Sulphates
plt.subplot(4,4,10)
sulphates_bad= badQuality_centered[9]
plt.hist(sulphates_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
sulphates_good = goodQuality_centered[9]
plt.hist(sulphates_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Sulphates')
plt.legend(loc = 'upper right')

# Alcohol
plt.subplot(4,4,11)
alcohol_bad= badQuality_centered[10]
plt.hist(alcohol_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
alcohol_good = goodQuality_centered[10]
plt.hist(alcohol_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Alcohol')
plt.legend(loc = 'upper right')

plt.suptitle("Features Histrograms (Centered Dataset")
plt.savefig(dirPath+'/features_histrograms.png')

## Boxplots

# All the features
plt.figure( figsize = (30,5) )
sns.boxplot(data = pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity, 
                    "Volatile_Acidity":volatile_acidity,
                    "Citric_Acidity":citric_acid,
                    "Residual_Sugar":residual_sugar,
                    "Chlorides":chlorides,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide,
                    "Density":density,
                    "pH":pH,
                    "Sulphates":sulphates,
                    "Alcohol":alcohol
                    }), palette="Paired", orient = "h")

plt.title("Boxplots")
plt.savefig(dirPath+'/boxplots.png')

# Without sulfur dioxide (Free and Total)
# (to show values intervals more in detail)
plt.figure( figsize = (15,5) )
sns.boxplot(data = pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity, 
                    "Volatile_Acidity":volatile_acidity,
                    "Citric_Acidity":citric_acid,
                    "Residual_Sugar":residual_sugar,
                    "Chlorides":chlorides,
                    #"Free_Sulfure_Dioxide":free_sulfur_dioxide,
                    #"Total_Sulfure_Dioxide":total_sulfur_dioxide,
                    "Density":density,
                    "pH":pH,
                    "Sulphates":sulphates,
                    "Alcohol":alcohol
                    }), palette="Paired", orient = "h")

plt.title("Boxplots(without sulfur dioxide")
plt.savefig(dirPath+'/boxplots_noSulfureDioxide.png')

## Correlation and HeatMap

#Dataset with label 0(bad quality)
l0 = (labels_array == 0)
badQuality = dataset_array[:, l0]
#'Bad' features
fixed_acidity_bad = badQuality[0]
volatile_acidity_bad = badQuality[1]
citric_acid_bad = badQuality[2]
residual_sugar_bad = badQuality[3]
chlorides_bad = badQuality[4]
free_sulfur_dioxide_bad = badQuality[5]
total_sulfur_dioxide_bad = badQuality[6]
density_bad = badQuality[7]
pH_bad = badQuality[8]
sulphates_bad = badQuality[9]
alcohol_bad = badQuality[10]

#Dataset with label 1(good quality)
l1 = (labels_array == 1)
goodQuality = dataset_array[:]
#'Good' features
fixed_acidity_good = goodQuality[0]
volatile_acidity_good = goodQuality[1]
citric_acid_good = goodQuality[2]
residual_sugar_good = goodQuality[3]
chlorides_good = goodQuality[4]
free_sulfur_dioxide_good = goodQuality[5]
total_sulfur_dioxide_good = goodQuality[6]
density_good = goodQuality[7]
pH_good = goodQuality[8]
sulphates_good = goodQuality[9]
alcohol_good = goodQuality[10]

# Total
corr_matrix = numpy.corrcoef(dataset_array)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity, 
                    "Volatile_Acidity":volatile_acidity,
                    "Citric_Acidity":citric_acid,
                    "Residual_Sugar":residual_sugar,
                    "Chlorides":chlorides,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide,
                    "Density":density,
                    "pH":pH,
                    "Sulphates":sulphates,
                    "Alcohol":alcohol
                    }).corr(), cmap="Reds", annot = True)

plt.title("Heatmap")
plt.savefig(dirPath+'/heatmap.png')


# Bad Quality Correlation 
corr_matrix_bad = numpy.corrcoef(badQuality)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity_bad, 
                    "Volatile_Acidity":volatile_acidity_bad,
                    "Citric_Acidity":citric_acid_bad,
                    "Residual_Sugar":residual_sugar_bad,
                    "Chlorides":chlorides_bad,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide_bad,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide_bad,
                    "Density":density_bad,
                    "pH":pH_bad,
                    "Sulphates":sulphates_bad,
                    "Alcohol":alcohol_bad
                    }).corr(), cmap="Blues", annot = True)

plt.title("Good Quality Heatmap")
plt.savefig(dirPath+'/heatmap_goodQuality.png')


# Good Quality Correlation 
corr_matrix_bad = numpy.corrcoef(goodQuality)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity_good, 
                    "Volatile_Acidity":volatile_acidity_good,
                    "Citric_Acidity":citric_acid_good,
                    "Residual_Sugar":residual_sugar_good,
                    "Chlorides":chlorides_good,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide_good,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide_good,
                    "Density":density_good,
                    "pH":pH_good,
                    "Sulphates":sulphates_good,
                    "Alcohol":alcohol_good
                    }).corr(), cmap="Greens", annot = True)

plt.title("Bad Quality Heatmap")
plt.savefig(dirPath+'/heatmap_badQuality.png')

normalized_dataset=normalization(dataset_array)

## Z-normalized dataset histograms

# Extracting Good Quality and Bad Quality Z-normalized datasets
l0 = (labels_array == 0) 
badQuality_normalized = normalized_dataset[:, l0]
l1 = (labels_array == 1)
goodQuality_normalized = normalized_dataset[:, l1]


# Plotting the histrograms
plt.figure(figsize=(30, 20))

# Fixed Acidity
plt.subplot(4,4,1)
fixed_acidity_bad= badQuality_normalized[0]
plt.hist(fixed_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
fixed_acidity_good = goodQuality_normalized[0]
plt.hist(fixed_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Fixed Acidity')
plt.legend(loc = 'upper right')

# Volatile Acidity
plt.subplot(4,4,2)
volatile_acidity_bad= badQuality_normalized[1]
plt.hist(volatile_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
volatile_acidity_good = goodQuality_normalized[1]
plt.hist(volatile_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Volatile Acidity')
plt.legend(loc = 'upper right')

# Citric Acid
plt.subplot(4,4,3)
citric_acid_bad= badQuality_normalized[2]
plt.hist(citric_acid_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
citric_acid_good = goodQuality_normalized[2]
plt.hist(citric_acid_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Citric Acid')
plt.legend(loc = 'upper right')


# Residual Sugar
plt.subplot(4,4,4)
residual_sugar_bad= badQuality_normalized[3]
plt.hist(residual_sugar_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
residual_sugar_good = goodQuality_normalized[3]
plt.hist(residual_sugar_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Residual Sugar')
plt.legend(loc = 'upper right')

# Chlorides
plt.subplot(4,4,5)
chlorides_bad= badQuality_normalized[4]
plt.hist(chlorides_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
chlorides_good = goodQuality_normalized[4]
plt.hist(chlorides_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Chlorides')
plt.legend(loc = 'upper right')

# Free Sulfur Dioxide
plt.subplot(4,4,6)
free_sulfur_dioxide_bad= badQuality_normalized[5]
plt.hist(free_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
free_sulfur_dioxide_good = goodQuality_normalized[5]
plt.hist(free_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Free Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Total Sulfur Dioxide
plt.subplot(4,4,7)
total_sulfur_dioxide_bad= badQuality_normalized[6]
plt.hist(total_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
total_sulfur_dioxide_good = goodQuality_normalized[6]
plt.hist(total_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Total Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Density
plt.subplot(4,4,8)
density_bad= badQuality_normalized[7]
plt.hist(density_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
density_good = goodQuality_normalized[7]
plt.hist(density_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Density')
plt.legend(loc = 'upper right')

# pH
plt.subplot(4,4,9)
pH_bad= badQuality_normalized[8]
plt.hist(pH_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
pH_good = goodQuality_normalized[8]
plt.hist(pH_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('pH')
plt.legend(loc = 'upper right')

# Sulphates
plt.subplot(4,4,10)
sulphates_bad= badQuality_normalized[9]
plt.hist(sulphates_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
sulphates_good = goodQuality_normalized[9]
plt.hist(sulphates_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Sulphates')
plt.legend(loc = 'upper right')

# Alcohol
plt.subplot(4,4,11)
alcohol_bad= badQuality_normalized[10]
plt.hist(alcohol_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
alcohol_good = goodQuality_normalized[10]
plt.hist(alcohol_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Alcohol')
plt.legend(loc = 'upper right')

plt.title("Z-Normalized Dataset histrograms")
plt.savefig(dirPath+"/zNormalized_dataset_histrograms.png")



dataset_gaussianized = ppf_gaussianization(features_list,normalized_dataset)

# 'Gaussianized' dataset histograms

# Extracting Good Quality and Bad Quality Z-normalized datasets
l0 = (labels_array == 0) 
badQuality_gaussianized = dataset_gaussianized[:, l0]
l1 = (labels_array == 1)
goodQuality_gaussianized = dataset_gaussianized[:, l1]


# Plotting the histrograms
plt.figure(figsize=(30, 20))

# Fixed Acidity
plt.subplot(4,4,1)
fixed_acidity_bad= badQuality_gaussianized[0]
plt.hist(fixed_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
fixed_acidity_good = goodQuality_gaussianized[0]
plt.hist(fixed_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Fixed Acidity')
plt.legend(loc = 'upper right')

# Volatile Acidity
plt.subplot(4,4,2)
volatile_acidity_bad= badQuality_gaussianized[1]
plt.hist(volatile_acidity_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
volatile_acidity_good = goodQuality_gaussianized[1]
plt.hist(volatile_acidity_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Volatile Acidity')
plt.legend(loc = 'upper right')

# Citric Acid
plt.subplot(4,4,3)
citric_acid_bad= badQuality_gaussianized[2]
plt.hist(citric_acid_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
citric_acid_good = goodQuality_gaussianized[2]
plt.hist(citric_acid_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Citric Acid')
plt.legend(loc = 'upper right')


# Residual Sugar
plt.subplot(4,4,4)
residual_sugar_bad= badQuality_gaussianized[3]
plt.hist(residual_sugar_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
residual_sugar_good = goodQuality_gaussianized[3]
plt.hist(residual_sugar_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Residual Sugar')
plt.legend(loc = 'upper right')

# Chlorides
plt.subplot(4,4,5)
chlorides_bad= badQuality_gaussianized[4]
plt.hist(chlorides_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
chlorides_good = goodQuality_gaussianized[4]
plt.hist(chlorides_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Chlorides')
plt.legend(loc = 'upper right')

# Free Sulfur Dioxide
plt.subplot(4,4,6)
free_sulfur_dioxide_bad= badQuality_gaussianized[5]
plt.hist(free_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
free_sulfur_dioxide_good = goodQuality_gaussianized[5]
plt.hist(free_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Free Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Total Sulfur Dioxide
plt.subplot(4,4,7)
total_sulfur_dioxide_bad= badQuality_gaussianized[6]
plt.hist(total_sulfur_dioxide_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
total_sulfur_dioxide_good = goodQuality_gaussianized[6]
plt.hist(total_sulfur_dioxide_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Total Sulfur Dioxide')
plt.legend(loc = 'upper right')

# Density
plt.subplot(4,4,8)
density_bad= badQuality_gaussianized[7]
plt.hist(density_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
density_good = goodQuality_gaussianized[7]
plt.hist(density_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Density')
plt.legend(loc = 'upper right')

# pH
plt.subplot(4,4,9)
pH_bad= badQuality_gaussianized[8]
plt.hist(pH_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
pH_good = goodQuality_gaussianized[8]
plt.hist(pH_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('pH')
plt.legend(loc = 'upper right')

# Sulphates
plt.subplot(4,4,10)
sulphates_bad= badQuality_gaussianized[9]
plt.hist(sulphates_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
sulphates_good = goodQuality_gaussianized[9]
plt.hist(sulphates_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Sulphates')
plt.legend(loc = 'upper right')

# Alcohol
plt.subplot(4,4,11)
alcohol_bad= badQuality_gaussianized[10]
plt.hist(alcohol_bad, bins=50, density = True, color='red', alpha= 0.5, label='Bad')
alcohol_good = goodQuality_gaussianized[10]
plt.hist(alcohol_good, bins=50, density = True, color='green', alpha= 0.5, label='Good')
plt.xlabel('Alcohol')
plt.legend(loc = 'upper right')

plt.title("Gaussianized Dataset Histrograms")
plt.savefig(dirPath+"/gaussianized_dataset_histograms.png")

## Correlation and HeatMap of 'Gaussiannized' dataset

# Isolated features
fixed_acidity= dataset_gaussianized[0]
volatile_acidity= dataset_gaussianized[1]
citric_acid= dataset_gaussianized[2]
residual_sugar= dataset_gaussianized[3]
chlorides= dataset_gaussianized[4]
free_sulfur_dioxide= dataset_gaussianized[5]
total_sulfur_dioxide= dataset_gaussianized[6]
density= dataset_gaussianized[7]
pH= dataset_gaussianized[8]
sulphates= dataset_gaussianized[9]
alcohol= dataset_gaussianized[10]

# Bad quality  features
fixed_acidity_bad = badQuality_gaussianized[0]
volatile_acidity_bad = badQuality_gaussianized[1]
citric_acid_bad = badQuality_gaussianized[2]
residual_sugar_bad = badQuality_gaussianized[3]
chlorides_bad = badQuality_gaussianized[4]
free_sulfur_dioxide_bad = badQuality_gaussianized[5]
total_sulfur_dioxide_bad = badQuality_gaussianized[6]
density_bad = badQuality_gaussianized[7]
pH_bad = badQuality_gaussianized[8]
sulphates_bad = badQuality_gaussianized[9]
alcohol_bad = badQuality_gaussianized[10]

# Good quality features
fixed_acidity_good = goodQuality_gaussianized[0]
volatile_acidity_good = goodQuality_gaussianized[1]
citric_acid_good = goodQuality_gaussianized[2]
residual_sugar_good = goodQuality_gaussianized[3]
chlorides_good = goodQuality_gaussianized[4]
free_sulfur_dioxide_good = goodQuality_gaussianized[5]
total_sulfur_dioxide_good = goodQuality_gaussianized[6]
density_good = goodQuality_gaussianized[7]
pH_good = goodQuality_gaussianized[8]
sulphates_good = goodQuality_gaussianized[9]
alcohol_good = goodQuality_gaussianized[10]

# Heatmap of the whole gaussianized dataset
corr_matrix = numpy.corrcoef(dataset_gaussianized)
plt.figure(figsize=(10, 8))

sns.heatmap(pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity, 
                    "Volatile_Acidity":volatile_acidity,
                    "Citric_Acidity":citric_acid,
                    "Residual_Sugar":residual_sugar,
                    "Chlorides":chlorides,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide,
                    "Density":density,
                    "pH":pH,
                    "Sulphates":sulphates,
                    "Alcohol":alcohol
                    }).corr(), cmap="Reds", annot = True)

plt.title("Heatmap of the Gaussianized Dataset")
plt.savefig(dirPath+"/gaussianized_heatmap.png")

# Heatmap of bad quality gaussianized dataset
corr_matrix_bad = numpy.corrcoef(badQuality_gaussianized)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity_bad, 
                    "Volatile_Acidity":volatile_acidity_bad,
                    "Citric_Acidity":citric_acid_bad,
                    "Residual_Sugar":residual_sugar_bad,
                    "Chlorides":chlorides_bad,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide_bad,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide_bad,
                    "Density":density_bad,
                    "pH":pH_bad,
                    "Sulphates":sulphates_bad,
                    "Alcohol":alcohol_bad
                    }).corr(), cmap="Blues", annot = True)

plt.title("Heatmap of the Bad Quality Gaussianized Dataset")
plt.savefig(dirPath+"/gaussianized_badQuality_heatmap.png")

# Heatmap of good quality gaussianized dataset 
corr_matrix_bad = numpy.corrcoef(goodQuality_gaussianized)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame({
                    "Fixed_Acidity":fixed_acidity_good, 
                    "Volatile_Acidity":volatile_acidity_good,
                    "Citric_Acidity":citric_acid_good,
                    "Residual_Sugar":residual_sugar_good,
                    "Chlorides":chlorides_good,
                    "Free_Sulfure_Dioxide":free_sulfur_dioxide_good,
                    "Total_Sulfure_Dioxide":total_sulfur_dioxide_good,
                    "Density":density_good,
                    "pH":pH_good,
                    "Sulphates":sulphates_good,
                    "Alcohol":alcohol_good
                    }).corr(), cmap="Greens", annot = True)

plt.title("Heatmap of the Good Quality Gaussianized Dataset")
plt.savefig(dirPath+"/gaussianized_goodQuality_heatmap.png")

dataset_mu = vcol(dataset_gaussianized.mean(1))
dataset_covariance_matrix = covariance_numpy(dataset_gaussianized)
#creating the dataset with only 2 variables (for plotting purposes) -- Eigh
pca_matrix = PCA_eigh(dataset_covariance_matrix, 2)
dataset_with_pca = numpy.dot(pca_matrix.T, dataset_gaussianized)

#extracting bad-quality
l0 = (labels_array == 0)
badQuality_pca = dataset_with_pca[:, l0]

#extracting good-quality
l1 = (labels_array == 1)
goodQuality_pca = dataset_with_pca[:, l1]

plt.figure(figsize=(10,8))
plt.scatter(badQuality_pca[0], badQuality_pca[1], color='red', label = 'badQuality')
plt.scatter(goodQuality_pca[0], goodQuality_pca[1], color='green', label = 'goodQuality')
plt.legend(loc = 'lower left')

plt.title("PCA (Eigh) with 2  variables")
plt.savefig(dirPath+"/pcaEigh.png")

#creating the dataset with only 2 variables (for plotting purposes) -- SVD
pca_matrix2 = PCA_svd(dataset_covariance_matrix, 2)
dataset_with_pca2 = numpy.dot(pca_matrix2.T, dataset_gaussianized)

#extracting bad-quality
l0 = (labels_array == 0)
badQuality_pca2 = dataset_with_pca2[:, l0]

#extracting good-quality
l1 = (labels_array == 1)
goodQuality_pca2 = dataset_with_pca2[:, l1]

plt.figure(figsize=(10,8))
plt.scatter(badQuality_pca2[0], badQuality_pca2[1], color='red', label = 'badQuality')
plt.scatter(goodQuality_pca2[0], goodQuality_pca2[1], color='green', label = 'goodQuality')
plt.legend(loc = 'lower left')

plt.title("PCA (SVD) with 2  variables")
plt.savefig(dirPath+"/pcaSVD.png")


# PCA WITH 3 DIMENSIONS

pca_matrix_3d = PCA_eigh(dataset_covariance_matrix, 3)
dataset_with_pca_3d = numpy.dot(pca_matrix_3d.T, dataset_gaussianized)

#extracting bad-quality
l0 = (labels_array == 0)
badQuality_pca_3d = dataset_with_pca_3d[:, l0]

#extracting good-quality
l1 = (labels_array == 1)
goodQuality_pca_3d = dataset_with_pca_3d[:, l1]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(badQuality_pca_3d[0], badQuality_pca_3d[1], badQuality_pca_3d[2], c='red', label = 'badQuality')
ax.scatter(goodQuality_pca_3d[0], goodQuality_pca_3d[1], goodQuality_pca_3d[2], c='green', label = 'goodQuality')

plt.title("PCA (Eigh) with 3 variables")
plt.savefig(dirPath+"/PCA_eigh_3d.png")

pca_matrix2_3d = PCA_svd(dataset_covariance_matrix, 3)
dataset_with_pca2_3d = numpy.dot(pca_matrix2_3d.T, dataset_gaussianized)

#extracting bad-quality
l0 = (labels_array == 0)
badQuality_pca2_3d = dataset_with_pca2_3d[:, l0]

#extracting good-quality
l1 = (labels_array == 1)
goodQuality_pca2_3d = dataset_with_pca2_3d[:, l1]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(badQuality_pca2_3d[0], badQuality_pca2_3d[1], badQuality_pca2_3d[2], c='red', label = 'badQuality')
ax.scatter(goodQuality_pca2_3d[0], goodQuality_pca2_3d[1], goodQuality_pca2_3d[2], c='green', label = 'goodQuality')
plt.title("PCA (SVD) with 3 variables")
plt.savefig(dirPath+"/PCA_SVD_3d.png")

##Single Fold with original dataset
(DTR, LTR), (DTE, LTE) = split_db_3to1(dataset_array, labels_array)
LTR=LTR.reshape(1471,)
LTE=LTE.reshape(368,)
LTR_0=(labels_array.reshape(1839,)==0)
DTR_0=dataset_array[:,LTR_0]
LTR_1=(labels_array.reshape(1839,)==1)
DTR_1=dataset_array[:,LTR_1]

# Class 0
LTR_0 = ( LTR == 0 )
DTR_0 = DTR[: , LTR_0]
# Class 1
LTR_1 = ( LTR == 1 )
DTR_1 = DTR[: , LTR_1]
# Function to compute data mean column vector
def dataset_mu(dataset):
    return vcol(dataset.mean(1))
mu_0 = dataset_mu(DTR_0)
cov_0 = covariance_numpy(DTR_0)

mu_1 = dataset_mu(DTR_1)
cov_1 = covariance_numpy(DTR_1)

wine_llr_labels=LTE
means = [mu_0, mu_1]
covariances = [cov_0, cov_1]
print("----------------SINGLE FOLD RAW FEATURES")
print("----------MVG FULL")
wine_llr_mvg = Compute_llr(score_matrix_logdensity_mvg(DTE, means, covariances)) ##mvg full covariances
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_mvg,wine_llr_labels))
print("----------NAIVE")
bayes_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## bayes diag covariances
wine_llr_bayes = Compute_llr(score_matrix_logdensity_mvg(DTE, means, bayes_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_bayes,wine_llr_labels))
print("----------TIED")
wine_llr_tied = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied,wine_llr_labels))
print("----------TIED DIAG")
tied_diag_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## tied diag covariances
wine_llr_tied_diag = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, tied_diag_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied_diag,wine_llr_labels))

##Single Fold with dataset gaussianized
(DTR, LTR), (DTE, LTE) = split_db_3to1(dataset_gaussianized, labels_array)

LTR = LTR.reshape(1471,)
LTE = LTE.reshape(368,)
# Class 0
LTR_0 = ( LTR == 0 )
DTR_0 = DTR[: , LTR_0]
# Class 1
LTR_1 = ( LTR == 1 )
DTR_1 = DTR[: , LTR_1]

mu_0 = dataset_mu(DTR_0)
cov_0 = covariance_numpy(DTR_0)

mu_1 = dataset_mu(DTR_1)
cov_1 = covariance_numpy(DTR_1)
means = [mu_0, mu_1]
covariances = [cov_0, cov_1]
print("----------------SINGLE FOLD GAUSS FEATURES")
print("----------MVG FULL")
wine_llr_mvg = Compute_llr(score_matrix_logdensity_mvg(DTE, means, covariances)) ##mvg full covariances
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_mvg,wine_llr_labels))
print("----------NAIVE")
bayes_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## bayes diag covariances
wine_llr_bayes = Compute_llr(score_matrix_logdensity_mvg(DTE, means, bayes_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_bayes,wine_llr_labels))
print("----------TIED")
wine_llr_tied = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied,wine_llr_labels))
print("----------TIED DIAG")
tied_diag_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## tied diag covariances
wine_llr_tied_diag = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, tied_diag_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied_diag,wine_llr_labels))

#Single Fold with dataset with PCA 10
dataset_mu_gaussianized = vcol(dataset_gaussianized.mean(1))
dataset_covariance_gaussianized = covariance_numpy(dataset_gaussianized)
pca_matrix = PCA_eigh(dataset_covariance_gaussianized, 10)
dataset_with_pca10 = numpy.dot(pca_matrix.T, dataset_gaussianized)
(DTR, LTR), (DTE, LTE) = split_db_3to1(dataset_with_pca10, labels_array)
LTR = LTR.reshape(1471,)
LTE = LTE.reshape(368,)
# Class 0
LTR_0 = ( LTR == 0 )
DTR_0 = DTR[: , LTR_0]
# Class 1
LTR_1 = ( LTR == 1 )
DTR_1 = DTR[: , LTR_1]

mu_0 = dataset_mu(DTR_0)
cov_0 = covariance_numpy(DTR_0)

mu_1 = dataset_mu(DTR_1)
cov_1 = covariance_numpy(DTR_1)

means = [mu_0, mu_1]
covariances = [cov_0, cov_1]
print("----------------SINGLE FOLD GAUSS FEATURES PCA 10")
print("----------MVG FULL")
wine_llr_mvg = Compute_llr(score_matrix_logdensity_mvg(DTE, means, covariances)) ##mvg full covariances
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_mvg,wine_llr_labels))
print("----------NAIVE")
bayes_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## bayes diag covariances
wine_llr_bayes = Compute_llr(score_matrix_logdensity_mvg(DTE, means, bayes_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_bayes,wine_llr_labels))
print("----------TIED")
wine_llr_tied = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied,wine_llr_labels))
print("----------TIED DIAG")
tied_diag_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## tied diag covariances
wine_llr_tied_diag = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, tied_diag_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied_diag,wine_llr_labels))

#Single Fold with dataset with PCA 9
dataset_mu_gaussianized = vcol(dataset_gaussianized.mean(1))
dataset_covariance_gaussianized = covariance_numpy(dataset_gaussianized)
pca_matrix = PCA_eigh(dataset_covariance_gaussianized, 9)
dataset_with_pca9 = numpy.dot(pca_matrix.T, dataset_gaussianized)
(DTR, LTR), (DTE, LTE) = split_db_3to1(dataset_with_pca9, labels_array)
LTR = LTR.reshape(1471,)
LTE = LTE.reshape(368,)
# Class 0
LTR_0 = ( LTR == 0 )
DTR_0 = DTR[: , LTR_0]
# Class 1
LTR_1 = ( LTR == 1 )
DTR_1 = DTR[: , LTR_1]

mu_0 = dataset_mu(DTR_0)
cov_0 = covariance_numpy(DTR_0)

mu_1 = dataset_mu(DTR_1)
cov_1 = covariance_numpy(DTR_1)
means = [mu_0, mu_1]
covariances = [cov_0, cov_1]
print("----------------SINGLE FOLD GAUSS FEATURES PCA 9")
print("----------MVG FULL")
wine_llr_mvg = Compute_llr(score_matrix_logdensity_mvg(DTE, means, covariances)) ##mvg full covariances
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_mvg,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_mvg,wine_llr_labels))
print("----------NAIVE")
bayes_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## bayes diag covariances
wine_llr_bayes = Compute_llr(score_matrix_logdensity_mvg(DTE, means, bayes_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_bayes,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_bayes,wine_llr_labels))
print("----------TIED")
wine_llr_tied = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied,wine_llr_labels))
print("----------TIED DIAG")
tied_diag_covariances=[element * numpy.identity(element.shape[0]) for element in covariances] ## tied diag covariances
wine_llr_tied_diag = Compute_llr(score_matrix_logdensity_tied(DTR,LTR,DTE, means, tied_diag_covariances))
print(BayesEmpiricalRisk_threshold(0.5,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.1,1,1,wine_llr_tied_diag,wine_llr_labels))
print(BayesEmpiricalRisk_threshold(0.9,1,1,wine_llr_tied_diag,wine_llr_labels))

k=5
##K-fold with raw features
raw_data_kfold = kfold(dataset_array, labels_array.flatten(),k)
##K-fold with dataset gaussianized
gauss_data_kfold = kfold(dataset_gaussianized, labels_array.flatten(),k)
##K-fold with dataset gaussianized PCA=10
dataset_mu_gaussianized = vcol(dataset_gaussianized.mean(1))
dataset_covariance_gaussianized = covariance_numpy(dataset_gaussianized)
pca_matrix = PCA_eigh(dataset_covariance_gaussianized, 10)
dataset_with_pca10 = numpy.dot(pca_matrix.T, dataset_gaussianized)
gauss_data_kfold_pca10 = kfold(dataset_with_pca10, labels_array.flatten(),k)
##K-fold with dataset gaussianized PCA=9
dataset_mu_gaussianized = vcol(dataset_gaussianized.mean(1))
dataset_covariance_gaussianized = covariance_numpy(dataset_gaussianized)
pca_matrix = PCA_eigh(dataset_covariance_gaussianized, 9)
dataset_with_pca9 = numpy.dot(pca_matrix.T, dataset_gaussianized)
gauss_data_kfold_pca9 = kfold(dataset_with_pca9, labels_array.flatten(),k)
print("K FOLD------------------------------------")
print("RAW DATA------------------------------MVG")
apply_MVG_Classifier(k,raw_data_kfold,0.5,1,1)
apply_MVG_Classifier(k,raw_data_kfold,0.1,1,1)
apply_MVG_Classifier(k,raw_data_kfold,0.9,1,1)
print("GAUSS DATA------------------------------MVG")
apply_MVG_Classifier(k,gauss_data_kfold,0.5,1,1)
apply_MVG_Classifier(k,gauss_data_kfold,0.1,1,1)
apply_MVG_Classifier(k,gauss_data_kfold,0.9,1,1)
print("GAUSS DATA PCA10------------------------------MVG")
apply_MVG_Classifier(k,gauss_data_kfold_pca10,0.5,1,1)
apply_MVG_Classifier(k,gauss_data_kfold_pca10,0.1,1,1)
apply_MVG_Classifier(k,gauss_data_kfold_pca10,0.9,1,1)
print("GAUSS DATA PCA9------------------------------MVG")
apply_MVG_Classifier(k,gauss_data_kfold_pca9,0.5,1,1)
apply_MVG_Classifier(k,gauss_data_kfold_pca9,0.1,1,1)
apply_MVG_Classifier(k,gauss_data_kfold_pca9,0.9,1,1)
print("RAW DATA------------------------------NAIVE")
apply_Naive_Bayes_Classifier(k,raw_data_kfold,0.5,1,1)
apply_Naive_Bayes_Classifier(k,raw_data_kfold,0.1,1,1)
apply_Naive_Bayes_Classifier(k,raw_data_kfold,0.9,1,1)
print("GAUSS DATA------------------------------NAIVE")
apply_Naive_Bayes_Classifier(k,gauss_data_kfold,0.5,1,1)
apply_Naive_Bayes_Classifier(k,gauss_data_kfold,0.1,1,1)
apply_Naive_Bayes_Classifier(k,gauss_data_kfold,0.9,1,1)
print("GAUSS DATA PCA10------------------------------NAIVE")
apply_Naive_Bayes_Classifier(k,gauss_data_kfold_pca10,0.5,1,1)
apply_Naive_Bayes_Classifier(k,gauss_data_kfold_pca10,0.1,1,1)
apply_Naive_Bayes_Classifier(k,gauss_data_kfold_pca10,0.9,1,1)
print("GAUSS DATA PCA9------------------------------NAIVE")
apply_Naive_Bayes_Classifier(k,gauss_data_kfold_pca9,0.5,1,1)
apply_Naive_Bayes_Classifier(k,gauss_data_kfold_pca9,0.1,1,1)
apply_Naive_Bayes_Classifier(k,gauss_data_kfold_pca9,0.9,1,1)
print("RAW DATA------------------------------TIED-FULL")
apply_Tied_Full_Classifier(k,raw_data_kfold,0.5,1,1)
apply_Tied_Full_Classifier(k,raw_data_kfold,0.1,1,1)
apply_Tied_Full_Classifier(k,raw_data_kfold,0.9,1,1)
print("GAUSS DATA------------------------------TIED-FULL")
apply_Tied_Full_Classifier(k,gauss_data_kfold,0.5,1,1)
apply_Tied_Full_Classifier(k,gauss_data_kfold,0.1,1,1)
apply_Tied_Full_Classifier(k,gauss_data_kfold,0.9,1,1)
print("GAUSS DATA PCA10------------------------------TIED-FULL")
apply_Tied_Full_Classifier(k,gauss_data_kfold_pca10,0.5,1,1)
apply_Tied_Full_Classifier(k,gauss_data_kfold_pca10,0.1,1,1)
apply_Tied_Full_Classifier(k,gauss_data_kfold_pca10,0.9,1,1)
print("GAUSS DATA PCA9------------------------------TIED-FULL")
apply_Tied_Full_Classifier(k,gauss_data_kfold_pca9,0.5,1,1)
apply_Tied_Full_Classifier(k,gauss_data_kfold_pca9,0.1,1,1)
apply_Tied_Full_Classifier(k,gauss_data_kfold_pca9,0.9,1,1)
print("RAW DATA------------------------------TIED-DIAG")
apply_Tied_Diag_Classifier(k,raw_data_kfold,0.5,1,1)
apply_Tied_Diag_Classifier(k,raw_data_kfold,0.1,1,1)
apply_Tied_Diag_Classifier(k,raw_data_kfold,0.9,1,1)
print("GAUSS DATA------------------------------TIED-DIAG")
apply_Tied_Diag_Classifier(k,gauss_data_kfold,0.5,1,1)
apply_Tied_Diag_Classifier(k,gauss_data_kfold,0.1,1,1)
apply_Tied_Diag_Classifier(k,gauss_data_kfold,0.9,1,1)
print("GAUSS DATA PCA10------------------------------TIED-DIAG")
apply_Tied_Diag_Classifier(k,gauss_data_kfold_pca10,0.5,1,1)
apply_Tied_Diag_Classifier(k,gauss_data_kfold_pca10,0.1,1,1)
apply_Tied_Diag_Classifier(k,gauss_data_kfold_pca10,0.9,1,1)
print("GAUSS DATA PCA9------------------------------TIED-DIAG")
apply_Tied_Diag_Classifier(k,gauss_data_kfold_pca9,0.5,1,1)
apply_Tied_Diag_Classifier(k,gauss_data_kfold_pca9,0.1,1,1)
apply_Tied_Diag_Classifier(k,gauss_data_kfold_pca9,0.9,1,1)
#logistic regression
k=5
lambdas = [1.E-6, 1.E-5,1.E-4,1.E-3,1.E-2,1.E-1,1,10,100,1000,10000,100000]
print("SINGLE FOLD--------------------------------------------------------")
print("RAW DATA------------------------------LOGISTIC REGRESSION")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,raw_data_kfold,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,raw_data_kfold,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,raw_data_kfold,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("GAUSS DATA------------------------------LOGISTIC REGRESSION")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,gauss_data_kfold,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,gauss_data_kfold,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,gauss_data_kfold,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("GAUSS DATA PCA 10------------------------------LOGISTIC REGRESSION")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,gauss_data_kfold_pca10,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,gauss_data_kfold_pca10,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(1,gauss_data_kfold_pca10,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("K FOLD--------------------------------------------------------")
k=5
print("RAW DATA------------------------------LOGISTIC REGRESSION")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,raw_data_kfold,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,raw_data_kfold,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,raw_data_kfold,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("GAUSS DATA------------------------------LOGISTIC REGRESSION")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,gauss_data_kfold,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,gauss_data_kfold,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,gauss_data_kfold,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("GAUSS DATA PCA 10------------------------------LOGISTIC REGRESSION")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,gauss_data_kfold_pca10,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,gauss_data_kfold_pca10,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg(k,gauss_data_kfold_pca10,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
#QUADRATIC LOGISTIC REGRESSION

print("K FOLD --------------------------------------------------------")
k=5
print("RAW DATA------------------------------LOGISTIC REGRESSION QUADRATIC")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,raw_data_kfold,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,raw_data_kfold,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,raw_data_kfold,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
k=5
print("GAUSS DATA------------------------------LOGISTIC REGRESSION QUADRATIC")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,gauss_data_kfold,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,gauss_data_kfold,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,gauss_data_kfold,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
k=5
print("GAUSS DATA PCA10------------------------------LOGISTIC REGRESSION QUADRATIC")
print("PI T: 0.5")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,gauss_data_kfold_pca10,0.5)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.1")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,gauss_data_kfold_pca10,0.1)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
print("PI T: 0.9")
dcf_along_lambda_05, dcf_along_lambda_01, dcf_along_lambda_09=apply_score_matrix_logReg_quadratic(k,gauss_data_kfold_pca10,0.9)
plt.figure(figsize=(10,8))
plt.xscale('log')
plt.xticks(lambdas)
plt.xlabel("λ")
plt.ylabel("minDCF")
plt.plot(lambdas, dcf_along_lambda_05, color="red", label="π = 0.5")
plt.plot(lambdas, dcf_along_lambda_01, color="blue", label="π = 0.1")
plt.plot(lambdas, dcf_along_lambda_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
#SVM 
#RAW DATA
cVector = numpy.array([1.E-3,1.E-2,2.E-2,4.E-2,6.E-2,8.E-2,1.E-1,2.E-1,4.E-1,6.E-1,8.E-1,1,10,100,1000])
print("RAW DATA KFOLD----------------SVM LINEAR")
dcf_along_c_05, dcf_along_c_01, dcf_along_c_09=apply_score_matrix_SVM(k,raw_data_kfold)
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("C")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_c_05, color="red", label="π = 0.5")
plt.plot(cVector, dcf_along_c_01, color="blue", label="π = 0.1")
plt.plot(cVector, dcf_along_c_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
#GAUSS DATA
print("GAUSS DATA KFOLD----------------SVM LINEAR")
dcf_along_c_05, dcf_along_c_01, dcf_along_c_09=apply_score_matrix_SVM(k,gauss_data_kfold)
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("C")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_c_05, color="red", label="π = 0.5")
plt.plot(cVector, dcf_along_c_01, color="blue", label="π = 0.1")
plt.plot(cVector, dcf_along_c_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
#GAUSS DATA PCA10
print("GAUSS DATA KFOLD PCA 10----------------SVM LINEAR")
dcf_along_c_05, dcf_along_c_01, dcf_along_c_09=apply_score_matrix_SVM(k,gauss_data_kfold_pca10)
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("C")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_c_05, color="red", label="π = 0.5")
plt.plot(cVector, dcf_along_c_01, color="blue", label="π = 0.1")
plt.plot(cVector, dcf_along_c_09, color="green", label="π = 0.9")
plt.legend(loc = 'lower right')
#SVM QUADRATIC
#RAW DATA
print("RAW DATA KFOLD----------------SVM POLY d=2")
dcf_along_c_05, dcf_along_c_01, dcf_along_c_09=apply_score_matrix_SVM_quadratic(k,raw_data_kfold)
#GAUSS DATA
print("GAUSS DATA KFOLD----------------SVM POLY d=2")
dcf_along_c_05, dcf_along_c_01, dcf_along_c_09=apply_score_matrix_SVM_quadratic(k,gauss_data_kfold)
#GAUSS DATA PCA10
print("GAUSS DATA KFOLD PCA 10----------------SVM POLY d=2")
dcf_along_c_05, dcf_along_c_01, dcf_along_c_09=apply_score_matrix_SVM_quadratic(k,gauss_data_kfold_pca10)
#SVM RBF
#RAW
print("RAW DATA KFOLD----------------SVM RBF")
dcf_along_gamma_10, dcf_along_gamma_9, dcf_along_gamma_8, dcf_along_gamma_7, dcf_along_gamma_6, dcf_along_gamma_5, dcf_along_gamma_4, dcf_along_gamma_3, dcf_along_gamma_2, dcf_along_gamma_1=apply_score_matrix_SVM_RBF(k,raw_data_kfold)
#GAUSS
print("GAUSS DATA KFOLD----------------SVM RBF")
dcf_along_gamma_10, dcf_along_gamma_9, dcf_along_gamma_8, dcf_along_gamma_7, dcf_along_gamma_6, dcf_along_gamma_5, dcf_along_gamma_4, dcf_along_gamma_3, dcf_along_gamma_2, dcf_along_gamma_1=apply_score_matrix_SVM_RBF(k,gauss_data_kfold)
cVector= numpy.array([1.E-3,1.E-2,5.E-2,1.E-1,5.E-1,8.E-1,1,10,100,1000])
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("c")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_gamma_5, color="blue", label="logγ = -5")
plt.plot(cVector, dcf_along_gamma_4, color="green", label="logγ = -4")
plt.plot(cVector, dcf_along_gamma_3, color="red", label="logγ = -3")
plt.plot(cVector, dcf_along_gamma_2, color="cyan", label="logγ = -2")
plt.plot(cVector, dcf_along_gamma_1, color="gray", label="logγ = -1")
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("c")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_gamma_6, color="blue", label="logγ = -6")
plt.plot(cVector, dcf_along_gamma_7, color="green", label="logγ = -7")
plt.plot(cVector, dcf_along_gamma_8, color="cyan", label="logγ = -8")
plt.plot(cVector, dcf_along_gamma_9, color="yellow", label="logγ = -9")
plt.plot(cVector, dcf_along_gamma_10, color="gray", label="logγ = -10")
plt.legend(loc = 'upper left')
#GAUSS PCA10
print("GAUSS DATA KFOLD PCA 10----------------SVM RBF")
dcf_along_gamma_10, dcf_along_gamma_9, dcf_along_gamma_8, dcf_along_gamma_7, dcf_along_gamma_6, dcf_along_gamma_5, dcf_along_gamma_4, dcf_along_gamma_3, dcf_along_gamma_2, dcf_along_gamma_1=apply_score_matrix_SVM_RBF(k,gauss_data_kfold_pca10)
cVector= numpy.array([1.E-3,1.E-2,5.E-2,1.E-1,5.E-1,8.E-1,1,10,100,1000])
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("c")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_gamma_5, color="blue", label="logγ = -5")
plt.plot(cVector, dcf_along_gamma_4, color="green", label="logγ = -4")
plt.plot(cVector, dcf_along_gamma_3, color="red", label="logγ = -3")
plt.plot(cVector, dcf_along_gamma_2, color="cyan", label="logγ = -2")
plt.plot(cVector, dcf_along_gamma_1, color="gray", label="logγ = -1")
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.xticks(cVector)
plt.xlabel("c")
plt.ylabel("minDCF")
plt.plot(cVector, dcf_along_gamma_6, color="blue", label="logγ = -6")
plt.plot(cVector, dcf_along_gamma_7, color="green", label="logγ = -7")
plt.plot(cVector, dcf_along_gamma_8, color="cyan", label="logγ = -8")
plt.plot(cVector, dcf_along_gamma_9, color="yellow", label="logγ = -9")
plt.plot(cVector, dcf_along_gamma_10, color="gray", label="logγ = -10")
plt.legend(loc = 'upper left')
#GMM
#RAW
score,label=apply_GMM(k,raw_data_kfold,5)
print("FULL RAW------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
score,label=apply_GMM(k,raw_data_kfold,5,2) #tied
print("TIED RAW------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
score,label=apply_GMM(k,raw_data_kfold,5,1) # diag tied
print("DIAG TIED RAW------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))

#GAUSS DATA
score,label=apply_GMM(k,gauss_data_kfold,5)
print("FULL GAUSS------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
score,label=apply_GMM(k,gauss_data_kfold,5,2) #tied
print("TIED GAUSS------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
score,label=apply_GMM(k,gauss_data_kfold,5,1) # diag tied
print("DIAG TIED GAUSS------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
#GAUSS DATA PCA10
score,label=apply_GMM(k,gauss_data_kfold_pca10,5)
print("FULL GAUSS PCA10------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
score,label=apply_GMM(k,gauss_data_kfold_pca10,5,2)
print("TIED GAUSS PCA10------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
score,label=apply_GMM(k,gauss_data_kfold_pca10,5,1) # diag tied
print("DIAG TIED GAUSS PCA10------------------------------------------------------")
for j in range(len(score)):
    print("-------------it j:",j)
    print("0.5:",BayesEmpiricalRisk_threshold(0.5,1,1, score[j],label[j]))
    print("0.1:",BayesEmpiricalRisk_threshold(0.1,1,1, score[j],label[j]))
    print("0.9:",BayesEmpiricalRisk_threshold(0.9,1,1, score[j],label[j]))
#ACTUAL DCF
#SVM RBF
labels,score=apply_score_matrix_SVM_RBF_actual(k,gauss_data_kfold_pca10)
print("ACTUAL DCF SVM RBF C=100, GAMMA=e^-1")
actual_dcf05=BayesEmpiricalRisk_threshold_actual(0.5,1,1,score,labels)
print("0.5:",actual_dcf05)
actual_dcf01=BayesEmpiricalRisk_threshold_actual(0.1,1,1,score,labels)
print("0.1:",actual_dcf01)
actual_dcf09=BayesEmpiricalRisk_threshold_actual(0.9,1,1,score,labels)
print("0.9:",actual_dcf09)
draw_bayes_error(score,labels)
#CALIBRATION (NOT EFFECTIVE)
score=mRow(score)
fun= kfold(score, labels,k)
newScore= score_matrix_logReg_Calibration(k,fun,0.5)
print("ACTUAL DCF SVM RBF C=100, GAMMA=e^-1 AFTER CALIBRATION (NOT EFFECTIVE)")
actual_dcf05=BayesEmpiricalRisk_threshold_actual(0.5,1,1,newScore,labels)
print("0.5",actual_dcf05)
actual_dcf01=BayesEmpiricalRisk_threshold_actual(0.1,1,1,newScore,labels)
print("0.1:",actual_dcf01)
actual_dcf09=BayesEmpiricalRisk_threshold_actual(0.9,1,1,newScore,labels)
print("0.9:",actual_dcf09)
#GAUSS DATA PCA10 GMM
score,label=apply_GMM(k,gauss_data_kfold_pca10,2)
score=score[1]
label=label[1]
print("ACTUAL DCF GMM FULL, 2 Gau")
print("0.5:",BayesEmpiricalRisk_threshold_actual(0.5,1,1, score,label))
print("0.1:",BayesEmpiricalRisk_threshold_actual(0.1,1,1, score,label))
print("0.9:",BayesEmpiricalRisk_threshold_actual(0.9,1,1, score,label))
draw_bayes_error(score,label)
#QUADRATIC LOGISTIC REGRESSION
labels,score=apply_score_matrix_logReg_quadratic_actual(k,gauss_data_kfold_pca10)
print("ACTUAL DCF LOGISTIC REGRESSION QUADRATIC, L=0.001 PiT=0.9")
actual_dcf05=BayesEmpiricalRisk_threshold_actual(0.5,1,1,score,labels)
print(actual_dcf05)
actual_dcf01=BayesEmpiricalRisk_threshold_actual(0.1,1,1,score,labels)
print(actual_dcf01)
actual_dcf09=BayesEmpiricalRisk_threshold_actual(0.9,1,1,score,labels)
print(actual_dcf09)
draw_bayes_error(score,labels)
#EVALUATION
features_list = [ 'Fixed_Acidity', 'Volatile_Acidity', 'Citric_Acid', 'Residual_sugar', 'Chlorides', 'Free_Sulfur_Dioxide', 'Total_Sulfur_Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol', 'Quality']
dataset_array_test = dataset_extraction("Datasets/Test.txt", features_list)[0]
labels_array_test = dataset_extraction("Datasets/Test.txt", features_list)[1]

normalized_dataset_test=normalization(dataset_array_test)
dataset_gaussianized_test = ppf_gaussianization(features_list,normalized_dataset_test)
dataset_mu_gaussianized_test = vcol(dataset_gaussianized_test.mean(1))
dataset_covariance_gaussianized_test = covariance_numpy(dataset_gaussianized_test)
pca_matrix_test = PCA_eigh(dataset_covariance_gaussianized_test, 10)
dataset_with_pca10_test = numpy.dot(pca_matrix_test.T, dataset_gaussianized_test)
print("EVALUATION SVM RBF C=100, GAMMA=e^-1")
f, test_score_svm_rbf=test_RBF_svm(dataset_with_pca10,labels_array.flatten(),dataset_with_pca10_test,100,numpy.exp(-1))
predicted_labels = 1*(test_score_svm_rbf > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
dual_obj = f
print("GAUSS DATA PCA10 ---------------------")
print("DUAL LOSS: ", -dual_obj)
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
f, test_score_svm_rbf=test_RBF_svm(dataset_array,labels_array.flatten(),dataset_array_test,100,numpy.exp(-1))
predicted_labels = 1*(test_score_svm_rbf > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
dual_obj = f
print("RAW DATA  ---------------------")
print("DUAL LOSS: ", -dual_obj)
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
f, test_score_svm_rbf=test_RBF_svm(dataset_gaussianized,labels_array.flatten(),dataset_gaussianized_test,100,numpy.exp(-1))
predicted_labels = 1*(test_score_svm_rbf > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
dual_obj = f
print("GAUSS DATA NO PCA---------------------")
print("DUAL LOSS: ", -dual_obj)
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
#GMM
print("EVALUATION GMM FULL 2 Gau")
test_score_gmm_std_2= test_GMM_std_2(dataset_with_pca10,labels_array.flatten(),dataset_with_pca10_test,2)
predicted_labels = 1*(test_score_gmm_std_2 > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
print("GAUSS DATA PCA10 ---------------------")
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
test_score_gmm_std_2= test_GMM_std_2(dataset_array,labels_array.flatten(),dataset_array_test,2)
predicted_labels = 1*(test_score_gmm_std_2 > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
print("RAW DATA  ---------------------")
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
test_score_gmm_std_2= test_GMM_std_2(dataset_gaussianized,labels_array.flatten(),dataset_gaussianized_test,2)
predicted_labels = 1*(test_score_gmm_std_2 > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
print("GAUSS DATA NO PCA---------------------")
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
#QUADRATIC LOGISTIC REGRESSION
print("EVALUATION QUADRATIC LOGISTI REGRESSION")
test_score_lr_quad=test_logReg_quadratic(dataset_with_pca10,labels_array.flatten(),dataset_with_pca10_test,0.001)
predicted_labels = 1*(test_score_lr_quad > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
print("GAUSS DATA PCA10 ---------------------")
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
test_score_lr_quad=test_logReg_quadratic(dataset_gaussianized,labels_array.flatten(),dataset_gaussianized_test,0.1,0.5)
predicted_labels = 1*(test_score_lr_quad > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
print("GAUSS DATA NO PCA---------------------")
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
test_score_lr_quad=test_logReg_quadratic(dataset_array,labels_array.flatten(),dataset_array_test,0.1,0.5)
predicted_labels = 1*(test_score_lr_quad > 0 )
accuracy, error = compute_accuracy_error(predicted_labels, mRow(labels_array_test))
print("RAW DATA  ---------------------")
print("ACCURACY: ", accuracy*100,"%")
print("ERROR RATE: ", error*100, "%")
#DET
labels_lr,score_lr=apply_score_matrix_logReg_quadratic_actual(k,gauss_data_kfold_pca10)
score_gmm,label_gmm=apply_GMM(k,gauss_data_kfold_pca10,2)
score_gmm=score_gmm[1]
label_gmm=label_gmm[1]
labels_rbf,score_rbf=apply_score_matrix_SVM_RBF_actual(k,gauss_data_kfold_pca10)
print("DET CURVE GAUSS PCA 10")
draw_det_curve(score_gmm,label_gmm,score_lr,labels_lr,score_rbf,labels_rbf,c1='r',c2='b',c3='green')
labels_lr,score_lr=apply_score_matrix_logReg_quadratic_actual(k,gauss_data_kfold,0.5,0.1)
score_gmm,label_gmm=apply_GMM(k,gauss_data_kfold,2)
score_gmm=score_gmm[1]
label_gmm=label_gmm[1]
labels_rbf,score_rbf=apply_score_matrix_SVM_RBF_actual(k,gauss_data_kfold)
print("DET CURVE GAUSS")
draw_det_curve(score_gmm,label_gmm,score_lr,labels_lr,score_rbf,labels_rbf,c1='r',c2='b',c3='green')








