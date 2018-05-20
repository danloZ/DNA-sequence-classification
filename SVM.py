
# coding: utf-8


#imports
import numpy as np
import collections
import itertools
import cvxopt
import cvxopt.solvers
import pandas as pd


path = 'Data/'
source_tr = ['Xtr0.csv','Xtr1.csv','Xtr2.csv']
source_te = ['Xte0.csv','Xte1.csv','Xte2.csv']
source_Ytr = ['Ytr0.csv','Ytr1.csv','Ytr2.csv']



# the data put together in files, 0 changed to -1 in the Y files

x_train = np.loadtxt(open(path + "Xtr0.csv", "rb"), delimiter=" ", dtype = str)
temp = np.loadtxt(open(path + "Xtr1.csv", "rb"), delimiter=" ", dtype = str)
x_train_temp = np.r_[x_train,temp]
x_test_temp = np.loadtxt(open(path + "Xtr2.csv", "rb"), delimiter=" ", dtype = str)
x_train = np.r_[x_train_temp,x_test_temp]


x_test = np.loadtxt(open(path + "Xte0.csv", "rb"), delimiter=" ", dtype = str)
temp = np.loadtxt(open(path + "Xte1.csv", "rb"), delimiter=" ", dtype = str)
x_test = np.r_[x_test,temp]
x_test_temp1 = np.loadtxt(open(path + "Xte2.csv", "rb"), delimiter=" ", dtype = str)
x_test = np.r_[x_test,x_test_temp1]

y_train = np.loadtxt(open(path + "Ytr0.csv", "rb"), delimiter=",", skiprows=1)
temp = np.loadtxt(open(path + "Ytr1.csv", "rb"), delimiter=",", skiprows=1)
y_train_temp = np.r_[y_train,temp]
y_test_temp = np.loadtxt(open(path + "Ytr2.csv", "rb"), delimiter=",", skiprows=1)
y_train = np.r_[y_train_temp, y_test_temp]
y_train[y_train[:,1] == 0] = -1

#this prepares the data according to the spectrum kernel
def prepare_data_kmers(x_data, y_data, test_data, k, l):
    '''
    x_data - Sequence data
    y_data - correctly labeled details
    test_data - if not None will output transformed data for test
    k - sequence lengths
    l - number of features to take #this was not properly implemented
    '''
    #this section finds all possible permutations of the four possibilities, then eleiminates duplicates
    poss = ['A','C','G','T','C','G','T','A','G','T','A','C','T','A','C','G']
    comb = []
    for j in itertools.combinations_with_replacement(poss, k):
        comb.append(list(j))
    comb = np.asarray(comb)
    comb= np.unique(comb, axis = 0)
    #joins as one string
    comb =["".join(i) for i in comb[:,:].astype(str)]
    #initialize the counting of occurences of each string in data occurences, stores for each one
    features= np.zeros(shape=(len(x_data), len(comb)))
    #saves the features
    for m in range(0,len(x_data)):
        s = x_data[m]
        li = [ s[i:i+k] for i in range(len(s)-k+1) ]
        counter = collections.Counter(li)
        i=0
        for j in comb:
            features[m][i] = counter[j]
            i=i+1
    temp1=features[y_data[:,1]==1]
    temp1_sum=temp1.sum(axis=0)
    ind1 = np.argpartition(temp1_sum, -l)[-l:]
    temp0=features[y_data[:,1]==0]
    temp0_sum=temp0.sum(axis=0)
    ind0 = np.argpartition(temp0_sum, -l)[-l:]
    index = np.append(ind0,ind1)
    extracted_feature_data = features[:,index]
    extracted_feature_data /= np.max(np.abs(extracted_feature_data),axis=0)
    #this says if we want the data for only one set or for two (ie test and train)
    if test_data.size != 0:
        extracted_test_data= np.zeros(shape=(len(test_data), len(comb)))
        for m in range(0,len(test_data)):
            s = test_data[m]
            li = [ s[i:i+k] for i in range(len(s)-k+1) ]
            counter = collections.Counter(li)
            i=0
            for j in comb:
                extracted_test_data[m][i] = counter[j]
                i=i+1
        extracted_test_data = extracted_test_data[:,[index]]
        extracted_test_data /= np.max(np.abs(extracted_test_data),axis=0)

        return extracted_feature_data, extracted_test_data
    else:
        return extracted_feature_data

#this was essentially identical, but allowed for some mistakes in the string according to the mismatch allowance
def prepare_data_kmers_gapped(x_data, y_data, test_data, k, l, gap, miss):
    '''
    x_data - Sequence data
    y_data - correctly labeled details
    test_data - if not None will
    k - sequence lengths
    l - number of features to take
    gap - permissible gap
    miss- permissible mismatch number
    '''
    #this section finds all possible permutations of the four possibilities, then eleiminates duplicates
    poss = ['A','C','G','T','C','G','T','A','G','T','A','C','T','A','C','G']
    comb = []
    for j in itertools.combinations_with_replacement(poss, k):
        comb.append(list(j))
    comb = np.asarray(comb)
    comb= np.unique(comb, axis = 0)
    #joins as one string
    comb =["".join(i) for i in comb[:,:].astype(str)]
    mismatch_index = np.zeros((len(comb), len(comb)))
    for j in range(0, len(comb)):
        for m in range(0, len(comb)):
            a= comb[j]
            b= comb[m]
            mismatch_index[j,m] = sum ( a[i] != b[i] for i in range(len(a)))
    mismatch_index[mismatch_index[:,:] > miss] = 0
    mismatch_index[mismatch_index[:,:] <= miss] = 1

    features= np.zeros((len(x_data), len(comb)))
    print(len(features))
    #saves the features
    for m in range(0,len(x_data)):
        s = x_data[m]
        li = [ s[i:i+k] for i in range(len(s)-k+1) ]
        counter = collections.Counter(li)
        i=0
        for j in comb:
            features[m][i] = counter[j]
            i=i+1
    if miss > 0:
        for j in range(0,len(x_data)):
            for m in range(0,len(comb)):
                features[j][m] = features[j][m] + np.dot(features[j],mismatch_index.T[m])
    temp1=features[y_data[:,1]==1]
    temp1_sum=temp1.sum(axis=0)
    ind1 = np.argpartition(temp1_sum, -l)[-l:]
    temp0=features[y_data[:,1]==0]
    temp0_sum=temp0.sum(axis=0)
    ind0 = np.argpartition(temp0_sum, -l)[-l:]
    index = np.append(ind0,ind1)
    extracted_feature_data = features[:,index]
    print(features.shape)
    print(temp1)
    extracted_feature_data /= np.max(np.abs(extracted_feature_data),axis=0)
    #this says if we want the data for only one set or for two (ie test and train)
    if test_data.size != 0:
        extracted_test_data= np.zeros(shape=(len(test_data), len(comb)))
        for m in range(0,len(test_data)):
            s = test_data[m]
            li = [ s[i:i+k] for i in range(len(s)-k+1) ]
            counter = collections.Counter(li)
            i=0
            for j in comb:
                extracted_test_data[m][i] = counter[j]
                i=i+1
        extracted_test_data = extracted_test_data[:,[index]]
        extracted_test_data /= np.max(np.abs(extracted_test_data),axis=0)

        return extracted_feature_data, extracted_test_data
    else:
        return extracted_feature_data


#definition of kernels
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

#nonlinear below, p and sigma are the important parts
def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def rbf_kernel(x, y, sigma=3):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class SVM(object):
    #initialization with kernel and C saying penalty for misclassification (small -> less penalty and vice versa)
    def __init__(self, kernel=rbf_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
    #fitting of the svm
    def fit(self, x_train, y_train):
        #shape
        n_samples, n_features = x_train.shape

        # create the gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(x_train[i], x_train[j])
        #create the necessary components for the quadratic program
        P = cvxopt.matrix(np.outer(y_train,y_train) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y_train, (1,n_samples))
        b = cvxopt.matrix(0.0)
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # solve quadratic progam
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers, cut off at 1e-5
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        #create the support vectors
        self.a = a[sv]
        self.sv = x_train[sv]
        self.sv_y = y_train[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))
        # fit the svs with intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        print(self.b)
        # weight vectors, difference between linear and non-linear
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
    #prediction, return the sign since we classify as -1 or +1
    def predict(self, X):
        if self.w is not None:
            Id = np.arange(len(X))
            Bound = np.sign(np.dot(X, self.w) + self.b)
            Bound[Bound == -1] = 0
            Id= Id.T
            Bound = Bound.T
            np.savetxt('Yte.csv',np.c_[Id,Bound],header="ID,Bound", delimiter = ",")
            names = ["Id", "Bound"]
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            #prepare the csv file, switch -1 to 0, add headers, etc
            Id = np.arange(len(X))
            Bound = np.sign(y_predict + self.b)
            Bound[Bound == -1] = 0
            Id= Id.T
            Bound = Bound.T
            np.savetxt('Yte.csv',np.c_[Id,Bound],header="ID,Bound", delimiter = ",")
            return np.sign(y_predict + self.b)
    #for testing purposes, fits and predicts, outputs accuracy
    def test(kernel, c, x_train, y_train, x_test, y_test):

        clf = SVM(kernel, C = c)
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)
        print("%d predicted as class -1" % (np.sum(y_predict == -1)))
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))


print("Features computation")
x,x_test_f = prepare_data_kmers(x_train, y_train, x_test, k=5, l=0)
print("Model class initialization")
init = SVM(polynomial_kernel, 5)
print("Model fitting")
init.fit(x, y_train[:,1])
print("Test set data prediction")
init.predict(x_test_f)


print("Exporting on file Yte.csv")
sub = np.loadtxt(open("Yte.csv"), delimiter=",", skiprows =1)
sub.shape
names = ["Id", "Bound"]
df = pd.DataFrame(sub, columns=names,dtype=int)
df1 = df.apply(pd.to_numeric, args=('coerce',))
df1.to_csv('Yte.csv', index=False, header=True, sep=',')
