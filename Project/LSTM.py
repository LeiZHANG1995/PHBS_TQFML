import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Utils:
    
    def __init__(self):
        pass
    
    def Ols(self, y, x):
        mx = np.asmatrix(x)*10    # we times 10 on both x and y 
                                  # to avoid singular error,
                                  # which is mainly caused by 
                                  # too small float dividing.
        my = np.asmatrix(y).transpose()*10
        A = np.matmul(mx.transpose(), mx)
        B = np.matmul(mx.transpose(), my)
        beta = np.matmul(np.linalg.inv(A), B)
        return beta
    
    def Sign(self, x):
        y = x.copy()
        y[y>0] = 1
        y[y<0] = -1
        return(y)
    
    def Standardization(self, x):
        mean = x.mean()
        std = x.std()
        return((x-mean)/std) 


class Predictor:
    
    W = None
    LR = None
    X_training = None
    Y_training = None
        
    def __init__(self):
        pass
    
    def ADL_fit(self, y, x, p):
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.DataFrame): 
            x = x.values
        if len(x) != len(y):
            print("Error: X and Y are inconsistent:", len(X), len(Y))
            return None
        window = len(x)
        dy = y[1:] - y[0:len(y)-1] 
        dx = x[1:] - x[0:len(x)-1] 
        X = None
        for i in range(0, p):
            A = np.asmatrix(dx[i:window-p+i]) 
            B = np.asmatrix(dy[i:window-p+i]).transpose()
            tmp = np.hstack((A, B))
            if X is None:
                X = tmp
            else:
                X = np.hstack((X, tmp))
        # get beta
        dy_1 = dy[p:] # len = 57
        X_0 = X[:len(X)-1]
        ut = Utils() 
        beta = ut.Ols(dy_1, X_0)  ## beta for dx and dy
        # get prediction
        y1 = np.asmatrix(y[p:]).transpose()
        y2 = y1 + np.matmul(X, beta)
        y2 = y2.transpose().tolist()[0]
        self.W = np.asmatrix(beta);
        return({'w' : beta, 'y_hat' : y2}) 
    
    def ADL_predict(self, y0, x, lag):
        X0 = None
        y0 = y0.tolist()
        x = x.values
        if len(y0) < lag+1:
            print("Error: y0 has at leaset lag+1 elements.")
            return None
        if len(x) < lag+1:
            print("Error: x has at leaset lag+1 rows.")
            return None
        len_re = len(x) - lag  ## the length of return
        re = []
        for j in range(0, len_re-1):
            x_1 = []
            x_2 = []
            for i in range(0, lag):
                x_1 = x_1 + x[j+i, :].tolist()
                x_1 = x_1 + [y0[j+i]]
                x_2 = x_2 + x[j+i+1, :].tolist()
                x_2 = x_2 + [y0[j+i+1]]
            x__ = np.array(x_2) - np.array(x_1)
            y_1 = y0[lag+j]
            y_2 = (y_1 + np.matmul(x__, self.W))[0,0]
            y0.append(y_2)
            re.append(y_2)
        return(re)
    
    def ADL_predict_accuracy(self, y, x, lag):
        y0 = y[0:lag+1]
        real_price = y[lag+1:]
        test_re = self.ADL_predict(y0, x, lag)
        u = Utils()
        pf = u.Sign(np.array(test_re[1:]) - np.array(test_re[:len(test_re)-1]))
        rf = u.Sign(np.array(real_price[1:]) - np.array(real_price[:len(real_price)-1]))
        return({'accuracy':np.sum(pf==rf)/(len(test_re)-1), 'pf':pf, 'rf':rf})
    
    def LSTM_fit(self, Y, X, lag, window):
        self.X_training = X
        self.Y_training = Y
        accuracy_adl = []
        accuracy_lr = []
        predict_adl = []
        predict_lr = []
        ty_hat = [] 
        trf = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        if len(X) != len(Y):
            print("Error: X and Y are inconsistent:", len(X), len(Y))
            return None
        sample_size = len(X)
        for i in range(0, sample_size-window): # This is a slice. 
            ## Step 1: Get sub-sample from the trainning set
            x_adl = X[i:(i+window)]
            y_adl = Y[i:(i+window)]
            ## Step 2: ADL Regression. Get h -- hiden layer
            p = Predictor()
            u = Utils()
            re = p.ADL_fit(y_adl, x_adl, lag)
            self.W = re['w']
            y_hat = re['y_hat']    ##--- index = lag+1 : window+1 ---##
            if i == 0:
                ty_hat += y_hat ## 38 elements, missing first 3, add last 1
            else:
                ty_hat.append(y_hat[-1]) 
            y_current = y_adl[lag:window] 
            h = y_hat - y_current
            ## Step 3: The prediction performance of pure ADL linear_model
            pf = u.Sign(h)   # predicted fluctuation
            rf = u.Sign(Y[i+lag+1:i+window+1] - y_current) # real f
            if i == 0:
                predict_adl += pf.tolist() 
                trf += rf.tolist()
            else:
                predict_adl.append(pf.tolist()[-1]) 
                trf.append(rf.tolist()[-1]) 
            accuracy_adl.append(np.sum(rf == pf)/(window-1)) 
            ## Step 4: Logistic Regression
            rf_0 = rf[0:len(rf)-1]
            rf_1 = rf[1:len(rf)]
            std_h_1 = u.Standardization(h[1:len(pf)])
            x_lr = np.array([rf_0, std_h_1]).transpose()
            lr = LogisticRegression(C = 1000.0, random_state = 0)
            lr.fit(x_lr, rf_1)
            self.LR = lr
            ## Step 5: The prediction performance after LR adjustment
            yt = lr.predict(x_lr)
            accuracy_lr.append(np.sum(yt==rf[1:])/(window-2))
            if i == 0:
                predict_lr += yt.tolist()
            else:
                predict_lr.append(yt[len(yt)-1])
        return({'ADL_accuracy':accuracy_adl, 'LR_accuracy': accuracy_lr, 'ADL_predict': predict_adl, 'LR_predict': predict_lr, 'Real_fluctuation': trf, 'y_hat': ty_hat})
    
    def LSTM_predict(self, X, lag, window): 
        train_size = len(self.X_training) 
        x0 = self.X_training.loc[train_size-window:, :] 
        Y = self.Y_training.loc[train_size-window:]
        adl_re = self.ADL_fit(Y, x0, lag) 
        y_hat_last = adl_re['y_hat'][-1] 
        Y[len(Y)] = y_hat_last 
        Y = Y.values 
        X = np.row_stack((x0.values, X.values)) 
        RESULT = [] 
        # print(Y.shape, X.shape) 
        # Recurse X window and Y window, len = window+1 
        for i in range(0, len(X)-window): 
            x_window = X[i:i+window+1] 
            y_window = Y[i:i+window+1] 
            re = self.LSTM_fit(y_window, x_window, lag, window) 
            y_hat = re['y_hat']  
            Y = np.hstack((Y, np.array(y_hat[-1]))) 
            RESULT.append(re['LR_predict'][-1]) 
        Y = Y[window:] 
        return({'y_hat':Y, 'prediction':RESULT}) 
        
        # print(x0) 
        # (1) From trainning set, get the x and y that I need
        #     and form a new window.
        # (2) Predict this window and get y_hat_151 using 
        #     LSTM_fit(...) directly
        # (3) Do (1) and (2) recursively with Trainning set data.
    
    def LSTM_predict_accuracy(self, y, x, lag, window):
        y = self.Y_training.append(y) 
        re = self.LSTM_predict(x, lag, window)
        pf = re['prediction']
        u = Utils()
        rf = u.Sign(y.diff()) 
        rf = rf[len(rf)-len(pf)-1: len(rf)-1]  
        accuracy = np.sum(rf==pf)/(len(pf))
        return({'y_hat':re['y_hat'], 'pf':pf, 'rf':rf, 'accuracy':accuracy})

