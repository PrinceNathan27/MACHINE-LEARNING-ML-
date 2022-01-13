from numpy import *
#from plotBoundary import *
from cvxopt import matrix, solvers
import random
import matplotlib.pyplot as plt
import sys

def myDualSVM(d,C_list):
    #data = loadtxt(open("MNIST-13.csv", "rb"), delimiter=",")
    data = loadtxt(open(d, "rb"), delimiter=",")
    # use deep copy here to make cvxopt happy
    X = data[:, 1:]
    Y = data[:, 0:1]
    Y[Y==1]=-1
    Y[Y==3]=1
    #C_list=[0.01,0.1,1,10,100]
    
    #10-folds
    folds=10
    window=int(ceil(len(data)/folds))
    solvers.options['show_progress'] = False
    error_table=zeros((len(C_list),5))
    train_error_table=zeros((len(C_list),5))
    sv=zeros((len(C_list),5))
    margin=zeros((len(C_list),5))
    
    for j in C_list:
        C=j
        print("C: "+str(C))
        
        for i in range(5):
            print("Iteration: "+str(i+1))
            
            t_idx=random.sample(range(10), 1)
            exclude=list(range(window*t_idx[0],window*(t_idx[0]+1)))
            t_idx=random.sample([i for i in range(10) if i!=t_idx[0]], 1)
            exclude.extend(list(range(window*t_idx[0],window*(t_idx[0]+1))))
            
            test = X[exclude]
            test_Y = Y[exclude]
            train = delete(X,exclude,0)
            train_Y = delete(Y,exclude,0)
            
            
            N=len(train)
            M=len(train[0])
            
            #setting matrix parameters for solver
            q = matrix(-1 * ones((N,1)))
            A = matrix(transpose(train_Y))
            b = matrix(double(0))
            P = matrix(multiply(matmul(train_Y,transpose(train_Y)),matmul(train,transpose(train))))
            G = matrix(concatenate((-1*identity(N),identity(N)),axis=0))
            h = matrix(concatenate((zeros((N,1)),C*ones((N,1))),axis=0))
            
            sol=solvers.qp(P, q, G, h, A, b)
            
            alpha=sol['x']
            a=array(alpha)
            print("Number of support vectors: "+str(len(a[a>(1e-8)])))
            sv[C_list.index(j)][i]=len(a[a>(1e-8)])
            
            #weight vector
            w=matmul(transpose(multiply(alpha,train_Y)),train)
            print("Margin: "+str(1/linalg.norm(w)))
            margin[C_list.index(j)][i]=1/linalg.norm(w)
            
            B=train_Y - matmul(train,transpose(w))
            
            #prediction
            pred = matmul(test,transpose(w)) + B[65]
            pred[pred>0]=1
            pred[pred<=0]=-1
            print("Test Error: "+str(sum(pred!=test_Y)/len(test)*100))
            error_table[C_list.index(j)][i]=sum(pred!=test_Y)/len(test)*100
            pred = matmul(train,transpose(w)) + B[65]
            pred[pred>0]=1
            pred[pred<=0]=-1
            print("Train Error: "+str(sum(pred!=train_Y)/len(train)*100))
            train_error_table[C_list.index(j)][i]=sum(pred!=train_Y)/len(train)*100
    
    Cl=['0.01','0.1','1','10','100']   
    plt.figure()
    plt.title("Test Errors")
    e=mean(error_table,axis=1)
    s=std(error_table,axis=1)   
    print("Mean Test errors for different C's respectively:")
    print(*(list(e)),sep=',')
    print("Standard Deviation for Test errors for different C's respectively:")
    print(*(list(s)),sep=',')
    plt.errorbar(Cl, e, s, label='Errors', fmt='o-', capthick=2)
    plt.show()
    plt.figure()
    plt.title("Train Errors")
    e=mean(train_error_table,axis=1)
    s=std(train_error_table,axis=1) 
    print("Mean Train errors for different C's respectively:")
    print(*(list(e)),sep=',')
    print("Standard Deviation for Train errors for different C's respectively:")
    print(*(list(s)),sep=',') 
    plt.errorbar(Cl, e, s, label='Errors', fmt='o-', capthick=2)
    plt.show()
    plt.figure()
    plt.title("Support Vectors")
    e=mean(sv,axis=1)
    s=std(sv,axis=1)   
    print("Mean Support Vectors for different C's respectively:")
    print(*(list(e)),sep=',')
    print("Standard Deviation for Support Vectors for different C's respectively:")
    print(*(list(s)),sep=',')     
    plt.errorbar(Cl, e, s, label='Support Vectors', fmt='o-', capthick=2)
    plt.show()
    plt.figure()
    plt.title("Margin")
    e=mean(margin,axis=1)
    s=std(margin,axis=1) 
    print("Mean Margin for different C's respectively:")
    print(*(list(e)),sep=',')
    print("Standard Deviation for Margin for different C's respectively:")
    print(*(list(s)),sep=',')   
    plt.errorbar(Cl, e, s, label='Margin', fmt='o-', capthick=2)
    plt.show()
def main():
    file=sys.argv[1]
    t=[float(i) for i in (sys.argv[2]).split(',')]
    myDualSVM(file,t)
if __name__ == "__main__":
	main()
