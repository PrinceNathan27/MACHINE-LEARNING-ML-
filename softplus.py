from numpy import *
import random
import matplotlib.pyplot as plt
import numpy.random
import time
import sys

def mySoftplus(d,numruns,K):
    #data = loadtxt(open("MNIST-13.csv", "rb"), delimiter=",")
    data = loadtxt(open(d, "rb"), delimiter=",")
    X = data[:, 1:]
    X=X/255.0
    Y = data[:, 0:1]
    Y[Y==1]=-1
    Y[Y==3]=1
    
    N=len(data)
    M=len(X[0])
    #numruns=5
    #K=[1,20,200,1000,2000]
    step=0.1
    a=0.1
    Lambda=0.01
    time_table=zeros((len(K),numruns))
    
    for k in K:
        print("k= "+str(k))
        plt.figure()
        for n in range(numruns):
            print("Run: "+str(n+1))
            start=time.time()
            loss_prev=[1e10]
            loss_curr=[]
            
            w=numpy.random.uniform(low=-1,high=1,size=(M,1))
            obj_fun=[]
            T=100*int(N/k)
            
            for t in range(1,T):
                if k==1:
                    idx=random.sample(range(N), 1)
                    Xt=X[idx]
                    Yt=Y[idx]
                elif k==2000:
                    Xt=X
                    Yt=Y
                else:
                    z=where(Y == 1)
                    s = int(len(z[0]) * k/N)
                    idx=random.sample(list(z[0]), s)
                    Xt=X[idx]
                    Yt=Y[idx]
                    z=where(Y == -1)
                    s = int(len(z[0]) * k/N)
                    idx=random.sample(list(z[0]), s)
                    Xt=concatenate((Xt,X[idx]),axis=0)
                    Yt=concatenate((Yt,Y[idx]),axis=0)
                delta_w=zeros((M,1))
                for i in range(len(Xt)):
                       delta_w = delta_w + (- multiply(Yt[i],Xt[i])[:,None]/(1 + exp( (multiply(Yt[i][:,None],matmul(transpose(w),Xt[i][:,None]))-1)/a)))
                delta_w = delta_w + multiply(2*Lambda,w)
                #weight-updtae
                w = w - (1/len(Xt))*multiply(step,delta_w)
                s=0
                for j in range(len(X)):
                   s=s + a*log(1 + exp((1 - multiply(Y[j][:,None],matmul(transpose(w),X[j][:,None])))/a)) 
                       
                obj = s[0][0]/len(X) + Lambda*linalg.norm(w)**2
                obj_fun.append(obj)
                #termination-condition
                if (t+1)%5==0:
                    if abs(mean(loss_prev)-mean(loss_curr))<1e-3:
                        #print(t)
                        break
                    loss_prev=loss_curr
                    loss_curr=[]
                else:
                    loss_curr.append(obj)
            
            end=time.time()
            time_table[K.index(k)][n]=end-start
            plt.plot(obj_fun)    
            #prediction
            pred = matmul(X,w)
            pred[pred>0]=1
            pred[pred<=0]=-1
            print("Error: "+str(sum(pred!=Y)/len(X)*100))
            plt.title("k= "+str(k))
            l=n+1
            plt.plot(obj_fun,label='Run %s' % l)
            plt.legend()
            plt.show()
    e=mean(time_table,axis=1)
    s=std(time_table,axis=1)  
    print("Mean time for different k's respectively:")
    print(*(list(e)),sep=',')
    print("Standard Deviation for different k's respectively:")
    print(*(list(s)),sep=',')

def main():
    file=sys.argv[1]
    file=sys.argv[1]
    num=int(sys.argv[2])
    mySoftplus(file,num,t)
if __name__ == "__main__":
	main()
