from numpy import *
import random
import matplotlib.pyplot as plt
import time
import sys

def myPegasos(d,numruns,K):
    #data = loadtxt(open("MNIST-13.csv", "rb"), delimiter=",")
    data = loadtxt(open(d, "rb"), delimiter=",")
    X = data[:, 1:]
    Y = data[:, 0:1]
    Y[Y==1]=-1
    Y[Y==3]=1
    
    N=len(data)
    M=len(X[0])
    #numruns=5
    #K=[1,20,200,1000,2000]
    Lambda=100
    time_table=zeros((len(K),numruns))
    
    for k in K:
        print("k= "+str(k))
        plt.figure()
        for i in range(numruns):
            print("Run: "+str(i+1))
            start=time.time()
            loss_prev=[1e10]
            loss_curr=[]
            
            T=100*int(N/k)
            
            #weight initialization
            w=ones((M,1))*(1/sqrt(Lambda*M) )
            obj_fun=[]
            
            for t in range(1,T):
                if k==1:
                    idx=random.sample(range(N), 1)
                    Xt=X[idx]
                    Yt=Y[idx]
                    p=multiply(Yt,matmul(Xt,w))
                    idx=where(p<1)
                    Xtplus=Xt[idx[0]]
                    Ytplus=Yt[idx[0]]
                elif k==2000:
                    p=multiply(Y,matmul(X,w))
                    idx=where(p<1)
                    Xtplus=X[idx[0]]
                    Ytplus=Y[idx[0]]
                else:
                    z=where(Y == 1)
                    s = int(len(z[0]) * k/N)
                    idx=random.sample(list(z[0]), s)
                    Xt=X[idx]
                    Yt=Y[idx]
                    z=where(Y == -1)
                    s = int(len(z[0]) * k/N)
                    ix=random.sample(list(z[0]), s)
                    Xt=concatenate((Xt,X[ix]),axis=0)
                    Yt=concatenate((Yt,Y[ix]),axis=0)
                    p=multiply(Yt,matmul(Xt,w))
                    dx=where(p<1)
                    Xtplus=Xt[dx[0]]
                    Ytplus=Yt[dx[0]]
                #weight-update
                eta=1/(Lambda*t)
                wt = multiply((1-eta*Lambda),w) + multiply((eta/k),(sum(multiply(Ytplus,Xtplus),axis=0))[:,None])
                w = multiply(min(1,(1/(sqrt(Lambda)*linalg.norm(wt)))),wt)
                loss = 1 - multiply(Y,matmul(X,w))
                obj = (Lambda/2)*(linalg.norm(w)**2) + (1/N)*sum(loss[loss>0])
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
            time_table[K.index(k)][i]=end-start
            #prediction
            pred = matmul(X,w)
            pred[pred>0]=1
            pred[pred<=0]=-1
            print("Error: "+str(sum(pred!=Y)/len(X)*100))
            plt.title("k= "+str(k))
            l=i+1
            plt.plot(obj_fun,label='Run %s' % l)
            plt.legend()
            plt.show()
    e=mean(time_table,axis=1)
    s=std(time_table,axis=1)  
    print("Mean time for different k's respectively in seconds:")
    print(*(list(e)),sep=',')
    print("Standard Deviation for time for different k's respectively in seconds:")
    print(*(list(s)),sep=',')

def main():
    file=sys.argv[1]
    t=[int(i) for i in (sys.argv[3]).split(',')]
    num=int(sys.argv[2])
    myPegasos(file,num,t)
if __name__ == "__main__":
	main()
