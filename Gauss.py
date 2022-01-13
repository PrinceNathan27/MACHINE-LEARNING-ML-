from sklearn.datasets import load_digits
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eig
from numpy.linalg import det
import matplotlib.pyplot as plt
import sys
import pandas as pd

def LDA2DGaussGM(filename,numcrossvals):
	digits = pd.read_csv(filename,header=None)
	digits = digits.values
	data = digits[:,:-1]
	target = digits[:,-1]
	folds=numcrossvals
	projected_data=[]

	N=np.zeros(10)
	m=np.zeros((10,len(data[0])))

	for i in range(10):
		x=0
		sum=np.zeros(len(data[0]))
		for j in range(len(data)):
			if target[j]==i:
				x=x+1
				sum=sum+data[j]
		m[i]=sum/x
		N[i]=x

	M=np.mean(m,axis=0)
	Sb=np.zeros((len(data[0]),len(data[0])))
	Sw=np.zeros((len(data[0]),len(data[0])))

	#Calculating within and between scatter:
	for i in range(10):
		t_sw=np.zeros((len(data[0]),len(data[0])))
		Sb=Sb + N[i]*(np.outer(m[i]-M,m[i]-M))
		for j in range(len(data)):
			if target[j] == i:
				t_sw = t_sw + np.outer((data[j]-m[i]),(data[j]-m[i]))
		Sw = Sw + t_sw

	#Projecting data to 2-dimensional space:
	S=np.matmul(pinv(Sw),Sb)
	values,vectors=eig(S)
	idx = values.argsort()[::-1]
	values = values[idx]
	vectors = vectors[:,idx]
	w=vectors[:,:2]
	projected_data = np.matmul(data,w)

	Cov=[]
	prior=[]

	for i in range(10):
		prior.append(N[i]/len(data))


	s=0
	st=0
	tr=[]
	te=[]
	#running k-folds:
	for i in range(folds):
		print("Fold: "+str(i+1))
		window=int(np.ceil(len(projected_data)/folds))
		exclude=list(range(window*i,window*(i+1)))
		test=projected_data[window*i:window*(i+1)]
		target_test=target[window*i:window*(i+1)]
		train=np.delete(projected_data,exclude,0)
		target_train=np.delete(target,exclude,0)
		Class=[]
		Class_train=[]
		Cov=[]
		M=[]
		prior=[]
		C=np.zeros((2,2))
		#Learning the generative model
		for j in range(10):
			k,=np.where(target_train == j)
			Cov.append(np.cov(train[k],rowvar=False))
			M.append(np.mean(train[k],axis=0))
			prior.append(len(k)/len(data))
			C=C+Cov[j]*prior[j]

		#Train Classification
		G=[]
		for j in range(len(train)):
			G.append([])
			for k in range(10):
				G[j].append(np.matmul(np.matmul(inv(C),M[k])[None,:],train[j]) - np.matmul(np.matmul(M[k],inv(C))[None,:],M[k][:,None])/2 + np.log(prior[k]))
			idx=np.argmax(G[j])
			Class_train.append(idx)

		#Test Classification
		G=[]
		for j in range(len(test)):
			G.append([])
			for k in range(10):
				G[j].append(np.matmul(np.matmul(inv(C),M[k])[None,:],test[j]) - np.matmul(np.matmul(M[k],inv(C))[None,:],M[k][:,None])/2 + np.log(prior[k]))
			idx=np.argmax(G[j])
			Class.append(idx)

		#Train error-rate calculation
		Class_train=np.array(Class_train)
		res=np.zeros(len(Class_train))
		res[target_train!=Class_train]=1
		print("Train error: "+str(np.sum(res)*100/len(train)))
		tr.append(np.sum(res)*100/len(train))
		st=st+np.sum(res)*100/len(train)

		#Test error-rate calculation
		Class=np.array(Class)
		res=np.zeros(len(Class))
		res[target_test!=Class]=1
		print("Test error: "+str(np.sum(res)*100/len(test)))
		te.append(np.sum(res)*100/len(test))
		s=s+np.sum(res)*100/len(test)

	print("Average train error ",st/folds)
	print("Average test error ",s/folds)
	print("Standard deviation of train error: "+str(np.std(tr)))
	print("Standard deviation of test error: "+str(np.std(te)))


def main():
	file=sys.argv[1]
	cross=int(sys.argv[2])
	LDA2DGaussGM(file,cross)
if __name__ == "__main__":
	main()
