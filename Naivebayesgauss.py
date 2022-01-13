from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eig
from numpy.linalg import det
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd

def naiveBayesGaussian(filename, num_splits, train_percent):

	data = pd.read_csv(filename,header=None)
	data = data.values
	target = data[:,-1]
	data = data[:,:-1]

	#setting target to boolean in case of boston data
	if len(set(target))>10:
		bool_target=np.zeros((len(target)))
		median = np.median(target)
		for i in range(len(target)):
			if target[i]>median:
				bool_target[i]=1
		target=bool_target

	folds=num_splits

# =============================================================================
# 	N=np.zeros(len(set(target)))
#
# 	for i in range(len(set(target))):
# 		x=0
# 		for j in range(len(data)):
# 			if target[j]==i:
# 				x=x+1
# 		N[i]=x
#
# 	prior=[]
#
# 	for i in range(len(set(target))):
# 		prior.append(N[i]/len(data))
# =============================================================================
	s=0
	errors=np.zeros((5,folds))

	#running folds times:
	for i in range(folds):
		print("Iteration: "+str(i+1))
		train=[]
		target_train=[]
		test=[]
		target_test=[]
		for j in range(len(set(target))):
			z,=np.where(target == j)
			X_class=data[z]
			target_class=target[z]
			k = int(len(X_class) * 0.8)
			indicies = random.sample(range(len(X_class)), k)
			train.extend(X_class[indicies].tolist())
			target_train.extend(target_class[indicies].tolist())
			test_indicies=set(range(len(X_class))).difference(set(indicies))
			test.extend(X_class[list(test_indicies)].tolist())
			target_test.extend(target_class[list(test_indicies)].tolist())

		train=np.array(train)
		target_train=np.array(target_train)
		test=np.array(test)
		target_test=np.array(target_test)
		splits=train_percent
		#splitting the data into train and test:
		for l in splits:
			print("Percentage: "+str(l))
			train_l=[]
			target_train_l=[]
			for j in range(len(set(target_train))):
				z,=np.where(target_train == j)
				X_class=train[z]
				target_class=target_train[z]
				k = int(len(X_class) * l)
				indicies = random.sample(range(len(X_class)), k)
				train_l.extend(X_class[indicies].tolist())
				target_train_l.extend(target_class[indicies].tolist())

			train_l=np.array(train_l)
			target_train_l=np.array(target_train_l)

			prior=[]
			Class=[]
			Cov=[]
			M=[]
			C=np.zeros((len(data[0]),len(data[0])))
			#Learning the generative model
			for j in range(len(set(target))):
				k,=np.where(target_train_l == j)
				c=(np.diagonal(np.cov(train_l[k],rowvar=False)))*np.identity(len(np.diagonal(np.cov(train_l[k],rowvar=False))))
				Cov.append(c)
				M.append(np.mean(train_l[k],axis=0))
				prior.append(len(k)/len(data))
				C=C+Cov[j]*prior[j]


			G=[]
			#Classification using discriminant function
			for j in range(len(test)):
				G.append([])
				for k in range(len(set(target))):
					G[j].append(np.matmul(np.matmul(pinv(C),M[k])[None,:],test[j]) - np.matmul(np.matmul(M[k],pinv(C))[None,:],M[k][:,None])/2 + np.log(prior[k]))
				idx=np.argmax(G[j])
				Class.append(idx)

			#Error-rate calculation
			Class=np.array(Class)
			res=np.zeros(len(Class))
			res[target_test!=Class]=1
			print("Test error: "+str(np.sum(res)*100/len(test)))
			errors[splits.index(l)][i]=np.sum(res)*100/len(test)
			s=s+np.sum(res)*100/len(test)

	print("Average error ",s/(folds*5))

	plt.figure()
	plt.title('Mean and Standard Deviation')
	plt.plot(splits,np.mean(errors,axis=1),'-o')
	plt.plot(splits,np.std(errors,axis=1),'-o')
	plt.legend(('Mean','Standard Deviation'))
	plt.xlabel('train percentage')
	plt.show()



def main():
	file=sys.argv[1]
	num=int(sys.argv[2])
	t=[float(i) for i in (sys.argv[3]).split(',')]
	naiveBayesGaussian(file,num,t)
if __name__ == "__main__":
	main()
