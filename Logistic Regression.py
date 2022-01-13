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

#softmax function
def softmax(x):
	mx = np.max(x, axis=-1, keepdims=True)
	numerator = np.exp(x - mx)
	denominator = np.sum(numerator, axis=-1, keepdims=True)
	return numerator/denominator

def logisticRegression(filename, num_splits, train_percent):

	data = pd.read_csv(filename,header=None)
	data = data.values
	target = data[:,-1]
	data = data[:,:-1]

#setting target to boolean
	if len(set(target))>10:
		binary=1
		bool_target=np.zeros((len(target)))
		median = np.median(target)
		for i in range(len(target)):
			if target[i]>median:
				bool_target[i]=1
		target=bool_target

	else:
		binary=0

	data=np.insert(data,0,1,axis=1)

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

	#running fold times:
	for i in range(folds):
		print("Iteration: "+str(i+1))
		train=[]
		target_train=[]
		test=[]
		target_test=[]
		#splitting the data into train and test:
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
		#getting different train splits
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

			Class=np.zeros(len(test))

			#binary class case
			if binary==1:
				w=np.zeros(len(data[0]))
				#w vector calculation
				for e in range(2):
					a = np.matmul(train_l,w)
					pi = 1/(1+np.exp(-a))
					R = pi*(1-pi)*np.identity(len(pi))
					z = np.matmul(train_l,w) - np.matmul(pinv(R),(pi-target_train_l))
					w = np.matmul(np.matmul(np.matmul(pinv(np.matmul(np.matmul(np.transpose(train_l),R),train_l)),(np.transpose(train_l))),R),z)
				#Classification
				a = np.matmul(test,np.transpose(w))
				pi = 1/(1+np.exp(-a))
				Class[pi>0.5]=1

			#Multi-class case
			else:
				w=np.zeros((len(set(target)),len(data[0])))
				Y_hot=np.eye(len(set(target)))[target_train_l]
				for j in range(2):
					pi = softmax(np.dot(train_l,np.transpose(w)))
					R = pi * (1-pi)
					for k in range(len(set(target))):
						r = np.diag(R[:,k])
						z = np.matmul(train_l,w[k]) - np.matmul(pinv(r),(pi[:,k]-Y_hot[:,k]))
						w[k] = np.matmul(np.matmul(np.matmul(pinv(np.matmul(np.matmul(np.transpose(train_l),r),train_l)),(np.transpose(train_l))),r),z)
				#Classification
				a = np.matmul(test,np.transpose(w))
				for j in range(len(a)):
					Class[j]=np.argmax(softmax(a[j]))

			#Error-rate calculation
			res=np.zeros(len(Class))
			res[target_test!=Class]=1
			print("Test error: "+str(np.sum(res)*100/len(test)))
			errors[splits.index(l)][i]=np.sum(res)*100/len(test)
			s=s+np.sum(res)*100/len(test)

	print("Average test error ",s/(folds*5))

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
	logisticRegression(file,num,t)
if __name__ == "__main__":
	main()
