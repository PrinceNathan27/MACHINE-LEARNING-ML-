from sklearn.datasets import load_boston
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys
import pandas as pd

def LDA1dProjection(filename):
	boston = pd.read_csv(filename,header=None)
	boston = boston.values
	data = boston[:,:-1]
	target = boston[:,-1]
	bool_target=np.zeros((len(target)))
	median = np.median(target)
	#Setiing target to boolean
	for i in range(len(target)):
		if target[i]>median:
			bool_target[i]=1

	projected_data=[]

	sum0=sum1=np.zeros(len(data[0]))
	for i in range(len(bool_target)):
		if bool_target[i]==0:
			sum0=sum0 + data[i]
		else:
			sum1=sum1 + data[i]

	#Mean of both classes:
	mean_c0=sum0/list(bool_target).count(0)
	mean_c1=sum1/list(bool_target).count(1)

	#Calculating within scatter:
	t_sw0 = np.zeros((len(data[0]),len(data[0])))
	t_sw1 = np.zeros((len(data[0]),len(data[0])))
	for j in range(len(data)):
		if bool_target[j] == 0:
			t_sw0 = t_sw0 + np.matmul((data[j]-mean_c0)[:,None],(data[j]-mean_c0)[None,:])
		else:
			t_sw1 = t_sw1 + np.matmul((data[j]-mean_c1)[:,None],(data[j]-mean_c1)[None,:])

	Sw = t_sw0 + t_sw1

	#Projection vector
	w = np.matmul(inv(Sw),(mean_c0-mean_c1))

	#Projecting data into 1-dimensional space
	projected_data = np.matmul(data,w[:,None])


	i,=np.where(bool_target == 0)
	proj_c0 = [projected_data[j] for j in i.tolist()]

	i,=np.where(bool_target == 1)
	proj_c1 = [projected_data[j] for j in i.tolist()]

	plt.figure()
	plt.hist([proj_c0,proj_c1],20,color=['r','b'])
	plt.show()

def main():
	file=sys.argv[1]
	LDA1dProjection(file)
if __name__ == "__main__":
	main()
