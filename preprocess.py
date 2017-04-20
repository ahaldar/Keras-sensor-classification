import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("input.txt", delimiter=',')
#print data.shape

output = [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]

#x = np.array([i for i in range(1200)])
#y = np.array([0 for i in range(1200)])
#print x.shape
#print y.shape

def clean(d):
	out = d
	inds = np.arange(1200)
	for i in range(16):
		zeros, x = np.nonzero(d[i]==0), lambda z: z.nonzero()[0]
		#print zeros
		#out[i][1199-zeros[0]] = np.interp(x(zeros[0]), x(~zeros[0]), out[i][~zeros[0]])
		out[i][zeros[0]] = np.interp(x(zeros[0]), inds, d[i])
	#for i in range(1200):
	#	inds = np.arange(1200)
	#	good = np.where(np.nonzero(d[i]))
	#	print good
	#	f = interp.interp1d(inds, d[i],bounds_error=False)
	#	out[i] = np.where(np.nonzero(d[i]),d[i],f(inds))
	
	return out

def window(d, n):

	out = np.zeros((16, 1200-n))

	print 'window size: ', n

	for i in range(0, 1200-n):
		window = d[:, i:i+n]
		#print window.shape
		cpower = power(window, n)
		#print cpower.shape
		out[:,i] = cpower

	return out


def power(d, n):
	power = np.zeros(16)

	#C = np.fft.fft(d)
	#C = abs(C)
	#print C.shape

	#plt.plot( 20*np.log10(C[0]) )
	#plt.show()

	for i in range(16):
		power[i] = sum(d[i])/n

	return power			

#temp = np.array([[1,2,3,0,0,2,4,0,1,0],[1,1,1,0,0,2,2,0,1,0]])
#print temp
#print data
#plt.plot(np.arange(1200),data[0], color='b')
plt.scatter(np.arange(1200),data[0], marker='x', color='b')
#plt.show()
clean1 = clean(data)
clean2 = clean(clean1)
clean3 = clean(clean2)
#print clean2
plt.plot(np.arange(1200),clean3[0], color='g')
#plt.scatter(np.arange(1200),clean3[0], marker='o', color='g')
plt.show()
out = window(clean3, 1000)
#print out.shape
#print out
#plt.plot(out[0])
#plt.show()

