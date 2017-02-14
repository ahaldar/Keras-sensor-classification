import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("input.txt", delimiter=',')
#print data.shape

#x = np.array([i for i in range(1200)])
#y = np.array([0 for i in range(1200)])
#print x.shape
#print y.shape

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
	
	C = np.fft.fft(d)
	C = abs(C)
	#print C.shape
	
	#plt.plot( 20*np.log10(C[0]) )
	#plt.show()
	
	for i in range(16):
		power[i] = sum(C[i])/n
	
	return power			

print data
out = window(data, 10)
print out.shape
print out
plt.plot(out[0])
plt.show
