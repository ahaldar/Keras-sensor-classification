import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory

data = np.loadtxt("input.txt", delimiter=',')

output = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
output.shape = (16,1)

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

clean1 = clean(data)
clean2 = clean(clean1)
clean3 = clean(clean2)

out = window(clean3, 1000)

data2 = np.column_stack([data, output])
print data2.shape

mapsize = (16, 1)
sm = SOMFactory().build(data2, mapsize, normalization = 'var', initialization='random', component_names=output)
sm.train(n_job=1, verbose=False, train_rough_len=20, train_finetune_len=20)

topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print "Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error)

from sompy.visualization.hitmap import HitMapView
sm.cluster(4)
hits  = HitMapView(10,10,"Clustering",text_size=12)
a=hits.show(sm)
