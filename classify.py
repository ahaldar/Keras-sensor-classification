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

print data
out = window(data, 1000)
print out.shape
print out
#plt.plot(out[0])
#plt.show


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X = out
Y = output

# create model
model = Sequential()
model.add(Dense(50, input_dim=200, init='normal', activation='relu'))
model.add(Dense(50, init='normal', activation='relu'))
model.add(Dense(4, init='normal', activation='sigmoid'))

'''
epochs = 500
learning_rate = 0.5
decay_rate = learning_rate / epochs
momentum = 0.9

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
'''
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, validation_split = 0.33, nb_epoch=200, batch_size=100)

# evaluate the model
scores = model.evaluate(X, Y)
print "\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [[round(x1),round(x2),round(x3),round(x4)] for [x1,x2,x3,x4] in predictions]
print rounded


