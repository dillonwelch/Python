import neuralNet

input1 = 1
input2 = 1

# 0 0, no backprop
w111 = 0.129952
w112 = -0.923123
w121 = 0.570345
w122 = -0.328932
w211 = 0.164732
w221 = 0.752621

# 0 1, backprop
# w111 = 0.129952
# w112 = -0.923123
# w121 = 0.570345
# w122 = -0.328932
# w211 = 0.110210
# w221 = 0.698099

# 1 0, backprop
# w111 = 0.129952
# w112 = -0.923123
# w121 = 0.572094
# w122 = -0.317103
# w211 = 0.154145
# w221 = 0.736616

# 1 1, with backprop?
# w111 = 0.132921
# w112 = -0.910900
# w121 = 0.572094
# w122 = -0.317103
# w211 = 0.195341
# w221 = 0.760716

for input1 in range(0, 2):
	for input2 in range(0, 2):
		a = input1 * w111 + input2 * w121
		b = input1 * w112 + input2 * w122
		print input1, w111
		print input2, w121
		print input1, w112
		print input2, w122
		c = neuralNet.sigmoid(a)
		d = neuralNet.sigmoid(b)
		e = c * w211 + d * w221
		print c, w211
		print d, w221
		print e
		print input1, input2, neuralNet.sigmoid(e)
		print "\n\n"