import cPickle
from urllib import urlretrieve
import os
import numpy as np
from classify_image import *
from scipy.misc import imsave



data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
data_path = os.path.dirname(os.path.abspath(__file__)) + "/data/cifar-10-batches-py/data_batch_1"
image_path = os.path.dirname(os.path.abspath(__file__)) + "/data/images/"

transfer_data_file = os.path.dirname(os.path.abspath(__file__)) + "/data/transfer.npy"

epochs = 100
batch_size = 32
display_step = 5
n_classes = 10




def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def get_inputs():
	data_dict = unpickle(data_path)

	data = data_dict['data']
	labels = np.array(data_dict['labels'])

	inputs = np.array(data).reshape(-1,3,32,32)
	inputs = inputs.transpose([0,2,3,1])

	return inputs, labels

inputs, labels = get_inputs()

def plot_img(img):
	import matplotlib.pyplot as plt
	plt.imshow()
	plt.show()


def save_all_images():
	for i in range(len(inputs)):
		imsave(os.path.join(image_path, "img_" + str(i) + ".jpg"), inputs[i])
		print("Saved image number " + str(i))

def get_image_paths():
	return np.array([ os.path.join(image_path, "img_" + str(i) + ".jpg") for i in range(5000) ])

def save_transfer_values(paths):
	transfer_values = run_inference_on_image(paths)
	np.save(transfer_data_file, transfer_values.reshape(-1, 2048))

X = np.load(transfer_data_file)
labels = labels[:5000]
print labels
y_ = np.zeros((len(labels), n_classes))
y_[np.arange(len(labels)), labels] = 1
print y_.shape


x = tf.placeholder("float", [None, 2048])
W = tf.Variable(tf.random_normal([2048, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))
y = tf.placeholder("float", [None, n_classes])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b, y_))
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:

	sess.run(init)

	for epoch in range(epochs):

		# Run optimization op (backprop) and cost op (to get loss value)
		_, c = sess.run([optimizer, cost], feed_dict={x: X, y: y_})
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", \
				"{:.9f}".format(c))
	print("Optimization Finished!")





# maybe_download_and_extract()
# save_all_images()
# save_transfer_values(get_image_paths())



