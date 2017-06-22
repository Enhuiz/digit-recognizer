import tensorflow as tf
import pandas as pd
import numpy as np

train_data = pd.read_csv('data/train.csv').values

np.random.seed(7) # change the seed to get different random result
np.random.shuffle(train_data) 

train_x = train_data[:, 1:] / 256 # divide 256 to limit the value of the images between 0 and 1
train_y = np.eye(10)[train_data[:, 0]] # one-hot ize
test_x = pd.read_csv('data/test.csv').values / 256

print('data loaded')

learning_rate = 1e-4
training_epochs = 100 # large enough
batch_size = 256 
display_step = 20

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.7 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([7, 7, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([14*14*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create session
sess = tf.InteractiveSession()

# Initializing the variables
tf.global_variables_initializer().run()

def to_csv(result, name="pred.csv"):
    df = pd.DataFrame(result)
    df.columns = ["Label"]
    df.index += 1
    df = df.to_csv(name, index_label="ImageId")
    print('result has been written to', name)

# Launch the graph
for epoch in range(training_epochs):
    # Calculate accuracy for 256 mnist test images
    for i in range(len(train_x) // batch_size):
        batch_x = train_x[i*batch_size: (i+1)*batch_size]
        batch_y = train_y[i*batch_size: (i+1)*batch_size]

        sess.run(step, {x: batch_x, y: batch_y, keep_prob: dropout})
    
        if i % display_step == 0:
            tr_loss, tr_acc = sess.run([loss, accuracy], feed_dict={x: batch_x,
                                                                    y: batch_y,
                                                                    keep_prob: 1.})

            print("Epoch", epoch, 
                  "Batch {}/{}".format(i , str(len(train_x) // batch_size)),
                  "Minibatch Loss =", "{:.6f}".format(tr_loss),
                  "Training Accuracy =", "{:.5f}".format(tr_acc))

    print('predicting')
    result = []
    for i in range(len(test_x) // batch_size): # seperate the calculation process to keep memory from running out
        batch_x = test_x[i*batch_size: (i+1)*batch_size]
        batch_y = np.argmax(pred.eval({x: batch_x, keep_prob: 1.}), axis=1)
        result.append(batch_y)
    
    rest = len(test_x) - len(test_x) // batch_size * batch_size # some samples left not calculated in the above 'for'
    batch_x = test_x[-rest:]
    batch_y = np.argmax(pred.eval({x: batch_x, keep_prob: 1.}), axis=1)
    result.append(batch_y)

    to_csv(np.concatenate(result))
    