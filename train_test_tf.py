import numpy as np
import tensorflow as tf
import image_process

img_w = 100
img_h = 100

batch_size = 100
learning_rate = 0.001
max_steps = 2000

data_sets = image_process.prepare_data()

# Define input placeholders and variables
images_placeholder = tf.placeholder(tf.float32, shape=[None, img_w*img_h*3])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])
weights = tf.Variable(tf.zeros([img_w*img_h*3, 2]))
biases = tf.Variable(tf.zeros([2]))

# Define the classifier's result
logits = tf.matmul(images_placeholder, weights) + biases

# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(max_steps):
        # Generate input data batch
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        # Print out the model's current accuracy every 100 steps
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

        # Perform a single training step
        sess.run(train_step, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})

    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))



