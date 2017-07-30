import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# download Mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.reset_default_graph()
sess = tf.Session()
Maxepoch = 2000

name_to_visualize_embedding = 'trainembedding'

LOGDIR = "/home/daiict/Documents/Neil/Tensor_flow/CNN/log/" # a directory
path_for_mnist_sprites = os.path.join(LOGDIR, 'sprite_train.png')
path_for_mnist_metadata = os.path.join(LOGDIR, 'metadata.tsv')

# define convolutional layer
def conv_layer(input, channel_in, channel_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5,5,channel_in, channel_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="B")
        conv = tf.nn.conv2d(input,w, strides=[1,2,2,1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activations", act)
        return act


# define fully connected layer
def fc_layer(input, channel_in, channel_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channel_in,channel_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

# define MNIST model
def mnist_model(learning_rate,use_two_conv,use_two_fc,hparam_str):
    tf.reset_default_graph()
    sess = tf.Session()

    # define placeholders
    x = tf.placeholder(tf.float32,shape=[None, 784], name="x") # _x784 (100 examples in one batch)
    y = tf.placeholder(tf.float32,shape=[None, 10], name="labels") # original labels, _x10

    # Reshape image(_x784) to 28x28 pixels so as to convolve on it
    x_image = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input', x_image, 3)

    # Convolutional layer
    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1") # 14x14x32
        conv_out = conv_layer(conv1,32,64, "conv2") # 7x7x64
    else:
        conv = conv_layer(x_image, 1, 64, "conv") # 14x14x64
        conv_out = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") #7x7x64

    # flattening output into 7x7x64 value, to give in FC layer
    flattened = tf.reshape(conv_out, [-1,7 * 7 * 64])

    # FC layer
    if use_two_fc:
        fc1 = fc_layer(flattened, 7*7*64, 1024, "fc1") 
        logits = fc_layer(fc1, 1024, 10, "fc2") # predicted labels
    else:
        logits = fc_layer(flattened, 7*7*64, 10, "fc")


    # compute cross entropy as our loss function
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    # use GD as optimizer to train network
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    # Compute accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # merging summary and writing that into a directory
    merged_summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR+hparam_str)
    writer.add_graph(sess.graph)

    # actual Training
    for i in range(Maxepoch):
        batch = mnist.train.next_batch(100)
        if i == Maxepoch-1:
            batch_x = batch[0] #100x784
            print(batch_x)
            batch_y_temp = batch[1] # 100x10
            batch_y = np.zeros([100,]) # 100 points, output labels
            for j in xrange(0,100):
                count = 0
                for k in xrange(0,10):
                    if batch_y_temp[j,k] == 1:
                        break
                    count = count + 1
                batch_y[j] = count

            ### Embedding
            embedding_var = tf.Variable(batch_x, name='name_to_visualize_variable')
            writer = tf.summary.FileWriter(LOGDIR)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # Specify where you find the metadata
            embedding.metadata_path = path_for_mnist_metadata #'metadata.tsv'

            # Specify where you find the sprite (we will create this later)
            embedding.sprite.image_path = path_for_mnist_sprites #'mnistdigits.png'
            embedding.sprite.single_image_dim.extend([28,28])

            # Say that you want to visualise the embeddings
            projector.visualize_embeddings(writer, config)
              
            ### start session
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), 1)

                ###
            def create_sprite_image(images):
                """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
                if isinstance(images, list):
                    images = np.array(images)
                img_h = images.shape[1]
                img_w = images.shape[2]
                n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
                spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
                for i in range(n_plots):
                    for j in range(n_plots):
                        this_filter = i * n_plots + j
                        if this_filter < images.shape[0]:
                            this_img = images[this_filter]
                            spriteimage[i * img_h:(i + 1) * img_h,
                                j * img_w:(j + 1) * img_w] = this_img
    
                return spriteimage

            def vector_to_matrix_mnist(mnist_digits):
                """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
                return np.reshape(mnist_digits,(-1,28,28))

            def invert_grayscale(mnist_digits):
                """ Makes black white, and white black """
                return 1-mnist_digits

            to_visualise = batch_x
            to_visualise = vector_to_matrix_mnist(to_visualise)
            to_visualise = invert_grayscale(to_visualise)

            sprite_image = create_sprite_image(to_visualise)

            # plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
            # plt.imshow(sprite_image,cmap='gray')
            # plt.show()

            with open(path_for_mnist_metadata,'w') as f:
                f.write("Index\tLabel\n")
                for index,label in enumerate(batch_y):
                    f.write("%d\t%d\n" % (index,label))

        # occasionally record parameters
        if i % 5 == 0:
            [error, train_accuracy, s] = sess.run([xent, accuracy, merged_summary], feed_dict={x:batch[0], y:batch[1]})
            print("step %d, training accuracy %g, error %g" % (i, train_accuracy*100, error))
            writer.add_summary(s, i)

        # feed placeholders and run train_step
        sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})

# creates a string of selected hyper-paramters
def make_hyper_string(learning_rate,use_two_conv,use_two_fc):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
      # Try few hyper-paramter
    for learning_rate in [1E-3,1E-4,1E-5]:
        # Try different model architecture
        for use_two_fc in [True,False]:
            for use_two_conv in [True,False]:
                # make unique string for each of this hyper-paramter setting
                # eg: (1E-3, fc=2, conv=2)
                hparam_str = make_hyper_string(learning_rate,use_two_conv,use_two_fc) # a string
                print "Starting run for %s" % hparam_str
                mnist_model(learning_rate,use_two_conv,use_two_fc,hparam_str)

if __name__ == '__main__':
    main()

    