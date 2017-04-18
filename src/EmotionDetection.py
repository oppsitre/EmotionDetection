import argparse
import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from reader import Reader
from model import Model
parser = argparse.ArgumentParser()
parser.add_argument("-lr", type = float, default = 0.01, help="learning rate")
parser.add_argument('-vector_size', type = int, default = 128, help = 'the size of word2vector')
parser.add_argument('-n_part_danmuku', type = int, default = 10, help = 'the lenght of lstm of danmuku')
parser.add_argument('-epoch_size', type = int, default = 10, help = 'the number of training epoch')
parser.add_argument('-batch_size', type = int, default = 1, help = 'the number of input in each batch')
parser.add_argument('-modality', type = list, default = [1, 0], help = 'choose which modalities will be used')
parser.add_argument('-ratio_split', type = list, default = [0.7, 0.3], help = 'ratio_split = [train_ration, test_ratio]')
parser.add_argument('-n_hidden_comment', type = int, default = 128, help = 'the size of hidden layer in modality of comment')
parser.add_argument('-n_hidden_danmuku', type = int, default = 128, help = 'the size of hidden layer in modality of danmuku')
parser.add_argument('-n_hidden_classifiation', type = int, default = 128, help = 'the size of hidden layer of classification')
parser.add_argument('-n_output_modality', type = list, default = [128,128], help = 'the list of output feature of each modality [comment, danmuku]')
parser.add_argument('-n_classes', type = int, default = 21, help = 'the number of classes will be predicted')
parser.add_argument('-optimize', type = str, default = 'adam', help = 'the method of optimization')
parser.add_argument('-loss', type = str, default = 'rmse', help = 'the function of loss')
parser.add_argument('-test_step', type = int, default = 2, help = 'step of print the loss of test set')

def fill_feed_dict(x_comment_placeholder, \
                x_danmuku_placeholder, \
                y_placeholder, \
                keep_prob_placeholder, \
                feed_data, keep_prob, modality):
    # print(type(feed_data))
    feed_dict = {}
    index = 0
    if modality[0] == 1:
        feed_dict[x_comment_placeholder] = feed_data[index]
        index += 1
    if modality[1] == 1:
        feed_dict[x_danmuku_placeholder] = feed_data[index]
        index += 1
    feed_dict[y_placeholder] = feed_data[-1]
    feed_dict[keep_prob_placeholder] = keep_prob

    return feed_dict

def do_eval(sess, evaluation, feed_dict):
    return sess.run(evaluation, feed_dict=feed_dict)

def batch_test(sess, x_comment, x_danmuku,  y_, keep_prob, modality, evaluation, get_data_set, batch_size):
    ptr=0
    results = []
    while True:
        data_set = get_data_set(ptr, batch_size)
        if len(data_set) == 0:
            break
        feed_dict = fill_feed_dict(x_comment, x_danmuku, y_, keep_prob, data_set, 1.0, modality)
        res = do_eval(sess, evaluation, feed_dict)
        results.append(res)
        ptr += batch_size

    return np.concatenate(results, axis=0)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01), name= 'W')

def init_bias(shape):
    return tf.Variable(tf.constant(0, shape=shape), name="bias")


# def build graph()
#
#

def main(args):
    x_comment_placeholder = tf.placeholder(tf.float32, [None, args.vector_size])
    x_danmuku_placeholder = tf.placeholder(tf.float32, [None, args.n_part_danmuku, args.vector_size])
    keep_prob_placeholder = tf.placeholder(tf.float32)
    y_placeholder = tf.placeholder(tf.int32, [None, 21])
    # with tf.variable
    # W = init_weights([128, 21])
    # b = init_bias([21])
    # Wx_b = tf.nn.xw_plus_b(x_comment_placeholder, W, b)
    # predict_op = tf.argmax(Wx_b, 1)
    # print(type(predict_op), predict_op.shape)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Wx_b, labels=y_placeholder))
    # train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # W = init_weights([128, 21])
    # b = init_bias([21])
    # Wx_b = tf.nn.xw_plus_b(x_comment_placeholder, W, b)
    # predict_op = tf.argmax(Wx_b, 1)
    # print(type(predict_op), predict_op.shape)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Wx_b, labels=y_placeholder))
    # train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

    reader = Reader(args)
    model = Model([x_comment_placeholder, x_danmuku_placeholder], y_placeholder, keep_prob_placeholder, args)
    # print(model.graph.shape, model.prediction.shape, model.loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # sess.run(tf.initialize_all_variables())
        for i in range(args.epoch_size):
            # if (i + 1) % args.test_step == 0:
            #     test_data = reader.get_test_data()
            #     print('Test Data', test_data[0].shape, test_data[1].shape)
            #     feed_dict = fill_feed_dict(x_comment_placeholder, x_danmuku_placeholder, y_placeholder, keep_prob_placeholder, test_data, 1.0, args.modality)
            #     sess.run(model.loss, feed_dict=feed_dict)
            #     print('Loss:', res)
            print('Trainning Round:', i + 1)
            batch = reader.next_batch()
            feed_dict = fill_feed_dict(x_comment_placeholder, x_danmuku_placeholder, y_placeholder, keep_prob_placeholder, batch, 0.6, args.modality)
            print(sess.run(model.loss, feed_dict=feed_dict))
            # print(sess.run(model.prediction, feed_dict=feed_dict))

            # sess.run(model.optimize, feed_dict=feed_dict)


            # sess.run(train_op, feed_dict=feed_dict)
            # print(sess.run(predict_op, feed_dict=feed_dict))
            # print(sess.run(cost, feed_dict = feed_dict))


if __name__ == '__main__':
    args = parser.parse_args()
    print(type(args), args.vector_size)
    main(args)
