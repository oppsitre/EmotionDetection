import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument("-lr", type = float, default = 0.0001, help="learning rate")
parser.add_argument('-n_input_comment', type = int, default = 128, help = 'the size of input feature of comment')
parser.add_argument('-n_input_danmuku', type = int, default = 128, help = 'the size of input feature of danmuku')
parser.add_argument('-n_step_danmuku', type = int, default = 10, help = 'the lenght of lstm of danmuku')
parser.add_argument('-epoch_size', type = int, default = 100, help = 'the number of training epoch')
parser.add_argument('-modality' type = list, default = [1,0], help = 'choose which modalities will be used')
def fill_feed_dict(x_comment_placeholder, \
                x_danmuku_placeholder, \
                y_placeholder, \
                keep_prob_placeholder, \
                feed_data, keep_prob, modality):
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

def main(args):
    n_input_comment = args.n_input_comment
    x_comment = tf.placeholder(tf.float32, [None, n_input_comment])
    x_danmuku = tf.placeholder(tf.float32, [None, n_step_comment, n_input_danmuku])
    keep_prob = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32, [None, 1])
    model = Model([x_comment, x_danmuku], y, keep_prob, args)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(args.epoch_size):
            if i % args.test_step == 0:
                test_resutl =
            batch = reader.next_batch()
            feed_dict = fill_feed_dict(x_comment, x_danmuku, y, keep_prob_placeholder, model.loss, batch, 0.6, args.modality)

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
