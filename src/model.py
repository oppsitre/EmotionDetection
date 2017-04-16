import tensorflow as tf
class Model:
    def __init__(self, data, config):
        self.data = data
        self.modality = config.modality
        self.prediction = None
        self.target = None

        self.inputsize_comment = config.inputsize_comment
        self.n_hidden_comment = config.n_hidden_comment
        self.inputsize_danmuku = config.inputsize_danmuku
        self.n_hidden_classifiation = config.hidden_unit_classifiation

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev = 0.01), name= 'W')

    def init_bias(out_size):
        return bias =  tf.Variable(tf.constant(0.1, shape=[out_size]), name="bias")

    def add_layer(inputs, in_size, out_size, layer_name, activity_func=None):
        '''
        args:
            inputs: 层的输入
            in_size: 输入的shape
            out_size: 输出的shape, 与in_size共同决定了权重的shape
            activity_fuc: 激活函数
        '''
        # 正太分布下，初始化权重W
        W = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0), name="W")
        # 偏置一般用一个常数来初始化就行
        bias =  tf.Variable(tf.constant(0.1, shape=[out_size]), name="bias")
        # Wx_Plus_b = tf.matmul(inputs, W) + bias 这种方式与下面的均可以
        Wx_Plus_b = tf.nn.xw_plus_b(inputs, W, bias)
        if activity_func is None:
            outputs = Wx_Plus_b
        else:
            outputs = activity_func(Wx_Plus_b)
        return outputs # 返回的是该层的输出

    def prediction(self):
        feature_fusion = []
        if self.modality[0] == 1:
            print('The comment modality is used')
            with tf.variable_scope('comment'):
                l1 = add_layer(self.comments, self.inputsize_comment, self.n_hidden_comment, 'hidden_layer_1', activity_func = tf.nn.tanh)
                l2 = add_layer(l1, self.n_hidden_comment, self.n_output_comment, 'output_layer_1', activity_func = tf.nn.tanh)
                feature_fusion.append(l2)

        if self.modality[1] == 1:
            print('The danmuku modality is used')
            with tf.variable_scope('danmuku'):
                cell = tf.nn.rnn_cell.LSTMCell(self.n_third_hidden, state_is_tuple=True)
                output, state = tf.nn.dynamic_rnn(cell, self.input_danmuku, dtype=tf.float32)
                feature_fusion.append(self._get_last_out(outputs))

        with tf.variable_scope('classification'):
            l1 = add_layer(feature_fusion, self.n_input, self.n_hidden_classifiation, 'hidden_layer_1', activity_func=tf.nn.tanh)
            prediction = add_layer(l1, self.self.n_hidden_classifiation, self.n_classes, 'prediction_layer', activity_func=tf.nn.softmax)
        return prediction

    def loss(self):
        if self._loss is None:
            return self.cross_entropy()
        if self._loss is 'mae':
            return self.mean_absolute_error()
        if self._loss is 'rmse':
            return self.root_mean_square_error()
        if self._loss is 'ce':
            return self.cross_entropy()

    @property
    def optimize(self):
        """
        Define the optimizer of the model used to train the model
        """
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize

    def root_mean_square_error(self):
        return tf.sqrt(tf.reduce_mean(tf.square(self.prediction - self.target)))
    def mean_absolute_error(self):
        return tf.reduce_mean(tf.abs(self.prediction - self.target))
    def cross_entropy(self):
        return tf.reduce_mean(tf.nn.softmax_crosssoftmax_cross_entropy_with_logits(self.prediction - self.target))
