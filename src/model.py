import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01), name= 'W')

def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name="bias")


class Model:
    def __init__(self, data, y, keep_prob, args):
        self.data = data
        self.modality = args.modality
        self._prediction = None
        self._optimize = args.optimize
        self._loss = args.loss
        self.target = y
        self.lr = args.lr
        self.data = data
        self.n_input_comment = args.vector_size
        self.n_input_danmuku = args.vector_size
        self.n_output_comment = args.n_output_modality[0]
        self.n_output_danmuku = args.n_output_modality[1]
        self.n_input_classification = 0
        if self.modality[0] == 1:
            self.n_input_classification += self.n_output_comment
        if self.modality[1] == 1:
            self.n_input_classification += self.n_output_danmuku
        self.n_hidden_comment = args.n_hidden_comment
        self.n_hidden_classifiation = args.n_hidden_classifiation
        self.n_classes = args.n_classes
        # self.labels = y
        self.keep_prob = keep_prob

        self.graph
        self.loss
        self.optimize
        self.prediction

    # def add_layer(self, inputs, in_size, out_size, layer_name, activity_func=None):
    #     '''
    #     args:
    #         inputs: 层的输入
    #         in_size: 输入的shape
    #         out_size: 输出的shape, 与in_size共同决定了权重的shape
    #         activity_fuc: 激活函数
    #     '''
    #     # 正太分布下，初始化权重W
    #     W = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0), name="W")
    #     # 偏置一般用一个常数来初始化就行
    #     bias =  tf.Variable(tf.constant(0.1, shape=[out_size]), name="bias")
    #     # Wx_Plus_b = tf.matmul(inputs, W) + bias 这种方式与下面的均可以
    #     Wx_Plus_b = tf.nn.xw_plus_b(inputs, W, bias)
    #     if activity_func is None:
    #         outputs = Wx_Plus_b
    #     else:
    #         outputs = activity_func(Wx_Plus_b)
    #     return outputs # 返回的是该层的输出



    @property
    def graph(self):
        feature_fusion = []
        # if self.modality[0] == 1:
        print('The comment modality is used')
        # with tf.variable_scope('comment'):
        W = init_weights([128, 21])
        b = init_bias([21])
        # W = tf.Variable(tf.random_uniform([128, 21]), dtype=tf.float32, name="W")
        # # 偏置一般用一个常数来初始化就行
        # bias =  tf.Variable(tf.constant(0.1, shape=[21]), dtype=tf.float32, name="bias")
        # Wx_b = tf.matmul(inputs, W) + b
        Wx_b = tf.nn.xw_plus_b(self.data[0], W, b)
            # out = tf.matmul(self.data[0], W)
        return Wx_b
                # l1 = self.add_layer(self.data[0], self.n_input_comment, self.n_hidden_comment, 'hidden_layer_1', activity_func = tf.nn.tanh)
                # l2 = self.add_layer(l1, self.n_hidden_comment, self.n_output_comment, 'output_layer_1', activity_func = tf.nn.tanh)
                # print(l2.shape)
        #         feature_fusion.append(l2)
        # feature_fusion = tf.concat(feature_fusion, 1)
        # print('Shape 1:',feature_fusion.shape, self.n_output_comment, self.n_input_classification)
        # if self.modality[1] == 1:
        #     print('The danmuku modality is used')
        #     with tf.variable_scope('danmuku'):
        #         cell = tf.nn.rnn_cell.LSTMCell(self.n_input_danmuku, state_is_tuple=True)
        #         output, state = tf.nn.dynamic_rnn(cell, self.data[1], dtype=tf.float32)
        #         l2 = self._get_last_out(outputs)
        #         feature_fusion.append(l2)
        #
        # feature_fusion = tf.concat(feature_fusion, 1)

        # print('Shape 2:',feature_fusion.shape, self.n_output_comment, self.n_input_classification)
        # with tf.variable_scope('classification'):
        #     l3 = self.add_layer(feature_fusion, self.n_input_classification, self.n_hidden_classifiation, 'hidden_layer_1', activity_func=tf.nn.tanh)
        #     self._prediction = self.add_layer(l3, self.n_hidden_classifiation, self.n_classes, 'prediction_layer', activity_func=tf.nn.softmax)
        # return self._predictio
        # return l1
    @property
    def prediction(self):
        self._prediction = tf.argmax(self.graph, 1)
        return self._prediction

    @property
    def loss(self):
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.graph, labels=self.target))
        return self._loss
        # if self._loss is None or self._loss is 'rmse':
        #     return tf.sqrt(tf.reduce_mean(tf.square(self.prediction - self.target)))
        # if self._loss is 'mae':
        #     return tf.reduce_mean(tf.abs(self.prediction - self.target))
        # if self._loss is 'mse':
        #     return tf.reduce_mean(tf.square(self.prediction - self.target))
        # if self._loss is 'ce':
        #     return tf.reduce_mean(tf.nn.softmax_crosssoftmax_cross_entropy_with_logits(self.prediction - self.target))

    @property
    def optimize(self):
        """
        Define the optimizer of the model used to train the model
        """
        if self._optimize is 'adam' or self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize
