import sys
import numpy as np
import tensorflow as tf
from modules import *
from tensorflow.contrib import rnn

class SDPP(object): #hist entities -> user embedding
    def __init__(self, config, sess, n_nodes):

        self.n_sequences = config.n_sequences
        self.learning_rate = config.learning_rate
        self.emb_learning_rate = config.emb_learning_rate
        self.training_iters = config.training_iters
        self.batch_size = config.batch_size
        self.display_step = config.display_step
        self.n_time_interval = config.n_time_interval
        self.ent_embedding_size = config.ent_embedding_size
        self.embedding_size = config.embedding_size
        self.n_input = config.n_input
        self.n_steps = config.n_steps
        self.n_entities = config.n_entities #111111111111111111111111111111111111111111111111
        #self.n_hidden_gru = config.n_hidden_gru
        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.n_nodes = n_nodes
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.initializer2 = tf.random_uniform_initializer(minval = 0,maxval = 1,dtype = tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)
        self.dropout_prob = config.dropout_prob
        self.sess = sess
        self.name = "undefined"


        self.build_input()
        self.build_var()
        self.pred = self.build_model()

        truth = self.y
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.pow(self.pred - truth, 2)) + self.scale*tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])
        error = tf.reduce_mean(tf.pow(self.pred - truth, 2))
        tf.summary.scalar("error", error)

        var_list1 = [var for var in tf.trainable_variables() if not 'embedding' in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'embedding0' in var.name]
        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)
        grads = tf.gradients(cost, var_list1 + var_list2)
        grads1 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1):]]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_op = tf.group(train_op1, train_op2)
        self.cost = cost
        self.error = error
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)


    def build_input(self):
        self.x = tf.placeholder(tf.int32, shape=[None, self.n_steps, self.n_entities], name="x")
        # (total_number of sequence,n_steps)
        self.x_indict = tf.placeholder(tf.int64,shape=[None,3])
        # (total number of sequence,dim_index)
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
        self.time_interval_index = tf.placeholder(tf.float32, [None,self.n_time_interval], name="time")
        # (total_number of sequence,n_time_interval)
        self.rnn_index = tf.placeholder(tf.float32, [None,self.n_steps], name="rnn_index")
        # self.rnn_index_u = tf.placeholder(tf.float32, [None,self.n_entities], name="rnn_index_u")
        # (total_number of sequence,n_steps)
        # self.rnn_length = tf.placeholder(tf.float32, [None,1], name="rnn_length")
        self.is_training = tf.placeholder(tf.bool, shape=())
    def build_var(self):
        print("build_var...")
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('embedding'):
                entity_embs = np.load('../entity_embeddings'+str(self.ent_embedding_size)+'.npy', allow_pickle=True)
                entity_embs = entity_embs.reshape(1)[0]['ent_embeddings']
                self.ent_embedding = tf.Variable(initial_value=entity_embs, dtype = tf.float32, name='ent_embedding')
                #self.embedding = tf.get_variable('embedding', initializer= self.initializer2([self.n_nodes,self.embedding_size]),dtype = tf.float32)
            '''with tf.variable_scope('BiGRU'):
                self.gru_fw_cell = tf.nn.rnn_cell.GRUCell(2*self.n_hidden_gru)'''
            with tf.variable_scope('SumPooling'):
                self.time_weight = tf.get_variable('time_weight', initializer=self.initializer([self.n_time_interval]), dtype=tf.float32)
        #self.time_weight = tf.multiply(self.time_weight_temp,self.time_weight_temp)
            ''' with tf.variable_scope('dense'):
                self.weights = {
                    'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                        self.n_hidden_dense1])),
                   'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                       self.n_hidden_dense2])),
                    'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1])),
                   'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
                }'''
        print("build_var done")


    def encoder(self, xs, d_model, enc_scope="encoder", posi_emb=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope(enc_scope, reuse=tf.AUTO_REUSE):

            #enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc = xs
            enc *= self.n_input**0.5 # scale

            if (posi_emb):
                enc += positional_encoding(enc, self.n_steps)
            enc = tf.layers.dropout(enc, rate = self.dropout_prob, training=self.is_training)

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_prob,
                                              training=self.is_training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[d_model, d_model])
        memory = enc
        return memory

    def build_model(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('model') as scope:
                with tf.variable_scope('embedding'):
                    x_ent_vector = tf.layers.dropout(tf.nn.embedding_lookup(self.ent_embedding, self.x),
                                             rate = self.dropout_prob, training=self.is_training)
                    x_ent_vector = tf.reshape(x_ent_vector,[-1,self.n_entities,self.ent_embedding_size])
                    '''x_vector = tf.layers.dropout(tf.nn.embedding_lookup(self.embedding, self.x),
                                             rate = self.dropout_prob, training=self.is_training)'''
                    # (total_number of sequence(u), n_steps, n_input)

                with tf.variable_scope('user_embedding'):
                    #print("before encoder: ", x_ent_vector.get_shape())
                    x_vector = self.encoder(x_ent_vector, self.ent_embedding_size)
                    #print("after encoder: ", x_vector.get_shape())
                    #x_vector = tf.reshape(x_vector,[-1,self.embedding_size])
                    #   (total_number of sequence*n_step, 2*n_hidden_gru)

                    #rnn_index_u = tf.reshape(self.rnn_index_u,[-1,1])
                    #   (total_number of sequence*n_step,1)

                    #x_vector = tf.multiply(rnn_index_u,x_vector)
                    #   (total_number of sequence*n_step, 2*n_hidden_gru)

                    x_vector = tf.reshape(x_vector,[-1,self.n_entities,self.ent_embedding_size]) #
                    #   (total_number of sequence,n_entities,2*n_hidden_gru)
                    x_vector = tf.reduce_sum(x_vector, axis=1) # (N, d_model)
                    x_vector = tf.reshape(x_vector,[-1,self.n_steps,self.ent_embedding_size])

                    x_vector = tf.layers.dense(x_vector, self.embedding_size)
                    x_vector = tf.layers.dropout(x_vector, rate = self.dropout_prob, training=self.is_training)

                '''with tf.variable_scope('RNN'):
                    x_vector = tf.transpose(x_vector, [1,0,2])
                    # (n_steps, total_number of sequence, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_input])
                    # (n_steps*total_number of sequence, n_input)


                    # Split to get a list of 'n_steps' tensors of shape (n_sequences*batch_size, n_input)
                    x_vector = tf.split(x_vector, self.n_steps, 0)#tf.split(0, self.n_steps, x_vector)

                    outputs, _ = rnn.static_rnn(self.gru_fw_cell, x_vector, dtype=tf.float32)

                    hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2])
                    # (total_number of sequence, n_steps, n_hidden_gru)

                    # filter according to the length
                    hidden_states = tf.reshape(hidden_states,[-1,2*self.n_hidden_gru])
                    #   (total_number of sequence*n_step, 2*n_hidden_gru)

                    rnn_index = tf.reshape(self.rnn_index,[-1,1])
                    #   (total_number of sequence*n_step,1)


                    hidden_states = tf.multiply(rnn_index,hidden_states)
                    #   (total_number of sequence*n_step, 2*n_hidden_gru)

                    hidden_states = tf.reshape(hidden_states,[-1,self.n_steps,2*self.n_hidden_gru])
                    #   (total_number of sequence,n_step,2*n_hidden_gru)

                    hidden_states = tf.reduce_sum(hidden_states, reduction_indices=[1])
                    #   (total_number of sequence,2*n_hidden_gru)'''

                hidden_states = self.encoder(x_vector, self.embedding_size)
                # filter according to the length
                hidden_states = tf.reshape(hidden_states,[-1,self.embedding_size])
                #   (total_number of sequence*n_step, 2*n_hidden_gru)

                rnn_index = tf.reshape(self.rnn_index,[-1,1])
                #   (total_number of sequence*n_step,1)


                hidden_states = tf.multiply(rnn_index,hidden_states)
                #   (total_number of sequence*n_step, 2*n_hidden_gru)

                hidden_states = tf.reshape(hidden_states,[-1,self.n_steps,self.embedding_size])
                #   (total_number of sequence,n_step,2*n_hidden_gru)
                hidden_states = tf.reduce_sum(hidden_states, axis=1) # (N, d_model)

                with tf.variable_scope('SumPooling'):
                    # sumpooling

                    time_weight = tf.reshape(self.time_weight,[-1,1])
                    #   (n_time_interval,1)
                    #   time_interval_index    (total_number of sequence,n_time_interval)
                    time_weight = tf.matmul(self.time_interval_index,time_weight)
                    #   (total_number of sequence,1)

                    hidden_graph_value = tf.multiply(time_weight,hidden_states)
                    #   (total_number of sequence,2*n_hidden_gru)

                    hidden_graph_value = tf.reshape(hidden_graph_value,[-1])
                    #   (total_number of sequence*2*n_hidden_gru)

                    hidden_graph = tf.SparseTensor(indices = self.x_indict, values=hidden_graph_value,
                                                   dense_shape=[self.batch_size, self.n_sequences, self.embedding_size])

                    hidden_graph = tf.sparse_reduce_sum(hidden_graph, axis=1)
                    # self.batch_size, 2 * self.n_hidden_gru

                with tf.variable_scope('dense'):
                    #dense1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights['dense1']), self.biases['dense1']))
                    #dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
                    #pred = self.activation(tf.add(tf.matmul(dense2, self.weights['out']), self.biases['out']))
                    pred = tf.layers.dense(hidden_graph, self.n_hidden_dense1)
                    pred = tf.layers.dropout(pred, rate = self.dropout_prob, training=self.is_training)
                    pred = tf.layers.dense(pred, self.n_hidden_dense2)
                    pred = tf.layers.dropout(pred, rate = self.dropout_prob, training=self.is_training)
                    pred = tf.layers.dense(pred, 1)
                    #print(pred.get_shape())
                return pred

    def train_batch(self, x, x_indict,y, time_interval_index,rnn_index):
        #merged = tf.summary.merge_all()
        _,time_weight = self.sess.run([self.train_op,self.time_weight],
                                                          feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index,
                                                                     self.is_training:True})
        # print rnn_state
        return time_weight
    def get_embedding(self, x, x_indict,y, time_interval_index,rnn_index):
        embedding = self.sess.run([self.ent_embedding],
                                                          feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index, self.is_training:False})
        # print rnn_state
        return embedding
    def get_error(self, x, x_indict,y, time_interval_index,rnn_index):
        return self.sess.run(self.error, feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index, self.is_training:False})
    def predict(self, x, x_indict,y, time_interval_index,rnn_index):
        return self.sess.run(self.pred, feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index, self.is_training:False})

