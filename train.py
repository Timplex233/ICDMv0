import sys
import numpy as np
import math
import tensorflow as tf
from model import SDPP
import sys
import pickle
import gzip
tf.set_random_seed(0)
import time
import config as timecas_config

NUM_THREADS = 20
# DATA_PATH = "data"


n_nodes, n_sequences, n_steps = pickle.load(open(timecas_config.information, 'rb'))
print("dataset information: ",n_nodes,n_sequences,n_steps)
tf.flags.DEFINE_integer("n_sequences", n_sequences, "num of sequences.")
tf.flags.DEFINE_integer("n_steps", n_steps, "num of step.")
tf.flags.DEFINE_integer("n_entities", 6, "num of entities for the user.")
tf.flags.DEFINE_integer("num_blocks", 3, "num of blocks.")
tf.flags.DEFINE_integer("num_heads", 1, "num of heads.")
tf.flags.DEFINE_integer("time_interval", timecas_config.time_interval, "the time interval")
tf.flags.DEFINE_integer("n_time_interval", timecas_config.n_time_interval, "the number of  time interval")
learning_rate = float(sys.argv[1])
emb_learning_rate = float(sys.argv[2])
l2 = float(sys.argv[3])
dropout = float(sys.argv[4])

tf.flags.DEFINE_float("learning_rate", learning_rate, "learning_rate.")
tf.flags.DEFINE_integer("batch_size", 32, "batch size.")
#tf.flags.DEFINE_integer("n_hidden_gru", 32, "hidden gru size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", l2, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("training_iters", 20*3200*8 + 1, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")
tf.flags.DEFINE_integer("ent_embedding_size", 24, "entity embedding size.")
tf.flags.DEFINE_integer("embedding_size", 64, "embedding size.")
tf.flags.DEFINE_integer("n_input", 50, "input size.")
tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", emb_learning_rate, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", dropout, "dropout probability.")

config = tf.flags.FLAGS

print("dropout prob:",config.dropout_prob)
print("l2",config.l2)
print("learning rate:",config.learning_rate)
print("emb_learning_rate:",config.emb_learning_rate)


# (total_number of sequence,n_steps)
def get_batch(x, y, sz, time,rnn_index,n_time_interval,step,batch_size=128):

    batch_y = np.zeros(shape= (batch_size, 1))
    batch_x = []
    batch_x_indict = []
    batch_time_interval_index = []
    batch_rnn_index = []

    start = step * batch_size % len(x)
    # print start
    for i in range(batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        for j in range(sz[id]):
            temp_x = np.zeros(shape=(len(x[id][j]), config.n_entities), dtype=np.int)
            for k in range(len(x[id][j])):
                #print(x[id][j][k], idmap[x[id][j][k]], user_table[idmap[x[id][j][k]]])
                temp_x[k] = user_table[idmap[x[id][j][k]]]
            batch_x.append(temp_x)
            #batch_x.append(x[id][j])
            #time_interval
            temp_time = np.zeros(shape=(n_time_interval))
            k = int(math.floor(time[id][j] / config.time_interval))
            temp_time[k] = 1
            batch_time_interval_index.append(temp_time)

            #rnn index
            temp_rnn = np.zeros(shape=(config.n_steps))
            if rnn_index[id][j]-1 >=0:
                temp_rnn[rnn_index[id][j]-1] = 1
            batch_rnn_index.append(temp_rnn)

            for k in range(config.embedding_size):#2*config.n_hidden_gru):
                batch_x_indict.append([i,j,k])

    # print(type(batch_x[0]), type(batch_time_interval_index), len(batch_x), len(batch_time_interval_index))
    return batch_x,batch_x_indict, batch_y,batch_time_interval_index,batch_rnn_index

version = config.version
x_train, y_train, sz_train,time_train,rnn_index_train, vocabulary_size = pickle.load(open(timecas_config.train_pkl,'rb'))
x_test, y_test, sz_test,time_test,rnn_index_test, _ = pickle.load(open(timecas_config.test_pkl,'rb'))
x_val, y_val, sz_val, time_val,rnn_index_val,_ = pickle.load(open(timecas_config.val_pkl,'rb'))
user_table = pickle.load(open(timecas_config.user_table_pkl,'rb'))
idmap = pickle.load(open(timecas_config.idmap_pkl,'rb'))


training_iters = config.training_iters
batch_size = config.batch_size
display_step = min(config.display_step, len(sz_train)/batch_size)

#determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
start = time.time()
model = SDPP(config, sess, n_nodes)
sess.graph.finalize()
step = 0
best_val_loss = 1000
best_test_loss = 1000


train_writer = tf.summary.FileWriter("./train", sess.graph)

# Keep training until reach max iterations or max_try
train_loss = []
max_try = 50
patience = max_try
while step * batch_size < training_iters:
    batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index = get_batch(x_train, y_train, sz_train, time_train,rnn_index_train,config.n_time_interval,step, batch_size=batch_size)
    lambda1 = model.train_batch(batch_x,batch_x_indict, batch_y,batch_time_interval_index,batch_rnn_index)
    #train_writer.add_summary(summary, step)
    train_loss.append(model.get_error(batch_x,batch_x_indict, batch_y,batch_time_interval_index,batch_rnn_index))
    if step % display_step == 0:
        # Calculate batch loss
    # print lambda1
        print(lambda1)
        val_loss = []
        for val_step in range(len(y_val)//batch_size):
            val_x, val_x_indict, val_y, val_time_interval_index,val_rnn_index = get_batch(x_val, y_val, sz_val, time_val,rnn_index_val,config.n_time_interval,val_step, batch_size=batch_size)
            val_loss.append(model.get_error(val_x, val_x_indict, val_y, val_time_interval_index,val_rnn_index))
        test_loss = []
        for test_step in range(len(y_test)//batch_size):
            test_x,test_x_indict, test_y, test_time_interval_index,test_rnn_index = get_batch(x_test, y_test, sz_test, time_test,rnn_index_test,config.n_time_interval,test_step, batch_size=batch_size)
            test_loss.append(model.get_error(test_x,test_x_indict, test_y, test_time_interval_index,test_rnn_index))
        
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try

        predict_result = []
        test_loss = []
        for test_step in range(len(y_test) // batch_size+1):
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_test, y_test, sz_test,time_test, rnn_index_test,config.n_time_interval,test_step,batch_size=batch_size)
            predict_result.extend(model.predict(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
            test_loss.append(model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
        print("last test error:",np.mean(test_loss))
        pickle.dump((predict_result, y_test,test_loss), open("prediction_result_total_l"+str(learning_rate)+"_embl"+str(emb_learning_rate)+"_l2"+str(l2)+"_dropout"+str(dropout)+".pkl",'wb'))

        print("#" + str(step//display_step) + 
              ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + 
              ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + 
              ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) + 
              ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) + 
              ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
             )
        train_loss = []
        patience -= 1
        if not patience:
            break
        #if best_val_loss <2.35:
    #    break
    step += 1

print(len(predict_result),len(y_test))
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time()-start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
print(lambda1)
