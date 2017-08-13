from __future__ import division
import math
import helper2 as helper
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from collections import Counter
import codecs
import os

np.random.seed(0)

class BILSTM_CLF(object):
    
    def __init__(self, num_chars, num_classes, num_steps=100, num_epochs=100, model_path='models/', \
        embedding_matrix=None, emb_dim=100,emb_trainable=False, is_training=True, weight=False, l2_reg_lambda=0.2):
        # Parameter
        self.max_f1 = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 128
        self.num_layers = 1   
        self.hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes
        self.model_path = model_path
        self.char2id, self.id2char = helper.loadMap(os.path.join(model_path, "char2id"))
        self.label2id, self.id2label = helper.loadMap(os.path.join(model_path, "label2id"))
        self.evaluate_labels = set() 
        self.emb_dim = emb_dim
        
        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.float32, [None, self.num_classes])
        
        # char embedding
        if embedding_matrix != None:
            self.embedding = tf.Variable(embedding_matrix, trainable=emb_trainable, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
    
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)   # shape: [batch_size, num_steps, emb_dim]
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])      # shape: [num_steps, batch_size, emb_dim]
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim]) # shape: [(num_steps * batch_size), emb_dim]
        self.inputs_emb = tf.split(0, self.num_steps, self.inputs_emb)  # num_steps tensor,[batch_size, emb_dim]

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample, shape [batch_size]
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)  
        
        # forward and backward
        # outputs: total num_steps tensors, each tensor's shape: [batch_size, hidden_dim * 2]
        self.outputs, _, _ = rnn.bidirectional_rnn(
            lstm_cell_fw, 
            lstm_cell_bw,
            self.inputs_emb, 
            dtype=tf.float32,
            sequence_length=self.length
        )

        # softmax
        self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, self.hidden_dim * 2]) #shape: [batch_size*num_steps, hidden_dim*2]
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])

        self.outputs_emb = tf.reshape(self.outputs, [-1, num_steps, self.hidden_dim*2])#[batch_size, num_steps, hidden_dim*2]
        self.attentions = tf.ones([self.batch_size, 1, num_steps])     #[batch_size, 1, num_steps]
        self.outputs_emb = tf.batch_matmul(self.attentions, self.outputs_emb)    #[batch_size, 1, hidden_dim*2]
        self.outputs_emb = tf.tanh(self.outputs_emb)
        self.outputs = tf.reshape(self.outputs_emb, [-1, self.hidden_dim*2] )    #[batch_size, hidden_dim*2]

        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b   #[batch_size, num_classes] 
        self.predictions = tf.argmax(self.logits, 1, name="predictions")    #[batch_size, 1]
        
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(self.softmax_w)
        l2_loss += tf.nn.l2_loss(self.softmax_b)
        losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
     
        # summary
        self.train_summary = tf.scalar_summary("loss", self.loss)
        self.val_summary = tf.scalar_summary("loss", self.loss)        
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) 

    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()
        summary_writer_train = tf.train.SummaryWriter(os.path.join(self.model_path, 'loss_log/train_loss'), sess.graph)  
        summary_writer_val = tf.train.SummaryWriter(os.path.join(self.model_path, 'loss_log/val_loss'), sess.graph)     
        
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            print "current epoch: %d" % (epoch)
            for iteration in range(num_iterations):
                # train
                X_train_batch, y_train_batch = helper.nextBatch(X_train, y_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                _, loss_train, predicts_train, accuracy, length, train_summary =\
                    sess.run([
                        self.optimizer, 
                        self.loss, 
                        self.predictions,
                        self.accuracy,
                        self.length,
                        self.train_summary
                    ], 
                    feed_dict={
                        self.inputs:X_train_batch, 
                        self.targets:y_train_batch
                    })

                if iteration % 10 == 0:
                    cnt += 1
                    summary_writer_train.add_summary(train_summary, cnt)
                    print "iteration: %5d, train loss: %5d, train accuracy: %.5f" % (iteration, loss_train, accuracy)  
                    print '---------------------------------------------------------------'
                
                # validation
                if iteration % 100 == 0:
                    X_val_batch, y_val_batch = helper.nextRandomBatch(X_val, y_val, batch_size=self.batch_size)
                    loss_val, predicts_val, accuracy, length, val_summary =\
                        sess.run([
                            self.loss, 
                            self.predictions,
                            self.accuracy,
                            self.length,
                            self.val_summary
                        ], 
                        feed_dict={
                            self.inputs:X_val_batch,
                            self.targets:y_val_batch
                        })
                    
                    summary_writer_val.add_summary(val_summary, cnt)

                    print "iteration: %5d, valid loss: %5d, valid accuracy: %.5f" % (iteration, loss_val, accuracy)
                    print '-----------------------------------------------------------------------'
                    
                    f1_val = accuracy
                    if f1_val > self.max_f1:
                        self.max_f1 = f1_val
                        save_path = saver.save(sess, save_file)
                        print "saved the best model with accuracy: %.5f" % (self.max_f1)

    def test(self, sess, X_test, X_test_str, y_test, output_path):
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print "number of iteration: " + str(num_iterations)
        with codecs.open(output_path, "w", 'utf8') as outfile:
            for i in range(num_iterations):
                X_test_batch = X_test[i * self.batch_size : (i + 1) * self.batch_size]
                y_test_batch = y_test[i * self.batch_size : (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    y_test_batch = list(y_test_batch)
                    last_size = len(X_test_batch)
                    for i in range(self.batch_size - last_size):
                        X_test_batch.append([0 for j in range(self.num_steps)])
                        y_test_batch.append([0 for j in range(self.num_classes)])
                    X_test_batch = np.array(X_test_batch)
                    y_test_batch = np.array(y_test_batch)
                    y_pred,accuracy,length = self.predictBatch(sess, X_test_batch, y_test_batch)
                    y_pred = y_pred[:last_size]
                else:
                    X_test_batch = np.array(X_test_batch)
                    y_pred,accuracy,length = self.predictBatch(sess, X_test_batch, y_test_batch)
                
                print "iteration: %5d, test accuracy: %.5f" % (i+1, accuracy)
                print '---------------------------------------------------------'

                for i in range(len(y_pred)):
                    label = y_pred[i]
                    outfile.write('%s\n'%(label))
        return y_pred

    def predictBatch(self, sess, X, X_str, y):
        length,predicts,accuracy = sess.run([self.length, self.predictions, self.accuracy], \
                feed_dict={self.inputs:X, self.targets:y})
        return predicts, accuracy, length

    


