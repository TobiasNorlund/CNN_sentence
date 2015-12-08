"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
from random import shuffle
import theano
import theano.tensor as T
import re
import warnings
import sys
import conv_net_classes
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets, # ( train list (doc,y) , validation list (doc,y) )
                   embedding, # Embedding object
                   longest_doc, # Max words allowed in doc
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = longest_doc + (max(filter_hs) - 1)*2
    img_w = embedding.d
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters
    
    #define model architecture
    #index = T.lscalar()
    #x = T.matrix('x')
    y = T.ivector('y')
    #Words = theano.shared(value = U, name = "Words")
    #zero_vec_tensor = T.vector()
    #zero_vec = np.zeros(img_w)
    #set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])

    layer0_input = embedding.get_embeddings_expr().reshape((batch_size,1,img_h,embedding.d))

    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = conv_net_classes.LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = conv_net_classes.MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    params += embedding.get_update_parameter_vars()
    #if non_static:
    #    #if word vectors are allowed to change, add them as model parameters
    #    params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)

    if len(datasets[0]) % batch_size > 0:
        extra_data_num = batch_size - len(datasets[0]) % batch_size
        shuffle(datasets[0])
        extra_data = datasets[0][:extra_data_num]
        datasets[0]=datasets[0] + extra_data

    if len(datasets[1]) % batch_size > 0:
        extra_data_num = batch_size - len(datasets[1]) % batch_size
        shuffle(datasets[0])
        extra_data = datasets[1][:extra_data_num]
        datasets[0]=datasets[0] + extra_data

    shuffle(datasets[0])
    n_batches = (len(datasets[0]) + len(datasets[1]))/batch_size
    n_train_batches = len(datasets[0])

    #divide train set into train/val sets 
    #test_set = datasets[1]
    #test_set_y = np.asarray(datasets[1][:,-1],"int32")
    #train_set = new_data[:n_train_batches*batch_size]
    #val_set = new_data[n_train_batches*batch_size:]
    #train_set_x, train_set_y = shared_dataset((train_set[:,:-1],train_set[:,-1]))
    #val_set_x, val_set_y = shared_dataset((val_set[:,:-1],val_set[:,-1]))
    n_val_batches = len(datasets[1])

    val_model = theano.function([y] + embedding.get_variable_vars(), classifier.errors(y) )
#         givens={
#            x: val_set_x[index * batch_size: (index + 1) * batch_size],     # batch_size x 64 (word indices)
#            y: val_set_y[index * batch_size: (index + 1) * batch_size]})
            
    #compile theano functions to get train/val/test errors
#    test_model = theano.function([y] + embedding.get_variable_vars(), classifier.errors(y) )
#             givens={
#                x: train_set_x[index * batch_size: (index + 1) * batch_size],
#                y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    train_model = theano.function([y] + embedding.get_variable_vars(), [cost, classifier.errors(y)], updates=grad_updates )
#          givens={
#            x: train_set_x[index*batch_size:(index+1)*batch_size],
#            y: train_set_y[index*batch_size:(index+1)*batch_size]})

#    test_pred_layers = []
#    test_size = len(test_set)
#    test_layer0_input = embedding.get_embeddings_expr().reshape((test_size,1,img_h,embedding.d))
#    for conv_layer in conv_layers:
#        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
#        test_pred_layers.append(test_layer0_output.flatten(2))
#    test_layer1_input = T.concatenate(test_pred_layers, 1)
#    test_y_pred = classifier.predict(test_layer1_input)
#    test_error = T.mean(T.neq(test_y_pred, y))
#    test_model_all = theano.function([y]+embedding.get_variable_vars(), test_error)
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0

    # helper function for building batch word list
    def get_padded_words(batch, max_l, filter_h):
        words = []
        pad = filter_h -1
        for (doc, y) in batch:
            doc_words = []
            for i in xrange(pad):
                doc_words.append("##zero##")
            for word in doc.split(" "):
                if embedding.has(word):
                    doc_words.append(word)
            while len(doc_words) < max_l+2*pad:
                doc_words.append("##zero##")
            words += doc_words
        return words

    # helper function for executing a model
    def exec_model(minibatch_index, model, dataset):
        # Go fetch all words in all batch documents and pad to img_h
        batch = dataset[minibatch_index*batch_size : (minibatch_index+1)*batch_size]
        words = get_padded_words(batch, longest_doc, max(filter_hs))
        y_vals = [lbl for doc,lbl in batch]

        return model(y_vals, *embedding.get_variables(words))

    while (epoch < n_epochs):        
        epoch = epoch + 1

        # The new_data set is implicitly divided into batches of size batch_size.
        # - the first n_train_batches are considered train set
        # - the remaining batches are considered validation set

        # Backpropagate training set errors

        batch_iter = range(n_train_batches)
        if shuffle_batch: shuffle(batch_iter)
        train_losses = [exec_model(minibatch_index, train_model, datasets[0])[1] for minibatch_index in batch_iter]
        train_perf = 1 - np.mean(train_losses)
        #for minibatch_index in batch_iter:

        #    # Go fetch all words in all batch documents and pad to img_h
        #    batch = train_set[minibatch_index*batch_size : (minibatch_index+1)*batch_size]
        #    words = get_padded_words(batch, embedding.has, img_h, filter_h)
        #    y_vals = [lbl for doc,lbl in batch]

        #    cost_epoch = train_model(y_vals, *embedding.get_variables(words))
        #    #set_zero(zero_vec)

        # Forward propagate to get training set accuracy
        #train_losses = [exec_model(i,test_model) for i in xrange(n_train_batches)]
        #train_perf = 1 - np.mean(train_losses)

        # Forward propagate to get validation set accuracy
        val_losses = [exec_model(i, val_model, datasets[1]) for i in range(n_train_batches,n_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.))

        # If we have a new peak validation performance, also forward propagate the test set and save its accuracy
        #if val_perf >= best_val_perf:
        #    best_val_perf = val_perf
#
#            words = get_padded_words(test_set, longest_doc, max(filter_hs))
#            y_vals = [lbl for doc,lbl in test_set]
#
#            test_loss = test_model_all(y_vals, *embedding.get_variables(words))
#            test_perf = 1- test_loss
    return val_perf

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     
  
   
if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,10)    
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=56,k=300, filter_h=5)
        perf = train_conv_net(datasets,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[100,2], 
                              shuffle_batch=True, 
                              n_epochs=25, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)  
    print str(np.mean(results))
