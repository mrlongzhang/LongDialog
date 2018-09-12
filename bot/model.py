# Copyright 2018 Long Zhang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Change Log:
#   Added attention mechanism
#   Changed BasicLSTMCell into GRUCell
#     
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf

from bot.textdata import Batch
from bot.textdata import TextData
from tensorflow.python.util import nest


class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class DynamicRNNS2SModel:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        4 GRU layers
    """

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs  = None
        self.encoderInputsLength = None
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    # Creation of the rnn cell
    def create_rnn_cell(self):
        def create_single_rnn_cell():
            encoDecoCell = tf.contrib.rnn.GRUCell(#BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                    self.args.hiddenSize,
                )
                
            #if not self.args.test:  # TODO: Should use a placeholder instead
            encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                                encoDecoCell,
                                #input_keep_prob=1.0,
                                output_keep_prob=self.keep_prob_placeholder)
                    
            return encoDecoCell
            
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
                    [create_single_rnn_cell() for _ in range(self.args.numLayers)],
                    )
        return encoDecoCell
        
    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt     = tf.cast(outputProjection.W_t,             tf.float32)
                localB      = tf.cast(outputProjection.b,               tf.float32)
                localInputs = tf.cast(inputs,                           tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                        self.textData.getVocabularySize()),  # The number of classes
                    self.dtype)


        # Network input (placeholders)

        #with tf.name_scope('placeholder_encoder'):
        self.encoderInputs  = tf.placeholder(tf.int32,   [None, None], name='encoderInputs')   # Batch size * sequence length * input dim
        self.encoderInputsLength=tf.placeholder(tf.int32, [None], name='encoderInputsLength')

        #with tf.name_scope('placeholder_decoder'):
        #self.decoderInputs  = tf.placeholder(tf.int32,   [None, None], name='decoderInputs')   # Same sentence length for input and output (Right ?)
        self.decoderTargets = tf.placeholder(tf.int32,   [None, None], name='decoderTargets')
        #self.decoderWeights = tf.placeholder(tf.float32, [None, None], name='weights')
        self.decoderTargetsLength = tf.placeholder(tf.int32, [None], name='decoderTargetsLength')
        
        self.batch_size = tf.placeholder(tf.int32, [], name='batchSize')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
        
        # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。用一个sequence_mask的例子来说明起作用
        #  tf.sequence_mask([1, 3, 2], 5)
        #  [[True, False, False, False, False],
        #  [True, True, True, False, False],
        #  [True, True, False, False, False]]
        self.maxTargetSequenceLength = tf.reduce_max(self.decoderTargetsLength, name='maxTargetLength')
        self.mask = tf.sequence_mask(self.decoderTargetsLength, self.maxTargetSequenceLength, dtype=tf.float32, name='masks')
        
        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        #=================================2, 定义模型的encoder部分
        with tf.variable_scope('encoder'):
            #创建LSTMCell，两层+dropout
            encoder_cell = self.create_rnn_cell()
            #构建embedding矩阵,encoder和decoder公用该词向量矩阵
            embedding = tf.get_variable('embedding', [self.args.vocabularySize, self.args.embeddingSize])
            encoderInputsEmbedded = tf.nn.embedding_lookup(embedding, self.encoderInputs)
            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
            # encoder_outputs用于attention，batch_size*encoder_inputs_length*rnn_size,
            # encoder_state用于decoder的初始化状态，batch_size*rnn_szie
            encoderOutputs, encoderState = tf.nn.dynamic_rnn(encoder_cell, encoderInputsEmbedded,
                                                               sequence_length=self.encoderInputsLength,
                                                               dtype=tf.float32)
                                                               
                                                               
        # =================================3, 定义模型的decoder部分
        with tf.variable_scope('decoder'):
            encoderInputsLength = self.encoderInputsLength
            if self.args.beamSearch:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoderOutputs = tf.contrib.seq2seq.tile_batch(encoderOutputs, multiplier=self.args.beamSize)
                encoderState = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.args.beamSize), encoderState)
                encoderInputsLength = tf.contrib.seq2seq.tile_batch(self.encoderInputsLength, multiplier=self.args.beamSize)

            #定义要使用的attention机制。
            attentionMechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.args.hiddenSize, memory=encoderOutputs,
                                                                     memory_sequence_length=encoderInputsLength)
            #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self.create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attentionMechanism,
                                                               attention_layer_size=self.args.hiddenSize, name='Attention_Wrapper')
            #如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            batchSize = self.batch_size if not self.args.beamSearch else self.batch_size * self.args.beamSize
            #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=batchSize, dtype=tf.float32).clone(cell_state=encoderState)
            output_layer = tf.layers.Dense(self.args.vocabularySize, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
            # training and reduce memory usage. Other solution, use sampling softmax
    
            # For testing only
            if self.args.test:
                startToken = tf.ones([self.batch_size,], tf.int32)*self.textData.goToken
                endToken = self.textData.eosToken
                
                #decoder:
                # if beamSearch, use BeamSearchDecoder
                # else use GreedyEmbeddingHelper + BasicDecoder (basic is greedy)
                if self.args.beamSearch:
                    infDecoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                      embedding = embedding,
                                                                      start_tokens = startToken,
                                                                      end_token = endToken,
                                                                      beam_width = self.args.beamSize,
                                                                      output_layer = output_layer)
                else:
                    decHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                         start_tokens=startToken,
                                                                         end_token = endToken,
                                                                         )
                    infDecoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decHelper,
                                                                 initial_state=decoder_initial_state,
                                                                 output_layer=output_layer)
                    
                decoderOutputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=infDecoder,
                                                                       maximum_iterations=10)
                
                if self.args.beamSearch:
                    self.outputs = decoderOutputs.predicted_ids
                else:
                    self.outputs = tf.expand_dims(decoderOutputs.sample_id, -1)
    
                # TODO: Attach a summary to visualize the output
    
            # For training only
            else:
                # more decoder staffs, 
                # decoder input: add <go> at taget, and del <end>, then embedding
                # decoder input embedded shape [batchSize, decoderTagetsLength, embeddingSize]
                ending = tf.strided_slice(self.decoderTargets,[0,0],[self.batch_size, -1], [1,1])
                decoderInput = tf.concat([tf.fill([self.batch_size,1],self.textData.goToken),ending], 1)
                decoderInputsEmbedded = tf.nn.embedding_lookup(embedding, decoderInput)
                
                #training decoder
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoderInputsEmbedded,
                                                            sequence_length=self.decoderTargetsLength,
                                                            time_major=False,name='training_helper')
                training_decoder =tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                          initial_state=decoder_initial_state,
                                                          output_layer = output_layer)
                
                #decoding output is namedtuple [rnn_outputs, sample_id]
                #rnn_output [batchSize, decoderTargetsLength, vocabularySize] to call loss
                # sample_id: [batchSize], tf.int32, the answer vector
                decoderOutputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations = self.maxTargetSequenceLength)
                
                # Finally, we define the loss function
                self.lossFct = tf.contrib.seq2seq.sequence_loss(
                    logits = tf.identity(decoderOutputs.rnn_output),
                    targets = self.decoderTargets,
                    weights = self.mask,
                    average_across_timesteps=True,
                    average_across_batch=True,
                    #self.textData.getVocabularySize(),
                    softmax_loss_function= None  # If None, use default SoftMax
                )
                
                #cal loss and grads
                tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost
                self.summaryOp = tf.summary.merge_all()
    
                # Initialize the optimizer
                opt = tf.train.AdamOptimizer(
                    learning_rate=self.args.learningRate,
                    #beta1=0.9,
                    #beta2=0.999,
                    #epsilon=1e-08
                )
                trainableParams = tf.trainable_variables()
                gradients = tf.gradients(self.lossFct, trainableParams)
                clipGradients,_ = tf.clip_by_global_norm(gradients, self.args.maxGradientNorm)
                
                self.optOp = opt.apply_gradients(zip(clipGradients, trainableParams)) #opt.minimize(self.lossFct)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            feedDict[self.encoderInputs]  = batch.encoderSeqs
            feedDict[self.encoderInputsLength] = batch.encoderSeqsLength
            feedDict[self.decoderTargets] = batch.targetSeqs
            feedDict[self.decoderTargetsLength] = batch.targetSeqsLength
            feedDict[self.keep_prob_placeholder] = 0.5
            feedDict[self.batch_size] = len(batch.encoderSeqs)

            ops = [self.optOp, self.lossFct,self.summaryOp]
        else:  # Testing (batchSize == 1)
            feedDict[self.encoderInputs]  = batch.encoderSeqs
            feedDict[self.encoderInputsLength] = batch.encoderSeqsLength
            feedDict[self.keep_prob_placeholder] = 1
            feedDict[self.batch_size] = len(batch.encoderSeqs)

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
