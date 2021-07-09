
# Importing the libraries

import numpy as np
import tensorflow
import re
import time

from tensorflow.python.ops.nn_ops import dropout

tf = tensorflow.compat.v1
tf.disable_v2_behavior()


### PART I: DATA PREPROCESSING ###

# Importing the dataset
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Creating a dictionary that maps each line to its id
id_to_line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id_to_line[_line[0]] = _line[-1]

# Creating a list of all conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","").split(',')
    conversations_ids.append(_conversation)

# Separating the questions and answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i+1]])

# Defining a function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"\'#/@;:<>{}+-=|.?,_]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its number of occurances
word_to_count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

# Creating two dictionaries that map the questions' and answers' words to a unique integer respectively while filtering out infrequent words
threshold = 20 #Minimum frequency for a unique integer (Can be adjusted for optimization)
questions_words_to_int = {}
word_numer = 0
for word, count in word_to_count.items():
    if count >= threshold:
        questions_words_to_int[word] = word_numer
        word_numer += 1
answers_words_to_int = {}
word_numer = 0
for word, count in word_to_count.items():
    if count >= threshold:
        answers_words_to_int[word] = word_numer
        word_numer += 1

# Adding tokens to the two dictionaries
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questions_words_to_int[token] = len(questions_words_to_int) + 1
    answers_words_to_int[token] = len(answers_words_to_int) + 1

# Creating the inverse dictionary of the answers_words_to_int dictionary
answers_into_to_word = {w_i:w for w, w_i in answers_words_to_int.items()}

# Adding End of String (EOS) token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all questions and answers into integers and replacing all the filtered words by <OUT>
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_to_int:
            ints.append(questions_words_to_int['<OUT>'])
        else:
            ints.append(questions_words_to_int[word])
    questions_to_int.append(ints)
answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_to_int:
            ints.append(answers_words_to_int['<OUT>'])
        else:
            ints.append(answers_words_to_int[word])
    answers_to_int.append(ints)

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


### PART II: BUILDING THE SEQ2SEQ MODEL ###

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word_to_int, batch_size):
    left_side = tf.fill([batch_size,1],word_to_int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets

# Creating the Encoder RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers,keep_prob,sequence_length):
    lstm = tf.contrib.BasicLSTMCell(rnn_size)
    lstm_dropout= tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.contrib.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,cell_bw = encoder_cell, sequence_length = sequence_length,inputs = rnn_inputs,dtype = tf.float32)
    return encoder_state 

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_function, scope = decoding_scope)
    return test_predictions

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word_to_int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout= tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiCell([lstm_dropout]*num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, num_words, None, scope = decoding_scope,weights_initializers = weights, biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,decoder_cell,decoder_embeddings_matrix,word_to_int['<SOS>'],word_to_int['<EOS>'],sequence_length-1,num_words,decoding_scope,output_function,keep_prob,batch_size)
    return training_predictions, test_predictions

# Building the Seq2Seq Model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questions_words_to_int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,answers_num_words+1,encoder_embedding_size,initializer= tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers,keep_prob,sequence_length)
    preprocessed_targets = preprocess_targets(targets,questions_words_to_int,batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    training_predictions,test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, questions_num_words, sequence_length, rnn_size, num_layers, questions_words_to_int, keep_prob, batch_size)
    return training_predictions, test_predictions