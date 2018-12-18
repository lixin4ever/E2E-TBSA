import dynet_config
dynet_config.set(mem='4096', random_seed=1314159)
import dynet as dy
import random
from utils import *
from evals import *
from nltk import word_tokenize


def norm_vec(vec):
    """
    normalize a dynet vector expression
    :param vec:
    :return:
    """
    sum_item = dy.sum_elems(vec)
    norm_vec = vec / sum_item.value()
    print(norm_vec.npvalue())
    return norm_vec


def calculate_confidence(vec, proportions=0.5):
    """
    calculate the value of alpha, the employed metric is GINI index
    :param vec:
    :return:
    """
    square_sum = dy.sum_elems(dy.cmult(vec, vec)).value()
    if not 0 <= square_sum <= 1:
        raise Exception("Invalid square sum %.3lf" % square_sum)
    return (1 - square_sum) * proportions


class WDEmb:
    def __init__(self, pc, n_words, dim_w, pretrained_embeddings=None):
        """
        constructor of Word Embedding Layer
        :param pc: parameter collection to hold the parameters
        :param n_words: number of words in the vocabulary
        :param dim_w: dimension of word embeddings
        :param pretrained_embeddings:
        """
        self.pc = pc.add_subcollection()
        self.n_words = n_words
        self.dim_w = dim_w
        # add word embedding as lookup parameters
        self.W = self.pc.add_lookup_parameters((self.n_words, self.dim_w))
        if pretrained_embeddings is not None:
            print("Use pre-trained word embeddings...")
            self.W.init_from_array(pretrained_embeddings)

    def parametrize(self):
        """
        note: lookup parameters do not need parametrization
        :return:
        """
        pass

    def __call__(self, xs):
        """
        map input words (or ngrams) into the corresponding word embeddings
        :param xs: a list of ngrams (or words if win is set to 1)
        :return: embeddings looked from tables
        """
        embeddings = [dy.concatenate([self.W[w] for w in ngram]) for ngram in xs]
        return embeddings


class CharEmb:
    # build character embedding layers from random initialization
    def __init__(self, pc, n_chars, dim_char, pretrained_embeddings=None):
        """

        :param pc: parameter collection
        :param n_chars: number of distinct characters
        :param dim_char: dimension of character embedding
        """
        self.pc = pc.add_subcollection()
        self.n_chars = n_chars
        self.dim_char = dim_char
        # network parameters
        #self.W = self.pc.add_lookup_parameters((self.n_chars, self.dim_char),
        #                                       init='uniform', scale=np.sqrt(3.0 / self.dim_char))
        self.W = self.pc.add_lookup_parameters((self.n_chars, self.dim_char),
                                               init=dy.UniformInitializer(np.sqrt(3.0 / self.dim_char)))
        if pretrained_embeddings is not None:
            print("Use pre-trained character embeddings...")
            self.W.init_from_array(pretrained_embeddings)

    def __call__(self, xs):
        """
        map input characters to low-dimensional character embeddings
        :param xs: input chars
        :return:
        """
        char_embs = [self.W[cid] for cid in xs]
        return char_embs


class Linear:
    # fully connected layer without non-linear activation
    def __init__(self, pc, n_in, n_out, use_bias=False, nonlinear=None):
        """

        :param pc: parameter collection to hold the parameters
        :param n_in: input dimension
        :param n_out: output dimension
        :param use_bias: if add bias or not, default NOT
        :param nonlinear: non-linear activation function
        """
        # create a sub-collection of the current parameters collection and returns it
        # the returned sub-collection is simply a ParameterCollection object tied to a parent collection
        self.pc = pc.add_subcollection()
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self.nonlinear = nonlinear
        # add a parameter to the ParameterCollection with a given initializer
        self._W = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        if self.use_bias:
            self._b = self.pc.add_parameters((self.n_out,), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        """
        put the parameters into the computational graph
        :return:
        """
        # add parameter to the computation graph (cg)
        self.W = dy.parameter(self._W)
        if self.use_bias:
            self.b = dy.parameter(self._b)

    def __call__(self, x):
        """

        :param x: input feature vector
        :return:
        """
        Wx = self._W * x
        if self.use_bias:
            Wx = Wx + self._b
        if self.nonlinear == 'sigmoid':
            return dy.logistic(Wx)
        elif self.nonlinear == 'tanh':
            return dy.tanh(Wx)
        elif not self.nonlinear:
            return Wx
        else:
            raise Exception("Unimplemented non-linear activation function %s" % self.nonlinear)


class Model:
    # cascaded LSTMs for joint aspect detection and sentiment prediction
    def __init__(self, params, vocab, embeddings, char_embeddings):
        """

        :param params:
        :param vocab:
        :param embeddings:
        :param char_embeddings:
        """
        self.params = params
        self.name = 'lstm_cascade'
        self.dim_char = params.dim_char
        self.dim_w = params.dim_w
        self.dim_char_h = params.dim_char_h
        self.dim_ote_h = params.dim_ote_h
        self.dim_ts_h = params.dim_ts_h
        self.input_win = params.input_win
        self.ds_name = params.ds_name
        # tag vocabulary of opinion target extraction and targeted sentiment
        self.ote_tag_vocab = params.ote_tag_vocab
        self.ts_tag_vocab = params.ts_tag_vocab
        self.dim_ote_y = len(self.ote_tag_vocab)
        self.dim_ts_y = len(self.ts_tag_vocab)
        self.n_epoch = params.n_epoch
        self.dropout_rate = params.dropout
        self.tagging_schema = params.tagging_schema
        self.clip_grad = params.clip_grad
        self.use_char = params.use_char
        # name of word embeddings
        self.emb_name = params.emb_name
        self.embeddings = embeddings
        self.vocab = vocab
        # character vocabulary
        self.char_vocab = params.char_vocab
        #self.td_proportions = params.td_proportions
        self.epsilon = params.epsilon
        #self.tc_proportions = params.tc_proportions
        self.pc = dy.ParameterCollection()

        if self.use_char:
            self.char_emb = CharEmb(pc=self.pc,
                                    n_chars=len(self.char_vocab),
                                    dim_char=self.dim_char,
                                    pretrained_embeddings=char_embeddings)
            self.lstm_char = dy.LSTMBuilder(1, self.dim_char, self.dim_char_h, self.pc)
            dim_input = self.input_win * self.dim_w + 2 * self.dim_char_h
        else:
            dim_input = self.input_win * self.dim_w
        # word embedding layer
        self.emb = WDEmb(pc=self.pc, n_words=len(vocab), dim_w=self.dim_w, pretrained_embeddings=embeddings)

        # lstm layers
        self.lstm_ote = dy.LSTMBuilder(1, dim_input, self.dim_ote_h, self.pc)
        self.lstm_ts = dy.LSTMBuilder(1, 2*self.dim_ote_h, self.dim_ts_h, self.pc)

        # fully connected layer
        self.fc_ote = Linear(pc=self.pc, n_in=2*self.dim_ote_h, n_out=self.dim_ote_y)
        self.fc_ts = Linear(pc=self.pc, n_in=2 * self.dim_ts_h, n_out=self.dim_ts_y)

        assert self.tagging_schema == 'BIEOS'
        transition_path = {'B': ['B-POS', 'B-NEG', 'B-NEU'],
                           'I': ['I-POS', 'I-NEG', 'I-NEU'],
                           'E': ['E-POS', 'E-NEG', 'E-NEU'],
                           'S': ['S-POS', 'S-NEG', 'S-NEU'],
                           'O': ['O']}
        self.transition_scores = np.zeros((self.dim_ote_y, self.dim_ts_y))
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = self.ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = self.ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)
        self.transition_scores = np.array(self.transition_scores, dtype='float32').transpose()

        # opinion target-opinion words co-occurrence modeling
        self.stm_lm = Linear(pc=self.pc, n_in=2*self.dim_ote_h, n_out=2*self.dim_ote_h, nonlinear='tanh')
        # fully connected layer for opinion-enhanced indicator prediction task
        self.fc_stm = Linear(pc=self.pc, n_in=2 * self.dim_ote_h, n_out=2)

        # gate for maintaining sentiment consistency
        self.W_gate = self.pc.add_parameters((2*self.dim_ote_h, 2*self.dim_ote_h),
                                             init=dy.UniformInitializer(0.2))

        # determine the optimizer
        if params.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.pc, params.sgd_lr)
        elif params.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.pc, 0.001, 0.9, 0.9)
        elif params.optimizer == 'adadelta':
            self.optimizer = dy.AdadeltaTrainer(self.pc)
        elif params.optimizer == 'momentum':
            self.optimizer = dy.MomentumSGDTrainer(self.pc, 0.01, 0.9)
        else:
            raise Exception("Unsupported optimizer type: %s" % params.optimizer)

    def forward(self, x, is_train=True):
        """
        feed the input x into the network
        :param x: input example
        :param is_train: model is in training stage or not, default yes
        :return: loss value, predicted ote labels, predicted ts labels
        """

        # renew computational graph
        dy.renew_cg()
        # push the parameters to the cg, no need to do this after v2.0.3
        # self.parametrize()

        wids = x['wids']
        cids = x['cids']
        gold_ote_labels = x['ote_labels']
        gold_ts_labels = x['ts_labels']
        gold_stm_labels = x['stm_lm_labels']

        seq_len = len(wids)

        if self.use_char:
            # using both character-level word representations and word-level representations
            ch_word_emb = []
            for t in range(seq_len):
                ch_seq = cids[t]
                input_ch_emb = self.char_emb(xs=ch_seq)
                ch_h0_f = self.lstm_char.initial_state()
                ch_h0_b = self.lstm_char.initial_state()
                ch_f = ch_h0_f.transduce(input_ch_emb)[-1]
                ch_b = ch_h0_b.transduce(input_ch_emb[::-1])[-1]
                ch_word_emb.append(dy.concatenate([ch_f, ch_b]))
            word_emb = self.emb(xs=wids)
            input_emb = [dy.concatenate([c, w]) for (c, w) in zip(ch_word_emb, word_emb)]
        else:
            # only using word-level representations
            input_emb = self.emb(xs=wids)

        # equivalent to applying partial dropout on the LSTM
        if is_train:
            input_emb = [dy.dropout(x, self.dropout_rate) for x in input_emb]

        # obtain initial rnn states
        ote_h0_f = self.lstm_ote.initial_state()
        ote_h0_b = self.lstm_ote.initial_state()

        ote_hs_f = ote_h0_f.transduce(input_emb)
        ote_hs_b = ote_h0_b.transduce(input_emb[::-1])[::-1]

        ote_hs = [dy.concatenate([f, b]) for (f, b) in zip(ote_hs_f, ote_hs_b)]

        # hidden states for opinion-enhanced target prediction, we refer it as stm_lm here
        stm_lm_hs = [self.stm_lm(h) for h in ote_hs]

        ts_h0_f = self.lstm_ts.initial_state()
        ts_h0_b = self.lstm_ts.initial_state()

        ts_hs_f = ts_h0_f.transduce(ote_hs)
        ts_hs_b = ts_h0_b.transduce(ote_hs[::-1])[::-1]

        ts_hs = [dy.concatenate([f, b]) for (f, b) in zip(ts_hs_f, ts_hs_b)]

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for t in range(seq_len):
            if t == 0:
                h_tilde_t = ts_hs[t]
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs[t]
                gt = dy.logistic(self.W_gate * ts_ht)
                h_tilde_t = dy.cmult(gt, ts_ht) + dy.cmult(1 - gt, h_tilde_tm1)
            ts_hs_tilde.append(h_tilde_t)
            h_tilde_tm1 = h_tilde_t

        if is_train:
            # perform dropout during training
            ote_hs = [dy.dropout(h, self.dropout_rate) for h in ote_hs]
            stm_lm_hs = [dy.dropout(h, self.dropout_rate) for h in stm_lm_hs]
            ts_hs_tilde = [dy.dropout(h, self.dropout_rate) for h in ts_hs_tilde]

        # weight matrix for boundary-guided transition
        self.W_trans_ote = dy.inputTensor(self.transition_scores.copy())

        losses = []
        pred_ote_labels, pred_ts_labels = [], []
        for i in range(seq_len):
            # probability distribution over ote tag
            p_y_x_ote = self.fc_ote(x=ote_hs[i])
            p_y_x_ote = dy.softmax(p_y_x_ote)
            loss_ote = -dy.log(dy.pick(p_y_x_ote, gold_ote_labels[i]))
            # probability distribution over ts tag
            p_y_x_ts = self.fc_ts(x=ts_hs_tilde[i])
            p_y_x_ts = dy.softmax(p_y_x_ts)
            # normalized the score
            alpha = calculate_confidence(vec=p_y_x_ote, proportions=self.epsilon)
            # transition score from ote tag to sentiment tag
            ote2ts = self.W_trans_ote * p_y_x_ote
            p_y_x_ts_tilde = alpha * ote2ts + (1 - alpha) * p_y_x_ts
            loss_ts = -dy.log(dy.pick(p_y_x_ts_tilde, gold_ts_labels[i]))
            loss_i = loss_ote / seq_len + loss_ts / seq_len

            # predict if the current word is a target word according to the opinion information
            p_y_x_stm = self.fc_stm(x=stm_lm_hs[i])
            loss_stm = dy.pickneglogsoftmax(p_y_x_stm, gold_stm_labels[i])
            loss_i += (loss_stm / seq_len)
            losses.append(loss_i)
            pred_ote_labels.append(np.argmax(p_y_x_ote.npvalue()))
            pred_ts_labels.append(np.argmax(p_y_x_ts_tilde.npvalue()))
        # total loss of the sequence predictions
        loss = dy.esum(losses)
        if is_train:
            # run the backward pass based on the expression
            loss.backward()
            # update the model parameters
            self.optimizer.update()
        return loss.value(), pred_ote_labels, pred_ts_labels

    def predict(self, dataset):
        """
        perform prediction
        :param dataset: dataset
        :return: ote scores, ts_scores, predicted ote labels, predicted ts labels
        """
        n_sample = len(dataset)
        gold_ote = [x['ote_tags'] for x in dataset]
        gold_ts = [x['ts_tags'] for x in dataset]

        if self.tagging_schema == 'BIO':
            gold_ote, gold_ts = bio2ot_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
        elif self.tagging_schema == 'OT':
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)

        pred_ote, pred_ts = [], []
        for i in range(n_sample):
            _, pred_ote_labels, pred_ts_labels = self.forward(x=dataset[i], is_train=False)
            pred_ote.append(label2tag(label_sequence=pred_ote_labels, tag_vocab=self.ote_tag_vocab))
            pred_ts.append(label2tag(label_sequence=pred_ts_labels, tag_vocab=self.ts_tag_vocab))
        # transform the output tag sequence to BIEOS tag sequence before evaluation
        if self.tagging_schema == 'BIO':
            pred_ote, pred_ts = bio2ot_batch(
                ote_tags=pred_ote, ts_tags=pred_ts)
            pred_ote, pred_ts = ot2bieos_batch(
                ote_tags=pred_ote, ts_tags=pred_ts)
        elif self.tagging_schema == 'OT':
            pred_ote, pred_ts = ot2bieos_batch(
                ote_tags=pred_ote, ts_tags=pred_ts)
        # evaluation
        ote_scores, ts_scores = evaluate(gold_ot=gold_ote, gold_ts=gold_ts,
                                         pred_ot=pred_ote, pred_ts=pred_ts)
        return ote_scores, ts_scores, pred_ote, pred_ts

    def decoding(self, dataset, model_name=None):
        """
        predict the tag sequence for the dataset
        :param dataset: dataset
        :param model_name: path of the model parameters
        :return:
        """
        model_path = './models/%s' % model_name
        if not os.path.exists(model_path):
            raise Exception("Invalid model path %s..." % model_path)
        self.pc.populate(model_path)
        n_sample = len(dataset)
        gold_ote = [x['ote_tags'] for x in dataset]
        gold_ts = [x['ts_tags'] for x in dataset]

        if self.tagging_schema == 'BIO':
            gold_ote, gold_ts = bio2ot_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
        elif self.tagging_schema == 'OT':
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
        # predicted tag sequences and the input words
        pred_ote, pred_ts, words = [], [], []
        for i in range(n_sample):
            _, pred_ote_labels, pred_ts_labels = self.forward(x=dataset[i], is_train=False)
            pred_ote.append(label2tag(label_sequence=pred_ote_labels, tag_vocab=self.ote_tag_vocab))
            pred_ts.append(label2tag(label_sequence=pred_ts_labels, tag_vocab=self.ts_tag_vocab))
            words.append(dataset[i]['words'])
        # transform the output tag sequence to BIEOS tag sequence before evaluation
        if self.tagging_schema == 'BIO':
            pred_ote, pred_ts = bio2ot_batch(
                    ote_tags=pred_ote, ts_tags=pred_ts)
            pred_ote, pred_ts = ot2bieos_batch(
                    ote_tags=pred_ote, ts_tags=pred_ts)
        elif self.tagging_schema == 'OT':
            pred_ote, pred_ts = ot2bieos_batch(
                    ote_tags=pred_ote, ts_tags=pred_ts)
        # evaluation
        ote_scores, ts_scores = evaluate(gold_ot=gold_ote, gold_ts=gold_ts,
                                         pred_ot=pred_ote, pred_ts=pred_ts)

        print("Evaluation scores: ote: f1: %.4f, ts: precision: %.4f, recall: %.4f, micro-f1: %.4f" %
              (ote_scores[2], ts_scores[1], ts_scores[2], ts_scores[3]))

        output_lines = ['Dataset: %s\n' % self.ds_name, 'Model: %s\n' % model_path, 'Parameter settings: \n']
        params_dict = vars(self.params)
        for k in params_dict:
            if k == 'char_vocab' or k == 'vocab':
                continue
            else:
                v = params_dict[k]
                output_lines.append('\t%s: %s\n' % (k, v))
        output_lines.append("==============================================\n\n")
        for i in range(n_sample):
            ote_seq = pred_ote[i]
            ts_seq = pred_ts[i]
            w_seq = words[i]
            assert len(ote_seq) == len(ts_seq) == len(w_seq)
            for j in range(len(ote_seq)):
                word = w_seq[j]
                ote_tag = ote_seq[j]
                ts_tag = ts_seq[j]
                output_lines.append('%s\t%s\t%s\n' % (word, ote_tag, ts_tag))
            # use empty lines as the separator
            output_lines.append('\n')


class LSTM_CRF:
    # LSTM CRF model for sequence tagging
    # NOT USED in the experiments
    def __init__(self, params, vocab, embeddings):
        """

        :param params: parameters
        :param vocab: vocabulary
        :param embeddings: pretrained word embeddings
        """
        self.params = params
        self.name = 'lstm_crf'
        self.dim_char = params.dim_char
        self.dim_w = params.dim_w
        self.dim_char_h = params.dim_char_h
        self.dim_ote_h = params.dim_ote_h
        self.dim_ts_h = params.dim_ts_h
        self.input_win = params.input_win
        self.ds_name = params.ds_name
        # tag vocabulary of opinion target extraction and targeted sentiment
        self.ote_tag_vocab = params.ote_tag_vocab
        self.ts_tag_vocab = params.ts_tag_vocab
        self.dim_ote_y = len(self.ote_tag_vocab)
        self.dim_ts_y = len(self.ts_tag_vocab)
        self.n_epoch = params.n_epoch
        self.dropout_rate = params.dropout
        self.tagging_schema = params.tagging_schema
        self.clip_grad = params.clip_grad
        self.use_char = params.use_char
        # name of word embeddings
        self.emb_name = params.emb_name
        self.embeddings = embeddings
        self.vocab = vocab
        # character vocabulary
        self.char_vocab = params.char_vocab
        self.pc = dy.ParameterCollection()

        # word embedding layer
        self.emb = WDEmb(pc=self.pc, n_words=len(vocab), dim_w=self.dim_w, pretrained_embeddings=embeddings)

        # input dimension
        dim_input = self.input_win * self.dim_w

        self.lstm_ts = dy.LSTMBuilder(1, dim_input, self.dim_ts_h, self.pc)

        # hidden layer between LSTM and CRF decoding layer
        self.hidden = Linear(pc=self.pc, n_in=2*self.dim_ts_h,
                                   n_out=self.dim_ts_h, use_bias=True, nonlinear='tanh')
        # map the word representation to the ts label space
        # in the label space, both BEG and END tag are considered
        self.fc_ts = Linear(pc=self.pc, n_in=self.dim_ts_h, n_out=self.dim_ts_y)

        # transition matrix, [i, j] is the transition score from tag i to tag j
        self.transitions = self.pc.add_lookup_parameters((self.dim_ts_y + 2, self.dim_ts_y + 2))

        # determine the optimizer
        if params.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.pc, params.sgd_lr)
        elif params.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.pc, 0.001, 0.9, 0.9)
        elif params.optimizer == 'adadelta':
            self.optimizer = dy.AdadeltaTrainer(self.pc)
        elif params.optimizer == 'momentum':
            self.optimizer = dy.MomentumSGDTrainer(self.pc, 0.01, 0.9)
        else:
            raise Exception("Unsupported optimizer type: %s" % params.optimizer)

    def log_sum_exp(self, scores):
        """

        :param scores: observation scores for all possible tag sequences
        :return: \log (\sum(exp(S(y))))
        """
        scores_val = scores.npvalue()
        max_idx = np.argmax(scores_val)
        # introduce max_scores to avoid underflow
        # if not, the results will be INF or -INF
        # dynet expression of maximum scores
        max_score = dy.pick(scores, max_idx)
        max_score_broadcast = dy.concatenate([max_score] * (self.dim_ts_y + 2))
        # shift the center of exponential sum to (scores - max)
        return max_score + dy.log(dy.sum_elems(dy.transpose(dy.exp(scores - max_score_broadcast))))

    def forward(self, x, is_train=True):
        # renew computational graph
        dy.renew_cg()
        # push the parameters to the cg, no need to do this after v 2.0.3
        # self.parametrize()

        wids = x['wids']
        gold_ts_labels = x['ts_labels']

        input_emb = self.emb(xs=wids)
        # add dropout on the embedding layer
        if is_train:
            input_emb = [dy.dropout(x, self.dropout_rate) for x in input_emb]
        ts_h0_f = self.lstm_ts.initial_state()
        ts_h0_b = self.lstm_ts.initial_state()
        # bi-directional lstm
        ts_hs_f = ts_h0_f.transduce(input_emb)
        ts_hs_b = ts_h0_b.transduce(input_emb[::-1])[::-1]

        ts_hs = [dy.concatenate([f, b]) for (f, b) in zip(ts_hs_f, ts_hs_b)]
        ts_cs = [self.hidden(x=h) for h in ts_hs]
        # tag scores output by the LSTM layer, shape: (n, dim_y)
        label_scores = [self.fc_ts(x=c) for c in ts_cs]
        min_val = -9999999
        observations = [dy.concatenate([score, dy.inputVector([min_val, min_val])]) for score in label_scores]
        assert len(observations) == len(gold_ts_labels)
        # score generated from the gold standard sequence
        gold_score = dy.scalarInput(0)
        # sum of the observation scores
        for t, score in enumerate(label_scores):
            gold_score = gold_score + dy.pick(score, gold_ts_labels[t])
        # <BEG> corresponds to dim_ts_y, <END> corresponds to dim_ts_y + 1
        padded_gold_ts_labels = [self.dim_ts_y] + gold_ts_labels
        # sum of the transition scores
        for t in range(len(observations)):
            # transition score A_{y_{t-1}, y_t}
            gold_score = gold_score + dy.pick(self.transitions[padded_gold_ts_labels[t]], padded_gold_ts_labels[t+1])
        # transition score from the last label to <END>
        gold_score = gold_score + dy.pick(self.transitions[padded_gold_ts_labels[-1]], self.dim_ts_y + 1)

        beg_obs = dy.inputVector([min_val] * self.dim_ts_y + [0, min_val])
        end_obs = dy.inputVector([min_val] * self.dim_ts_y + [min_val, 0])
        padded_observations = [beg_obs] + observations + [end_obs]
        # observations at t=0,
        init = padded_observations[0]
        prev = init
        for t, obs in enumerate(padded_observations[1:]):
            alphas_t = []
            for next_y in range(self.dim_ts_y + 2):
                # dy.pick(obs, t), get the score of the tag t in the current observation vector (i.e., current word)
                # transitions[:, next_y] is the transition scores ends in next_y
                obs_broadcast = dy.concatenate([dy.pick(obs, next_y)] * (self.dim_ts_y + 2))
                next_y_expr = prev + dy.transpose(self.transitions)[next_y] + obs_broadcast
                alphas_t.append(self.log_sum_exp(scores=next_y_expr))
            prev = dy.concatenate(alphas_t)
        # dim_ts_y + 1 corresponds to the END tag
        #final = prev + dy.transpose(self.transitions)[self.dim_ts_y + 1]
        final = prev
        all_path_score = self.log_sum_exp(scores=final)
        loss = - (gold_score - all_path_score)

        if is_train:
            loss.backward()
            self.optimizer.update()

        pred_ts_labels, _ = self.viterbi_decoding(observations=padded_observations)
        return loss.value(), [], pred_ts_labels

    def predict(self, dataset):
        """

        :param dataset:
        :return:
        """
        n_sample = len(dataset)
        gold_ote = [x['ote_tags'] for x in dataset]
        gold_ts = [x['ts_tags'] for x in dataset]

        if self.tagging_schema == 'BIO':
            gold_ote, gold_ts = bio2ot_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
        elif self.tagging_schema == 'OT':
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)

        pred_ote, pred_ts = [], []
        for i in range(n_sample):
            _, _, pred_ts_labels = self.forward(x=dataset[i], is_train=False)
            #pred_ote.append(label2tag(label_sequence=pred_ote_labels, tag_vocab=self.ote_tag_vocab))
            pred_ts.append(label2tag(label_sequence=pred_ts_labels, tag_vocab=self.ts_tag_vocab))
        # transform the output tag sequence to BIEOS tag sequence before evaluation
        if self.tagging_schema == 'BIO':
            pred_ote, pred_ts = bio2ot_batch(
                ote_tags=pred_ote, ts_tags=pred_ts)
            pred_ote, pred_ts = ot2bieos_batch(
                ote_tags=pred_ote, ts_tags=pred_ts)
        elif self.tagging_schema == 'OT':
            pred_ote, pred_ts = ot2bieos_batch(
                ote_tags=pred_ote, ts_tags=pred_ts)
        # evaluation
        ts_scores = evaluate_ts(gold_ts=gold_ts, pred_ts=pred_ts)
        return None, ts_scores

    def viterbi_decoding(self, observations):
        """
        viterbi decoding for CRF decoding layer
        :param observations: observation scores
        :return:
        """
        back_pointers = []
        # observation score for BEG tag
        init = observations[0]
        prev = init
        transition_T = dy.transpose(self.transitions)
        trans_exprs = [transition_T[idx] for idx in range(self.dim_ts_y + 2)]
        for obs in observations[1:]:
            bpts_t = []
            vvars_t = []
            for next_y in range(self.dim_ts_y + 2):
                # trans_exprs[next_y], transition probabilities that ends with next_y
                next_y_expr = prev + trans_exprs[next_y]
                next_y_arr = next_y_expr.npvalue()
                best_y = np.argmax(next_y_arr)
                bpts_t.append(best_y)
                vvars_t.append(dy.pick(next_y_expr, best_y))
            prev = dy.concatenate(vvars_t) + obs
            back_pointers.append(bpts_t)
        # end tags
        #terminal_expr = prev + trans_exprs[self.dim_ts_y+1]
        #terminal_arr = terminal_expr.npvalue()
        final = prev
        final_arr = final.npvalue()
        best_y = np.argmax(final_arr)
        assert best_y == (self.dim_ts_y + 1)
        path_score = dy.pick(final, best_y)
        # reverse over the backpointers to get the best path
        # backtracking
        best_path = []
        for bpts_t in reversed(back_pointers):
            best_y = bpts_t[best_y]
            best_path.append(best_y)
        # remove the beg label
        BEG = best_path.pop()
        best_path.reverse()
        assert BEG == self.dim_ts_y
        return best_path, path_score

















