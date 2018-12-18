import argparse
from model import *
from utils import *
from evals import evaluate
import random
import os

separator = '========================================================================================'

def run(dataset, model, params):
    """
    run the experiment
    :param dataset: dataset
    :param model: constructed neural model
    :param params: settings of hyper-parameter
    :return:
    """
    train_set, val_set, test_set = dataset
    n_train = len(train_set)
    best_val_ote_score, best_val_ts_score = -999.0, -999.0
    best_pred_ote, best_pred_ts = [], []
    best_iter = -1
    ote_tag_vocab = params.ote_tag_vocab
    ts_tag_vocab = params.ts_tag_vocab
    tagging_schema = params.tagging_schema
    n_epoch = params.n_epoch
    init_lr = model.optimizer.learning_rate
    decay_rate = params.lr_decay
    for n_iter in range(n_epoch):
        cur_lr = init_lr / (1 + decay_rate * n_iter)
        model.optimizer.learning_rate = cur_lr
        total_train_loss = 0.0
        train_pred_ote, train_pred_ts = [], []
        print("In Epoch %s / %s (current lr: %.4f):" % (n_iter + 1, n_epoch, cur_lr))
        # shuffle the training set in each epoch
        random.shuffle(train_set)

        train_gold_ote = [x['ote_tags'] for x in train_set]
        train_gold_ts = [x['ts_tags'] for x in train_set]

        if tagging_schema == 'BIO':
            train_gold_ote, train_gold_ts = bio2ot_batch(
                ote_tags=train_gold_ote, ts_tags=train_gold_ts)
            train_gold_ote, train_gold_ts = ot2bieos_batch(
                ote_tags=train_gold_ote, ts_tags=train_gold_ts)
        elif tagging_schema == 'OT':
            train_gold_ote, train_gold_ts = ot2bieos_batch(
                ote_tags=train_gold_ote, ts_tags=train_gold_ts)

        for i in range(n_train):
            loss, pred_ote_labels, pred_ts_labels = model.forward(x=train_set[i], is_train=True)
            total_train_loss += loss
            if pred_ote_labels:
                # if pred_ts_labels is empty, skip this expression
                train_pred_ote.append(label2tag(label_sequence=pred_ote_labels, tag_vocab=ote_tag_vocab))
            train_pred_ts.append(label2tag(label_sequence=pred_ts_labels, tag_vocab=ts_tag_vocab))

        # before evaluation, transform the output tag sequence to BIEOS tag sequence
        if tagging_schema == 'BIO':
            if train_pred_ote:
                train_pred_ote = bio2ot_ote_batch(ote_tag_seqs=train_pred_ote)
                train_pred_ote = ot2bieos_ote_batch(ote_tag_seqs=train_pred_ote)
            train_pred_ts = bio2ot_ts_batch(ts_tag_seqs=train_pred_ts)
            train_pred_ts = ot2bieos_ts_batch(ts_tag_seqs=train_pred_ts)
        elif tagging_schema == 'OT':
            if train_pred_ote:
                train_pred_ote = ot2bieos_ote_batch(ote_tag_seqs=train_pred_ote)
            train_pred_ts = ot2bieos_ts_batch(ts_tag_seqs=train_pred_ts)
        # evaluation
        ts_scores = evaluate_ts(gold_ts=train_gold_ts, pred_ts=train_pred_ts)
        ts_macro_f1, ts_micro_p, ts_micro_r, ts_micro_f1 = ts_scores
        if train_pred_ote:
            ote_scores = evaluate_ote(gold_ot=train_gold_ote, pred_ot=train_pred_ote)
            ote_p, ote_r, ote_f1 = ote_scores
            print("\ttrain loss: %.4f, ote: f1: %.4f, ts: precision: %.4f, recall: %.4f, "
                  "micro-f1: %.4f" % (total_train_loss / n_train, ote_f1, ts_micro_p, ts_micro_r, ts_micro_f1))
        else:
            print("\ttrain_loss: %.4f, ts: precision: %.4f, recall: %.4f, micro-f1: %.4f" %
                  (total_train_loss / n_train, ts_micro_p, ts_micro_r, ts_micro_f1))
        val_outputs = model.predict(dataset=val_set)
        val_ote_scores, val_ts_scores = val_outputs[0], val_outputs[1]
        val_ts_macro_f1, val_ts_micro_p, val_ts_micro_r, val_ts_micro_f1 = val_ts_scores
        if val_ote_scores:
            val_ote_p, val_ote_r, val_ote_f1 = val_ote_scores
            print("\tval performance: ote: f1: %.4f, ts: precision: %.4f, recall: %.4f, "
                  "micro-f1: %.4f" % (val_ote_f1, val_ts_micro_p, val_ts_micro_r, val_ts_micro_f1))
        else:
            print("\tval performance: ts: precision: %.4f, recall: %.4f, micro-f1: %.4f"
                  % (val_ts_micro_p, val_ts_micro_r, val_ts_micro_f1))
        if val_ts_micro_f1 > best_val_ts_score:
            best_val_ts_score = val_ts_micro_f1
            test_outputs = model.predict(dataset=test_set)
            test_ote_scores, test_ts_scores = test_outputs[0], test_outputs[1]
            if len(test_outputs) > 2:
                best_pred_ote, best_pred_ts = test_outputs[2], test_outputs[3]
            best_iter = n_iter + 1
            if test_ote_scores:
                print("\tExceed: test performance: ote: f1: %.4f, ts: precision: %.4f, recall: %.4f, micro-f1: %.4f" % (test_ote_scores[2], test_ts_scores[1], test_ts_scores[2], test_ts_scores[3]))
            else:
                print("\tExceed: test performance: ts: precision: %.4f, recall: %.4f, micro-f1: %.4f"
                      % (test_ts_scores[1], test_ts_scores[2], test_ts_scores[3]))
            model_path = './models/%s_%.6lf.model' % (params.ds_name, test_ts_scores[3])
            print("Save the model to %s..." % model_path)
            if not os.path.exists('./models'):
                os.mkdir('./models')
            model.pc.save(model_path)
    if test_ote_scores:
        final_res_string = "\nBest results obtained at %s: ote f1: %.4f, ts: precision: %.4f, recall: %.4f, " \
                           "ts micro-f1: %.4f" % (best_iter, test_ote_scores[2], test_ts_scores[1],
                                                  test_ts_scores[2], test_ts_scores[3])
    else:
        final_res_string = "\nBest results obtained at %s, ts: precision: %.4f, recall: %.4f, " \
                           "ts micro-f1: %.4f" % (best_iter, test_ts_scores[1],
                                                  test_ts_scores[2], test_ts_scores[3])
    if best_pred_ote:
        n_sample = len(test_set)
        gold_ote = [x['ote_tags'] for x in test_set]
        gold_ts = [x['ts_tags'] for x in test_set]
        if model.tagging_schema == 'BIO':
            gold_ote, gold_ts = bio2ot_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
        elif model.tagging_schema == 'OT':
            gold_ote, gold_ts = ot2bieos_batch(
                ote_tags=gold_ote, ts_tags=gold_ts)
        output_lines = ['Dataset: %s\n' % params.ds_name, 'Parameter settings: \n']
        params_dict = vars(params)
        for k in params_dict:
            if k == 'char_vocab' or k == 'vocab':
                continue
            else:
                v = params_dict[k]
                output_lines.append('\t%s: %s\n' % (k, v))
        output_lines.append("==============================================\n\n")
        for i in range(n_sample):
            ote_seq = best_pred_ote[i]
            ts_seq = best_pred_ts[i]
            w_seq = test_set[i]['words']
            ote_seq_gold = gold_ote[i]
            ts_seq_gold = gold_ts[i]
            assert len(ote_seq) == len(ts_seq) == len(w_seq)
            for j in range(len(ote_seq)):
                word = w_seq[j]
                ote_tag = ote_seq[j]
                ote_tag_gold = ote_seq_gold[j]
                ts_tag = ts_seq[j]
                ts_tag_gold = ts_seq_gold[j]
                output_lines.append('%s\t%s\t%s\t%s\t%s\n' % (word, ote_tag, ote_tag_gold, ts_tag, ts_tag_gold))
            # use empty lines as the separator
            output_lines.append('\n')
        if not os.path.exists('./predictions'):
            os.mkdir('./predictions')
        model_path = './predictions/%s_%.6lf.txt' % (params.ds_name, test_ts_scores[3])
        with open(model_path, 'w+') as fp:
            fp.writelines(output_lines)
    print(final_res_string)
    return final_res_string, model_path


if __name__ == '__main__':
    # random_seed = 1234
    # random.seed(random_seed)

    parser = argparse.ArgumentParser(description="Open Domain ABSA")
    parser.add_argument("-ds_name", type=str, default='rest14', help="dataset name")
    # dimension of LSTM hidden representations
    parser.add_argument("-dim_char", type=int, default=30, help="dimension of char embeddings")
    parser.add_argument("-dim_char_h", type=int, default=50, help="dimension of char hidden representations")
    parser.add_argument("-dim_ote_h", type=int, default=50, help="hidden dimension for opinion target extraction")
    parser.add_argument("-dim_ts_h", type=int, default=50, help="hidden dimension for targeted sentiment")
    parser.add_argument("-input_win", type=int, default=3, help="window size of input")
    parser.add_argument("-stm_win", type=int, default=3, help="window size of OE component")
    parser.add_argument("-optimizer", type=str, default="sgd", help="optimizer (or, trainer)")
    parser.add_argument("-n_epoch", type=int, default=40, help="number of training epoch")
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout rate for final representations")
    parser.add_argument("-emb_name", type=str, default="glove_840B", help="name of word embedding")
    # Note: tagging schema is OT in the original data record
    parser.add_argument("-tagging_schema", type=str, default="BIO", help="tagging schema")
    parser.add_argument("-rnn_type", type=str, default="LSTM",
                        help="type of rnn unit, currently only LSTM and GRU are supported")
    parser.add_argument("-sgd_lr", type=float, default=0.1,
                        help="learning rate for sgd, only used when the optimizer is sgd")
    parser.add_argument("-clip_grad", type=float, default=5.0, help="maximum gradients")
    parser.add_argument("-lr_decay", type=float, default=0.05, help="decay rate of learning rate")
    parser.add_argument("-use_char", type=int, default=0, help="if use character-level word embeddings")
    parser.add_argument('-epsilon', type=float, default=0.5, help="maximum proportions of the boundary-based scores")
    dy_seed = 1314159
    random_seed = 1234
    #random_seed = 1972
    args = parser.parse_args()
    if args.ds_name == 'laptop14':
        random_seed = 13456
    if args.ds_name.startswith("twitter"):
        random_seed = 7788
    args.dynet_seed = dy_seed
    args.random_seed = random_seed
    random.seed(random_seed)
    emb_name = args.emb_name

    emb2path = {
        'glove_6B': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_6B_300d.txt',
        'glove_42B': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_42B_300d.txt',
        'glove_840B': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_840B_300d.txt',
        'glove_27B100d': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_twitter_27B_100d.txt',
        'glove_27B200d': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_twitter_27B_200d.txt',
        'yelp_rest1': '/projdata9/info_fil/lixin/Research/yelp/yelp_vec_200_2_win5_sent.txt',
        'yelp_rest2': '/projdata9/info_fil/lixin/Research/yelp/yelp_vec_200_2_new.txt',
        'amazon_laptop': '/projdata9/info_fil/lixin/Resources/amazon_full/vectors/amazon_laptop_vec_200_5.txt'
    }

    emb_path = emb2path[emb_name]

    input_win = args.input_win
    stm_win = args.stm_win
    ds_name = args.ds_name
    tagging_schema = args.tagging_schema

    # build dataset
    train, val, test, vocab, char_vocab, ote_tag_vocab, ts_tag_vocab = build_dataset(
        ds_name=ds_name, input_win=input_win,
        tagging_schema=tagging_schema, stm_win=stm_win
    )

    # obtain the pre-trained word embeddings
    embeddings = load_embeddings(path=emb_path, vocab=vocab, ds_name=ds_name, emb_name=emb_name)

    # obtain the pre-trained character embeddings
    char_embeddings = None

    # convert the datasets to the conll format and write them back to the specified folder
    #to_conll(train=train, val=val, test=test, ds_name=ds_name, embeddings=embeddings, vocab=vocab)

    args.dim_w = len(embeddings[0])
    #args.dim_char = len(char_embeddings[0])
    args.dim_char = 10
    args.ote_tag_vocab = ote_tag_vocab
    args.ts_tag_vocab = ts_tag_vocab

    if ds_name.startswith("twitter"):
        args.epsilon = 0.8

    # content need to write to the log file
    log_lines = [separator+"\n"]
    #print(args)
    print(separator)
    for arg in vars(args):
        arg_string = "\t-%s: %s" % (arg, str(getattr(args, arg)))
        print(arg_string)
        log_lines.append(arg_string+"\n")

    args.char_vocab = char_vocab
    model = Model(params=args, vocab=vocab, embeddings=embeddings, char_embeddings=char_embeddings)

    mode = 'train-test'
    if mode == 'train-test':
        final_res_string, model_path = run(dataset=[train, val, test], model=model, params=args)
        log_lines.append(final_res_string + "\n")
        log_lines.append("Best model is saved at: %s\n" % model_path)
        log_lines.append(separator + "\n\n")
        print(separator)
        if not os.path.exists('log'):
            os.mkdir('log')
        with open('log/%s.txt' % ds_name, 'a') as fp:
            fp.writelines(log_lines)
    else:
        model.decoding(dataset=test, model_name='lstm_cascade_laptop14_0.573138.model')







