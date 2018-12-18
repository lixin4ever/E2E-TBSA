# coding: UTF-8
__author__ = 'lixin77'

from scrapy.selector import Selector
#import cPickle
import nltk
from nltk import word_tokenize
import sys
import string

def process_text(text):
    """
    process the text and filter some special symbol
    :param text:
    :return:
    """
    # string preprocessing and aspect term will not be processed
    dot_exist = ('.' in text)
    cur_text = text.replace('.', '')
    #cur_text = cur_text.replace('-', ' ')
    cur_text = cur_text.replace(' - ', ', ').strip()
    cur_text = cur_text.replace('- ', ' ').strip()

    # split words and punctuations
    if '? ' not in cur_text:
        cur_text = cur_text.replace('?', '? ').strip()
    if '! ' not in cur_text:
        cur_text = cur_text.replace('!', '! ').strip()
    cur_text = cur_text.replace('(', '')
    cur_text = cur_text.replace(')', '')
    cur_text = cur_text.replace('...', ', ').strip('.').strip().strip(',')
    # remove quote
    cur_text = cur_text.replace('"', '')
    cur_text = cur_text.replace(" '", " ")
    cur_text = cur_text.replace("' ", " ")

    cur_text = cur_text.replace(':', ', ')
    if dot_exist:
        cur_text += '.'
        # correct some typos
    # mainly for processing English texts
    cur_text = cur_text.replace('cant', "can't")
    cur_text = cur_text.replace('wouldnt', "wouldn't")
    cur_text = cur_text.replace('dont', "don't")
    cur_text = cur_text.replace('didnt', "didn't")
    cur_text = cur_text.replace("you 're", "you're")

    # replace some special symbol
    cur_text = cur_text.replace(u' – ', ', ').strip()

    cur_text = cur_text.replace(u"‘", "")
    # filter the non-ascii character
    cur_text = ''.join([ch if ord(ch) < 128 else ' ' for ch in cur_text])
    return cur_text

def extract_aspect(aspects, text, dataset_name):
    """
    extract aspects from xml tags
    :param aspects: a list of aspect tags / selectors
    :param text: corresponding sentence
    :param dataset_name: name of dataset
    :return:
    """
    counter = 0
    # mapping between aspect id and aspect name
    id2aspect = {}
    # mapping between aspect id and the sentiment polarity of the aspect
    id2polarity = {}
    # number of aspects, singleton, multi-word-aspects in the sentence, respectively
    n_aspect, n_singleton, n_mult_word = 0, 0, 0
    cur_text = text
    from_to_pairs = []
    for t in aspects:
        _from = int(t.xpath('.//@from').extract()[0])
        _to = int(t.xpath('.//@to').extract()[0])
        if _from == 0 and _to == 0:
            # NULL target
            continue
        if not '14' in dataset_name:
            target = t.xpath('.//@target').extract()[0].replace(u'\xa0', ' ')
        else:
            target = t.xpath('.//@term').extract()[0].replace(u'\xa0', ' ')
        if target == 'NULL':
            # there is no aspect in the text
            continue
        # for SemEval challenge, polarity can be positive, negative or neutral
        polarity = t.xpath('.//@polarity').extract()[0]
        if polarity == 'positive':
        	pol_val = 'POS'
        elif polarity == 'negative':
        	pol_val = 'NEG'
        elif polarity == 'neutral':
        	pol_val = 'NEU'
        elif polarity == 'conflict':
            # ignore the confilct aspects
            continue
        else:
        	raise Exception("Invalid polarity value #%s#" % polarity)
        flag = False
        # remove special symbol in aspect term
        #if 'english' in dataset_name:
        target = target.replace(u'é', 'e')
        target = target.replace(u'’', "'")
        if text[_from:_to] == target:
            flag = True
        elif (_from - 1 >= 0) and text[(_from - 1):(_to - 1)] == target:
            _from -= 1
            _to -= 1
            flag = True
        elif (_to + 1 < len(text)) and text[(_from + 1):(_to + 1)] == target:
            _from += 1
            _to += 1
            flag = True
        # we can find the aspect in the raw text
        assert flag

        if (_from, _to) in from_to_pairs:
            continue
        aspect_temp_value = 'ASPECT%s' % counter
        counter += 1
        id2aspect[aspect_temp_value] = target
        id2polarity[aspect_temp_value] = pol_val
        cur_text = cur_text.replace(target, aspect_temp_value)
        from_to_pairs.append((_from, _to))
        n_aspect += 1
        if len(target.split()) > 1:
            n_mult_word += 1
        else:
            n_singleton += 1
    return id2aspect, id2polarity, n_aspect, n_singleton, n_mult_word, cur_text


def format_output(x, y, text):
    """
    format the dataset output
    :param x: word sequence
    :param y: tag sequence
    :param text: raw text
    :return:
    """
    tag_sequence = ''
    for i in range(len(x)):
        if i == (len(x) - 1):
            tag_sequence = '%s%s=%s' % (tag_sequence, x[i], y[i])
        else:
            tag_sequence = '%s%s=%s ' % (tag_sequence, x[i], y[i])
    data_line = '%s####%s\n' % (text, tag_sequence)
    #print(data_line)
    return data_line



def extract_text(dataset_name):
    """
    extract textual information from the xml file
    :param dataset_name: dataset name
    """
    delset = string.punctuation
    fpath = './raw_data/%s.xml' % dataset_name
    print("Process %s..." % fpath)
    page_source = ''
    with open(fpath) as fp:
        for line in fp:
            page_source = '%s%s' % (page_source, line.strip())
    reviews = []
    # regard one sentence as an example
    sentences = Selector(text=page_source).xpath('//sentences/sentence')
    reviews = [sentences]

    n_sen = 0
    n_word = 0
    # number of aspects, singletons and multi-words in the dataset, respectively
    n_aspect, n_singleton, n_mult_word = 0, 0, 0
    n_sen_with_no_aspect = 0
    lines = []
    for sentences in reviews:
        # scan all of the reviews
        x, y, review_text = [], [], ''
        for sid in range(len(sentences)):
            sen = sentences[sid]
            prev = ''
            n_sen += 1
            text = sen.xpath('.//text/text()').extract()[0]
            text = text.replace(u'\xa0', ' ')
            # note: preprocessing in the raw text should not change the index
            # perform this only for English texts
            # in spanish, it can be a normal word
            text = text.replace(u'é', 'e')
            text = text.replace(u'’', "'")
            cur_text = text

            assert isinstance(dataset_name, str)
            if '14' in dataset_name:
                aspects = sen.xpath('.//aspectterms/aspectterm')
            else:
                aspects = sen.xpath('.//opinions/opinion')

            if not aspects:
                # sent with no aspect
                n_sen_with_no_aspect += 1
            else:
                id2aspect, id2polarity, n_a, n_s, n_m, cur_text = extract_aspect(aspects=aspects, text=cur_text,
                                                                    dataset_name=dataset_name)
                n_aspect += n_a
                n_singleton += n_s
                n_mult_word += n_m
            # flush output buffer every sentence
            x, y = [], []
            # process the text and filter the unnecessary characters
            cur_text = process_text(text=cur_text)
            tokens = word_tokenize(cur_text)
            for t in tokens:
                if t.startswith('ASPECT'):
                    # in this case, t is actually the id of aspect
                    # raw_string is the aspect word or aspect phrase
                    raw_string = id2aspect[t[:7]]
                    pol_val = id2polarity[t[:7]]
                    aspect_words = raw_string.split()
                    n_aw = len(aspect_words)
                    x.extend(aspect_words)
                    y.extend(['T-%s' % pol_val] * n_aw)
                    n_word += n_aw
                else:
                    # t is the literal value
                    if not t.strip() == '':
                        # t is not blank space or empty string
                        x.append(t.strip())
                        y.append('O')
                        n_word += 1
            # length check for every sentence
            assert len(x) == len(y)
            # write back after processing a sentence
            lines.append(format_output(x=x, y=y, text=text))

    with open('./data/%s.txt' % (dataset_name), 'w+') as fp:
        fp.writelines(lines)

    print("dataset:", dataset_name)
    print("n_sen:", n_sen)
    print("average length:", int(n_word / n_sen))
    print("total aspects:", n_aspect)
    print("n_singleton:", n_singleton)
    print("n_mult_words:", n_mult_word)
    print("n_without_aspect:", n_sen_with_no_aspect)
    print("n_tokens:", n_word)
    print("\n\n")

if __name__ == '__main__':
    # this script is used for converting the original xml files into the formatted files 
    dataset_names = ['laptop14_train', 'laptop14_test',
                     'rest14_train', 'rest14_test',
                     'rest15_train', 'rest15_test', 'hotel15_test',
                     'rest16_train', 'rest16_test']
    for ds_name in dataset_names:
    	extract_text(ds_name)

