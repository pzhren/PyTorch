import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO

#nltk.download()
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    '''
    __init__(self)是一种特殊的方法，可以在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。
    https://blog.csdn.net/github_40122084/article/details/79375369
    '''
    #定义两个词典
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    #构建词典
    def add_word(self, word):
        if not word in self.word2idx:#单词如果不在词典中
            self.word2idx[word] = self.idx#word:idx
            self.idx2word[self.idx] = word#idx:word
            self.idx += 1#下标后移

    '''
    __call__():Python中有一个有趣的语法，只要定义类型的时候，实现__call__函数，这个类型就成为可调用的。
    换句话说，我们可以把这个类型的对象当作函数来使用，相当于 重载了括号运算符。
    https://www.cnblogs.com/lovemo1314/archive/2011/04/29/2032871.html
    '''
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    #json:训练集的解释文本
    #threshold：单词次数的阈值
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        #分词word tokenize：使用nltk.word_tokenize(text)
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':#直接作为脚本执行
    ''''https://blog.csdn.net/mameng1/article/details/54409910'''
    parser = argparse.ArgumentParser()#创建解析器对象ArgumentParser
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
