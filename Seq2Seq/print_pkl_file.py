import pickle
from pprint import pprint

# 加载 .pkl 文件
with open('trg_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 打印内容
print("Loaded object type:", type(vocab))  # 打印对象类型
pprint(vocab)  # 格式化打印对象内容

if hasattr(vocab, 'word2index'):
    print("\nWord to Index:")
    print(vocab.word2index)

if hasattr(vocab, 'index2word'):
    print("\nIndex to Word:")
    print(vocab.index2word)

if hasattr(vocab, 'word2count'):
    print("\nWord Counts:")
    print(vocab.word2count)

if hasattr(vocab, 'n_words'):
    print("\nTotal Words:", vocab.n_words)