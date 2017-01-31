import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import cPickle as pickle
import numpy as np

from sklearn.manifold import TSNE

def load_data():
    train_data_path = './data/train.p'
    val_data_path = './data/val.p'
    reverse_dictionary_path = './data/reverse_dictionary.p'

    train_data = pickle.load(open(train_data_path, 'rb'))
    print "Loaded train data!"
    val_data = pickle.load(open(val_data_path, 'rb'))
    print "Loaded val data!"
    reverse_dictionary = pickle.load(open(reverse_dictionary_path, 'rb'))
    print "Loaded reverse dictionary!"
    return train_data, val_data, reverse_dictionary

def print_closest_words(val_index, nearest, reverse_dictionary):
    val_word = reverse_dictionary[val_index]                 
    log_str = "Nearest to %s:" % val_word                          
    for k in xrange(len(nearest)):                                        
        close_word = reverse_dictionary[nearest[k]]                
        log_str = "%s %s," % (log_str, close_word)                 
    print(log_str)

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

def visualize_embeddings(final_embeddings, reverse_dictionary):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
