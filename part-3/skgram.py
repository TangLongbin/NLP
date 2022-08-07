# -*- coding: utf-8 -*-
"""
最后修改时间：08/08/1：13
最后修改人：熊翔翔

part3.2的skip-gram算法以及后面有加入part4.1的 Visualise the word representations by t-SNE

存在的问题：最后的loss：11.4635 很大 学习次数太少了 学习效果不好
也可以参考朱一诺上传的那份skip-gram 与这份存在一样的问题：最终的loss太大了
"""

import os

import re
from collections import Counter

import random
import numpy as np

import torch
from torch import nn
import torch.optim as optim

from torch import optim

# 数据读写文件路径
TextPath = "D:/try/data_for_p3/text.txt" 
#OutTemplateFile_skipgram = "D:/try/skipgram_text/skipgram"

with open(TextPath) as f:
    text = f.read()

def preprocess(text):

    # Convert all text to lowercase
    text = text.lower()

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with 5 or less occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words

# 分词
words = preprocess(text)

'''
print(type(words)) #list

'''


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    - words: list of words
    - return: 2 dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # descending freq order
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

# 对当前文档构建字典：
vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

# Perform Word Subsampling 二次采样舍弃一些词
# Words that show up often such as "the", "of", and "for" don't provide much context to the nearby words
# discard if random.random() < probs[word] 
# threshold (something small like 1e-5)
def subsample_words(int_words, threshold = 1e-5):
    word_counts = Counter(int_words)
    total_n_words = len(int_words)

    freq_ratios = {word: count/total_n_words for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freq_ratios[word]) for word in word_counts}

    return [word for word in int_words if random.random() < (1 - p_drop[word])]

# Generate Context Targets
# For each word in the text, define Context by grabbing the surrounding words in a window of size
def get_target(words, idx, max_window_size=5):
    R = random.randint(1, max_window_size)
    start = max(0,idx-R)
    end = min(idx+R,len(words)-1)
    targets = words[start:idx] + words[idx+1:end+1] # +1 since doesn't include this idx
    return targets

# Generate Batches 
# Grab batch_size words from a words list. Then for each of those words, get the context target words in a window.
# batch_x 是中心词 batch_y 是附近的词
def get_batches(words, batch_size, max_window_size=5):
  # only full batches
    n_batches = len(words)//batch_size
    words = words[:n_batches*batch_size]
    for i in range(0, len(words), batch_size):
        batch_of_center_words = words[i:i+batch_size]   # current batch of words
        batch_x, batch_y = [], []  

    for ii in range(len(batch_of_center_words)):  # range(batch_size) unless truncated at the end
        x = [batch_of_center_words[ii]]             # single word
        y = get_target(words=batch_of_center_words, idx=ii, max_window_size=max_window_size)  # list of context words

        batch_x.extend(x * len(y)) # repeat the center word (n_context_words) times
        batch_y.extend(y)
  
    yield batch_x, batch_y       # ex) [1,1,2,2,2,2,3,3,3,3], [0,2,0,1,3,4,1,2,4,5]


# Define COSINE SIMILARITY Function for Validation Metric 余弦相似度函数
# Print their closest words using Cosine Similarity
# sim = (a . b) / |a||b|
def cosine_similarity(embedding, n_valid_words=16, valid_window=100):
    """ 
    Returns the cosine similarity of validation words with words in the embedding matrix.
    embedding: pytorch中nn.Embedding模块
    n_valid_words: numbers of validation words (recommended to have even numbers)

    """
    all_embeddings = embedding.weight  # (n_vocab, n_embed) 
    
    magnitudes = all_embeddings.pow(2).sum(dim=1).sqrt().unsqueeze(0) # (1, n_vocab)
  
    # Pick validation words from 2 ranges: (0, window): common words & (1000, 1000+window): uncommon words 
    valid_words = random.sample(range(valid_window), n_valid_words//2) + random.sample(range(1000, 1000+valid_window), n_valid_words//2)
    device=torch.device("cpu")
    valid_words = torch.LongTensor(np.array(valid_words)).to(device) # (n_valid_words, 1)

    valid_embeddings = embedding(valid_words) # (n_valid_words, n_embed)
    # (n_valid_words, n_embed) * (n_embed, n_vocab) --> (n_valid_words, n_vocab) / 1, n_vocab)
    similarities = torch.mm(valid_embeddings, all_embeddings.t()) / magnitudes  # (n_valid_words, n_vocab)
  
    return valid_words, similarities

# Define SkipGram model with Negative Sampling 
# 使用nn.Embedding layer
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors  # input vector embeddings
    

    def forward_target(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors  # output vector embeddings
    

    def forward_noise(self, batch_size, n_samples=5):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        # If no Noise Distribution specified, sample noise words uniformly from vocabulary
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # torch.multinomial :
        # Returns a tensor where each row contains (num_samples) **indices** sampled from 
        # multinomial probability distribution located in the corresponding row of tensor input.
        noise_words = torch.multinomial(input       = noise_dist,           # input tensor containing probabilities
                                        num_samples = batch_size*n_samples, # number of samples to draw
                                        replacement = True)
        device=torch.device("cpu")
        noise_words = noise_words.to(device)
        
        # use context matrix for embedding noise samples
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors
    

# Define Loss Class
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input_vectors,output_vectors,noise_vectors):
      
        batch_size, embed_size = input_vectors.shape

        input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors

        # log-sigmoid loss for correct pairs
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()

        # log-sigmoid loss for incorrect pairs
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        return -(out_loss + noise_loss).mean()  # average batch loss

# Define Noise Distribution
# As defined in the paper by Mikolov et all.
freq = Counter(int_words)
freq_ratio = {word:cnt/len(vocab_to_int) for word, cnt in freq.items()}        
freq_ratio = np.array(sorted(freq_ratio.values(), reverse=True))
unigram_dist = freq_ratio / freq_ratio.sum() 
noise_dist = torch.from_numpy(unigram_dist**0.75 / np.sum(unigram_dist**0.75))

# Define Model, Loss, & Optimizer
embedding_dim = 300
model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist )
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.003)

def train_skipgram(model,
                   criterion,
                   optimizer,
                   int_words,
                   n_negative_samples=5,
                   batch_size=512,
                   n_epochs=5,
                   print_every=5,
                   ):
    device=torch.device("cpu")
    model.to(device)
  
    step = 0
    for epoch in range(n_epochs):
        for inputs, targets in get_batches(int_words, batch_size=batch_size):
            step += 1
            inputs = torch.LongTensor(inputs).to(device)    # [b*n_context_words]
            targets = torch.LongTensor(targets).to(device)  # [b*n_context_words]

            embedded_input_words = model.forward_input(inputs)
            embedded_target_words = model.forward_target(targets)
            embedded_noise_words = model.forward_noise(batch_size=inputs.shape[0], n_samples=n_negative_samples)

            loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            if (step % print_every) == 0:
                print("Epoch: {}/{}".format((epoch+1), n_epochs))
                print("Loss: {:.4f}".format(loss.item()))
                valid_idxs, similarities = cosine_similarity(model.in_embed)
                _, closest_idxs = similarities.topk(6)
                valid_idxs, closest_idxs = valid_idxs.to('cpu'), closest_idxs.to('cpu')
        
                for ii, v_idx in enumerate(valid_idxs):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[v_idx.item()] + " | "+ ", ".join(closest_words))

    print("\n...\n")
    print(step)
    
                
#run

train_skipgram(model,
               criterion,
               optimizer,
               int_words,
               n_negative_samples=5)


'''
part4.1
t-sne 可视化
'''

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
'''
以下语句在jupyter notebook 中使用：
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
在spyder中改用：
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
'''
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

embeddings = model.in_embed.weight.to('cpu').data.numpy()

viz_words = 380
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    
