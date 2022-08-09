# -*- coding: utf-8 -*-
"""
@author: hyl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2 #由前面两个词来预测这个单词
EMBEDDING_DIM = 10  # 词向量嵌入的维度

with open("output.txt", "r", encoding=('utf-8')) as f:
    data = f.read() #输入数据集
ngrams = [
    (
        [data[i - j - 1] for j in range(CONTEXT_SIZE)],
        data[i]
    )
    for i in range(CONTEXT_SIZE, len(data))
]#将词分成三个为一组的词块，前两个为给定词，后一个为目标词

vocab = set(data)#消除重复的词语
word_to_ix = {word: i for i, word in enumerate(vocab)}#此处建立词典，建立对每个词的索引

class NGramLanguageModeler(nn.Module):
    
    #初始化时定义单词表大小，想要嵌入的维度大小，上下文的长度
    def __init__(self, vocab_size, embedding_dim, context_size):
        # 继承自nn.Module，例行执行父类super 初始化方法
        super(NGramLanguageModeler, self).__init__()
        # 建立词嵌入模块
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #建立线性层
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # 将输入进行“嵌入”，并转化为“行向量”
        embeds = self.embeddings(inputs).view((1, -1))
        #通过两层线性层
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        #将结果映射为概率的log
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
#开始训练模型

losses = []
# 设置损失函数为 负对数似然损失函数(Negative Log Likelihood)
loss_function = nn.NLLLoss()
# 实例化我们的模型，传入：单词表的大小、嵌入维度、上下文长度
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
# 优化函数使用随机梯度下降算法，学习率设置为0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):#训练100代
    total_loss = 0
    #循环context与target
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
