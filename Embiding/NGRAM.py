
import torch

import numpy as np

from torch.nn import functional as F

import torch.optim as optim

import torch.nn as nn



"""
 This is an implementation of word embedding with pytorch
 In this project we are going to learn how embedding word are created and how they are used efore this word 
 
 N-Gram Language Modeling is what we used. 
 
    Given a sequence of word we want to compute the :
    
        P(Wi | Wi-1, Wi-2 ... Wi-t) 
        
        where : 
            Wi is the word of the sequence.
            Wi-t: are word of the context that came b
            
            
     Exemple : 
     
     Let's take the sentence "par exemple" (french)
     
     there are 10 letters. On the ten letters :
     
     * p : 2 / 10 => 1/5 
     * e : 3 / 10
     * x : 1 / 10 
     
     
     knowing that probability sum is 1.
     
     there is 9 possible word couples here are some : 
     
     * p-a : 1 / 9
     
     * p-l : 1 / 9
     
     * p-e : 0 / 9
     
     
     if the character 'p' appear. knowing the word couples there p(a | p) = 1/2  
     
     [ref] : https://fr.wikipedia.org/wiki/N-gramme
"""


"""
Embiding are used to represent word. Embiding are better for word representation compare to one-hot encoder
because they represent the semantic of a word (ex : apple and orange are fruit they will have a high similarity score )

We compute the Word embiding by determining the probability of the word in specific contexts. 

We optimize the objective function : 

            -log(L0) with L0 = 1/m * sum (sum(-log(Pk))) 
            
            Similarity(physicist,mathematician)= qphysicist⋅qmathematician / ‖qphysicist‖‖qmathematician‖ = cos(ϕ)
            
            ϕ est l'angle entre les deux vecteur. 
            Un angle proche de  0 signifie une forte similarité. 
            Un angle proche de  pi signifie une forte dissimilarité 
            


"""





def createDico(texte):
    """
     We need to create a dictionnary that contains the word of the vocab and the index of the word.
     We need those index to be able to use the lookup table that will contains all the
     embedding vectors.
    """

    vocab = set(texte) # ex => set("hello world world for") => {"hello", "world", "for"}

    dico = {word : i for i, word in enumerate(texte)}

    return dico



def createTrigram(texte):
    """
    We are going to create 3-Gram this mean that the only take the two previous word in the sentence as context
    and the word to form the trigram.

    :param texte: texte from wich we are going to create the trigram
    :return: trigram array of tuple ([word-2, word-1], word)
    """

    trigram = [([texte[i], texte[i + 1]], texte[i + 2]) for i in range(len(texte)-2)]

    return trigram


def getTexte():
    """
    Shakespeare Sonnet 2
    :return: test_sentence
    """
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    return test_sentence




def showEmbiding():
    """
    If you to see what is the role of index dictionnary and what is the return value after a call of
    embidings
    :return:
    """

    word_to_ix = {"hello": 0, "world": 1}
    embeds = torch.nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    lookup_tensor = torch.tensor([word_to_ix["hello"], word_to_ix["world"]],
                                 dtype=torch.long)  # contient l'index de hello et l'index du mot work

    hello_embed = embeds(lookup_tensor)  # retourn les vector des deux mots
    print(hello_embed)
    print("Use view(1, -1) ")

    print(hello_embed.view((1, -1))) # display a flatten output a matrix with one row and all the element are in column



"""  
 Embiding is a matrix that contains on each row the vector that represent a word.
 the width D is the dimension of the embiding 
 
 In order to train we are going to use our trigram. 
 
 the two first words will be pass to the netword as context. and we will train the network to predict the next word based
 on the two previous word. It's the N-Gram technique. You try to predict the next word based on the previous words(context).
 
 N-Gram used the probability of "word sequence"  to predict the next word. The next word is the one that have the 
 higher probability to appear next.   
"""

class NGramModel(nn.Module):

    hidden = 200

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramModel, self).__init__()

        self.embing = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # embedding_dim(number of column in the vector) * context_size (number_of_word in the context)
        self.fc = nn.Linear(in_features=embedding_dim * context_size, out_features=self.hidden)

        self.output = nn.Linear(in_features=self.hidden, out_features=vocab_size)



    def forward(self, input):

        # input is a longTensor that contains the index of the words of the context
        # embeds : is a matrix of (context_size, embedding_dim)
        embeds = self.embing(input).view((1, -1))

        # shape(Wfc) => (context_size * embedding_dim, hidden_size)
        # shape(embeds) => (1, context_size * embedding_dim)
        x = F.relu(self.fc(embeds))

        # shape(out) => (1, vocab_size)
        out = F.log_softmax(self.output(x), dim=1)

        return out



def train():
    losses = []
    EMBEDDING_DIM=5
    CONTEXT_SIZE=2
    loss_function = nn.NLLLoss()
    texte = getTexte()
    trigrams = createTrigram(texte)
    vocab = set(texte)
    word_to_ix = createDico(vocab)
    model = NGramModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, context_size=CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for context, target in trigrams:
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



if __name__ == "__main__":
    train()






