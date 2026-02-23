import numpy as np
import scipy.io
import torch



def lire_alpha_digits(indices= [], path = 'data/Binary_Alpha_Digits/binaryalphadigs.mat'):
    mat = scipy.io.loadmat(path)
    dat = mat['dat']

    X=[]
    
    for i in indices:
        for j in range(dat.shape[1]):
            X.append(dat[i,j].flatten())
    
    return torch.tensor(X, dtype=torch.float32)

def init_RBM(n_visible= 1, n_hidden=1):
    W = torch.randn(n_visible, n_hidden) * 0.01
    a = torch.zeros(n_visible)
    b = torch.zeros(n_hidden)
    return W, a, b


def entree_sortie_RBM(data, W, a, b):
    # Calcul des activations des unités cachées
    hidden_activations = torch.matmul(data, W) + b
    hidden_probabilities = torch.sigmoid(hidden_activations)
    
    return hidden_probabilities

def sortie_entree_RBM(data_h, W, a, b):
    # Calcul des activations des unités visibles
    visible_activations = torch.matmul(data_h, W.t()) + a
    visible_probabilities = torch.sigmoid(visible_activations)

    return visible_probabilities

def train_RBM(W, a, b, epochs, learning_rate, batch_size, data):
    n_samples = data.shape[0]

    #Apprentissage par CD-1
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            batch = data[i:i+batch_size]

            # Phase positive
            hidden_probabilities = entree_sortie_RBM(batch, W, a, b)
            hidden_states = (hidden_probabilities > torch.rand_like(hidden_probabilities)).float()

            # Phase négative
            visible_probabilities = sortie_entree_RBM(hidden_states, W, a, b)
            visible_states = (visible_probabilities > torch.rand_like(visible_probabilities)).float()

            hidden_probabilities_neg = entree_sortie_RBM(visible_states, W, a, b)

            # Mise à jour des poids et des biais
            W += learning_rate * (torch.matmul(batch.t(), hidden_probabilities) - torch.matmul(visible_states.t(), hidden_probabilities_neg)) / batch_size
            a += learning_rate * torch.mean(batch - visible_states, dim=0)
            b += learning_rate * torch.mean(hidden_probabilities - hidden_probabilities_neg, dim=0)

        print(f'Epoch {epoch+1}/{epochs}.\n Erreur quadratique entrée/reconstruction : {torch.mean((data - visible_probabilities) ** 2).item():.4f}')

    return W, a, b

def generer_image_RBM(W, a, b, n_iterations=1000, n_images=1):
    images = []
    
    for _ in range(n_images):
        # Initialisation aléatoire d'une image
        visible = torch.rand(1, W.shape[0])

    for _ in range(n_iterations):
        hidden_probabilities = entree_sortie_RBM(visible, W, a, b)
        hidden_states = (hidden_probabilities > torch.rand_like(hidden_probabilities)).float()
        visible_probabilities = sortie_entree_RBM(hidden_states, W, a, b)
        visible = (visible_probabilities > torch.rand_like(visible_probabilities)).float()

        images.append(visible.reshape(20, 16))
    return images