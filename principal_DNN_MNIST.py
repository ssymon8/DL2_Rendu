import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import principal_DBN_alpha as dbn_module
import principal_RBM_alpha as rbm_module

def init_DNN(taille_reseau= [320, 100, 10]):
    # 10 classes pour MNIST
    dnn = []
    
    dnn = dbn_module.init_DBN(taille_reseau[:-1])

    couche_classification = rbm_module.init_RBM(n_visible= taille_reseau[-2], n_hidden=taille_reseau[-1])
    dnn.append(couche_classification) 
    return dnn

def pretrain_DNN(dnn, epochs, learning_rate, batch_size, data):
    dbn = dnn[:-1]
    dbn = dbn_module.train_DBN(dbn, epochs, learning_rate, batch_size, data)
    dnn[:-1] = dbn
    return dnn

def calcul_softmax(rbm, input_data):
    with torch.no_grad():
        hidden_probabilities = rbm_module.entree_sortie_RBM(input_data, rbm)
        return torch.softmax(hidden_probabilities, dim=1)
    
def entree_sortie_reseau(dnn, input_data):
    sorties = [input_data]
    
    for i in range(len(dnn) - 1):
        input_data = rbm_module.entree_sortie_RBM(input_data, dnn[i])
        sorties.append(input_data)
        
    probas = calcul_softmax(dnn[-1], input_data)
    sorties.append(probas)
    
    return sorties

def retropropagation(dnn, epochs, learning_rate, batch_size, data, labels):
    criterion = nn.CrossEntropyLoss()
    # on pourrait utiliser AdamW mais ce sera trop puissant pour voir l'intérêt de la rétropropagation
    optimizer = optim.SGD(dnn[-1]['W'].parameters(), lr=learning_rate)

    params_to_optimize = []
    for layer in dnn:
        layer['W'].requires_grad_(True)
        layer['b'].requires_grad_(True)
        params_to_optimize.extend([layer['W'], layer['b']])
    
    n_samples = data.shape[0]
    
    for epoch in range(epochs):
        #forward pass par batch de données
        for i in range(0, n_samples, batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            sorties = entree_sortie_reseau(dnn, batch_data)
            loss = criterion(sorties[-1], batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # on regarde l'entropie croisée sur l'ensemble des données à la fin
        # de chaque epoch pour voir l'évolution de la perte
        with torch.no_grad():
            probas = entree_sortie_reseau(dnn, data)[-1]
            loss = criterion(probas, labels)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    for layer in dnn:
        layer['W'].requires_grad_(False)
        layer['b'].requires_grad_(False)

    return dnn

def test_DNN(dnn, data, labels):
    with torch.no_grad():
        probas = entree_sortie_reseau(dnn, data)[-1]
        predicted_labels = torch.argmax(probas, dim=1)
        taux_erreur = (predicted_labels == labels).float().mean().item()
    print(f"Test Accuracy: {taux_erreur * 100:.2f}%")

    return taux_erreur


