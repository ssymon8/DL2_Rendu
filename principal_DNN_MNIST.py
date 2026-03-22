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
    
    n_samples = data.shape[0]
    
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Forward pass
            sorties= entree_sortie_reseau(dnn, batch_data)
            probas = sorties[-1]

            #Gradient de la couche de classification (softmax)
            dZ = probas - batch_labels

            dW= sorties[-2].t() @ dZ / batch_size
            db = torch.mean(dZ, dim=0)

            dA = dZ @ dnn[-1]['W'].t()

            # Mise à jour de la couche de classification
            dnn[-1]['W'] -= learning_rate * dW
            dnn[-1]['b'] -= learning_rate * db

            #Rétropropagation à travers les couches RBM
            for j in range(len(dnn) - 2, -1, -1):
                A= sorties[j+1]
                dZ = dA * A * (1 - A)

                dW = sorties[j].t() @ dZ / batch_size
                db = torch.mean(dZ, dim=0)
                dA = dZ @ dnn[j]['W'].t()

                dnn[j]['W'] -= learning_rate * dW
                dnn[j]['b'] -= learning_rate * db
        
        probas_totales = entree_sortie_reseau(dnn, data)[-1]
        loss = -torch.mean(torch.sum(labels * torch.log(probas_totales + 1e-8), dim=1)).item()
        print(f'Epoch {epoch+1}/{epochs}.\n Loss: {loss:.4f}')

    return dnn

def test_DNN(dnn, data, labels):
    probas = entree_sortie_reseau(dnn, data)[-1]
    predictions = torch.argmax(probas, dim=1)
    labels_indices = torch.argmax(labels, dim=1)
    taux_erreur = (predictions != labels_indices).float().mean().item()

    print(f'Taux d\'erreur sur le test : {taux_erreur:.4f}')

    return taux_erreur


