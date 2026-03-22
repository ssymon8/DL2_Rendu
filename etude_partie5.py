import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
import principal_DNN_MNIST as dnn_module
import copy

def lire_et_binariser_mnist(chemin='./data'):
    print("Chargement et binarisation de MNIST...")
    train_set = datasets.MNIST(root=chemin, train=True, download=True)
    test_set = datasets.MNIST(root=chemin, train=False, download=True)

    X_train = (train_set.data.view(-1, 28 * 28).float() >= 127).float()
    X_test = (test_set.data.view(-1, 28 * 28).float() >= 127).float()

    # One-Hot Encoding pour la Cross-Entropy manuelle
    Y_train = F.one_hot(train_set.targets, num_classes=10).float()
    Y_test = F.one_hot(test_set.targets, num_classes=10).float()

    return X_train, Y_train, X_test, Y_test

def evaluer_modeles(taille_reseau, X_train, Y_train, X_test, Y_test, epochs_pretrain=50, epochs_backprop=100, lr=0.1, batch_size=256):
    print(f"\n--- Évaluation pour l'architecture : {taille_reseau} ---")
    
    # 1. Modèle initialisé aléatoirement (Sans pré-entraînement)
    print("-> Entraînement du modèle SANS pré-entraînement...")
    dnn_rand = dnn_module.init_DNN(taille_reseau)
    dnn_rand = dnn_module.retropropagation(dnn_rand, epochs_backprop, lr, batch_size, X_train, Y_train)
    erreur_rand = dnn_module.test_DNN(dnn_rand, X_test, Y_test)

    # 2. Modèle pré-entraîné (Avec RBM)
    print("-> Entraînement du modèle AVEC pré-entraînement...")
    dnn_pre = dnn_module.init_DNN(taille_reseau)
    dnn_pre = dnn_module.pretrain_DNN(dnn_pre, epochs_pretrain, lr, batch_size, X_train)
    dnn_pre = dnn_module.retropropagation(dnn_pre, epochs_backprop, lr, batch_size, X_train, Y_train)
    erreur_pre = dnn_module.test_DNN(dnn_pre, X_test, Y_test)

    return erreur_rand, erreur_pre

def main():
    X_train, Y_train, X_test, Y_test = lire_et_binariser_mnist()
    
    epochs_pretrain = 50
    epochs_backprop = 100
    lr = 0.1
    batch_size = 256

    # =========================================================
    # FIG 1 : Impact du nombre de couches
    # =========================================================
    print("\n\n=== EXPÉRIENCE 1 : NOMBRE DE COUCHES ===")
    architectures_couches = [
        [784, 200, 200, 10],                     # 2 couches cachées
        [784, 200, 200, 200, 10],                # 3 couches cachées
        [784, 200, 200, 200, 200, 200, 10]       # 5 couches cachées
    ]
    nb_couches = [2, 3, 5]
    err_rand_couches, err_pre_couches = [], []

    for arch in architectures_couches:
        err_r, err_p = evaluer_modeles(arch, X_train, Y_train, X_test, Y_test, epochs_pretrain, epochs_backprop, lr, batch_size)
        err_rand_couches.append(err_r * 100)
        err_pre_couches.append(err_p * 100)

    plt.figure(figsize=(8, 5))
    plt.plot(nb_couches, err_rand_couches, marker='o', label='Initialisation aléatoire')
    plt.plot(nb_couches, err_pre_couches, marker='s', label='Pré-entraînement DBN')
    plt.title('Fig 1: Taux d\'erreur selon le nombre de couches (200 neurones/couche)')
    plt.xlabel('Nombre de couches cachées')
    plt.ylabel('Taux d\'erreur sur le test (%)')
    plt.xticks(nb_couches)
    plt.legend()
    plt.grid(True)
    plt.savefig('Fig1_couches.png')
    
    # =========================================================
    # FIG 2 : Impact du nombre de neurones
    # =========================================================
    print("\n\n=== EXPÉRIENCE 2 : NOMBRE DE NEURONES ===")
    architectures_neurones = [
        [784, 100, 100, 10],
        [784, 300, 300, 10],
        [784, 700, 700, 10]
    ]
    nb_neurones = [100, 300, 700]
    err_rand_neu, err_pre_neu = [], []

    for arch in architectures_neurones:
        err_r, err_p = evaluer_modeles(arch, X_train, Y_train, X_test, Y_test, epochs_pretrain, epochs_backprop, lr, batch_size)
        err_rand_neu.append(err_r * 100)
        err_pre_neu.append(err_p * 100)

    plt.figure(figsize=(8, 5))
    plt.plot(nb_neurones, err_rand_neu, marker='o', label='Initialisation aléatoire')
    plt.plot(nb_neurones, err_pre_neu, marker='s', label='Pré-entraînement DBN')
    plt.title('Fig 2: Taux d\'erreur selon le nombre de neurones (2 couches)')
    plt.xlabel('Nombre de neurones par couche')
    plt.ylabel('Taux d\'erreur sur le test (%)')
    plt.xticks(nb_neurones)
    plt.legend()
    plt.grid(True)
    plt.savefig('Fig2_neurones.png')

    # =========================================================
    # FIG 3 : Impact de la taille des données d'entraînement
    # =========================================================
    print("\n\n=== EXPÉRIENCE 3 : TAILLE DU JEU DE DONNÉES ===")
    tailles_train = [1000, 3000, 7000, 10000, 30000, 60000]
    arch_base = [784, 200, 200, 10]
    err_rand_data, err_pre_data = [], []

    for size in tailles_train:
        # Sous-échantillonnage des données d'entraînement
        X_train_sub = X_train[:size]
        Y_train_sub = Y_train[:size]
        
        err_r, err_p = evaluer_modeles(arch_base, X_train_sub, Y_train_sub, X_test, Y_test, epochs_pretrain, epochs_backprop, lr, batch_size)
        err_rand_data.append(err_r * 100)
        err_pre_data.append(err_p * 100)

    plt.figure(figsize=(8, 5))
    plt.plot(tailles_train, err_rand_data, marker='o', label='Initialisation aléatoire')
    plt.plot(tailles_train, err_pre_data, marker='s', label='Pré-entraînement DBN')
    plt.title('Fig 3: Taux d\'erreur selon la taille du jeu d\'entraînement')
    plt.xlabel('Nombre d\'images d\'entraînement')
    plt.ylabel('Taux d\'erreur sur le test (%)')
    plt.xscale('log') # Échelle logarithmique utile ici
    plt.xticks(tailles_train, tailles_train)
    plt.legend()
    plt.grid(True)
    plt.savefig('Fig3_donnees.png')
    
    print("\nTerminé ! Les figures ont été sauvegardées.")

if __name__ == "__main__":
    main()