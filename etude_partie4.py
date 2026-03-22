import matplotlib.pyplot as plt
import torch
import principal_RBM_alpha as rbm_module
import principal_DBN_alpha as dbn_module

indices_caracteres = [10, 11, 12] 
taille_reseau_dbn = [320, 200, 100]
epochs = 100
learning_rate = 0.1
batch_size = 10
n_iterations_gibbs = 1000
n_images_a_generer = 5

print("Chargement des données...")
data = rbm_module.lire_alpha_digits(indices_caracteres)

print("\n--- Entraînement du RBM ---")
rbm = rbm_module.init_RBM(n_visible=320, n_hidden=200)
rbm = rbm_module.train_RBM(rbm, epochs, learning_rate, batch_size, data)

print("Génération des images par le RBM...")
images_rbm = rbm_module.generer_image_RBM(rbm, n_iterations=n_iterations_gibbs, n_images=n_images_a_generer)

fig, axes = plt.subplots(1, n_images_a_generer, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(images_rbm[i].numpy(), cmap='gray')
    ax.axis('off')
plt.suptitle("Images générées par le RBM")
plt.show()

print("\n--- Entraînement du DBN ---")
dbn = dbn_module.init_DBN(taille_reseau_dbn)
dbn = dbn_module.train_DBN(dbn, epochs, learning_rate, batch_size, data)

print("Génération des images par le DBN...")
images_dbn = dbn_module.generer_image_DBN(dbn, n_iterations=n_iterations_gibbs, n_images=n_images_a_generer)

fig, axes = plt.subplots(1, n_images_a_generer, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(images_dbn[i].numpy(), cmap='gray')
    ax.axis('off')
plt.suptitle("Images générées par le DBN")
plt.show()