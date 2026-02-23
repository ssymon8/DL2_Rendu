import principal_RBM_alpha as rbm_module
import torch

def init_DBN(taille_reseau= [320, 100]):
    dbn = []
    for i in range(len(taille_reseau)-1):
        dbn.append(rbm_module.init_RBM(n_visible= taille_reseau[i], n_hidden=taille_reseau[i+1]))
    return dbn

def train_DBN(dbn, epochs, learning_rate, batch_size, data):
    input_data = data
    for i, rbm in enumerate(dbn):
        print(f"entrainement de la couche RBM {i+1}/{len(dbn)}")
        dbn[i] = rbm_module.train_RBM(rbm, epochs, learning_rate, batch_size, input_data)
        input_data = rbm_module.entree_sortie_RBM(input_data, dbn[i])
    return dbn

def generer_image_DBN(dbn, n_iterations=1000, n_images=1):
    images = []
    for _ in range(n_images):
        visible = torch.rand(1, dbn[0]['W'].shape[0])
        for rbm in dbn:
            for _ in range(n_iterations):
                hidden_probabilities = rbm_module.entree_sortie_RBM(visible, rbm)
                hidden_states = (hidden_probabilities > torch.rand_like(hidden_probabilities)).float()
                visible_probabilities = rbm_module.sortie_entree_RBM(hidden_states, rbm)
                visible = (visible_probabilities > torch.rand_like(visible_probabilities)).float()
        images.append(visible)
    return images