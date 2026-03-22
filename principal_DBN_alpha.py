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
    # On isole le RBM supérieur pour l'échantillonnage de Gibbs
    top_rbm = dbn[-1]
    
    for _ in range(n_images):
        # Initialisation aléatoire sur la couche visible du RBM supérieur
        v = torch.rand(1, top_rbm['W'].shape[0])
        
        # 1. Échantillonnage de Gibbs sur le RBM supérieur
        for _ in range(n_iterations):
            h_prob = rbm_module.entree_sortie_RBM(v, top_rbm)
            h_state = (h_prob > torch.rand_like(h_prob)).float()
            
            v_prob = rbm_module.sortie_entree_RBM(h_state, top_rbm)
            v = (v_prob > torch.rand_like(v_prob)).float()
        
        # On parcourt le réseau à l'envers, en excluant le top_rbm déjà traité
        for rbm in reversed(dbn[:-1]):
            v_prob = rbm_module.sortie_entree_RBM(v, rbm)
            v = (v_prob > torch.rand_like(v_prob)).float()
        
        images.append(v.reshape(20, 16))
        
    return images