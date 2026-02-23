import principal_RBM_alpha as rbm

def init_DBN(nb_couches=1):
    dbn = []
    for _ in range(nb_couches):
        dbn.append(rbm.init_RBM(n_visible= 320, n_hidden=100))
    return dbn

def train_DBN(dbn, epochs, learning_rate, batch_size, data):
    input_data = data
    for i, rbm in enumerate(dbn):
        print(f"Training RBM layer {i+1}/{len(dbn)}")
        rbm.train_RBM(rbm, epochs, learning_rate, batch_size, input_data)
        input_data = rbm.entree_sortie_RBM(input_data, rbm)
    return dbn

def generer_image_DBN(dbn, n_iterations=1000, n_images=1):
    images = []
    for _ in range(n_images):
        visible = torch.rand(1, dbn[0]['W'].shape[0])
        for rbm in dbn:
            for _ in range(n_iterations):
                hidden_probabilities = rbm.entree_sortie_RBM(visible, rbm)
                hidden_states = (hidden_probabilities > torch.rand_like(hidden_probabilities)).float()
                visible_probabilities = rbm.sortie_entree_RBM(hidden_states, rbm)
                visible = (visible_probabilities > torch.rand_like(visible_probabilities)).float()
        images.append(visible)
    return images