import tensorflow as tf

def constr_phi_2D(center,scale):
    """
    constructs function phi for an unnormalized 2D Gaussian
    center: 2 x instances
    scale: 2 x instances
    """
    cen_x, cen_y = tf.split(center,2,axis=0) # 1 x instances, 1 x instances
    scale = tf.exp(scale*10) # taken from the neven implementation
    sca_x, sca_y = tf.split(scale,2,axis=0)  # 1 x instances, 1 x instances
    def phi(e):
        # e : pixels x 2 or h x w x 2
        e_x,e_y = tf.split(e,2,axis=-1) # pixels x 1, pixels x 1 <- pixels x 2
        res_x = tf.square(tf.add(e_x,-cen_x))/(2*tf.square(tf.squeeze(sca_x))+1e-8) # pixels x inst 
        res_y = tf.square(tf.add(e_y,-cen_y))/(2*tf.square(tf.squeeze(sca_y))+1e-8) # pixels x inst
        return tf.exp(-res_x - res_y) # pixels x inst
    return phi

@tf.function
def joint_loss_single(embeddings, sigma, correct_label):
    """
    2D version
    embeddings: 2 x h x w
    sigma :  2 x h x w
    correct label:  1 x h x w
    """
    # prepare and reshape input
    num_dims = tf.rank(embeddings)-1 # 2
    reshaped_correct_label = tf.reshape(correct_label, [-1]) # pixels 
    unique_labels,_,_ = tf.unique_with_counts(reshaped_correct_label)
    num_instances = tf.size(unique_labels) # including background instance
    correct_label_oh = tf.one_hot(tf.cast(reshaped_correct_label,tf.int32), num_instances) # pixels x instances
    reshaped_sigma = tf.transpose(tf.reshape(sigma, [num_dims,-1])) # pixels x sigma_dim
    reshaped_embeddings = tf.transpose(tf.reshape(embeddings, [num_dims,-1])) # pixels x 2
    # Calculate mean embeddings mu for each instance
    e_seg_sum = tf.matmul(tf.transpose(reshaped_embeddings), correct_label_oh) # 2 x inst  
    e_mu = e_seg_sum/tf.reduce_sum(correct_label_oh,axis=0) # 2 x instances
    # Calculate mean sigma for each instance
    sigma_seg_sum = tf.matmul(tf.transpose(reshaped_sigma), correct_label_oh)
    sigma_mu = sigma_seg_sum/tf.reduce_sum(correct_label_oh,axis=0) # sigma_dim x instances
    # constr_phi_2D was located here
    phi = constr_phi_2D(e_mu, sigma_mu)
    phi_e_reshaped = phi(reshaped_embeddings) # pixels x instances
    correct_label_oh = tf.one_hot(tf.cast(reshaped_correct_label,tf.int32), num_instances) # pixels x instances    ## DOUBLE
    # lovasz loss
    label_phi = tf.stack([correct_label_oh[:,1:], phi_e_reshaped[:,1:]], axis=0) # 2 x pixels x instances # background excluded by [:,1:]
    label_phi = tf.transpose(label_phi, [2,0,1]) # instances x 2 x pixels
    lovasz_loss = tf.reduce_sum(tf.map_fn(map_lovasz, label_phi)) # maps over instances
    ### smoothness loss punishes intra instance sigma variance
    # sigma_i has only non-zero values where it belongs to the instance
    # inst x pixels x sigma_dim = inst x pixels x 1 * 1 x pixels x sigma_dim 
    sigma_i = tf.expand_dims(tf.transpose(correct_label_oh[:,1:],[1,0]),2) * tf.expand_dims(reshaped_sigma,0)  # check smooth loss again
    # pixels x inst x sigma_dim = pixels x instances x 1 * 1 x instances x sigma_dim
    sigma_mu_ = tf.multiply(tf.expand_dims(correct_label_oh[:,1:],2),tf.transpose(sigma_mu)[tf.newaxis,1:,:])
    sigma_mu_ = tf.transpose(sigma_mu_, [1,0,2]) # pixels x inst x sigma_dim -> inst x pixels x sigma_dim
    #
    smooth = sigma_i - tf.stop_gradient(sigma_mu_) # treat sigma_mu as constant
    smooth = tf.square(tf.norm(smooth, 'euclidean', axis =-1))
    smooth_loss = tf.cast(1/num_instances,tf.float32) * tf.reduce_sum(smooth)
    phi_e = phi(tf.transpose(embeddings,[1,2,0]))
    phi_e = tf.transpose(phi_e, [2,0,1])[1:,:,:] # instances x h x w # except background
    return lovasz_loss, smooth_loss, phi_e

@tf.function
def seed_loss_single(seed_map, phi_e, correct_label):
    """
    2D version
    seed_map : 1 x h x w
    correct_ : 1 x h x w
    phi_e    : instances x h x w
    """
    num_dims = tf.rank(phi_e)-1 # 2
    shape = tf.shape(seed_map) # 1 x h x w
    unique_labels, idx = tf.unique(tf.reshape(correct_label,[-1]))
    num_instances = tf.size(unique_labels)
    #
    foreground = tf.squeeze(tf.one_hot(tf.cast(correct_label,tf.int32),num_instances, axis=0)) # instances x h x w 
    foreground_2d = tf.reduce_max(foreground[1:,:,:], axis=0) # h x w
    background_2d = foreground[0,:,:] # h x w 
    foreground = foreground[1:,:,:] # remove background dim
    # fg loss
    res = foreground_2d * seed_map # h x w
    res = res - tf.stop_gradient(tf.reduce_max(phi_e, axis=0)) # h x w # treat phi_e as constant
    seed_loss_fg = tf.square(res)
    # bg loss and sum
    seed_loss_bg = background_2d*tf.square(seed_map)
    seed_loss = seed_loss_fg + seed_loss_bg 
    seed_loss = tf.reduce_mean(seed_loss)  
    return seed_loss


def map_lovasz(label_phi):
    label_temp,phi_temp = tf.split(label_phi,2,axis=0)
    lovasz = lovasz_hinge(logits=phi_temp*2-1, labels=label_temp, per_image=True)
    return lovasz

# --------------------------- LOVASZ LOSSES ---------------------------
# Code taken from https://github.com/bermanmaxim/LovaszSoftmax/tree/master/tensorflow

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss
    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels
