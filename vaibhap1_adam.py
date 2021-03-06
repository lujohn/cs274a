
def get_AdaM_update(alpha_0, grad, adam_values, b1=.95, b2=.999, e=1e-8):
    adam_values['t'] += 1
    
    # update mean
    adam_values['mean'] = b1 * adam_values['mean'] + (1-b1) * grad
    m_hat = adam_values['mean'] / (1-b1**adam_values['t'])

    # update variance
    adam_values['var'] = b2 * adam_values['var'] + (1-b2) * grad**2
    v_hat = adam_values['var'] / (1-b2**adam_values['t'])

    return alpha_0 * m_hat/(np.sqrt(v_hat) + e)

# Initialize a dictionary that keeps track of the mean, variance, and update counter
alpha_0 = 1e-3
adam_values = \
    {'mean': np.zeros(beta.shape), 'var': np.zeros(beta.shape), 't': 0}

### Inside the training loop do ###
beta_grad = # compute gradient w.r.t. the weight vector (beta) as usual
beta_update = get_AdaM_update(alpha_0, beta_grad, adam_values)
beta += beta_update


