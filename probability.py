import jax.numpy as jnp
import jax as jx


########################################################################
# Probability Helper Functions
########################################################################

def log_gaussian_probability(x, params):
    mu = params['mu']
    sigma = params['sigma']
    return -(jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi) + 0.5 * ((x - mu) / sigma)**2)


def gaussian_cross_entropy(params_1, params_2):
    mu_1 = params_1['mu']
    sigma_1 = params_1['sigma']
    mu_2 = params_2['mu']
    sigma_2 = params_2['sigma']
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma_2) + (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2)


def gaussian_entropy(params):
    sigma = params['sigma']
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma)


def gaussian_KL(params_1, params_2):
    return gaussian_cross_entropy(params_1, params_2) - gaussian_entropy(params_1)


def log_binary_probability(x, params):
    logit = params['logit']
    return jnp.where(x, jx.nn.log_sigmoid(logit), jx.nn.log_sigmoid(-logit))


def binary_entropy(params):
    logit = params['logit']
    return jx.nn.sigmoid(logit) * jx.nn.log_sigmoid(logit) + jx.nn.sigmoid(-logit) * jx.nn.log_sigmoid(-logit)


def categorical_cross_entropy(params_1, params_2):
    probs_1 = params_1['probs']
    log_probs_2 = params_2['log_probs']
    return -jnp.sum(probs_1 * log_probs_2, axis=(-1))


def categorical_entropy(params):
    probs = params['probs']
    log_probs = params['log_probs']
    return -jnp.sum(probs * log_probs, axis=(-1))


def categorical_KL(params_1, params_2):
    return categorical_cross_entropy(params_1, params_2) - categorical_entropy(params_1)
