import jax as jx
import jax.numpy as jnp
import haiku as hk

########################################################################
# All the neural networks used in this implementation.
########################################################################

activation_dict = {"silu":jx.nn.silu, "elu": jx.nn.elu}
std_activation_dict = {"softplus": jx.nn.softplus, "sigmoid": jx.nn.sigmoid, "sigmoid2": lambda x: 2*jx.nn.sigmoid(x/2)}

class recurrent_func(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, phi, a, h):
        x = jnp.concatenate([jnp.ravel(phi), a])
        #GRU returns a 2-tuple but I belive both elements are the same, just return the next hidden state
        return hk.GRU(self.num_hidden_units)(x,h)[1]

class reward_func(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.learn_reward_variance = config.learn_reward_variance

    def __call__(self, phi, h, key=None):
        x = jnp.concatenate([jnp.ravel(phi),h])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        mu = hk.Linear(1)(x)[0]
        if(self.learn_reward_variance):
            sigma = jx.nn.softplus(hk.Linear(1)(x))[0]
        else:
            sigma = jnp.ones(mu.shape)
        return {'mu':mu, 'sigma':sigma}

class termination_func(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, phi, h, key=None):
        x = jnp.concatenate([jnp.ravel(phi),h])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(1)(x)[0]
        return {'logit':logit}

class next_phi_func(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_features = config.num_features
        self.num_hidden_units = config.num_hidden_units
        self.feature_width = config.feature_width
        self.activation_function = activation_dict[config.activation]
        self.latent_type = config.latent_type

        if(self.latent_type=='gaussian'):
            self.std_activation_function = std_activation_dict[config.std_act]
            self.min_std = config.min_std

    def __call__(self, h, key):
        x = h
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))+self.min_std

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(jnp.ravel(x)), [self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=1)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class phi_conv_func(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_features = config.num_features
        self.num_hidden_layers = config.num_hidden_layers
        self.feature_width = config.feature_width
        self.conv_depth = config.conv_depth
        self.num_conv_filters = config.num_conv_filters
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.latent_type = config.latent_type

        if(self.latent_type=='gaussian'):
            self.std_activation_function = std_activation_dict[config.std_act]
            self.min_std = config.min_std

    def __call__(self, s, h, key):
        #encode image
        x = s
        for i in range(self.conv_depth):
            x = self.activation_function(hk.Conv2D(self.num_conv_filters*(2**i), 3, padding='VALID')(x))

        #combine image and recurrent state
        x = jnp.concatenate([h,jnp.ravel(x)])

        #pass both through a MLP
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))+self.min_std

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(jnp.ravel(x)), [self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=1)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class phi_flat_func(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_features = config.num_features
        self.feature_width = config.feature_width
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.latent_type = config.latent_type

        if(self.latent_type=='gaussian'):
            self.std_activation_function = std_activation_dict[config.std_act]
            self.min_std = config.min_std

    def __call__(self, s, h, key):
        x = jnp.concatenate([h,jnp.ravel(s)])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))+self.min_std

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(jnp.ravel(x)), [self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=1)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class state_conv_func(hk.Module):
    def __init__(self, config, state_shape, name=None):
        super().__init__(name=name)
        self.num_features = config.num_features
        self.feature_width = config.feature_width
        self.conv_depth = config.conv_depth
        self.num_conv_filters = config.num_conv_filters
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.binary = config.binary_state
        self.state_shape = state_shape

    def __call__(self, phi, h):
        desired_shape = [self.state_shape[0]-2*self.conv_depth,self.state_shape[1]-2*self.conv_depth,self.num_conv_filters*(2**(self.conv_depth-1))]
        num_units = 1
        for j in desired_shape:
            num_units*=j

        x = jnp.concatenate([h,jnp.ravel(phi)])
        x = self.activation_function(hk.Linear(num_units)(x))
        x = jnp.reshape(x, [-1]+desired_shape)
        for i in range(self.conv_depth-1):
            x = self.activation_function(hk.Conv2DTranspose(self.num_conv_filters*2**(self.conv_depth-i-1), 3, padding='VALID')(x))

        if(self.binary):
            logit = hk.Conv2DTranspose(self.state_shape[2], 3, output_shape=self.state_shape[:2], padding='VALID')(x)
            #Note this returns the logit of S, we wish to apply a sigmoid after to keep it bounded
            return {'logit':logit}
        else:
            mu = hk.Conv2DTranspose(self.state_shape[2], 3, output_shape=self.state_shape[:2], padding='VALID')(x)
            return {'mu':mu, 'sigma':1.0}

class state_flat_func(hk.Module):
    def __init__(self, config, state_width, name=None):
        super().__init__(name=name)
        self.num_features = config.num_features
        self.feature_width = config.feature_width
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.binary = config.binary_state
        self.state_width = state_width

    def __call__(self, phi, h):
        x = jnp.concatenate([h,jnp.ravel(phi)])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        if(self.binary):
            logit = hk.Linear(self.state_width)(x)
            return {'logit':logit}
        else:
            mu = hk.Linear(self.state_width)(x)
            return {'mu':mu, 'sigma':1.0}

class V_func(hk.Module):
    def __init__(self, config, name=None):
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        super().__init__(name=name)

    def __call__(self, phi, h):
        x = jnp.concatenate([jnp.ravel(phi), h])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        V = hk.Linear(1)(x)[0]
        return V

class pi_func(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.num_actions = num_actions
        self.activation_function = activation_dict[config.activation]

    def __call__(self, phi, h):
        x = jnp.concatenate([h,jnp.ravel(phi), h])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        pi_logit = hk.Linear(self.num_actions)(x)
        return pi_logit
