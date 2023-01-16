import jax as jx
from jax import grad, vmap, jit
import jax.numpy as jnp
from jax.example_libraries import optimizers

import haiku as hk

import copy

from optimizers import adamw
import networks as nets

from types import SimpleNamespace

from replay_buffer import replay_buffer

from losses_and_evaluation import get_model_loss_and_latents_function, get_AC_loss_function, get_model_eval_function

########################################################################
# Initializes both state and functions used in the interaction loop.
########################################################################

def get_init_fns(env, config):
    def get_update_function(opt_update, grad_clip):
        def update(t, grads, opt_state):
            grads = optimizers.clip_grads(grads, grad_clip)
            opt_state = opt_update(t, grads, opt_state)
            return opt_state
        return update

    def init_fn(key):
        num_actions = env.num_actions()

        # Model related initialization
        #============================
        model_opt_init, model_opt_update, get_model_params = adamw(config.model_alpha, eps=config.eps_adam, wd=config.wd_adam)
        model_params = {}
        model_funcs = {}

        # dummy variables for network initialization
        key, dummy_key = jx.random.split(key)
        _, dummy_obs = env.reset(dummy_key)
        obs_shape = dummy_obs.shape

        dummy_phi = jnp.zeros((config.num_features,config.feature_width)) if config.latent_type=='categorical' else jnp.zeros((config.num_features))
        dummy_a = jnp.zeros((), dtype=int)
        dummy_r = jnp.zeros(())
        dummy_terminal = jnp.zeros((),dtype=bool)
        dummy_h = jnp.zeros((config.num_hidden_units))

        key, subkey = jx.random.split(key)
        buffer = replay_buffer(config.buffer_size, [dummy_obs, dummy_a, dummy_r, dummy_terminal], terminal_index=3)
        buffer_state = buffer.initialize()

        # initialize recurrent network
        recurrent_net = hk.without_apply_rng(hk.transform(lambda phi, a, h: nets.recurrent_func(config)(phi,a,h)))
        key, subkey = jx.random.split(key)
        recurrent_params = recurrent_net.init(subkey,dummy_phi, jnp.eye(num_actions)[dummy_a], dummy_h)
        recurrent_apply = recurrent_net.apply
        model_params['recurrent']=recurrent_params
        model_funcs['recurrent']=recurrent_apply

        # initialize phi network
        if(len(obs_shape)>1 and not config.no_conv):
            image_state = True
            phi_net = hk.without_apply_rng(hk.transform(lambda s, h, k: nets.phi_conv_func(config)(s, h, k)))
        else:
            image_state = False
            phi_net = hk.without_apply_rng(hk.transform(lambda s, h, k: nets.phi_flat_func(config)(s, h, k)))
        phi_apply = phi_net.apply
        key, subkey = jx.random.split(key)
        phi_params = phi_net.init(subkey,dummy_obs, dummy_h, dummy_key)
        model_params['phi']=phi_params
        model_funcs['phi']=phi_apply

        # initialize reward network
        reward_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.reward_func(config)(phi, h)))
        key, subkey = jx.random.split(key)
        reward_params = reward_net.init(subkey, dummy_phi, dummy_h)
        reward_apply = reward_net.apply
        model_params['reward']=reward_params
        model_funcs['reward']=reward_apply

        # initialize termination network
        termination_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.termination_func(config)(phi, h)))
        key, subkey = jx.random.split(key)
        termination_params = termination_net.init(subkey, dummy_phi, dummy_h)
        termination_apply = termination_net.apply
        model_params['termination']=termination_params
        model_funcs['termination']=termination_apply

        # initialize phi prediction network
        next_phi_net = hk.without_apply_rng(hk.transform(lambda h, k: nets.next_phi_func(config)(h, k)))
        next_phi_apply = next_phi_net.apply
        key, subkey = jx.random.split(key)
        next_phi_params = next_phi_net.init(subkey, dummy_h, dummy_key)
        model_params['next_phi']=next_phi_params
        model_funcs['next_phi']=next_phi_apply

        # initialize state reconstruction network
        if(image_state):
            state_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.state_conv_func(config, obs_shape)(phi, h)))
        else:
            state_width = 1
            for j in obs_shape:
                state_width*=j
            state_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.state_flat_func(config, state_width)(phi, h)))
        state_apply = state_net.apply
        key, subkey = jx.random.split(key)
        state_params = state_net.init(subkey, dummy_phi, dummy_h)
        model_params['state'] = state_params
        model_funcs['state'] = state_apply

        model_opt_state = model_opt_init(model_params)
        model_opt_update = model_opt_update
        model_update = get_update_function(model_opt_update, config.grad_clip)

        model_loss_and_latents = get_model_loss_and_latents_function(model_funcs, image_state, num_actions, config)

        def batch_model_loss_and_latents(*args):
            model_loss, phis, hs = vmap(model_loss_and_latents, in_axes=(None, 0, 0, 0, 0, 0))(*args)
            return jnp.mean(model_loss), (phis,hs)

        # This returns model_grads, (phis,hs)
        model_grad_and_latents = grad(lambda *args: batch_model_loss_and_latents(*args), has_aux=True)

        # AC related initialization
        #=========================
        pi_opt_init, pi_opt_update, get_pi_params = adamw(config.pi_alpha, eps=config.eps_adam, wd=config.wd_adam)
        V_opt_init, V_opt_update, get_V_params = adamw(config.V_alpha, eps=config.eps_adam, wd=config.wd_adam)

        # initialize pi network
        pi_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.pi_func(config, num_actions)(phi, h)))
        pi_func = pi_net.apply
        key, subkey = jx.random.split(key)
        pi_params = pi_net.init(subkey, dummy_phi, dummy_h)

        # initialize V network
        V_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.V_func(config)(phi, h)))
        V_func = V_net.apply
        key, subkey = jx.random.split(key)
        V_params = V_net.init(subkey ,dummy_phi, dummy_h)

        V_target_params = copy.deepcopy(V_params)

        pi_opt_state = pi_opt_init(pi_params)
        V_opt_state = V_opt_init(V_params)

        AC_loss = get_AC_loss_function(pi_func, V_func, model_funcs, env.num_actions(), config)
        AC_grads = grad(lambda *args: jnp.mean(vmap(AC_loss, in_axes=(None, None, None, None, 0, 0, 0))(*args)), argnums=(0,1))

        V_update = get_update_function(V_opt_update, config.grad_clip)
        pi_update = get_update_function(pi_opt_update, config.grad_clip)

        def act(pi_params, recurrent_params, phi_params, observation, phi, h, key, random):
            key, subkey = jx.random.split(key)
            phi, _ = model_funcs['phi'](phi_params, observation, h, subkey)

            pi_logit = pi_func(pi_params, phi, h)
            key, subkey = jx.random.split(key)
            a = jnp.where(random, jx.random.choice(subkey, num_actions), jx.random.categorical(subkey, pi_logit))

            h = model_funcs["recurrent"](recurrent_params,phi,jnp.eye(num_actions)[a],h)
            return a, h, phi

        model_eval = get_model_eval_function(model_funcs, buffer, get_model_params, image_state, num_actions, config)

        # Maintain state information for acting in the real world
        h = jnp.zeros(dummy_h.shape)
        phi = jnp.zeros(dummy_phi.shape)

        key, subkey = jx.random.split(key)
        env_state, _ = env.reset(subkey)

        episode_return = jnp.array(0.0)
        last_reward = jnp.array(0.0)
        last_terminal = jnp.array(False)
        opt_t = jnp.array(0)
        env_t = jnp.array(0)

        var_dict = locals()
        function_names = ["V_update", "pi_update", "model_update", "model_funcs", "pi_func", "V_func", "get_V_params", "get_pi_params",
            "get_model_params", "AC_grads", "model_grad_and_latents", "act", "model_eval"]
        class_names = ["buffer", "env"]
        run_state_names = ["env_state", "h", "phi", "V_opt_state", "pi_opt_state", "model_opt_state", "V_target_params", "env_t",
            "opt_t", "buffer_state", "last_reward", "last_terminal", "episode_return", "key"]
        # Contains state variables
        run_state = SimpleNamespace(**{name:var_dict[name] for name in run_state_names})

        # jit compile all returned functions
        function_dict = {name:jx.tree_map(lambda x: jit(x),var_dict[name]) for name in function_names}
        # Add classes as well (methods are already jit compiled)
        function_dict.update({name:var_dict[name] for name in class_names})

        # Contains functions and classes
        functions = SimpleNamespace(**function_dict)

        return run_state, functions

    def init_state(key):
        return init_fn(key)[0]

    def init_functions():
        #key is irrelevant for this part
        return init_fn(jx.random.PRNGKey(0))[1]
    return init_state, init_functions
