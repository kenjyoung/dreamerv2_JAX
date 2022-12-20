import jax as jx
import jax.numpy as jnp
from tree_utils import tree_stack

# Non JAX implementations of scan and cond to use for non JAX envs.
def nonjax_scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, tree_stack(ys)

def nonjax_cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)

########################################################################
# Define agent environment interaction loop.
########################################################################

def get_agent_environment_interaction_loop_function(F, iterations, config):
    # Choose whether to use jax implementations of scan and cond here based on whether the environment is written in JAX
    if(config.jax_env):
        scan = jx.lax.scan
        cond = jx.lax.cond
    else:
        scan = nonjax_scan
        cond = nonjax_cond

    def agent_environment_interaction_loop_function(S):
        def train_act_loop(S,data):
            def act_loop(S, data):
                model_params = F.get_model_params(S.model_opt_state)
                pi_params = F.get_pi_params(S.pi_opt_state)
                obs = F.env.get_observation(S.env_state)

                # Choose action
                random_action = S.env_t<config.training_start_time
                S.key, subkey = jx.random.split(S.key)
                action, S.h, S.phi = F.act(pi_params, model_params['recurrent'], model_params['phi'], obs, S.phi, S.h, subkey, random_action)

                # Step environment
                S.key, subkey = jx.random.split(S.key)
                last_obs = obs
                S.env_state, obs, reward, terminal, _ = F.env.step(subkey, S.env_state, action)

                # Update buffer
                # Note: buffer contains observation-action pairs and following reward and terminal
                S.buffer_state = F.buffer.add(S.buffer_state, last_obs, action, reward, terminal)

                S.episode_return += reward

                S.last_reward = reward
                S.last_terminal = terminal

                # Compute desired output metrics
                M = {}
                M["return"] = S.episode_return
                M["episode_complete"] = terminal

                # Reset things whenever termination occurs
                def reset(S):
                    S.last_reward = 0.0
                    S.last_terminal = False

                    # Reset env
                    S.key, subkey = jx.random.split(S.key)
                    S.env_state, _ = F.env.reset(subkey)

                    # Reset latent state on termination
                    S.h = jnp.zeros(S.h.shape)
                    S.phi = jnp.zeros(S.phi.shape)

                    # Reset episode returns to zero
                    S.episode_return = 0.0
                    return S
                S = cond(terminal, reset, lambda S: S, S)

                S.env_t += 1
                return S, M
            S, M = scan(act_loop, S, None, length=config.train_frequency)

            def train(S):
                model_params = F.get_model_params(S.model_opt_state)
                pi_params = F.get_pi_params(S.pi_opt_state)
                V_params = F.get_V_params(S.V_opt_state)
                # Sample from buffer
                S.key, subkey = jx.random.split(S.key)
                if(config.maximize_nonterminal):
                    sample = F.buffer.sample_sequences_maximize_nonterminal(S.buffer_state, config.batch_size, config.sequence_length, subkey)
                else:
                    sample = F.buffer.sample_sequences(S.buffer_state, config.batch_size, config.sequence_length, subkey)

                # Update model and compute latent states
                S.key, subkey = jx.random.split(S.key)
                subkeys = jx.random.split(subkey, num=config.batch_size)
                model_grads, (phis, hs) = F.model_grad_and_latents(model_params, subkeys, *sample)
                S.model_opt_state = F.model_update(S.opt_t, model_grads, S.model_opt_state)

                # returned phis and hs have shape [batch_size, sequence_length,...], flatten batch and sequence together
                phis = jnp.reshape(phis, [config.sequence_length*config.batch_size]+list(phis.shape[2:]))
                hs = jnp.reshape(hs, [config.sequence_length*config.batch_size]+list(hs.shape[2:]))

                # Update actor and critic
                S.key, subkey = jx.random.split(S.key)
                subkeys = jx.random.split(subkey, num=config.batch_size*config.sequence_length)
                pi_grads, V_grads = F.AC_grads(pi_params, V_params, S.V_target_params, model_params, subkeys, phis, hs)
                S.pi_opt_state = F.pi_update(S.opt_t, pi_grads, S.pi_opt_state)
                S.V_opt_state = F.V_update(S.opt_t, V_grads, S.V_opt_state)

                # Sync V_target params periodically
                S.V_target_params = jx.tree_map(lambda x,y: jnp.where(S.opt_t%config.target_update_frequency==0,x,y),F.get_V_params(S.V_opt_state), S.V_target_params)

                S.opt_t+=1
                return S
            # Train every training frequency steps if time is greater than training start time
            S = cond(S.env_t>=config.training_start_time, train, lambda S: S, S)
            return S, M
        S, M = scan(train_act_loop, S, None, length=config.eval_frequency//config.train_frequency)
        #aggregate metrics from seperate calls to act_loop
        M = jx.tree_map(lambda x: x.reshape([x.shape[0]*x.shape[1]]+list(x.shape[2:])), M)
        return S, M
    return agent_environment_interaction_loop_function
