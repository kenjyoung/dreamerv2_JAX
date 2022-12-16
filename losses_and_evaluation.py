import jax.numpy as jnp
import jax as jx
from jax import vmap
import probability as prob
from types import SimpleNamespace

########################################################################
# Define Loss Functions.
########################################################################

#This returns the model loss along with intermediate latent states which are then reused to initialize the actor-critic training
def get_model_loss_and_latents_function(funcs, image_state, num_actions, config):
    if(config.latent_type=='gaussian'):
        latent_KL = prob.gaussian_KL
        # Posterior entropy regularizes toward univariate Gaussian distribution
        base_dist ={'mu':jnp.zeros(config.num_features), 'sigma':jnp.ones(config.num_features)}
    elif(config.latent_type=='categorical'):
        latent_KL = prob.categorical_KL
        base_logits = jnp.zeros((config.num_features, config.feature_width))
        base_probs = jx.nn.softmax(base_logits)
        base_log_probs = jx.nn.log_softmax(base_logits)
        # Posterior entropy regularizes toward uniform distribution (equivalent to just entropy regularization)
        base_dist = {'probs':base_probs, 'log_probs':base_log_probs}
    else:
        raise ValueError('Unrecognized latent type.')

    def model_loss_and_latents_function(params, key, observations, actions, rewards, terminals):
        recurrent_params = params['recurrent']
        phi_params = params['phi']
        next_phi_params = params['next_phi']
        reward_params = params['reward']
        termination_params = params['termination']
        state_params = params['state']

        recurrent_func = funcs['recurrent']
        phi_func = funcs['phi']
        next_phi_func = funcs['next_phi']
        reward_func = funcs['reward']
        termination_func = funcs['termination']
        state_func = funcs['state']

        # initialize hidden state for recurrent func
        h = jnp.zeros((config.num_hidden_units))
        loss = 0.0
        # record whether trajectory has terminated
        terminated = False
        nonterminal_steps = 0

        # generate sequential predictions
        def model_loss_loop_function(carry, data):
            h, loss, key, terminated, nonterminal_steps = carry
            observation, action, reward, terminal = data

            key, subkey = jx.random.split(key)
            phi, phi_dist = phi_func(phi_params, observation, h, subkey)

            S_hat_params = state_func(state_params, phi, h)
            if(config.binary_state):
                S_log_probs = prob.log_binary_probability(observation, S_hat_params)
            else:
                S_log_probs = prob.log_gaussian_probability(observation, S_hat_params)
            log_P_S = jnp.sum(S_log_probs)
            state_prediction_loss = -log_P_S

            key, subkey = jx.random.split(key)
            phi_hat, phi_hat_dist = next_phi_func(next_phi_params, h, subkey)
            # KL loss applied to make current phi closer to prediction
            KL_posterior_loss = jnp.sum(latent_KL(phi_dist,jx.lax.stop_gradient(phi_hat_dist)))

            posterior_entropy_loss = jnp.sum(latent_KL(phi_dist,base_dist))

            # KL loss applied to make prediction closer to current phi
            KL_prior_loss = jnp.sum(latent_KL(jx.lax.stop_gradient(phi_dist),phi_hat_dist))

            # h = jnp.where(terminal, jnp.zeros((config.num_hidden_units)), recurrent_func(recurrent_params,phi,jnp.eye(num_actions)[action],h))
            h = recurrent_func(recurrent_params,phi,jnp.eye(num_actions)[action],h)

            r_dist = reward_func(reward_params, phi, h)
            reward_loss = -prob.log_gaussian_probability(reward, r_dist)

            gamma_dist = termination_func(termination_params, phi, h)
            termination_loss = -prob.log_binary_probability(jnp.logical_not(terminal), gamma_dist)

            step_loss = (config.KL_posterior_weight*KL_posterior_loss+
                    config.KL_prior_weight*KL_prior_loss+
                    config.posterior_entropy_weight*posterior_entropy_loss+
                    config.reward_weight*reward_loss+
                    config.termination_weight*termination_loss+
                    config.state_prediction_weight*state_prediction_loss)

            # no need to predict anything that occurs after termination
            step_loss = jnp.where(terminated, 0.0, step_loss)

            loss += jnp.sum(step_loss)

            # Reset h on termination
            h = jnp.where(terminal, jnp.zeros((config.num_hidden_units)), h)

            nonterminal_steps+=jnp.logical_not(terminated)

            terminated = jnp.logical_or(terminated, terminal)
            return (h, loss, key, terminated, nonterminal_steps), (phi, h)

        (h, loss, key, terminated, nonterminal_steps), (phis, hs) =\
            jx.lax.scan(model_loss_loop_function, (h, loss, key, terminated, nonterminal_steps), (observations, actions, rewards, terminals))
        return loss/nonterminal_steps, phis, hs
    return model_loss_and_latents_function

def get_AC_loss_function(pi_func, V_func, model_funcs, num_actions, config):
    def AC_loss(pi_params, V_params, V_target_params, model_params, key, phi, h):
        reward_params = model_params['reward']
        recurrent_params = model_params['recurrent']
        termination_params = model_params['termination']
        next_phi_params = model_params['next_phi']

        reward_func = model_funcs['reward']
        recurrent_func = model_funcs['recurrent']
        termination_func = model_funcs['termination']
        next_phi_func = model_funcs['next_phi']

        def model_trajectory_loop_function(carry, data):
            h, phi, key = carry

            curr_V = V_func(V_params, phi, h)
            curr_V_target = V_func(V_target_params, phi, h)

            curr_pi_logit = pi_func(pi_params, phi, h)

            key, subkey = jx.random.split(key)
            action = jx.random.categorical(subkey, curr_pi_logit)
            one_hot_action = jnp.eye(num_actions)[action]

            h = recurrent_func(recurrent_params,phi,one_hot_action,h)

            reward_dist = reward_func(reward_params, phi, h)
            reward = reward_dist['mu']

            gamma_dist = termination_func(termination_params, phi, h)
            gamma = jnp.exp(prob.log_binary_probability(True, gamma_dist))*config.discount

            key, subkey = jx.random.split(key)
            phi, _ = jx.lax.stop_gradient(next_phi_func(next_phi_params, h, subkey))

            return (h,phi,key), (curr_pi_logit, curr_V, curr_V_target, one_hot_action, gamma, reward)

        # gather model trajectory
        (h,phi,key), (pi_logits, Vs, target_Vs, actions, gammas, rewards) = jx.lax.scan(model_trajectory_loop_function, (h, phi, key),jnp.arange(config.rollout_length))

        # Compute final value estimate to initialize lambda return
        curr_V_target = V_func(V_target_params, phi, h)

        def compute_loss_loop_function(carry, data):
            G, loss = carry
            pi_logit, V, target_V, action, gamma, reward = data

            # Reward and gamma are those that follow from current action, thus added here
            G = reward+gamma*G

            critic_loss = jnp.mean(0.5*(G-V)**2)
            entropy = -jnp.sum(jx.nn.log_softmax(pi_logit)*jx.nn.softmax(pi_logit))

            actor_loss = jnp.mean(-0.5*jx.lax.stop_gradient(G-target_V)*jnp.sum(jx.nn.log_softmax(pi_logit)*action)-config.beta*entropy)

            loss += jnp.mean(critic_loss+actor_loss)

            #values are associated with states in which current action is executed, thus added for previous actions lambda return
            G = ((1-config.lmbda)*jx.lax.stop_gradient(target_V)+config.lmbda*G)

            return (G, loss), None

        loss = 0.0
        #initialize G with final state value after trajectory
        G = jx.lax.stop_gradient(curr_V_target)
        # process model trajectory in reverse
        (G, loss), _ = jx.lax.scan(compute_loss_loop_function, (G, loss), (pi_logits, Vs, target_Vs, actions, gammas, rewards), reverse=True)

        return loss/config.rollout_length
    return AC_loss


########################################################################
# Define model evaluation function.
########################################################################

def get_model_eval_function(model_funcs, buffer, get_model_params, image_state, num_actions, config):
    if(config.latent_type=='gaussian'):
        latent_entropy = prob.gaussian_entropy
        latent_cross_entropy = prob.gaussian_cross_entropy
        # base_dist ={'mu':jnp.zeros(config.num_features), 'sigma':jnp.ones(config.num_features)}
    elif(config.latent_type=='categorical'):
        latent_entropy = prob.categorical_entropy
        latent_cross_entropy = prob.categorical_cross_entropy
        # base_logits = jnp.zeros((config.num_features, config.feature_width))
        # base_probs = jx.nn.softmax(base_logits)
        # base_log_probs = jx.nn.log_softmax(base_logits)
        # base_dist = {'probs':base_probs, 'log_probs':base_log_probs}
    else:
        raise ValueError('Unrecognized latent type.')

    def model_eval(params, key, observations, actions, rewards, terminals):
        recurrent_params = params['recurrent']
        phi_params = params['phi']
        next_phi_params = params['next_phi']
        reward_params = params['reward']
        termination_params = params['termination']
        state_params = params['state']

        recurrent_func = model_funcs['recurrent']
        phi_func = model_funcs['phi']
        next_phi_func = model_funcs['next_phi']
        reward_func = model_funcs['reward']
        termination_func = model_funcs['termination']
        state_func = model_funcs['state']

        r_0_count = 0
        r_1_count = 0

        gamma_0_count = 0
        gamma_1_count = 0

        gamma_hat_0_tot = 0.0
        gamma_hat_1_tot = 0.0
        r_hat_0_tot = 0.0
        r_hat_1_tot = 0.0

        phi_mean_cross_entropy = 0.0
        phi_mean_entropy = 0.0

        S_mean_logprob_tot = 0.0
        S_nonzero_tot = 0

        nonterminal_steps = 0

        # initialize hidden state for recurrent network
        h = jnp.zeros(config.num_hidden_units)
        # record whether trajectory has terminated
        terminated = False

        def evaluate_model_loop_function(C, data):
            observation, action, reward, terminal = data

            C.key, subkey = jx.random.split(C.key)
            phi, phi_dist = phi_func(phi_params, observation, C.h, subkey)

            S_hat_params = state_func(state_params, phi, C.h)
            if(config.binary_state):
                S_log_probs = prob.log_binary_probability(observation, S_hat_params)
            else:
                S_log_probs = prob.log_gaussian_probability(observation, S_hat_params)

            log_P_S = jnp.mean(S_log_probs)
            S_nonzero = jnp.mean(observation)
            C.S_mean_logprob_tot += jnp.where(terminated, 0.0,log_P_S)
            C.S_nonzero_tot += jnp.where(terminated, 0.0,S_nonzero)

            C.key, subkey = jx.random.split(C.key)
            phi_hat, phi_hat_dist = next_phi_func(next_phi_params, C.h, subkey)

            C.phi_mean_cross_entropy += jnp.sum(jnp.where(terminated, 0.0,jnp.mean(latent_cross_entropy(phi_dist,phi_hat_dist))))
            C.phi_mean_entropy += jnp.sum(jnp.where(terminated, 0.0, jnp.mean(latent_entropy(phi_dist))))

            C.r_1_count += jnp.sum(jnp.where(terminated, 0.0, reward==1.0))
            C.r_0_count += jnp.sum(jnp.where(terminated, 0.0, reward==0.0))

            # C.h = jnp.where(terminal, jnp.zeros((config.num_hidden_units)), recurrent_func(recurrent_params,phi,jnp.eye(num_actions)[action],C.h))
            C.h = recurrent_func(recurrent_params,phi,jnp.eye(num_actions)[action],C.h)

            r_dist = reward_func(reward_params, phi, C.h)
            r_hat = r_dist['mu']

            gamma_dist = termination_func(termination_params, phi, C.h)
            gamma_hat = jnp.exp(prob.log_binary_probability(1.0,gamma_dist))

            r_hat_1 = jnp.where(reward==1.0, r_hat, 0.0)
            r_hat_0 = jnp.where(reward==0.0, r_hat, 0.0)
            C.r_hat_1_tot += jnp.sum(jnp.where(terminated, 0.0, r_hat_1))
            C.r_hat_0_tot += jnp.sum(jnp.where(terminated, 0.0, r_hat_0))

            C.gamma_1_count += jnp.sum(jnp.where(terminated, 0.0, jnp.logical_not(terminal)))
            C.gamma_0_count += jnp.sum(jnp.where(terminated, 0.0, terminal))

            gamma_hat_1 = jnp.where(jnp.logical_not(terminal), gamma_hat, 0.0)
            gamma_hat_0 = jnp.where(terminal, gamma_hat,0.0)
            C.gamma_hat_1_tot += jnp.sum(jnp.where(terminated, 0.0, gamma_hat_1))
            C.gamma_hat_0_tot += jnp.sum(jnp.where(terminated, 0.0, gamma_hat_0))

            # Reset h on termination
            C.h = jnp.where(terminal, jnp.zeros((config.num_hidden_units)), C.h)

            C.nonterminal_steps += jnp.logical_not(terminated)

            C.terminated = jnp.logical_or(C.terminated, terminal)

            return C, None

        key, subkey = jx.random.split(key)

        var_dict = locals()
        carry_names = ['h','S_mean_logprob_tot','S_nonzero_tot','phi_mean_cross_entropy','phi_mean_entropy',
            'r_1_count','r_0_count','r_hat_1_tot','r_hat_0_tot','gamma_1_count','gamma_0_count','gamma_hat_1_tot',
            'gamma_hat_0_tot','terminated','nonterminal_steps']
        carry = {name:var_dict[name] for name in carry_names}
        key, subkey = jx.random.split(key)
        carry["key"] = subkey
        carry = SimpleNamespace(**carry)

        C, _ = jx.lax.scan(evaluate_model_loop_function, carry, (observations, actions, rewards, terminals))

        metrics={'gamma_0_tot' : C.gamma_hat_0_tot, 'gamma_1_tot' : C.gamma_hat_1_tot, 'r_0_tot' : C.r_hat_0_tot, 'r_1_tot' : C.r_hat_1_tot,
            'gamma_0_count': C.gamma_0_count, 'gamma_1_count': C.gamma_1_count, 'r_0_count': C.r_0_count, 'r_1_count': C.r_1_count,
            'phi_cross_entropy' : C.phi_mean_cross_entropy, 'phi_entropy' : C.phi_mean_entropy, 'S_logprob_tot' : C.S_mean_logprob_tot,
            'S_nonzero_tot' : C.S_nonzero_tot, 'nonterminal_steps' : C.nonterminal_steps }
        return metrics

    def batch_model_eval(buffer_state, model_opt_state, key):
        key, subkey = jx.random.split(key)
        if(config.maximize_nonterminal):
            sample = buffer.sample_sequences_maximize_nonterminal(buffer_state, config.batch_size, config.sequence_length, subkey)
        else:
            sample = buffer.sample_sequences(buffer_state, config.batch_size, config.sequence_length, subkey)
        key, subkey = jx.random.split(key)
        subkeys = jx.random.split(subkey, num=config.batch_size)
        metrics = vmap(model_eval, in_axes=(None, 0, 0, 0, 0, 0))(get_model_params(model_opt_state), subkeys, *sample)
        nonterminal_steps = jnp.sum(metrics["nonterminal_steps"])
        r_0_count = jnp.sum(metrics["r_0_count"])
        r_1_count = jnp.sum(metrics["r_1_count"])
        gamma_0_count = jnp.sum(metrics["gamma_0_count"])
        gamma_1_count = jnp.sum(metrics["gamma_1_count"])
        gamma_0_tot = jnp.sum(metrics["gamma_0_tot"])
        gamma_1_tot = jnp.sum(metrics["gamma_1_tot"])
        r_0_tot = jnp.sum(metrics["r_0_tot"])
        r_1_tot = jnp.sum(metrics["r_1_tot"])
        S_logprob_tot = jnp.sum(metrics["S_logprob_tot"])
        S_nonzero_tot = jnp.sum(metrics["S_nonzero_tot"])
        phi_cross_entropy = jnp.sum(metrics["phi_cross_entropy"])
        phi_entropy = jnp.sum(metrics["phi_entropy"])

        # Note: r_1 and r_0 predictions are useful in MinAtar in particular because rewards are almost always 1 or 0
        # thus we can observe how accurate the model is for each case
        combined_metrics={'gamma_0_pred' : gamma_0_tot/gamma_0_count,'gamma_1_pred' : gamma_1_tot/gamma_1_count,'r_0_pred' : r_0_tot/r_0_count,
            'r_1_pred' : r_1_tot/r_1_count,'gamma_0_frac': gamma_0_count/nonterminal_steps,'gamma_1_frac': gamma_1_count/nonterminal_steps,
            'r_0_frac': r_0_count/nonterminal_steps,'r_1_frac': r_1_count/nonterminal_steps,'phi_cross_entropy' : phi_cross_entropy/nonterminal_steps,
            'phi_entropy' : phi_entropy/nonterminal_steps,'S_logprob' : S_logprob_tot/nonterminal_steps,'S_nonzero_tot' : S_nonzero_tot/nonterminal_steps}
        return combined_metrics
    return batch_model_eval
