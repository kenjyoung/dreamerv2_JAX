import jax as jx
import jax.numpy as jnp
from jax import jit
from jax.example_libraries import optimizers
from jax.tree_util import register_pytree_node

import json
import pickle as pkl
import argparse
import time
from tqdm import tqdm
import os

import environments
from initialization import get_init_fns
from agent_environment_interaction_loop import get_agent_environment_interaction_loop_function

import wandb

from types import SimpleNamespace

# Tell JAX how to handle SimpleNamespace as a pytree (allows for more compact notation than dicts)
def SimpleNamespace_flatten(v):
    return (v.__dict__.values(), v.__dict__.keys())

def SimpleNamespace_unflatten(aux_data, children):
    return SimpleNamespace(**{k:v for k,v in zip(aux_data, children)})

register_pytree_node(SimpleNamespace, SimpleNamespace_flatten, SimpleNamespace_unflatten)

activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--group", "-g", type=str, default=None)
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--config", "-c", type=str)
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--save_checkpoint", type=str, default="checkpoint.pkl")
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config=json.load(f)
config.update({"agent_type":"dreamerv2", "seed":args.seed})

config = SimpleNamespace(**config)

assert(config.training_start_time>config.sequence_length)

########################################################################
# Define logging and checkpointing functions.
########################################################################

def update_log_dict(d, u):
    if d is None:
        for k,v in u.items():
            if(jnp.ndim(v)==0):
                u[k] = jnp.expand_dims(v,axis=0)
        d = u
    else:
        for k,v in u.items():
            if(jnp.ndim(v)==0):
                v = jnp.expand_dims(v,axis=0)
            d[k]=jnp.concatenate([d[k],v])
    return d

def save_log(log_dicts, config):
    with open(args.output, 'wb') as f:
        data = log_dicts
        data["config"]=dict(config)
        pkl.dump(data, f)

def get_log_function(F):
    model_eval = jit(F.model_eval)

    def log(S, M, log_dicts, wallclock, key):
        curr_time = S.env_t

        # Log returns and associated times
        returns = M["return"][M["episode_complete"]]
        return_times = curr_time-config.eval_frequency+jnp.arange(config.eval_frequency)[M["episode_complete"]]
        for ret, t in zip(returns, return_times):
            wandb.log({"return":ret,"return_time":t})
        log_dicts["returns_and_times"] = update_log_dict(log_dicts["returns_and_times"],{"return":returns,"return_time":return_times})

        # Log model metrics
        key, subkey = jx.random.split(S.key)
        metrics = model_eval(S.buffer_state,S.model_opt_state,subkey)
        metrics["time"] = curr_time
        metrics["time_per_step"] = wallclock/config.eval_frequency
        wandb.log(metrics)
        log_dicts["metrics"] = update_log_dict(log_dicts["metrics"],metrics)
        return log_dicts
    return log

def save_checkpoint(run_state, log_dicts, i, wandb_id, opt_state_names):
    temp_filename = args.save_checkpoint+str(time.time())
    with open(temp_filename, 'wb') as f:
        unpacked_run_state = {}
        for k, v in run_state.__dict__.items():
            if k in opt_state_names:
                unpacked_run_state[k]=optimizers.unpack_optimizer_state(v)
            else:
                unpacked_run_state[k]=v
        pkl.dump({
            'run_state':unpacked_run_state,
            'log_dicts':log_dicts,
            'i':i,
            'wandb_id':wandb_id
        }, f)

    os.rename(temp_filename, args.save_checkpoint)

def load_checkpoint(opt_state_names):
    with open(args.load_checkpoint, 'rb') as f:
        checkpoint = pkl.load(f)
        run_state = checkpoint["run_state"]
        for k in opt_state_names:
            run_state[k] = optimizers.pack_optimizer_state(run_state[k])
        run_state = SimpleNamespace(**run_state)
        log_dicts = checkpoint["log_dicts"]
        start_i = checkpoint["i"]+1
        wandb_id = checkpoint["wandb_id"]
    return run_state, log_dicts, start_i, wandb_id

########################################################################
# Initialization
########################################################################

Environment = getattr(environments, config.environment)
env_config = config.env_config

env = Environment(**env_config)
num_actions = env.num_actions()

# Initialize run_state and functions
init_state, init_functions = get_init_fns(env, config)
key, subkey = jx.random.split(key)
run_state = init_state(subkey)
functions = init_functions()
start_i = 0
log_dicts = {"returns_and_times":None, "metrics":None}

resumed = False
opt_state_names = ["V_opt_state", "pi_opt_state", "model_opt_state"]
if(args.load_checkpoint is not None):
    if(os.path.exists(args.load_checkpoint)):
        run_state, log_dicts, start_i, wandb_id = load_checkpoint(opt_state_names)
        resumed = True
    else:
        print("Warning! load_checkpoint does not exist, starting run from scratch.")

# Resume wandb session as well if loading from checkpoint
if(resumed):
    wandb.init(config=config, resume="must", id=wandb_id, project='dreamerv2_pure_jax', group=args.group)
else:
    wandb_id = wandb.util.generate_id()
    wandb.init(config=config, id=wandb_id, project='dreamerv2_pure_jax', group=args.group)

log = get_log_function(functions)

# Build the agent environment interaction loop function
agent_environment_interaction_loop_function = get_agent_environment_interaction_loop_function(functions, config.eval_frequency, config)

# If the env itself is written in JAX we can compile the interaction loop
if(config.jax_env):
    agent_environment_interaction_loop_function = jit(agent_environment_interaction_loop_function)

time_since_checkpoint = 0
last_time = time.time()

########################################################################
# Main training loop
########################################################################

i = start_i
tqdm.write("Beginning run...")
for i in tqdm(range(start_i,config.num_steps//config.eval_frequency), initial=start_i, total=config.num_steps//config.eval_frequency):
    run_state, metrics = agent_environment_interaction_loop_function(run_state)

    ellapsed_time = time.time()-last_time
    last_time = time.time()

    run_state.key, subkey = jx.random.split(run_state.key)
    log_dicts = log(run_state, metrics, log_dicts, ellapsed_time, subkey)

    # periodically save checkpoint to disk
    time_since_checkpoint+=config.eval_frequency
    if(time_since_checkpoint>=config.checkpoint_frequency):
        save_checkpoint(run_state, log_dicts, i, wandb_id, opt_state_names)
        time_since_checkpoint = 0

# Save Data and final checkpoint
save_log(log_dicts, config)
save_checkpoint(run_state, log_dicts, i, wandb_id, opt_state_names)
wandb.finish()
