import jax as jx
import jax.numpy as jnp
from jax import jit, vmap
from jax.example_libraries import optimizers
from jax.tree_util import register_pytree_node

import numpy as np

import json
import pickle as pkl
import argparse
import time
import copy
from tqdm import tqdm
import os

import environments
from initialization import get_init_fns
from agent_environment_interaction_loop import get_agent_environment_interaction_loop_function

from tree_utils import tree_unstack

import wandb
from multi_wandb import multi_wandb

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

configs = []
for i in range(config["num_seeds"]):
    seed_config = copy.copy(config)
    seed_config["seed_index"]=i
    configs+=[seed_config]

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
        for d in data:
            d["config"]=config.__dict__
        pkl.dump(data, f)

def get_log_function(F):
    model_eval = vmap(jit(F.model_eval))

    def log(S, M, log_dicts_list, logger, wallclock, keys):
        curr_time = run_states.env_t

        # Log complete episode returns for each run seperately
        unstacked_M = tree_unstack(M)
        unstacked_t = tree_unstack(curr_time)
        for i,(M_i,t_i,log_dicts) in enumerate(zip(unstacked_M, unstacked_t, log_dicts_list)):
            returns = M_i["return"][M_i["episode_complete"]]
            return_times = t_i-config.eval_frequency+jnp.arange(config.eval_frequency)[M_i["episode_complete"]]
            log_dicts["returns_and_times"] = update_log_dict(log_dicts["returns_and_times"],{"return":returns,"return_time":return_times})
            for ret, t in zip(returns,return_times):
                data = {"return":np.array(ret),"return_time":np.array(t)}
                logger.log_at_index(data, i)

        # Log model metrics
        run_states.key, subkeys = jnp.transpose(vmap(jx.random.split)(run_states.key),axes=(1,0,2))
        stacked_metrics = model_eval(S.buffer_state,S.model_opt_state,subkeys)
        stacked_metrics["time"] = curr_time
        unstacked_metrics = tree_unstack(stacked_metrics)
        for m in unstacked_metrics:
            m["time_per_step"] = wallclock/config.eval_frequency
        data = [{key:np.array(value) for key, value in m.items()} for m in unstacked_metrics]
        logger.log(data)

        for (metrics,log_dicts) in zip(unstacked_metrics, log_dicts_list):
            log_dicts["metrics"] = update_log_dict(log_dicts["metrics"],metrics)
        return log_dicts_list
    return log

def save_checkpoint(run_state, log_dicts, i, wandb_ids, opt_state_names):
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
            'wandb_ids':wandb_ids
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
        wandb_ids = checkpoint["wandb_ids"]
    return run_state, log_dicts, start_i, wandb_ids

########################################################################
# Initialization
########################################################################

Environment = getattr(environments, config.environment)
env_config = config.env_config

env = Environment(**env_config)
num_actions = env.num_actions()

# Initialize run_states and functions
# Note:vmap over run state but not functions
init_state, init_functions = get_init_fns(env,config)
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
run_states = vmap(init_state)(subkeys)
functions = init_functions()
start_i = 0
log_dicts = [{"returns_and_times":None, "metrics":None} for i in range(config.num_seeds)]

resumed = False
opt_state_names = ["V_opt_state", "pi_opt_state", "model_opt_state"]
if(args.load_checkpoint is not None):
    if(os.path.exists(args.load_checkpoint)):
        run_states, log_dicts, start_i, wandb_ids = load_checkpoint(opt_state_names)
        resumed = True
    else:
        print("Warning! load_checkpoint does not exist, starting run from scratch.")

# Resume wandb sessions as well if loading from checkpoint
if(resumed):
    logger = multi_wandb(configs, wandb_ids, resume="must", project='dreamerv2_pure_jax', group=args.group)
else:
    wandb_ids = [wandb.util.generate_id() for i in range(config.num_seeds)]
    logger = multi_wandb(configs, wandb_ids, project='dreamerv2_pure_jax', group=args.group)

log = get_log_function(functions)

# Build the agent environment interaction loop function
agent_environment_interaction_loop_function = jit(vmap(get_agent_environment_interaction_loop_function(functions, config.eval_frequency, config)))

time_since_checkpoint = 0
last_time = time.time()

########################################################################
# Main training loop
########################################################################

i = start_i
tqdm.write("Beginning run...")
try:
    for i in tqdm(range(start_i,config.num_steps//config.eval_frequency), initial=start_i, total=config.num_steps//config.eval_frequency):
        run_states, metrics = agent_environment_interaction_loop_function(run_states)

        ellapsed_time = time.time()-last_time
        last_time = time.time()

        run_states.key, subkeys = jnp.transpose(vmap(jx.random.split)(run_states.key),axes=(1,0,2))
        log_dicts = log(run_states, metrics, log_dicts, logger, ellapsed_time, subkeys)

        # periodically save checkpoint to disk
        time_since_checkpoint+=config.eval_frequency
        if(time_since_checkpoint>=config.checkpoint_frequency):
            save_checkpoint(run_states, log_dicts, i, wandb_ids, opt_state_names)
            time_since_checkpoint = 0
except Exception as e:
    logger.end()
    raise e

# Save Data and final checkpoint
save_log(log_dicts, config)
save_checkpoint(run_states, log_dicts, i, wandb_ids, opt_state_names)
logger.end()
