import argparse
import pickle as pkl
import numpy as np
from jax import numpy as jnp
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--window", "-w", type=int, default=1)
parser.add_argument("--frequency", "-f", type=int, default=1)
parser.add_argument("--metric", "-m", type=str)
parser.add_argument("--savefile", "-s", type=str)
parser.add_argument("--multiseed", action="store_true")
parser.add_argument("--timescale", type=float, default=1.0)
args = parser.parse_args()


with open(args.output, 'rb') as f:
    data = pkl.load(f)

if(args.multiseed):
    # Handle returns as special case, each seed may have a different number of completed episodes and episodes 
    # may complete at different times. In order to plot return v.s. time with error bars we simply fill each list 
    # of returns to have the same size as the list of all times at which a return was recorded in any seed.
    # For a given seed, as long as there is a remaining completed episode in the future, the next completed 
    # episode is copied into each time slot. Once a given seed has no more completed episodes, we copy the final 
    # return to fill any remaining slots.
    if(args.metric=="return"):
        values = [d["returns_and_times"]['return'][::len(d["returns_and_times"]['return'])//args.frequency] for d in data]
        times = [d["returns_and_times"]['return_time'][::len(d["returns_and_times"]['return'])//args.frequency] for d in data]
        all_times = np.unique(np.concatenate(times))
        expanded_values = []
        for v,t in zip(values,tqdm(times)):
            i=0
            expanded_v = []
            for j in tqdm(range(len(t))):
                while i<len(all_times) and (all_times[i]<=t[j]):
                    expanded_v+=[v[j]]
                    i+=1
            #fill in expanded v with the final recorded return
            expanded_v+=[v[j]]*(len(all_times)-len(expanded_v))
            expanded_values+=[expanded_v]
        values = expanded_values
        times = [all_times]*len(times)

    else:
        values = np.array([d['metrics'][args.metric][::len(d['metrics'][args.metric])//args.frequency] for d in data])
        times = np.array([d['metrics']["time"][::len(d['metrics']["time"])//args.frequency] for d in data])
    values = vmap(lambda a,v: jnp.convolve(a,v,mode='valid'),in_axes=(0,None))(jnp.asarray(values),jnp.ones(args.window)/args.window)
    times = vmap(lambda a,v: jnp.convolve(a,v,mode='valid'),in_axes=(0,None))(jnp.asarray(times),jnp.ones(args.window)/args.window)/args.timescale

    values = values.flatten()
    times = times.flatten()
else:
    # Handle returns as special case
    if(args.metric=="return"):
        values = data["returns_and_times"]['return'][::args.frequency]
        times = data["returns_and_times"]['return_time'][::args.frequency]
    else:
        values = np.array(data['metrics'][args.metric][::args.frequency])
        times = np.array(data['metrics']["time"][::args.frequency])
    values = jnp.convolve(values,np.ones(args.window)/args.window, mode='valid')
    times = jnp.convolve(times,np.ones(args.window)/args.window, mode='valid')/args.timescale

data_frame = pd.DataFrame({args.metric:values,"time":times})

plot = sns.lineplot(x="time",y=args.metric,data=data_frame)

plt.xlabel('')
plt.ylabel('')

if(args.savefile is not None):
    plt.savefig(args.savefile)
else:
    plt.show()
