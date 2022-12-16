import jax.numpy as jnp
import jax as jx
from jax import jit, vmap
from functools import partial

########################################################################
# Define Replay Buffer
########################################################################

class replay_buffer:
    def __init__(self, buffer_size, dummy_items, terminal_index=None):
        self.buffer_size = buffer_size
        self.dummy_items = dummy_items
        self.terminal_index = terminal_index

    @partial(jit, static_argnums=(0,))
    def initialize(self):
        location = 0
        full = False
        # construct a buffer for each dummy item by repeating them along each axis
        buffers = jx.tree_map(lambda x: jnp.zeros_like(jnp.repeat(jnp.expand_dims(x,0),self.buffer_size, axis=0)), self.dummy_items)
        state = (location, full, buffers)
        return state

    #add a sample to the buffer
    @partial(jit, static_argnums=(0,))
    def add(self, state, *args):
        location, full, buffers = state

        buffers = jx.tree_map(lambda x,y: x.at[location].set(y),buffers,list(args))

        full = jnp.where(location+1 >= self.buffer_size, True, full)
        # Increment the buffer location
        location = (location + 1) % self.buffer_size
        state = (location, full, buffers)
        return state


    @partial(jit, static_argnums=(0,2))
    def sample(self, state, batch_size, key):
        location, full, buffers = state
        key, subkey = jx.random.split(key)
        indices = jx.random.randint(subkey, minval=0, maxval=jnp.where(full, self.buffer_size, location),shape=(batch_size,))

        sample = jx.tree_map(lambda x: x.take(indices, axis=0),buffers)
        return sample

    @partial(jit, static_argnums=(0,2,3))
    def sample_sequences(self, state, batch_size, sequence_length, key):
        location, full, buffers = state
        # Note: this will not work right if sequence_length is larger than the current number of items in the buffer and may return empty items
        # This is simpler than the original dreamer implementation which always shifts sequences to be as long as possible if they include a terminal state
        key, subkey = jx.random.split(key)
        start_indices = jnp.mod(
                            jx.random.randint(subkey,
                                minval=jnp.where(full,location-self.buffer_size, 0),
                                maxval=(location-sequence_length+1),
                                shape=(batch_size,)),
                            self.buffer_size)

        sample_sequence = lambda i,b: jx.tree_map(lambda x: x[jnp.mod(jnp.arange(sequence_length)+i, self.buffer_size)], b)
        sample = vmap(sample_sequence, in_axes=(0,None))(start_indices, buffers)
        return sample

    @partial(jit, static_argnums=(0,2,3))
    def sample_sequences_maximize_nonterminal(self, state, batch_size, sequence_length, key):
        location, full, buffers = state
        # Note: this will not work right if sequence_length is larger than the current number of items in the buffer and may return empty items
        key, subkey = jx.random.split(key)
        start_indices = jnp.mod(
                            jx.random.randint(subkey,
                                minval=jnp.where(full,location-self.buffer_size, 0),
                                maxval=(location-sequence_length+1),
                                shape=(batch_size,)),
                            self.buffer_size)

        def adjust_index(i, state):
            location, full, buffers = state
            terminal_buffer = buffers[self.terminal_index]
            terms = terminal_buffer[jnp.mod(jnp.arange(sequence_length)+i, self.buffer_size)]
            first_terminal_index = jnp.nonzero(terms,size=1, fill_value=-1)[0]

            # shift such that first terminal index is the last item in sequence
            i = jnp.where(first_terminal_index!=-1,(i-(sequence_length-1-first_terminal_index))%self.buffer_size, i)

            terms = terminal_buffer[jnp.mod(jnp.arange(sequence_length)+i, self.buffer_size)]

            # find last terminal in sequence besides the one found in the last step (if any others are present)
            last_terminal_index = jnp.nonzero(jnp.flip(terms[:-1]),size=1, fill_value=-1)[0]
            # invert flip, maintaining -1 as special case
            last_terminal_index = jnp.where(jnp.equal(last_terminal_index,-1),-1, sequence_length-2-last_terminal_index)

            buffer_end_in_sample = jnp.sum(jnp.equal(jnp.mod(jnp.arange(sequence_length-1)+i, self.buffer_size),(location-1)%self.buffer_size))
            last_terminal_index = jnp.where(buffer_end_in_sample, jnp.maximum(last_terminal_index,(location-i)%self.buffer_size), last_terminal_index)

            # shift i to just after the second last terminal index or buffer end (only if we found a terminal in the first step and another, or buffer end in the second)
            i = jnp.where(jnp.logical_and(first_terminal_index!=-1,last_terminal_index!=-1),(i+last_terminal_index+1)%self.buffer_size,i)
            return i

        start_indices = vmap(adjust_index, in_axes=(0,None))(start_indices,state)

        sample_sequence = lambda i,b: jx.tree_map(lambda x: x[jnp.mod(jnp.arange(sequence_length)+i, self.buffer_size)], b)
        sample = vmap(sample_sequence, in_axes=(0,None))(start_indices, buffers)
        return sample