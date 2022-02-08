# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A seq2seq task (simple addition).

Ref: https://github.com/google/flax/tree/main/examples/seq2seq
"""

import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray


class CharacterTable(object):
    """Encode/decodes between strings and integer representations."""

    def __init__(self):
        self._chars = '0123456789+= '
        self.pad_id = len(self._chars)
        self.eos_id = self.pad_id + 1
        self.vocab_size = len(self._chars) + 2
        self._indices_char = dict(
            (idx, ch) for idx, ch in enumerate(self._chars))
        self._indices_char[self.pad_id] = '_'

    def encode(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([inputs, jnp.array([self.eos_id])])

    def decode(self, inputs):
        """Decode from list of integers to string."""
        chars = []
        for elem in inputs.tolist():
            if elem == self.eos_id:
                break
            chars.append(self._indices_char[elem])
        return ''.join(chars)


class Seq2seqTask(VectorizedTask):
    """Seq2seq task: encoder's input is "x+y", decoder's output is "=z"."""

    def __init__(self,
                 batch_size: int = 128,
                 max_len_query_digit: int = 3,
                 test: bool = False):

        char_table = CharacterTable()
        max_input_len = max_len_query_digit + 2 + 2
        max_output_len = max_len_query_digit + 3
        max_num = pow(10, max_len_query_digit)
        self.obs_shape = tuple([max_input_len, char_table.vocab_size])
        self.act_shape = tuple([max_output_len, char_table.vocab_size])
        self.max_steps = 1

        def encode_onehot(batch_inputs, max_len):
            def encode_str(s):
                tokens = char_table.encode(s)
                org_len = len(tokens)
                tokens = jnp.pad(
                    tokens, [(0, max_len - org_len)], mode='constant')
                return jax.nn.one_hot(
                    tokens, char_table.vocab_size, dtype=jnp.float32)
            return jnp.array([encode_str(inp) for inp in batch_inputs])

        def decode_onehot(batch_inputs):
            return np.array(list(map(
                lambda x: char_table.decode(x.argmax(axis=-1)), batch_inputs)))
        self.decode_embeddings = decode_onehot

        def breakdown_int(n):
            return jnp.where(
                n == 0,
                jnp.zeros(max_len_query_digit),
                jnp.array([(n // (10**i)) % 10
                           for i in range(max_len_query_digit - 1, -1, -1)]))

        def get_batch_data(key):
            keys = random.split(key, 2)
            add_op1 = random.randint(
                keys[0], minval=0, maxval=100, shape=(batch_size,))
            add_op2 = random.randint(
                keys[1], minval=0, maxval=max_num, shape=(batch_size,))
            for op1, op2 in zip(add_op1, add_op2):
                inputs = jnp.concatenate([
                    breakdown_int(op1)[1:],
                    jnp.array([10, ]),   # 10 is '+'
                    breakdown_int(op2)], axis=0)
                outputs = jnp.concatenate([
                    jnp.array([11, ]),  # 11 is '='
                    breakdown_int(op1 + op2)], axis=0)
                yield inputs, outputs

        def reset_fn(key):
            inputs, outputs = zip(*get_batch_data(key[0]))
            batch_data = encode_onehot(inputs, max_input_len)
            batch_labels = encode_onehot(outputs, max_output_len)
            return State(
                obs=jnp.repeat(batch_data[None, :], key.shape[0], axis=0),
                labels=jnp.repeat(batch_labels[None, :], key.shape[0], axis=0))
        self._reset_fn = jax.jit(reset_fn)

        def get_sequence_lengths(sequence_batch):
            # sequence_batch.shape = (batch_size, seq_length, vocab_size)
            eos_row = sequence_batch[:, :, char_table.eos_id]
            eos_idx = jnp.argmax(eos_row, axis=-1)
            return jnp.where(
                eos_row[jnp.arange(eos_row.shape[0]), eos_idx],
                eos_idx + 1, sequence_batch.shape[1])

        def mask_sequences(sequence_batch, lengths):
            return sequence_batch * (
                lengths[:, np.newaxis] >
                np.arange(sequence_batch.shape[1])[np.newaxis])

        def cross_entropy_loss(logits, labels, lengths):
            xe = jnp.sum(jax.nn.log_softmax(logits) * labels, axis=-1)
            return -jnp.mean(mask_sequences(xe, lengths))

        def step_fn(state, action):
            labels = state.labels[:, 1:]
            lengths = get_sequence_lengths(labels)
            if test:
                token_acc = jnp.argmax(action, -1) == jnp.argmax(labels, -1)
                sequence_acc = (jnp.sum(
                    mask_sequences(token_acc, lengths), axis=-1) == lengths)
                reward = jnp.mean(sequence_acc)
            else:
                reward = -cross_entropy_loss(action, labels, lengths)
            done = jnp.ones((), dtype=jnp.int32)
            return state, reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
