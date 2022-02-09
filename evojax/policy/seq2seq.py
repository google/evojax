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

"""Implementation of seq2seq model.

The model is based on: https://github.com/google/flax/tree/main/examples/seq2seq
"""

import logging
import numpy as np
from typing import Tuple
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


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


char_table = CharacterTable()


class EncoderLSTM(nn.Module):
    """LSTM in the encoder part of the seq2seq model."""

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        lstm_state, is_eos = carry
        new_lstm_state, y = nn.LSTMCell()(lstm_state, x)

        # Pass forward the previous state if EOS has already been reached.
        def select_carried_state(new_state, old_state):
            return jnp.where(is_eos[:, np.newaxis], old_state, new_state)

        # LSTM state is a tuple (c, h).
        carried_lstm_state = tuple(
            select_carried_state(*s) for s in zip(new_lstm_state, lstm_state))
        # Update `is_eos`.
        is_eos = jnp.logical_or(is_eos, x[:, char_table.eos_id])
        return (carried_lstm_state, is_eos), y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # use dummy key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size)


class Encoder(nn.Module):
    """LSTM encoder, returning state after EOS is input."""

    hidden_size: int

    @nn.compact
    def __call__(self, inputs):
        # inputs.shape = (batch_size, seq_length, vocab_size).
        batch_size = inputs.shape[0]
        lstm = EncoderLSTM(name='encoder_lstm')
        init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)
        init_is_eos = jnp.zeros(batch_size, dtype=np.bool)
        init_carry = (init_lstm_state, init_is_eos)
        (final_state, _), _ = lstm(init_carry, inputs)
        return final_state


class DecoderLSTM(nn.Module):
    """LSTM in the decoder part of the seq2seq model."""

    teacher_force: bool

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        lstm_state, last_prediction = carry
        if not self.teacher_force:
            x = last_prediction
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        logits = nn.Dense(features=char_table.vocab_size)(y)
        predicted_token = jnp.argmax(logits, axis=-1)
        prediction = jax.nn.one_hot(
            predicted_token, char_table.vocab_size, dtype=jnp.float32)

        return (lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
    """LSTM decoder."""

    init_state: Tuple[Any]
    teacher_force: bool

    @nn.compact
    def __call__(self, inputs):
        # inputs.shape = (seq_length, vocab_size).
        lstm = DecoderLSTM(teacher_force=self.teacher_force)
        init_carry = (self.init_state, inputs[:, 0])
        _, (logits, predictions) = lstm(init_carry, inputs)
        return logits, predictions


class Seq2seq(nn.Module):
    """Sequence-to-sequence class using encoder/decoder architecture."""

    teacher_force: bool
    hidden_size: int

    @nn.compact
    def __call__(self, encoder_inputs, decoder_inputs):
        # Encode inputs.
        init_decoder_state = Encoder(
            hidden_size=self.hidden_size)(encoder_inputs)
        # Decode outputs.
        logits, predictions = Decoder(
            init_state=init_decoder_state,
            teacher_force=self.teacher_force)(decoder_inputs[:, :-1])
        return logits, predictions


class Seq2seqPolicy(PolicyNetwork):
    """A seq2seq policy that deals with simple additions."""

    def __init__(self,
                 hidden_size: int = 256,
                 teacher_force: bool = False,
                 max_len_query_digit: int = 3,
                 logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('Seq2seqPolicy')
        else:
            self._logger = logger

        max_input_len = max_len_query_digit + 2 + 2
        max_output_len = max_len_query_digit + 3
        encoder_shape = jnp.ones(
            (1, max_input_len, char_table.vocab_size), dtype=jnp.float32)
        decoder_shape = jnp.ones(
            (1, max_output_len, char_table.vocab_size), dtype=jnp.float32)
        model = Seq2seq(hidden_size=hidden_size, teacher_force=teacher_force)
        key = random.PRNGKey(0)
        params = model.init({'params': key, 'lstm': key},
                            encoder_shape, decoder_shape)['params']
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(
            'Seq2seqPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)

        def forward_fn(p, o):
            x = jax.nn.one_hot(
                char_table.encode(jnp.array([11]))[0:1], char_table.vocab_size,
                dtype=jnp.float32)
            x = jnp.tile(x, (o.shape[0], max_output_len, 1))
            logits, predictions = model.apply({'params': p}, o, x)
            return logits
        self._forward_fn = jax.vmap(forward_fn)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
