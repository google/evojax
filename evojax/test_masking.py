# import unittest
#
# import numpy as np
# from jax import random
# import jax.numpy as jnp
#
# from evojax.util import create_logger
# from evojax.policy.mask import Mask, set_bias_and_weights
#
# from evojax.datasets import DATASET_LABELS
# from evojax.train_mnist_cnn import run_mnist_training, linear_layer_name, CNN
#
#
# logger = create_logger('masking-tests', debug=False)
#
#
# class MaskingTests(unittest.TestCase):
#
#     def test_initial_setup(self):
#         """
#         Testing that the mask is different for the different dataset classes initially, then after setting the weights
#         and bias for the final layer that it will return a mask of ones.
#         """
#         cnn = CNN()
#         cnn_params = cnn.init(random.PRNGKey(0), jnp.ones([1, 28, 28, 1]))['params']
#         linear_weights = cnn_params[linear_layer_name]["kernel"]
#         mask_size = linear_weights.shape[0]
#
#         model = Mask(mask_size=mask_size, test_no_mask=False)
#         # batch_size = 8
#         mask_params = model.init(random.PRNGKey(0), jnp.ones([1, ]))
#
#         model_output = model.apply({"params": mask_params["params"]}, jnp.array(list(DATASET_LABELS.values())))
#         self.assertFalse(jnp.array_equal(model_output[0], model_output[1]))
#         self.assertFalse(jnp.array_equal(model_output[0], model_output[2]))
#         self.assertFalse(jnp.array_equal(model_output[1], model_output[2]))
#
#         # Now set the final bias and weights
#         mask_params = set_bias_and_weights(mask_params)
#
#         model_output = model.apply({"params": mask_params["params"]}, jnp.array(list(DATASET_LABELS.values())))
#
#         self.assertTrue(jnp.array_equal(model_output, jnp.ones_like(model_output)))
#
#     def test_mask_of_ones(self):
#         """
#         Test that a mask of ones doesn't change the CNN output, but that a binary mask will.
#         """
#         cnn_params = run_mnist_training(logger=logger, return_model=True, num_epochs=3)
#
#         linear_weights = cnn_params[linear_layer_name]["kernel"]
#         unity_mask = jnp.ones((linear_weights.shape[0],))
#         binary_mask = jnp.array(np.random.randint(1, size=linear_weights.shape[0]))
#
#         pretend_images = jnp.array(np.random.uniform(size=(5, 28, 28, 1)))
#
#         unity_masked_output = CNN().apply({'params': cnn_params}, pretend_images, unity_mask)
#         binary_masked_output = CNN().apply({'params': cnn_params}, pretend_images, binary_mask)
#         non_masked_output = CNN().apply({'params': cnn_params}, pretend_images, None)
#
#         self.assertTrue(jnp.array_equal(unity_masked_output, non_masked_output))
#         self.assertTrue(jnp.array_equal(binary_masked_output, non_masked_output))
