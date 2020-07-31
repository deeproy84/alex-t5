print("Installing dependencies...")
%tensorflow_version 2.x
!pip install -q t5

import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5

BASE_DIR = "gs://" #@param { type: "string" }
if not BASE_DIR or BASE_DIR == "gs://":
  raise ValueError("You must enter a BASE_DIR.")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ON_CLOUD = True


if ON_CLOUD:
  print("Setting up GCS access...")
  import tensorflow_gcs_config
  from google.colab import auth
  # Set credentials for GCS reading/writing from Colab and TPU.
  TPU_TOPOLOGY = "v2-8"
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    TPU_ADDRESS = tpu.get_master()
    print('Running on TPU:', TPU_ADDRESS)
  except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
  auth.authenticate_user()
  tf.config.experimental_connect_to_host(TPU_ADDRESS)
  tensorflow_gcs_config.configure_gcs_from_colab_auth()

tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager
import logging as py_logging

if ON_CLOUD:
  tf.get_logger().propagate = False
  py_logging.root.setLevel('INFO')

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)