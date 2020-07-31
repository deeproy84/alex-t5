print("Installing dependencies...")
!pip3 install -q t5

import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5

BASE_DIR = "gs://alex-t5" #@param { type: "string" }
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
  TPU_TOPOLOGY = "v3-8"
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    TPU_NAME = tpu.get_master()
    print('Running on TPU:', TPU_NAME)
  except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
  auth.authenticate_user()
  tf.config.experimental_connect_to_host(TPU_NAME)
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

import gzip
import json

# Public directory of Natural Questions data on GCS.
NQ_JSONL_DIR = "gs://natural_questions/v1.0-simplified/"
NQ_SPLIT_FNAMES = {
    "train": "simplified-nq-train.jsonl.gz",
    "validation": "nq-dev-all.jsonl.gz"
}
nq_counts_path = os.path.join(DATA_DIR, "nq-counts.json")
nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "nq-train.tsv"),
    "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
}

def nq_jsonl_to_tsv(in_fname, out_fname):

  def extract_answer(tokens, span):
    """Reconstruct answer from token span and remove extra spaces."""
    start, end = span["start_token"], span["end_token"]  
    ans = " ".join(tokens[start:end])
    # Remove incorrect spacing around punctuation.
    ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
    ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
    ans = ans.replace("( ", "(").replace(" )", ")")
    ans = ans.replace("`` ", "\"").replace(" ''", "\"")
    ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
    return ans

  count = 0
  with tf.io.gfile.GFile(in_fname, "rb") as infile,\
       tf.io.gfile.GFile(out_fname, "w") as outfile:
    for line in gzip.open(infile):
      ex = json.loads(line)
      # Remove any examples with more than one answer.
      if len(ex['annotations'][0]['short_answers']) != 1:
        continue
      # Questions in NQ do not include a question mark.
      question = ex["question_text"] + "?"
      answer_span = ex['annotations'][0]['short_answers'][0]
      # Handle the two document formats in NQ (tokens or text).
      if "document_tokens" in ex:
        tokens = [t["token"] for t in ex["document_tokens"]]
      elif "document_text" in ex:
        tokens = ex["document_text"].split(" ")
      answer = extract_answer(tokens, answer_span)
      # Write this line as <question>\t<answer>
      outfile.write("%s\t%s\n" % (question, answer))
      count += 1
      tf.logging.log_every_n(
          tf.logging.INFO,
          "Wrote %d examples to %s." % (count, out_fname),
          1000)
    return count

if tf.io.gfile.exists(nq_counts_path):
  # Used cached data and counts.
  tf.logging.info("Loading NQ from cache.")
  num_nq_examples = json.load(tf.io.gfile.GFile(nq_counts_path))
else:
  # Create TSVs and get counts.
  tf.logging.info("Generating NQ TSVs.")
  num_nq_examples = {}
  for split, fname in NQ_SPLIT_FNAMES.items():
    num_nq_examples[split] = nq_jsonl_to_tsv(
        os.path.join(NQ_JSONL_DIR, fname), nq_tsv_path[split])
  json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))

#Step 1

ds = tfds.load(
    "trivia_qa/unfiltered.nocontext",
    data_dir=DATA_DIR,
    # Download data locally for preprocessing to avoid using GCS space.
    download_and_prepare_kwargs={"download_dir": "./downloads"})
print("A few raw validation examples...")
for ex in tfds.as_numpy(ds["validation"].take(2)):
  print(ex)

#Step 2

def tiviaqa_extract_qa(ds):
  def exract_qa(ex):
    return {
        "question": ex["question"],
        "answer": ex["answer"]["value"]
    }
  return ds.map(exract_qa, num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.add(
    "triviaqa_context_free",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    t5.data.TfdsTask,
    tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
    tfds_data_dir=DATA_DIR,
    text_preprocessor=[tiviaqa_extract_qa, trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy]
)

# Load and print a few examples.
triviaqa_task = t5.data.TaskRegistry.get("triviaqa_context_free")
ds = triviaqa_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(3)):
  print(ex)

#Step 3

t5.data.MixtureRegistry.remove("trivia_all")
t5.data.MixtureRegistry.add(
    "trivia_all",
    ["nq_context_free", "triviaqa_context_free"],
     default_rate=1.0

#Step 1

MODEL_SIZE = "11B" #@param["small", "base", "large", "3B", "11B"]
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)

MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

# Set parallelism and batch size to fit on vv-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

tf.io.gfile.makedirs(MODEL_DIR)
# The models from our paper are based on the Mesh Tensorflow Transformer.
model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_NAME,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 128, "targets": 32},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    iterations_per_loop=100,
)

#Step 2

if ON_CLOUD:
  %reload_ext tensorboard
  import tensorboard as tb
tb.notebook.start("--logdir " + MODELS_DIR)


FINETUNE_STEPS = 25000 #@param {type: "integer"}

model.finetune(
    mixture_or_task_name="trivia_all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)

#Step 2

# Use a larger batch size for evaluation, which requires less memory.
model.batch_size = train_batch_size * 4
model.eval(
    mixture_or_task_name="trivia_all",
    checkpoint_steps="all"
)

#Step 3

import random

def print_random_predictions(task_name, n=10):
  """Print n predictions from the validation split of a task."""
  # Grab the dataset for this task.
  ds = t5.data.TaskRegistry.get(task_name).get_dataset(
      split="validation",
      sequence_length={"inputs": 128, "targets": 32},
      shuffle=False)

  def _prediction_file_to_ckpt(path):
    """Extract the global step from a prediction filename."""
    return int(path.split("_")[-2])

  # Grab the paths of all logged predictions.
  prediction_files = tf.io.gfile.glob(
      os.path.join(
          MODEL_DIR,
          "validation_eval/%s_*_predictions" % task_name))
  # Get most recent prediction file by sorting by their step.
  latest_prediction_file = sorted(
      prediction_files, key=_prediction_file_to_ckpt)[-1]

  # Collect (inputs, targets, prediction) from the dataset and predictions file
  results = []
  with tf.io.gfile.GFile(latest_prediction_file) as preds:
    for ex, pred in zip(tfds.as_numpy(ds), preds):
      results.append((tf.compat.as_text(ex["inputs_plaintext"]),
                      tf.compat.as_text(ex["targets_plaintext"]),
                      pred.strip()))

  print("<== Random predictions for %s using checkpoint %s ==>\n" %
        (task_name, 
         _prediction_file_to_ckpt(latest_prediction_file)))

  for inp, tgt, pred in random.choices(results, k=10):
    print("Input:", inp)
    print("Target:", tgt)
    print("Prediction:", pred)
    print("Counted as Correct?", tgt == pred)
    print()

print_random_predictions("triviaqa_context_free")
print_random_predictions("nq_context_free")


export_dir = os.path.join(MODEL_DIR, "- Trained")

model.batch_size = 1 # make one prediction per call
saved_model_path = model.export(
    export_dir,
    checkpoint_step=-1,  # use most recent
    beam_size=1,  # no beam search
    temperature=1.0,  # sample according to predicted distribution
)
print("Model saved to:", saved_model_path)


question_1 = "Where is the Google headquarters located?" #@param {type:"string"}
question_2 = "What is the most populous country in the world?" #@param {type:"string"}
question_3 = "Who are the 4 members of The Beatles?" #@param {type:"string"}
question_4 = "How many teeth do humans have?" #@param {type:"string"}

questions = [question_1, question_2, question_3, question_4]

now = time.time()
# Write out the supplied questions to text files.
predict_inputs_path = os.path.join(MODEL_DIR, "predict_inputs_%d.txt" % now)
predict_outputs_path = os.path.join(MODEL_DIR, "predict_outputs_%d.txt" % now)
# Manually apply preprocessing by prepending "triviaqa question:".
with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
  for q in questions:
    f.write("trivia question: %s\n" % q.lower())

# Ignore any logging so that we only see the model's answers to the questions.
with tf_verbosity_level('ERROR'):
  model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
  model.predict(
      input_file=predict_inputs_path,
      output_file=predict_outputs_path,
      # Select the most probable output token at each step.
      temperature=0,
  )

# The output filename will have the checkpoint appended so we glob to get 
# the latest.
prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
print("\nPredictions using checkpoint %s:\n" % prediction_files[-1].split("-")[-1])
with tf.io.gfile.GFile(prediction_files[-1]) as f:
  for q, a in zip(questions, f):
    if q:
      print("Q: " + q)
      print("A: " + a)






