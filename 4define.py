#Step 1

MODEL_SIZE = "3B" #@param["small", "base", "large", "3B", "11B"]
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)

MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

if ON_CLOUD and MODEL_SIZE == "3B":
  tf.logging.warn(
      "The `3B` model is too large to use with the 5GB GCS free tier. "
      "Make sure you have at least 25GB on GCS before continuing."
  )
elif ON_CLOUD and MODEL_SIZE == "11B":
  raise ValueError(
      "The `11B` parameter is too large to fine-tune on the `v2-8` TPU "
      "provided by Colab. Please comment out this Error if you're running "
      "on a larger TPU."
  )

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
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
    tpu=TPU_ADDRESS,
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

