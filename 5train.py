#Step 1

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