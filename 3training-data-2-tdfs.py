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


