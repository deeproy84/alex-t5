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
      print()