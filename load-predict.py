#Step 1

import tensorflow as tf
import tensorflow_text  # Required to run exported model.

def load_predict_fn(model_path):
  if tf.executing_eagerly():
    print("Loading SavedModel in eager mode.")
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures['serving_default'](tf.constant(x))['outputs'].numpy()
  else:
    print("Loading SavedModel in tf 1.x graph mode.")
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    meta_graph_def = tf.compat.v1.saved_model.load(sess, ["serve"], model_path)
    signature_def = meta_graph_def.signature_def["serving_default"]
    return lambda x: sess.run(
        fetches=signature_def.outputs["outputs"].name, 
        feed_dict={signature_def.inputs["input"].name: x}
    )

predict_fn = load_predict_fn(saved_model_path)



#Step 2

def answer(question):
  return predict_fn([question])[0].decode('utf-8')

for question in ["trivia question: where is the google headquarters?",
                 "trivia question: what is the most populous country in the 			world?",
                 "trivia question: who are the 4 members of the beatles?",
                 "trivia question: how many teeth do humans have?"]:
    print(answer(question))

#You can now deploy your SavedModel for serving (e.g., with TensorFlow Serving).