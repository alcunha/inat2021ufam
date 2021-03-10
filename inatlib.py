import pandas as pd
import tensorflow as tf

def get_top_5(pred):
  _, indexes = tf.math.top_k(pred, k=5)
  indexes = indexes.numpy().astype(str)
  parsed = " ".join(indexes)

  return parsed

def _decode_predictions(predictions):
  preds = [get_top_5(pred) for pred in predictions]

  return preds

def generate_submission(instance_ids, predictions, csv_file):
  predictions = _decode_predictions(predictions)
  df = pd.DataFrame(list(zip(instance_ids, predictions)),
                    columns=['Id', 'Predicted'])
  df = df.sort_values('Id')
  df.to_csv(csv_file, index=False, header=True, sep=',')

  return df
