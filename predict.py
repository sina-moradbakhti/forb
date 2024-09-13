import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('./my_model')
tokenizer = BertTokenizer.from_pretrained('./my_model')

# Load profession labels from CSV
def load_profession_labels(csv_file):
    df = pd.read_csv(csv_file)
    profession_labels = dict(zip(df['label_id'], df['profession']))
    return profession_labels

# Load the labels dynamically
profession_labels = load_profession_labels('datasets/profession_labels.csv')

def predict_profession(user_prompt):
    # Tokenize and encode the user prompt
    inputs = tokenizer(user_prompt, return_tensors="tf", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Make prediction
    outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
    
    # Get the predicted profession
    predicted_profession = profession_labels.get(predicted_class, "I cannot answer this, please ask a relevant question")
    
    return predicted_profession

# Example usage
user_prompt = "I need someone for making a mobile app for my business."
predicted_profession = predict_profession(user_prompt)
print(f"Predicted Profession: {predicted_profession}")