import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

# Load and preprocess data
def preprocess_data(texts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    input_ids = inputs['input_ids'].numpy()
    attention_mask = inputs['attention_mask'].numpy()
    labels = np.array(labels)
    return input_ids, attention_mask, labels

# Main function to train the model
def main():
    # Load dataset
    texts, labels = load_data('datasets/dataset.csv')

    # Preprocess data
    input_ids, attention_mask, labels = preprocess_data(texts, labels)
    
    # Split data into training and validation sets
    input_ids_train, input_ids_val, attention_mask_train, attention_mask_val, labels_train, labels_val = train_test_split(
        input_ids, attention_mask, labels, test_size=0.1, random_state=42
    )
    
    # Load model
    print(len(set(labels)))
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=42)
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train model
    model.fit({'input_ids': input_ids_train, 'attention_mask': attention_mask_train}, labels_train, epochs=3, batch_size=8, validation_data=({'input_ids': input_ids_val, 'attention_mask': attention_mask_val}, labels_val))
    
    # Save model and tokenizer
    model_save_path = './my_model'
    tokenizer_save_path = './my_model'
    
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Model and tokenizer saved successfully to {model_save_path}")
    except Exception as e:
        print(f"Error saving model or tokenizer: {e}")

if __name__ == "__main__":
    main()