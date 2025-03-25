import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report
import io
from tensorflow.keras.models import load_model

# Cache loaded models to avoid reloading every time
_loaded_models = {}



def list_available_models():
    """List folders with valid model files."""
    base_dir = os.getcwd()
    models = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, "tokenizer.pkl")) and \
               os.path.exists(os.path.join(folder_path, "lstm_model.h5")) and \
               os.path.exists(os.path.join(folder_path, "history.pkl")):
                models.append(folder)
    return models

def load_model_and_assets(model_path):
    """Load model, tokenizer, and training history (cached for better performance)."""
    if model_path in _loaded_models:
        return _loaded_models[model_path]
    
    tokenizer_path = os.path.join(model_path, "tokenizer.pkl")
    model_file = os.path.join(model_path, "lstm_model.h5")
    history_path = os.path.join(model_path, "history.pkl")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    model = load_model(model_file)
    _loaded_models[model_path] = (tokenizer, model, history, X_test, y_test)

    

    return tokenizer, model, history, X_test, y_test

def predict_next_word(input_text, model, tokenizer):
    """Predict the next word based on input text."""
    sequence = tokenizer.texts_to_sequences([input_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=model.input_shape[1], padding='pre')

    predictions = model.predict(padded)[0]
    predicted_index = np.argmax(predictions)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word, predictions[predicted_index]  # Return predicted word and confidence score
    
    return "No prediction", 0.0


def get_model_summary(model):
    """Capture the model summary as a formatted string."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str

def plot_training_history(history):
    """Plot model training history."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss
    axs[0].plot(history['loss'], label='Train Loss', color='blue')
    axs[0].plot(history['val_loss'], label='Validation Loss', color='orange')
    axs[0].set_title('Loss Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot accuracy
    axs[1].plot(history['accuracy'], label='Train Accuracy', color='blue')
    axs[1].plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
    axs[1].set_title('Accuracy Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    return fig

def plot_confusion_matrix(model, tokenizer, X_test, y_test):
    """Generate and return the confusion matrix plot."""
    y_pred = model.predict(X_test, batch_size=64)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Get the top 30 most frequent classes
    top_classes = np.argsort(np.sum(cm, axis=1))[-30:]
    cm_filtered = cm[np.ix_(top_classes, top_classes)]
    class_labels_filtered = [list(tokenizer.word_index.keys())[i] for i in top_classes]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm_filtered, annot=True, fmt="d", cmap="Blues", linewidths=0.5, square=True, ax=ax)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Top 30 Classes)")
    ax.set_xticks(np.arange(len(class_labels_filtered)))
    ax.set_xticklabels(class_labels_filtered, rotation=45)
    ax.set_yticks(np.arange(len(class_labels_filtered)))
    ax.set_yticklabels(class_labels_filtered, rotation=45)

    return fig  

from sklearn.metrics import classification_report

def evaluate_model(model, tokenizer, X_test, y_test):
    """Evaluate the model and return metrics like accuracy, perplexity, and classification report."""
    # Predict on test data
    y_pred = model.predict(X_test, batch_size=64)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Compute accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)

    # Compute perplexity
    loss = model.evaluate(X_test, y_test, verbose=0)
    perplexity = np.exp(loss[0]) if isinstance(loss, list) else np.exp(loss)

    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "classification_report_df": report_df
    }
