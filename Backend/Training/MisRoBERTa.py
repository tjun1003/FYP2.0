# -*- coding: utf-8 -*-
"""
MisRoBÆRTa Hybrid Neural Network Model - Enhanced Version
New Features:
1. Automatic saving of the trained model
2. Demonstration of random sample classification
3. Saving of training history and evaluation metrics
4. Use of Hugging Face mirror source
"""

# ============================================================================
# 🔧 Configure Hugging Face Mirror Source (Must be before importing other libraries)
# ============================================================================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("✅ Hugging Face mirror source configured: https://hf-mirror.com")

# System and utility libraries
import sys
import time
import numpy as np
import pandas as pd
import random as rnd
import pickle
import json

# Machine learning evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Deep learning framework
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Input, Concatenate, Conv1D, Flatten, MaxPooling1D, Reshape
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Pre-trained language models
from simpletransformers.language_representation import RepresentationModel
from sentence_transformers import SentenceTransformer

# --- Configuration and Hyperparameters ---
config = {
    'epochs_n': 100,
    'filters': 64,
    'units': 128,
    'dropout_rate': 0.2,
    'recurrent_dropout_rate': 0.2,
    'batch_size': 1000,
    'patience': 20,
    'max_features': 5000,
    'min_df': 5,
    'test_size': 0.30,
}

config['no_attributes'] = config['units'] * 2
config['kernel_size'] = int(config['no_attributes'] / 2)

# --- Global Metrics List ---
accuracies = []
precisions_micro = []
precisions_macro = []
recalls_micro = []
recalls_macro = []
execution_time = []

# --- Helper Functions ---

def evaluate(y_test, y_pred, modelName='GRU', iters=0):
    """Evaluate model performance"""
    y_pred = np.asarray(y_pred)
    y_pred[np.isnan(y_pred)] = 0
    
    y_p = np.argmax(y_pred, axis=1)
    y_t = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_t, y_p)
    accuracies.append(accuracy)
    precision_micro = precision_score(y_t, y_p, average='micro', zero_division=0)
    precisions_micro.append(precision_micro)
    precision_macro = precision_score(y_t, y_p, average='macro', zero_division=0)
    precisions_macro.append(precision_macro)
    recall_micro = recall_score(y_t, y_p, average='micro', zero_division=0)
    recalls_micro.append(recall_micro)
    recall_macro = recall_score(y_t, y_p, average='macro', zero_division=0)
    recalls_macro.append(recall_macro)

    print(f"\n--- {modelName} Evaluation Results (Iteration {iters+1}) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro-average Precision: {precision_micro:.4f}")
    print(f"Macro-average Precision: {precision_macro:.4f}")
    print(f"Micro-average Recall: {recall_micro:.4f}")
    print(f"Macro-average Recall: {recall_macro:.4f}")
    print("Classification Report:\n", classification_report(y_t, y_p, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_t, y_p))
    return y_p, y_t

def splitDataset(X, y):
    """Split dataset into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], shuffle=True, stratify=y)
    return X_train, X_test, np.array(y_train), np.array(y_test)

def getBERTEncoding(X_train, X_test, use_cuda=False):
    """Get BERT Encoding"""
    model = RepresentationModel(model_type="bert", model_name="bert-large-uncased", use_cuda=use_cuda)
    X_train_BERT = model.encode_sentences(X_train, combine_strategy="mean")
    X_test_BERT = model.encode_sentences(X_test, combine_strategy="mean")
    X_train_BERT = X_train_BERT.reshape((X_train_BERT.shape[0], 1, X_train_BERT.shape[1]))
    X_test_BERT = X_test_BERT.reshape((X_test_BERT.shape[0], 1, X_test_BERT.shape[1]))
    return X_train_BERT, X_test_BERT

def getRoBERTaEncoding(X_train, X_test, use_cuda=False):
    """Get RoBERTa Encoding"""
    model = RepresentationModel(model_type="roberta", model_name="roberta-base", use_cuda=use_cuda)
    X_train_RoBERTa = model.encode_sentences(X_train, combine_strategy="mean")
    X_test_RoBERTa = model.encode_sentences(X_test, combine_strategy="mean")
    X_train_RoBERTa = X_train_RoBERTa.reshape((X_train_RoBERTa.shape[0], 1, X_train_RoBERTa.shape[1]))
    X_test_RoBERTa = X_test_RoBERTa.reshape((X_test_RoBERTa.shape[0], 1, X_test_RoBERTa.shape[1]))
    return X_train_RoBERTa, X_test_RoBERTa

def getBARTEncoding(X_train, X_test, use_cuda=False):
    """Get BART Encoding"""
    print("📥 Loading BART model...")
    model = SentenceTransformer('facebook/bart-large')
    print("✅ BART model loaded successfully")
    
    print("🔄 Encoding training set...")
    X_train_BART = model.encode(X_train, show_progress_bar=True)
    print("🔄 Encoding test set...")
    X_test_BART = model.encode(X_test, show_progress_bar=True)
    
    X_train_BART = X_train_BART.reshape((X_train_BART.shape[0], 1, X_train_BART.shape[1]))
    X_test_BART = X_test_BART.reshape((X_test_BART.shape[0], 1, X_test_BART.shape[1]))
    return X_train_BART, X_test_BART

def prepareYTrainTestNN(y_train, y_test, num_classes):
    """Prepare Y training and testing data for Neural Network (One-hot encoding)"""
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    return y_train_cat, y_test_cat

def test_random_samples(X_test, y_test, y_pred, id2label, num_samples=5):
    """Demonstrate classification test with random samples"""
    print("\n" + "="*80)
    print("🔍 Random Sample Classification Test")
    print("="*80)
    
    # Get predicted labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Select random samples
    random_indices = rnd.sample(range(len(X_test)), min(num_samples, len(X_test)))
    
    for i, idx in enumerate(random_indices, 1):
        sample_text = X_test[idx]
        true_label_id = y_test[idx]
        pred_label_id = y_pred_labels[idx]
        confidence = y_pred[idx][pred_label_id] * 100
        
        # Get label names
        true_label = id2label[true_label_id]
        pred_label = id2label[pred_label_id]
        
        # Check if prediction is correct
        is_correct = "✅ Correct" if true_label_id == pred_label_id else "❌ Incorrect"
        
        print(f"\n【Sample {i}】{is_correct}")
        print(f"Text Content: {sample_text[:150]}{'...' if len(sample_text) > 150 else ''}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label} (Confidence: {confidence:.2f}%)")
        print("-" * 80)

def save_model_and_results(model, id2label, config, save_dir="saved_models"):
    """Save model, label map, and configuration"""
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Save Keras Model
    model_path = os.path.join(save_dir, f"model_{timestamp}.h5")
    model.save(model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # 2. Save label map
    label_path = os.path.join(save_dir, f"id2label_{timestamp}.pkl")
    with open(label_path, 'wb') as f:
        pickle.dump(id2label, f)
    print(f"✅ Label map saved: {label_path}")
    
    # 3. Save configuration
    config_path = os.path.join(save_dir, f"config_{timestamp}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✅ Configuration saved: {config_path}")
    
    # 4. Save evaluation metrics
    metrics = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_micro_mean": float(np.mean(precisions_micro)),
        "precision_micro_std": float(np.std(precisions_micro)),
        "precision_macro_mean": float(np.mean(precisions_macro)),
        "precision_macro_std": float(np.std(precisions_macro)),
        "recall_micro_mean": float(np.mean(recalls_micro)),
        "recall_micro_std": float(np.std(recalls_micro)),
        "recall_macro_mean": float(np.mean(recalls_macro)),
        "recall_macro_std": float(np.std(recalls_macro)),
        "execution_time_mean": float(np.mean(execution_time)),
        "execution_time_std": float(np.std(execution_time)),
        "all_accuracies": [float(x) for x in accuracies],
        "all_execution_times": [float(x) for x in execution_time]
    }
    
    metrics_path = os.path.join(save_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"✅ Evaluation metrics saved: {metrics_path}")
    
    return model_path, label_path, config_path, metrics_path

# --- Main Execution Block ---
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_file.csv> <use_cuda_flag> <num_iterations>")
        sys.exit(1)

    # Parse command line arguments
    FIN = sys.argv[1]
    USE_CUDA = bool(int(sys.argv[2]))
    print(f"Using CUDA: {USE_CUDA}")
    NUM_ITER = int(sys.argv[3])

    # Load and preprocess dataset
    dataSet = pd.read_csv(FIN, encoding="utf-8")
    labels = dataSet['label'].unique()
    num_classes = len(labels)
    id2label = {}
    for idx, label in enumerate(labels):
        id2label[idx] = label
        dataSet.loc[dataSet['label'] == label, 'label'] = idx

    print("Label Mapping:")
    for key in id2label:
        print(f"{key}: {id2label[key]}")

    X = dataSet['content'].astype(str).to_list()
    y = dataSet['label'].astype(int).to_list()

    # --- Model Architecture Definition ---
    BART_EMBEDDING_DIM = 1024
    ROBERTA_EMBEDDING_DIM = 768

    # BART Branch
    input_bart = Input(shape=(1, BART_EMBEDDING_DIM), name='BART_Input')
    model_bart = Bidirectional(LSTM(units=config['units'], dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout_rate'], return_sequences=True), name='BART_BiLSTM_1')(input_bart)
    model_bart = Bidirectional(LSTM(units=config['units'], dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout_rate'], return_sequences=True), name='BART_BiLSTM_2')(model_bart)
    model_bart = Bidirectional(LSTM(units=config['units'], dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout_rate'], return_sequences=True), name='BART_BiLSTM_3')(model_bart)
    model_bart = Reshape((config['no_attributes'], 1), name='BART_Reshape_1')(model_bart)
    model_bart = Conv1D(filters=config['filters'], kernel_size=config['kernel_size'], activation='relu', name='BART_CNN_1')(model_bart)
    model_bart = MaxPooling1D(name='BART_MaxPooling_1')(model_bart)
    model_bart = Flatten(name='BART_Flatten_1')(model_bart)

    # RoBERTa Branch
    input_roberta = Input(shape=(1, ROBERTA_EMBEDDING_DIM), name='RoBERTa_Input')
    model_roberta = Bidirectional(LSTM(units=config['units'], dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout_rate'], return_sequences=True), name='RoBERTa_BiLSTM_1')(input_roberta)
    model_roberta = Bidirectional(LSTM(units=config['units'], dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout_rate'], return_sequences=True), name='RoBERTa_BiLSTM_2')(model_roberta)
    model_roberta = Bidirectional(LSTM(units=config['units'], dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout_rate'], return_sequences=True), name='RoBERTa_BiLSTM_3')(model_roberta)
    model_roberta = Reshape((config['no_attributes'], 1), name='RoBERTa_Reshape_1')(model_roberta)
    model_roberta = Conv1D(filters=config['filters'], kernel_size=config['kernel_size'], activation='relu', name='RoBERTa_CNN_1')(model_roberta)
    model_roberta = MaxPooling1D(name='RoBERTa_MaxPooling_1')(model_roberta)
    model_roberta = Flatten(name='RoBERTa_Flatten_1')(model_roberta)

    # Combine Branches
    combined = Concatenate(name='Model_Concat')([model_bart, model_roberta])
    
    # Final Layers
    combined = Dense(config['units'], activation='relu', name='Dense_1')(combined)
    combined = Dense(config['units'], activation='relu', name='Dense_2')(combined)
    output = Dense(num_classes, activation='softmax', name='Output')(combined)

    # Create Model
    model = Model(inputs=[input_bart, input_roberta], outputs=output, name='MisRoBERTa_Hybrid_Model')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("\n✅ Model architecture created and compiled.")
    model.summary()

    # --- Training Loop ---
    for idx in range(NUM_ITER):
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Starting Training - Iteration {idx+1}/{NUM_ITER}")
        print(f"{'='*80}")

        # Dataset split
        X_train, X_test, y_train, y_test = splitDataset(X, y)
        y_vec_train, y_vec_test = prepareYTrainTestNN(y_train, y_test, num_classes)

        # Get encodings
        print("Getting BART Encoding...")
        x_vec_train_bart, x_vec_test_bart = getBARTEncoding(X_train, X_test, USE_CUDA)
        
        print("Getting RoBERTa Encoding...")
        x_vec_train_roberta, x_vec_test_roberta = getRoBERTaEncoding(X_train, X_test, USE_CUDA)

        # Early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config['patience'])

        # Train model
        print(f"\n🔄 Starting model training (Epoch: {config['epochs_n']})...")
        history = model.fit(
            x=[x_vec_train_bart, x_vec_train_roberta],
            y=y_vec_train,
            epochs=config['epochs_n'],
            verbose=True,
            validation_data=([x_vec_test_bart, x_vec_test_roberta], y_vec_test),
            batch_size=config['batch_size'],
            callbacks=[es]
        )

        # Evaluate
        y_pred = model.predict([x_vec_test_bart, x_vec_test_roberta], verbose=False)
        evaluate(y_vec_test, y_pred, modelName=model.name, iters=idx)
        
        # Test random samples
        test_random_samples(X_test, y_test, y_pred, id2label)

        end_time = time.time()
        execution_time.append(end_time - start_time)
        print(f"⏱️ Iteration time: {end_time - start_time:.2f} seconds")

    # --- Results Summary ---
    print("\n" + "="*80)
    print("📊 Training Complete - Overall Results")
    print("="*80)
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision (Micro-average): {np.mean(precisions_micro):.4f} ± {np.std(precisions_micro):.4f}")
    print(f"Precision (Macro-average): {np.mean(precisions_macro):.4f} ± {np.std(precisions_macro):.4f}")
    print(f"Recall (Micro-average): {np.mean(recalls_micro):.4f} ± {np.std(recalls_micro):.4f}")
    print(f"Recall (Macro-average): {np.mean(recalls_macro):.4f} ± {np.std(recalls_macro):.4f}")

    # Save model and results
    print("\n" + "="*80)
    print("💾 Saving model and results...")
    print("="*80)
    save_model_and_results(model, id2label, config)
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)