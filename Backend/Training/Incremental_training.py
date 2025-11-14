# -*- coding: utf-8 -*-
"""
MisRoBÆRTa Hybrid Neural Network Model - Incremental Training Version
Functionality:
1. Load saved model to continue training
2. Support additional training with new data
3. Save the updated model
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import time
import numpy as np
import pandas as pd
import random as rnd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from simpletransformers.language_representation import RepresentationModel
from sentence_transformers import SentenceTransformer

# --- Configuration (Consistent with original training) ---
config = {
    'epochs_n': 50,  # Incremental training can reduce epochs
    'batch_size': 1000,
    'patience': 10,
    'test_size': 0.30,
}

# --- Global Metrics List ---
accuracies = []
precisions_micro = []
precisions_macro = []
recalls_micro = []
recalls_macro = []
execution_time = []

# --- Helper Functions ---
def load_saved_model(model_path, label_path, config_path):
    """Load saved model, label map, and configuration"""
    print("\n" + "="*80)
    print("📥 Loading saved model...")
    print("="*80)
    
    # 1. Load model
    model = load_model(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # 2. Load label map
    with open(label_path, 'rb') as f:
        id2label = pickle.load(f)
    print(f"✅ Label map loaded: {label_path}")
    print(f"   Number of classes: {len(id2label)}")
    
    # 3. Load configuration (Optional)
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        print(f"✅ Configuration loaded: {config_path}")
        # Merge configuration (new config overwrites old config)
        saved_config.update(config)
        return model, id2label, saved_config
    
    return model, id2label, config

def evaluate(y_test, y_pred, modelName='Model', iters=0):
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
    return y_p, y_t

def splitDataset(X, y, test_size=0.3):
    """Split dataset"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y
    )
    return X_train, X_test, np.array(y_train), np.array(y_test)

def getBARTEncoding(X_train, X_test):
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

def getRoBERTaEncoding(X_train, X_test):
    """Get RoBERTa Encoding"""
    print("📥 Loading RoBERTa model...")
    model = RepresentationModel(model_type="roberta", model_name="roberta-base", use_cuda=False)
    X_train_RoBERTa = model.encode_sentences(X_train, combine_strategy="mean")
    X_test_RoBERTa = model.encode_sentences(X_test, combine_strategy="mean")
    X_train_RoBERTa = X_train_RoBERTa.reshape((X_train_RoBERTa.shape[0], 1, X_train_RoBERTa.shape[1]))
    X_test_RoBERTa = X_test_RoBERTa.reshape((X_test_RoBERTa.shape[0], 1, X_test_RoBERTa.shape[1]))
    return X_train_RoBERTa, X_test_RoBERTa

def prepareYTrainTestNN(y_train, y_test, num_classes):
    """Prepare Y data for Neural Network"""
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    return y_train_cat, y_test_cat

def save_incremental_model(model, id2label, config, original_path, save_dir="saved_models"):
    """Save the model after incremental training"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save model (labeled as incremental training version)
    model_path = os.path.join(save_dir, f"model_incremental_{timestamp}.h5")
    model.save(model_path)
    print(f"\n✅ Incremental training model saved: {model_path}")
    
    # Save label map
    label_path = os.path.join(save_dir, f"id2label_incremental_{timestamp}.pkl")
    with open(label_path, 'wb') as f:
        pickle.dump(id2label, f)
    print(f"✅ Label map saved: {label_path}")
    
    # Save configuration (including original model path)
    config['original_model_path'] = original_path
    config['incremental_training_timestamp'] = timestamp
    config_path = os.path.join(save_dir, f"config_incremental_{timestamp}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✅ Configuration saved: {config_path}")
    
    # Save evaluation metrics
    metrics = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_micro_mean": float(np.mean(precisions_micro)),
        "precision_macro_mean": float(np.mean(precisions_macro)),
        "recall_micro_mean": float(np.mean(recalls_micro)),
        "recall_macro_mean": float(np.mean(recalls_macro)),
        "execution_time_mean": float(np.mean(execution_time)),
        "all_accuracies": [float(x) for x in accuracies]
    }
    
    metrics_path = os.path.join(save_dir, f"metrics_incremental_{timestamp}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"✅ Evaluation metrics saved: {metrics_path}")
    
    return model_path, label_path, config_path, metrics_path

def merge_datasets(old_data_path, new_data_path):
    """Merge old and new datasets (optional)"""
    print("\n" + "="*80)
    print("🔄 Merging datasets...")
    print("="*80)
    
    old_data = pd.read_csv(old_data_path, encoding="utf-8")
    new_data = pd.read_csv(new_data_path, encoding="utf-8")
    
    print(f"Original dataset size: {len(old_data)}")
    print(f"New dataset size: {len(new_data)}")
    
    # Merge data
    merged_data = pd.concat([old_data, new_data], ignore_index=True)
    print(f"Merged dataset size: {len(merged_data)}")
    
    return merged_data

# --- Main Execution Block ---
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:")
        print("  Train with new data only: python incremental_training.py <model_path.h5> <label_path.pkl> <config_path.json> <new_data.csv> [num_iterations]")
        print("  Train with merged data: python incremental_training.py <model_path.h5> <label_path.pkl> <config_path.json> <new_data.csv> <old_data.csv> [num_iterations]")
        sys.exit(1)

    # Parse arguments
    MODEL_PATH = sys.argv[1]
    LABEL_PATH = sys.argv[2]
    CONFIG_PATH = sys.argv[3]
    NEW_DATA_PATH = sys.argv[4]
    
    # Check if old data path is provided
    if len(sys.argv) >= 6 and sys.argv[5].endswith('.csv'):
        OLD_DATA_PATH = sys.argv[5]
        NUM_ITER = int(sys.argv[6]) if len(sys.argv) >= 7 else 1
        USE_MERGED = True
    else:
        OLD_DATA_PATH = None
        NUM_ITER = int(sys.argv[5]) if len(sys.argv) >= 6 else 1
        USE_MERGED = False

    # Load model
    model, id2label, loaded_config = load_saved_model(MODEL_PATH, LABEL_PATH, CONFIG_PATH)
    config.update(loaded_config)
    
    # Load data
    if USE_MERGED:
        print("\n📊 Using merged dataset mode")
        dataSet = merge_datasets(OLD_DATA_PATH, NEW_DATA_PATH)
    else:
        print("\n📊 Using new dataset only mode")
        dataSet = pd.read_csv(NEW_DATA_PATH, encoding="utf-8")
        print(f"New dataset size: {len(dataSet)}")
    
    # Process labels
    num_classes = len(id2label)
    print(f"\nClass mapping: {id2label}")
    
    # Check for new classes in new data
    new_labels = dataSet['label'].unique()
    has_new_labels = False
    for label in new_labels:
        if label not in id2label.values():
            print(f"⚠️ Warning: New class found '{label}', but model does not support new classes")
            has_new_labels = True
    
    if has_new_labels:
        print("❌ Error: Incremental training does not support new classes. Please use the original script for retraining.")
        sys.exit(1)
    
    # Convert labels to numbers
    label_to_id = {v: k for k, v in id2label.items()}
    dataSet['label'] = dataSet['label'].map(label_to_id)
    
    X = dataSet['content'].astype(str).to_list()
    y = dataSet['label'].astype(int).to_list()

    # Recompile model (ensure correct optimizer state is used)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("\n✅ Model recompiled")

    # --- Incremental Training Loop ---
    for idx in range(NUM_ITER):
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Starting Incremental Training - Iteration {idx+1}/{NUM_ITER}")
        print(f"{'='*80}")

        # Dataset split
        X_train, X_test, y_train, y_test = splitDataset(X, y, config['test_size'])
        y_vec_train, y_vec_test = prepareYTrainTestNN(y_train, y_test, num_classes)

        # Get encodings
        print("Getting BART Encoding...")
        x_vec_train_bart, x_vec_test_bart = getBARTEncoding(X_train, X_test)
        
        print("Getting RoBERTa Encoding...")
        x_vec_train_roberta, x_vec_test_roberta = getRoBERTaEncoding(X_train, X_test)

        # Early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config['patience'])

        # Continue training
        print(f"\n🔄 Performing incremental training (Epoch: {config['epochs_n']})...")
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

        end_time = time.time()
        execution_time.append(end_time - start_time)
        print(f"⏱️ Iteration time: {end_time - start_time:.2f} seconds")

    # --- Results Summary ---
    print("\n" + "="*80)
    print("📊 Incremental Training Complete - Overall Results")
    print("="*80)
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision (Micro-average): {np.mean(precisions_micro):.4f} ± {np.std(precisions_micro):.4f}")
    print(f"Precision (Macro-average): {np.mean(precisions_macro):.4f} ± {np.std(precisions_macro):.4f}")
    print(f"Recall (Micro-average): {np.mean(recalls_micro):.4f} ± {np.std(recalls_micro):.4f}")
    print(f"Recall (Macro-average): {np.mean(recalls_macro):.4f} ± {np.std(recalls_macro):.4f}")

    # Save the incrementally trained model
    print("\n" + "="*80)
    print("💾 Saving incrementally trained model...")
    print("="*80)
    save_incremental_model(model, id2label, config, MODEL_PATH)
    
    print("\n" + "="*80)
    print("✅ Incremental Training Complete!")
    print("="*80)