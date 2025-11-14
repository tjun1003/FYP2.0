# -*- coding: utf-8 -*-
"""
Model Wrapper
Handles model loading, encoding, prediction logic, and RAG integration.
"""
import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from keras.models import load_model
from sentence_transformers import SentenceTransformer
from simpletransformers.language_representation import RepresentationModel

from mongodb_manager import MongoDBManager
from input_validator import InputValidator, validator as input_validator
from web_retriever import WebRetriever

logger = logging.getLogger(__name__)

class ModelConfig:
    USE_DEFAULT_CACHE = True
    LOCAL_MODEL_PATHS = {'bart': None, 'roberta': None, 'minilm': None}
    CACHE_DIR = os.getenv('TRANSFORMERS_CACHE', None)
    BART_MODEL_NAME = os.getenv('BART_MODEL_NAME', 'facebook/bart-large')
    ROBERTA_MODEL_NAME = os.getenv('ROBERTA_MODEL_NAME', 'roberta-base')
    MINILM_MODEL_NAME = os.getenv('MINILM_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
    USE_CUDA = False
    ENABLE_SEMANTIC_CHECK = os.getenv('ENABLE_SEMANTIC_CHECK', 'false').lower() == 'true'


class ModelWrapper:
    def __init__(self, model_path: str, label_path: str, config: ModelConfig = None, 
                 mongodb_manager: MongoDBManager = None):
        if config is None:
            config = ModelConfig()
        self.config = config
        self.retriever = WebRetriever()
        self.mongodb = mongodb_manager
        
        logger.info("🚀 Starting model loading...")
        self.model = load_model(model_path)
        with open(label_path, 'rb') as f:
            self.id2label = pickle.load(f)
        self.label2id = {v: k for k, v in self.id2label.items()}
        self._load_encoders()
        logger.info("✅ Model and encoders loaded successfully!")

    def _load_encoders(self):
        """Load BART, RoBERTa, and MiniLM encoders"""
        
        # Set cache directory (if specified)
        if self.config.CACHE_DIR:
            os.environ['TRANSFORMERS_CACHE'] = self.config.CACHE_DIR
            os.environ['HF_HOME'] = self.config.CACHE_DIR
            logger.info(f"📁 Using custom cache directory: {self.config.CACHE_DIR}")
        
        # ===== Load MiniLM Text Encoder (Optional local or HF) =====
        self.text_encoder = None
        try:
            minilm_name = self.config.MINILM_MODEL_NAME
            logger.info("🔥 Loading MiniLM encoder...")

            if not self.config.USE_DEFAULT_CACHE and self.config.LOCAL_MODEL_PATHS.get('minilm'):
                minilm_path = self.config.LOCAL_MODEL_PATHS['minilm']
                logger.info(f"   Loading from local path: {minilm_path}")

                if not Path(minilm_path).exists():
                    raise FileNotFoundError(f"MiniLM model path does not exist: {minilm_path}")

                self.text_encoder = SentenceTransformer(minilm_path)
            else:
                logger.info(f"   Loading from Hugging Face: {minilm_name}")
                self.text_encoder = SentenceTransformer(minilm_name, cache_folder=self.config.CACHE_DIR)

            logger.info("✅ MiniLM encoder loaded successfully")
            
            # Initialize semantic check in InputValidator
            if self.config.ENABLE_SEMANTIC_CHECK:
                try:
                    # Calculate reference embeddings for the validator
                    ref_embeddings = self.text_encoder.encode(input_validator.ref_sentences, convert_to_tensor=True)
                    input_validator.set_semantic_model(self.text_encoder, ref_embeddings)
                    logger.info("✓ InputValidator semantic check enabled.")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize semantic check in InputValidator: {e}")
                    input_validator.set_semantic_model(None, None)

        except Exception as e:
            logger.warning(f"⚠️ MiniLM encoder failed to load, semantic check disabled: {e}")
            input_validator.set_semantic_model(None, None)


        # ===== Load BART Encoder =====
        bart_name = self.config.BART_MODEL_NAME
        logger.info("🔥 Loading BART encoder...")
        
        if not self.config.USE_DEFAULT_CACHE and self.config.LOCAL_MODEL_PATHS['bart']:
            bart_path = self.config.LOCAL_MODEL_PATHS['bart']
            logger.info(f"   Loading from local path: {bart_path}")
            
            if not Path(bart_path).exists():
                raise FileNotFoundError(f"BART model path does not exist: {bart_path}")
            
            self.bart_model = SentenceTransformer(bart_path)
        else:
            logger.info(f"   Loading from Hugging Face: {bart_name}")
            if self.config.CACHE_DIR:
                logger.info(f"   Cache directory: {self.config.CACHE_DIR}")
            else:
                logger.info(f"   Using default cache: ~/.cache/huggingface/")
            
            self.bart_model = SentenceTransformer(
                bart_name,
                cache_folder=self.config.CACHE_DIR
            )
        
        logger.info("✅ BART encoder loaded successfully")
        
        # ===== Load RoBERTa Encoder =====
        roberta_name = self.config.ROBERTA_MODEL_NAME
        logger.info("🔥 Loading RoBERTa encoder...")
        
        if not self.config.USE_DEFAULT_CACHE and self.config.LOCAL_MODEL_PATHS['roberta']:
            roberta_path = self.config.LOCAL_MODEL_PATHS['roberta']
            logger.info(f"   Loading from local path: {roberta_path}")
            
            if not Path(roberta_path).exists():
                raise FileNotFoundError(f"RoBERTa model path does not exist: {roberta_path}")
            
            self.roberta_model = RepresentationModel(
                model_type="roberta",
                model_name=roberta_path,
                use_cuda=self.config.USE_CUDA
            )
        else:
            logger.info(f"   Loading from Hugging Face: {roberta_name}")
            if self.config.CACHE_DIR:
                logger.info(f"   Cache directory: {self.config.CACHE_DIR}")
            else:
                logger.info(f"   Using default cache: ~/.cache/huggingface/")
            
            self.roberta_model = RepresentationModel(
                model_type="roberta",
                model_name=roberta_name,
                use_cuda=self.config.USE_CUDA,
                args={'cache_dir': self.config.CACHE_DIR} if self.config.CACHE_DIR else {}
            )
        
        logger.info("✅ RoBERTa encoder loaded successfully")

    def _get_encodings(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get BART and RoBERTa encodings for a list of texts"""
        
        # BART Encoding
        bart_embeddings = self.bart_model.encode(texts, show_progress_bar=False)
        x_vec_bart = bart_embeddings.reshape((bart_embeddings.shape[0], 1, bart_embeddings.shape[1]))
        
        # RoBERTa Encoding
        roberta_embeddings = self.roberta_model.encode_sentences(texts, combine_strategy="mean")
        x_vec_roberta = roberta_embeddings.reshape((roberta_embeddings.shape[0], 1, roberta_embeddings.shape[1]))
        
        return x_vec_bart, x_vec_roberta

#        Perform prediction for a single text. 
    def predict(self, text: str, context_text: Optional[str] = None) -> Dict[str, Any]:
    
        # 1. Prepare input text
        if context_text:
            input_text = f"Context: {context_text}. Statement: {text}"
            logger.info(f"RAG enabled. Input length: {len(input_text)}")
        else:
            input_text = text
        
        texts = [input_text]
        
        # 2. Get encodings
        x_vec_bart, x_vec_roberta = self._get_encodings(texts)
        
        # 3. Predict
        predictions = self.model.predict([x_vec_bart, x_vec_roberta], verbose=0)[0]
        
        # 4. Process results
        predicted_class_id = np.argmax(predictions)
        predicted_label = self.id2label[predicted_class_id]
        confidence = float(predictions[predicted_class_id] * 100)
        
        probabilities = {
            self.id2label[i]: float(p * 100)
            for i, p in enumerate(predictions)
        }
        
        return {
            'predicted_label': predicted_label,
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Perform batch prediction for a list of texts"""
        
        # 1. Get encodings
        x_vec_bart, x_vec_roberta = self._get_encodings(texts)
        
        # 2. Predict
        predictions = self.model.predict([x_vec_bart, x_vec_roberta], verbose=0)
        
        # 3. Process results
        results = []
        for text, preds in zip(texts, predictions):
            predicted_class_id = np.argmax(preds)
            predicted_label = self.id2label[predicted_class_id]
            confidence = float(preds[predicted_class_id] * 100)
            
            probabilities = {
                self.id2label[i]: float(p * 100)
                for i, p in enumerate(preds)
            }
            
            results.append({
                'text': text,
                'predicted_label': predicted_label,
                'confidence': round(confidence, 2),
                'probabilities': probabilities
            })
            
        return results