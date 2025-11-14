# -*- coding: utf-8 -*-
"""
MongoDB Configuration and Manager
Handles connection to MongoDB Atlas and data operations (save, import, stats).
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from pymongo import MongoClient, errors
from pymongo.server_api import ServerApi

logger = logging.getLogger(__name__)

# ===== MongoDB Configuration =====
class MongoDBConfig:
    """MongoDB Atlas Configuration"""
    CONNECTION_STRING = os.getenv(
        'MONGODB_URI',
        'mongodb+srv://Jun_Fyp2:Juns170275@jun.wo9qn0p.mongodb.net/?retryWrites=true&w=majority'
    )
    DATABASE_NAME = os.getenv('MONGODB_DATABASE', 'Model_Training')
    COLLECTION_NAME = os.getenv('MONGODB_COLLECTION', 'User_Record')
    ENABLED = os.getenv('MONGODB_ENABLED', 'true').lower() == 'true'


# ===== MongoDB Manager =====
class MongoDBManager:
    """MongoDB Atlas Data Manager"""
    
    def __init__(self, config: MongoDBConfig = None):
        if config is None:
            config = MongoDBConfig()
        
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
        
        if self.config.ENABLED:
            self._connect()
    
    def _connect(self):
        """Connect to MongoDB Atlas"""
        try:
            self.client = MongoClient(
                self.config.CONNECTION_STRING,
                server_api=ServerApi('1'),
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ Successfully connected to MongoDB Atlas")
            
            # Select database and collection
            self.db = self.client[self.config.DATABASE_NAME]
            self.collection = self.db[self.config.COLLECTION_NAME]
            self.connected = True
            
            logger.info(f"📦 Using database: {self.config.DATABASE_NAME}, collection: {self.config.COLLECTION_NAME}")
            
        except errors.ServerSelectionTimeoutError as e:
            logger.error(f"❌ Failed to connect to MongoDB Atlas: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ MongoDB initialization failed: {e}")
            self.connected = False
    
    def save_prediction(self, text: str, label: str, confidence: float, 
                       probabilities: Dict[str, float], context: Dict = None) -> Optional[str]:
        """
        Save prediction result to MongoDB
        
        Returns: Inserted document ID (string) or None
        """
        if not self.connected:
            logger.warning("⚠️ MongoDB not connected, skipping save")
            return None
        
        try:
            document = {
                'content': text,
                'label': label,
                'confidence': confidence,
                'probabilities': probabilities,
                'context': context,
                'created_at': datetime.utcnow(),
                'source': 'api_prediction'
            }
            
            result = self.collection.insert_one(document)
            logger.info(f"✅ Prediction result saved to MongoDB: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to save to MongoDB: {e}")
            return None
    
    def import_from_list(self, data_list: List[Dict[str, str]]) -> int:
        """Batch import data"""
        if not self.connected:
            logger.warning("⚠️ MongoDB not connected, skipping import")
            return 0
        
        try:
            documents = []
            for item in data_list:
                if 'label' in item and 'content' in item:
                    doc = {
                        'label': item['label'],
                        'content': item['content'],
                        'created_at': datetime.utcnow(),
                        'source': 'bulk_import'
                    }
                    # Preserve other fields
                    for key, value in item.items():
                        if key not in ['label', 'content']:
                            doc[key] = value
                    documents.append(doc)
            
            if documents:
                result = self.collection.insert_many(documents)
                count = len(result.inserted_ids)
                logger.info(f"✅ Successfully imported {count} records to MongoDB")
                return count
            else:
                logger.warning("⚠️ No valid data to import")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Batch import failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.connected:
            return {'error': 'MongoDB not connected'}
        
        try:
            total_count = self.collection.count_documents({})
            
            # Group by label
            pipeline = [
                {"$group": {"_id": "$label", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            label_stats = list(self.collection.aggregate(pipeline))
            
            # Group by source
            source_pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            source_stats = list(self.collection.aggregate(source_pipeline))
            
            return {
                'total_documents': total_count,
                'label_distribution': {item['_id']: item['count'] for item in label_stats},
                'source_distribution': {item['_id']: item['count'] for item in source_stats}
            }
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def query_by_label(self, label: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query data by label"""
        if not self.connected:
            return []
        
        try:
            results = list(self.collection.find({'label': label}).limit(limit))
            for doc in results:
                doc['_id'] = str(doc['_id'])
            return results
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return []
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("🔌 MongoDB connection closed")