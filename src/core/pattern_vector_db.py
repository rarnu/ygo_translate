#!/usr/bin/env python3
"""
翻译范式向量数据库
使用FAISS存储和检索翻译范式
"""

import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os

@dataclass
class PatternMatch:
    """范式匹配结果"""
    pattern_id: str
    source_pattern: str
    target_pattern: str
    category: str
    confidence: float
    similarity: float
    examples: List[Tuple[str, str]]
    match_type: str = "similarity"  # 添加匹配类型字段：exact, substring, superstring, term_based, similarity


class JapaneseToChinesePatternDB:
    """日译中范式数据库"""
    
    def __init__(self, index_type: str = "flat", dimension: int = 768):
        self.index_type = index_type
        self.dimension = dimension
        self.index = None
        self.patterns = {}  # pattern_id -> pattern_data
        self.embeddings = {}  # pattern_id -> embedding
        self.pattern_ids = []  # 保持与FAISS索引一致的pattern_id列表
        self._create_index()
    
    def _create_index(self):
        """创建FAISS索引"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def add_patterns(self, patterns: List[Dict], embeddings: List[np.ndarray]):
        """添加范式到数据库"""
        if len(patterns) != len(embeddings):
            raise ValueError("范式数量与向量数量不匹配")
        
        # 只添加日译中的范式
        valid_patterns = []
        valid_embeddings = []
        
        for i, pattern in enumerate(patterns):
            if (pattern.get('source_lang') == 'ja' and 
                pattern.get('target_lang') == 'zh'):
                self.patterns[pattern['id']] = pattern
                self.pattern_ids.append(pattern['id'])  # 保持顺序
                valid_embeddings.append(embeddings[i])
                valid_patterns.append(pattern)
        
        # 添加到FAISS索引
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings).astype('float32')
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
            self.index.add(embeddings_array)
            print(f"已添加 {len(valid_patterns)} 个日译中范式到数据库")
    
    def search_similar_patterns(self, query_embedding: np.ndarray, top_k: int = 5) -> List[PatternMatch]:
        """搜索相似的日译中范式"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1)
        
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        matches = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            
            # 使用pattern_ids列表来确保索引一致性
            if idx < len(self.pattern_ids):
                pattern_id = self.pattern_ids[idx]
                pattern = self.patterns[pattern_id]
                
                match = PatternMatch(
                    pattern_id=pattern_id,
                    source_pattern=pattern['source_pattern'],
                    target_pattern=pattern['target_pattern'],
                    category=pattern['category'],
                    confidence=pattern['confidence'],
                    similarity=float(similarity),
                    examples=pattern.get('examples', [])
                )
                matches.append(match)
        
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]
    
    def save_to_file(self, base_path: str):
        """保存数据库到文件"""
        index_path = f"{base_path}.index"
        faiss.write_index(self.index, index_path)
        
        metadata_path = f"{base_path}_metadata.json"
        metadata = {
            'patterns': self.patterns,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_patterns': len(self.patterns)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"日译中范式数据库已保存: {base_path}")
    
    def load_from_file(self, base_path: str):
        """从文件加载数据库"""
        index_path = f"{base_path}.index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        self.index = faiss.read_index(index_path)
        
        metadata_path = f"{base_path}_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.patterns = metadata['patterns']
        self.index_type = metadata['index_type']
        self.dimension = metadata['dimension']
        
        # 重建pattern_ids列表，保持与FAISS索引一致的顺序
        self.pattern_ids = []
        for pattern_id, pattern_data in self.patterns.items():
            if (pattern_data.get('source_lang') == 'ja' and 
                pattern_data.get('target_lang') == 'zh'):
                self.pattern_ids.append(pattern_id)
        
        print(f"已加载日译中范式数据库: {len(self.patterns)} 个范式")


class ChineseToJapanesePatternDB:
    """中译日范式数据库"""
    
    def __init__(self, index_type: str = "flat", dimension: int = 768):
        self.index_type = index_type
        self.dimension = dimension
        self.index = None
        self.patterns = {}  # pattern_id -> pattern_data
        self.embeddings = {}  # pattern_id -> embedding
        self.pattern_ids = []  # 保持与FAISS索引一致的pattern_id列表
        self._create_index()
    
    def _create_index(self):
        """创建FAISS索引"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def add_patterns(self, patterns: List[Dict], embeddings: List[np.ndarray]):
        """添加范式到数据库"""
        if len(patterns) != len(embeddings):
            raise ValueError("范式数量与向量数量不匹配")
        
        # 只添加中译日的范式
        valid_patterns = []
        valid_embeddings = []
        
        for i, pattern in enumerate(patterns):
            if (pattern.get('source_lang') == 'zh' and 
                pattern.get('target_lang') == 'ja'):
                self.patterns[pattern['id']] = pattern
                self.pattern_ids.append(pattern['id'])  # 保持顺序
                valid_embeddings.append(embeddings[i])
                valid_patterns.append(pattern)
        
        # 添加到FAISS索引
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings).astype('float32')
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
            self.index.add(embeddings_array)
            print(f"已添加 {len(valid_patterns)} 个中译日范式到数据库")
    
    def search_similar_patterns(self, query_embedding: np.ndarray, top_k: int = 5) -> List[PatternMatch]:
        """搜索相似的中译日范式"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1)
        
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        matches = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            
            # 使用pattern_ids列表来确保索引一致性
            if idx < len(self.pattern_ids):
                pattern_id = self.pattern_ids[idx]
                pattern = self.patterns[pattern_id]
                
                match = PatternMatch(
                    pattern_id=pattern_id,
                    source_pattern=pattern['source_pattern'],
                    target_pattern=pattern['target_pattern'],
                    category=pattern['category'],
                    confidence=pattern['confidence'],
                    similarity=float(similarity),
                    examples=pattern.get('examples', [])
                )
                matches.append(match)
        
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]
    
    def save_to_file(self, base_path: str):
        """保存数据库到文件"""
        index_path = f"{base_path}.index"
        faiss.write_index(self.index, index_path)
        
        metadata_path = f"{base_path}_metadata.json"
        metadata = {
            'patterns': self.patterns,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_patterns': len(self.patterns)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"中译日范式数据库已保存: {base_path}")
    
    def load_from_file(self, base_path: str):
        """从文件加载数据库"""
        index_path = f"{base_path}.index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        self.index = faiss.read_index(index_path)
        
        metadata_path = f"{base_path}_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.patterns = metadata['patterns']
        self.index_type = metadata['index_type']
        self.dimension = metadata['dimension']
        
        # 重建pattern_ids列表，保持与FAISS索引一致的顺序
        self.pattern_ids = []
        for pattern_id, pattern_data in self.patterns.items():
            if (pattern_data.get('source_lang') == 'zh' and 
                pattern_data.get('target_lang') == 'ja'):
                self.pattern_ids.append(pattern_id)
        
        print(f"已加载中译日范式数据库: {len(self.patterns)} 个范式")


def create_pattern_embeddings(patterns: List[Dict]) -> List[np.ndarray]:
    """
    为范式创建向量表示
    这是一个简化的实现，实际应该使用专门的embedding模型
    
    Args:
        patterns: 范式列表
        
    Returns:
        向量列表
    """
    embeddings = []
    
    for pattern in patterns:
        # 简单的文本特征向量化
        text = f"{pattern['source_pattern']} {pattern['target_pattern']}"
        
        # 创建一个基于字符的简单向量
        # 实际应用中应该使用预训练的语言模型
        vector = np.zeros(768)
        
        # 简单的特征提取
        features = [
            len(text),  # 文本长度
            text.count('「卡名」'),  # 卡名占位符数量
            text.count('N'),  # 数字占位符数量
            hash(text) % 1000 / 1000,  # 文本哈希特征
        ]
        
        # 填充向量前几个维度
        for i, feature in enumerate(features):
            if i < len(vector):
                vector[i] = feature
        
        # 其余维度用简单的字符编码填充
        char_features = []
        for char in text[:100]:  # 只取前100个字符
            char_features.append(ord(char) / 65536.0)
        
        for i, char_feature in enumerate(char_features):
            idx = len(features) + i
            if idx < len(vector):
                vector[idx] = char_feature
        
        embeddings.append(vector)
    
    return embeddings