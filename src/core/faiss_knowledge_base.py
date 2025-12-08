import json
import re
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from collections import defaultdict


@dataclass
class TranslationPair:
    """翻译对数据结构"""
    original: str
    translated: str
    metadata: Optional[Dict] = None
    id: str = ""
    category: str = ""
    priority: float = 1.0


class FAISSKnowledgeBase:
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 index_type: str = "flat",
                 max_results: int = 100):
        """
        初始化FAISS知识库
        
        Args:
            model_name: 语义模型名称
            embedding_dim: 向量维度
            index_type: 索引类型 ("flat", "hnsw", "ivf")
            max_results: 最大返回结果数
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.max_results = max_results
        
        # 存储数据
        self.pairs: List[TranslationPair] = []
        self.category_index: Dict[str, List[int]] = defaultdict(list)
        self.id_to_index: Dict[str, int] = {}
        
        # FAISS索引
        self.faiss_index = None
        self.embeddings = None
        self.is_built = False
        
        # 初始化FAISS索引
        self._init_faiss_index()
    
    def _init_faiss_index(self):
        """初始化FAISS索引"""
        if self.index_type == "flat":
            # 精确搜索，最准确但最慢
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "hnsw":
            # 层次化小世界图，快速且准确
            self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            # HNSW参数调优
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.hnsw.efSearch = 50
        elif self.index_type == "ivf":
            # 倒排文件索引，适合超大规模数据
            # 根据数据量动态调整聚类中心数量
            nlist = min(100, max(10, len(self.pairs) // 10))
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
    
    def add_translation_pair(self, 
                            original: str, 
                            translated: str, 
                            metadata: Optional[Dict] = None,
                            category: str = "",
                            priority: float = 1.0) -> str:
        """添加翻译对到知识库"""
        # 生成唯一ID
        pair_id = hashlib.md5(f"{original}||{translated}".encode()).hexdigest()[:8]
        
        pair = TranslationPair(
            original=original.strip(),
            translated=translated.strip(),
            metadata=metadata or {},
            id=pair_id,
            category=category,
            priority=priority
        )
        
        # 添加到列表
        index = len(self.pairs)
        self.pairs.append(pair)
        self.id_to_index[pair_id] = index
        
        # 分类索引
        if category:
            self.category_index[category].append(index)
        
        # 标记需要重建索引
        self.is_built = False
        
        return pair_id
    
    def batch_add_pairs(self, pairs_data: List[Dict]):
        """批量添加翻译对（性能优化版本）"""
        batch_originals = []
        batch_translated = []
        
        for data in pairs_data:
            original = data['original']
            translated = data['translated']
            metadata = data.get('metadata', {})
            category = data.get('category', '')
            priority = data.get('priority', 1.0)
            
            pair_id = hashlib.md5(f"{original}||{translated}".encode()).hexdigest()[:8]
            
            pair = TranslationPair(
                original=original.strip(),
                translated=translated.strip(),
                metadata=metadata,
                id=pair_id,
                category=category,
                priority=priority
            )
            
            index = len(self.pairs)
            self.pairs.append(pair)
            self.id_to_index[pair_id] = index
            
            if category:
                self.category_index[category].append(index)
            
            batch_originals.append(original)
            batch_translated.append(translated)
        
        # 批量编码（更高效）
        if batch_originals:
            new_embeddings = self.embedding_model.encode(
                batch_originals, 
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            if self.embeddings is None:
                self.embeddings = new_embeddings.astype('float32')
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings.astype('float32')])
        
        self.is_built = False

    def find_pair_id(self, original: str, translated: str) -> str | None:
        """查找翻译对ID"""
        pair_id = hashlib.md5(f"{original}||{translated}".encode()).hexdigest()[:8]
        if pair_id not in self.id_to_index:
            return None
        return pair_id

    def find_original_list(self, original: str) -> list[str]:
        """查找原始文本对应的pair_id列表"""
        pair_list = []
        for pair in self.pairs:
            if pair.original == original or pair.translated == original:
                pair_list.append(pair.id)
        return pair_list

    def delete_pair(self, pair_id: str)-> bool:
        """删除翻译对"""
        if pair_id not in self.id_to_index:
            return False

        index = self.id_to_index[pair_id]
        pair = self.pairs[index]

        # 从分类索引中移除
        if pair.category and pair.category in self.category_index:
            category_indices = self.category_index[pair.category]
            if index in category_indices:
                category_indices.remove(index)
            # 如果类别为空，移除该类别
            if not category_indices:
                del self.category_index[pair.category]

        # 从pairs列表中移除
        del self.pairs[index]
        # 从id_to_index中移除
        self.id_to_index.pop(pair_id)

        # 更新所有大于被删除索引的索引映射
        new_id_to_index = {}
        for i, updated_pair in enumerate(self.pairs):
            new_id_to_index[updated_pair.id] = i
        self.id_to_index = new_id_to_index

        # 更新分类索引中的所有索引值
        for category in self.category_index:
            updated_indices = []
            for idx in self.category_index[category]:
                if idx < index:
                    updated_indices.append(idx)
                elif idx > index:
                    updated_indices.append(idx - 1)
            self.category_index[category] = updated_indices

        # 从embeddings中移除对应的向量（如果存在）
        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, index, axis=0)

        # 标记需要重建FAISS索引
        self.is_built = False

        return True


    def build_index(self):
        """构建FAISS索引"""
        if not self.pairs:
            raise ValueError("知识库为空，无法构建索引")
        
        print(f"正在构建FAISS索引，数据量: {len(self.pairs)}")
        
        # 如果还没有embeddings，则生成
        if self.embeddings is None:
            print("正在生成文本嵌入...")
            originals = [pair.original for pair in self.pairs]
            self.embeddings = self.embedding_model.encode(
                originals, 
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            ).astype('float32')
        
        # 构建索引
        if self.index_type == "ivf":
            # IVF索引需要先训练
            print("正在训练IVF索引...")
            
            # 检查数据量是否足够训练IVF索引
            n_training_points = len(self.embeddings)
            n_clusters = self.faiss_index.nlist
            
            if n_training_points < n_clusters:
                print(f"⚠️  训练数据不足 ({n_training_points} < {n_clusters} 聚类中心)")
                print("🔄 自动切换到Flat索引...")
                # 切换到Flat索引
                self.index_type = "flat"
                self._init_faiss_index()
            else:
                self.faiss_index.train(self.embeddings)
        
        print("正在添加向量到索引...")
        self.faiss_index.add(self.embeddings)
        
        self.is_built = True
        print(f"✓ FAISS索引构建完成，类型: {self.index_type}")
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = 10,
                      category_filter: Optional[str] = None,
                      min_similarity: float = 0.1) -> List[Tuple[TranslationPair, float]]:
        """搜索相似的翻译对"""
        if not self.pairs:
            print("Debug - 知识库为空，无法搜索")
            return []
            
        if not self.is_built:
            try:
                self.build_index()
            except Exception as e:
                print(f"Debug - 构建索引失败: {e}")
                return []
        
        # 1. 首先检查精确匹配
        exact_matches = []
        for i, pair in enumerate(self.pairs):
            if pair.original.strip() == query.strip():
                # 类别过滤
                if category_filter and pair.category != category_filter:
                    continue
                
                # 精确匹配给予最高相似度和优先级
                exact_matches.append((pair, 10.0 * pair.priority))  # 10.0为精确匹配基础分
        
        # 如果有精确匹配，优先返回
        if exact_matches:
            exact_matches.sort(key=lambda x: x[1], reverse=True)
            return exact_matches[:top_k]
        
        # 2. 没有精确匹配时进行向量搜索
        # 编码查询
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True
        ).astype('float32')
        
        # 搜索
        search_k = min(top_k * 3, self.faiss_index.ntotal)  # 搜索更多候选
        distances, indices = self.faiss_index.search(query_embedding, search_k)
        
        # 转换距离为相似度（L2距离转余弦相似度）
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # 无效索引
                continue
            
            # 计算相似度（改进版：更好的距离到相似度转换）
            similarity = max(0.0, 1.0 - dist / 2.0)
            
            if similarity < min_similarity:
                continue
            
            pair = self.pairs[idx]
            
            # 类别过滤
            if category_filter and pair.category != category_filter:
                continue
            
            # 应用优先级
            adjusted_similarity = similarity * pair.priority
            
            results.append((pair, adjusted_similarity))
        
        # 排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def smart_search(self, 
                    query: str,
                    max_tokens: int = 4096,
                    diversity: bool = True) -> List[Tuple[TranslationPair, float]]:
        """智能搜索：考虑token限制和多样性"""
        print(f"Debug - smart_search 被调用，查询: {query}")
        candidates = self.search_similar(query, top_k=50)
        print(f"Debug - search_similar 返回了 {len(candidates)} 个候选")
        
        if not candidates:
            print("Debug - 没有候选结果，返回空列表")
            return []
        
        # 智能筛选
        selected = []
        current_tokens = 0
        used_categories = set()
        
        for pair, similarity in candidates:
            # 估算tokens
            estimated_tokens = self._estimate_tokens(pair.original, pair.translated)
            
            if current_tokens + estimated_tokens > max_tokens:
                continue
            
            # 多样性控制
            if diversity and pair.category and pair.category in used_categories:
                category_count = sum(1 for p, _ in selected if p.category == pair.category)
                if category_count >= 2:  # 每个类别最多2个
                    continue
            
            selected.append((pair, similarity))
            current_tokens += estimated_tokens
            
            if pair.category:
                used_categories.add(pair.category)
            
            if len(selected) >= 10:  # 最多10个
                break
        
        return selected
    
    def _estimate_tokens(self, original: str, translated: str) -> int:
        """估算文本的token数量"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', translated))
        english_words = len(re.findall(r'\b\w+\b', original + translated))
        punctuation = len(re.findall(r'[^\w\s]', original + translated))
        
        return int(chinese_chars * 1.5 + english_words * 1.3 + punctuation * 0.5)
    
    def save_to_file(self, filepath_prefix: str):
        """保存知识库到文件"""
        # 保存元数据
        metadata = {
            'pairs': [asdict(pair) for pair in self.pairs],
            'category_index': dict(self.category_index),
            'id_to_index': self.id_to_index,
            'config': {
                'model_name': self.embedding_model._modules['0'].auto_model.name_or_path,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'max_results': self.max_results
            }
        }
        
        with open(f"{filepath_prefix}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存FAISS索引
        if self.is_built and self.faiss_index:
            faiss.write_index(self.faiss_index, f"{filepath_prefix}.index")
        
        # 保存embeddings
        if self.embeddings is not None:
            np.save(f"{filepath_prefix}_embeddings.npy", self.embeddings)
        
        print(f"✓ 知识库已保存到 {filepath_prefix}")
    
    def load_from_file(self, filepath_prefix: str):
        """从文件加载知识库"""
        # 加载元数据
        with open(f"{filepath_prefix}_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 恢复数据
        self.pairs = [TranslationPair(**pair_data) for pair_data in metadata['pairs']]
        self.category_index = defaultdict(list, metadata['category_index'])
        self.id_to_index = metadata['id_to_index']
        
        config = metadata['config']
        self.embedding_dim = config['embedding_dim']
        self.index_type = config['index_type']
        self.max_results = config['max_results']
        
        # 重新初始化FAISS索引
        self._init_faiss_index()
        
        # 加载embeddings
        embeddings_path = f"{filepath_prefix}_embeddings.npy"
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        
        # 加载FAISS索引
        index_path = f"{filepath_prefix}.index"
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
            self.is_built = True
        else:
            self.is_built = False
        
        print(f"✓ 知识库已从 {filepath_prefix} 加载")
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        stats = {
            'total_pairs': len(self.pairs),
            'categories': {cat: len(indices) for cat, indices in self.category_index.items()},
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'is_built': self.is_built,
            'faiss_index_size': self.faiss_index.ntotal if self.is_built else 0
        }
        
        if self.embeddings is not None:
            stats['embeddings_shape'] = self.embeddings.shape
        
        return stats