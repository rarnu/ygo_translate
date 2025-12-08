"""
知识库数据准备工具
提供数据清洗、格式转换、预处理等功能
"""

import json
import re
import pandas as pd
from typing import List, Dict
from pathlib import Path
import hashlib
from collections import Counter, defaultdict


class DataPreparator:
    """知识库数据准备工具"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.quality_filters = {
            'min_length': 2,
            'max_length': 1000,
            'min_unique_chars': 2
        }
    
    def _load_stop_words(self) -> set:
        """加载停用词"""
        # 中英文常用停用词
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那',
            'text', 'string', 'data', 'file', 'content', 'message'
        }
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符但保留基本标点
        # text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】]', '', text)
        
        # 移除HTML标签
        # text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def validate_pair(self, original: str, translated: str) -> bool:
        """验证翻译对质量"""
        # 基本长度检查
        if not original or not translated:
            return False
        
        if (len(original) < self.quality_filters['min_length'] or 
            len(original) > self.quality_filters['max_length']):
            return False
        
        if (len(translated) < self.quality_filters['min_length'] or 
            len(translated) > self.quality_filters['max_length']):
            return False
        
        # 唯一字符检查
        if (len(set(original)) < self.quality_filters['min_unique_chars'] or
            len(set(translated)) < self.quality_filters['min_unique_chars']):
            return False
        
        # 重复文本检查
        if original == translated:
            return False
        
        # 过长的重复字符检查
        if len(set(original)) / len(original) < 0.1:
            return False
        
        return True
    
    def extract_category(self, text: str, default: str = "general") -> str:
        """从文本中提取类别"""
        text_lower = text.lower()
        
        # 游戏王相关关键词
        monster_keywords = ['dragon', 'warrior', 'magician', 'monster', 'beast', 'fiend']
        spell_keywords = ['spell', 'magic', 'activate', 'effect', 'power']
        trap_keywords = ['trap', 'counter', 'response', 'trigger']
        action_keywords = ['summon', 'attack', 'defend', 'destroy', 'battle']
        
        if any(keyword in text_lower for keyword in monster_keywords):
            return "monster"
        elif any(keyword in text_lower for keyword in spell_keywords):
            return "spell"
        elif any(keyword in text_lower for keyword in trap_keywords):
            return "trap"
        elif any(keyword in text_lower for keyword in action_keywords):
            return "action"
        
        return default
    
    def calculate_priority(self, original: str, translated: str, category: str) -> float:
        """计算优先级"""
        base_priority = 1.0
        
        # 根据类别调整
        category_priorities = {
            "monster": 2.0,
            "spell": 1.5,
            "trap": 1.5,
            "action": 1.3,
            "general": 1.0
        }
        
        priority = category_priorities.get(category, base_priority)
        
        # 根据文本长度调整（中等长度优先）
        length_factor = 1.0
        text_len = len(original + translated)
        if 10 <= text_len <= 100:
            length_factor = 1.2
        elif text_len > 200:
            length_factor = 0.8
        
        # 根据质量指标调整
        quality_factor = 1.0
        
        # 检查是否包含数字（可能是重要信息）
        if re.search(r'\d+', original) or re.search(r'\d+', translated):
            quality_factor += 0.1
        
        # 检查是否包含大写字母（可能是专有名词）
        if re.search(r'[A-Z][a-z]+', original):
            quality_factor += 0.1
        
        return priority * length_factor * quality_factor
    
    def prepare_from_csv(self, 
                        csv_path: str,
                        original_col: str = 'original',
                        translated_col: str = 'translated',
                        metadata_cols: List[str] = None) -> List[Dict]:
        """从CSV文件准备数据"""
        print(f"正在从CSV文件加载数据: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"CSV读取失败: {e}")
            return []
        
        if original_col not in df.columns or translated_col not in df.columns:
            print(f"CSV中缺少必要的列: {original_col}, {translated_col}")
            return []
        
        prepared_data = []
        
        for idx, row in df.iterrows():
            original = str(row[original_col])
            translated = str(row[translated_col])
            
            # 清洗文本
            original = self.clean_text(original)
            translated = self.clean_text(translated)
            
            # 验证质量
            if not self.validate_pair(original, translated):
                continue
            
            # 提取元数据
            metadata = {}
            if metadata_cols:
                for col in metadata_cols:
                    if col in df.columns and pd.notna(row[col]):
                        metadata[col] = str(row[col])
            
            # 自动分类
            category = self.extract_category(original)
            
            # 计算优先级
            priority = self.calculate_priority(original, translated, category)
            
            prepared_data.append({
                'original': original,
                'translated': translated,
                'metadata': metadata,
                'category': category,
                'priority': priority
            })
        
        print(f"✓ 从CSV加载了 {len(prepared_data)} 条有效数据")
        return prepared_data
    
    def prepare_from_json(self, json_path: str) -> List[Dict]:
        """从JSON文件准备数据"""
        print(f"正在从JSON文件加载数据: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON读取失败: {e}")
            return []
        
        prepared_data = []
        
        # 支持不同的JSON格式
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'pairs' in data:
            items = data['pairs']
        else:
            print("不支持的JSON格式")
            return []
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            original = item.get('original', '')
            translated = item.get('translated', '')
            
            # 清洗文本
            original = self.clean_text(original)
            translated = self.clean_text(translated)
            
            # 验证质量
            if not self.validate_pair(original, translated):
                continue
            
            # 提取元数据
            metadata = item.get('metadata', {})
            category = item.get('category', self.extract_category(original))
            priority = item.get('priority', self.calculate_priority(original, translated, category))
            
            prepared_data.append({
                'original': original,
                'translated': translated,
                'metadata': metadata,
                'category': category,
                'priority': priority
            })
        
        print(f"✓ 从JSON加载了 {len(prepared_data)} 条有效数据")
        return prepared_data
    
    def prepare_from_txt(self, txt_path: str) -> List[Dict]:
        """从文本文件准备数据（支持||分隔格式）"""
        print(f"正在从文本文件加载数据: {txt_path}")
        
        prepared_data = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"文本读取失败: {e}")
            return []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析格式：原文||译文||元数据||类别||优先级
            parts = line.split('||')
            
            if len(parts) < 2:
                print(f"第{line_num}行格式错误，跳过: {line}")
                continue
            
            original = parts[0].strip()
            translated = parts[1].strip()
            
            # 清洗文本
            original = self.clean_text(original)
            translated = self.clean_text(translated)
            
            # 验证质量
            if not self.validate_pair(original, translated):
                continue
            
            # 解析元数据
            metadata = {}
            category = ""
            priority = 1.0
            
            if len(parts) >= 3 and parts[2].strip():
                try:
                    metadata = json.loads(parts[2].strip())
                except:
                    pass
            
            if len(parts) >= 4 and parts[3].strip():
                category = parts[3].strip()
            else:
                category = self.extract_category(original)
            
            if len(parts) >= 5 and parts[4].strip():
                try:
                    priority = float(parts[4].strip())
                except:
                    priority = self.calculate_priority(original, translated, category)
            else:
                priority = self.calculate_priority(original, translated, category)
            
            prepared_data.append({
                'original': original,
                'translated': translated,
                'metadata': metadata,
                'category': category,
                'priority': priority
            })
        
        print(f"✓ 从文本文件加载了 {len(prepared_data)} 条有效数据")
        return prepared_data
    
    def deduplicate(self, data: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """去重处理"""
        print(f"正在去重，原始数据量: {len(data)}")
        
        unique_data = []
        seen_hashes = set()
        
        for item in data:
            # 生成哈希值
            content = f"{item['original']}|||{item['translated']}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_data.append(item)
        
        print(f"✓ 去重后数据量: {len(unique_data)}")
        return unique_data
    
    def balance_categories(self, data: List[Dict], max_per_category: int = 10000) -> List[Dict]:
        """平衡各类别数据量"""
        print("正在平衡各类别数据量...")
        
        category_groups = defaultdict(list)
        for item in data:
            category_groups[item['category']].append(item)
        
        balanced_data = []
        
        for category, items in category_groups.items():
            if len(items) > max_per_category:
                # 按优先级排序，保留高优先级的
                items.sort(key=lambda x: x['priority'], reverse=True)
                items = items[:max_per_category]
                print(f"类别 {category}: {len(category_groups[category])} -> {len(items)}")
            
            balanced_data.extend(items)
        
        print(f"✓ 平衡后数据量: {len(balanced_data)}")
        return balanced_data
    
    def generate_statistics(self, data: List[Dict]) -> Dict:
        """生成数据统计信息"""
        stats = {
            'total_count': len(data),
            'category_distribution': Counter(item['category'] for item in data),
            'priority_distribution': {
                'high': sum(1 for item in data if item['priority'] >= 2.0),
                'medium': sum(1 for item in data if 1.0 <= item['priority'] < 2.0),
                'low': sum(1 for item in data if item['priority'] < 1.0)
            },
            'length_stats': {
                'avg_original_length': sum(len(item['original']) for item in data) / len(data) if data else 0,
                'avg_translated_length': sum(len(item['translated']) for item in data) / len(data) if data else 0
            }
        }
        
        return stats
    
    def prepare_dataset(self, 
                       input_path: str,
                       output_path: str,
                       file_format: str = "auto",
                       deduplicate: bool = True,
                       balance_categories: bool = True) -> List[Dict]:
        """完整的数据准备流程"""
        print(f"=== 开始数据准备 ===")
        print(f"输入文件: {input_path}")
        print(f"输出路径: {output_path}")
        
        # 自动检测文件格式
        if file_format == "auto":
            suffix = Path(input_path).suffix.lower()
            if suffix == '.csv':
                file_format = 'csv'
            elif suffix == '.json':
                file_format = 'json'
            elif suffix == '.txt':
                file_format = 'txt'
            else:
                raise ValueError(f"无法识别文件格式: {suffix}")
        
        # 加载数据
        if file_format == 'csv':
            data = self.prepare_from_csv(input_path)
        elif file_format == 'json':
            data = self.prepare_from_json(input_path)
        elif file_format == 'txt':
            data = self.prepare_from_txt(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        if not data:
            print("没有有效数据，处理结束")
            return []
        
        # 去重
        if deduplicate:
            data = self.deduplicate(data)
        
        # 平衡类别
        if balance_categories:
            data = self.balance_categories(data)
        
        # 生成统计信息
        stats = self.generate_statistics(data)
        print(f"\n=== 数据统计 ===")
        print(f"总数据量: {stats['total_count']}")
        print(f"类别分布: {dict(stats['category_distribution'])}")
        print(f"优先级分布: {stats['priority_distribution']}")
        print(f"平均长度: {stats['length_stats']}")
        
        # 保存准备好的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 数据准备完成，已保存到: {output_path}")
        
        return data


