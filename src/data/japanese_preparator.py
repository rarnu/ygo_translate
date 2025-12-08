"""
日语数据准备工具
专门处理日语到中文的翻译数据
"""

import re
import json
from typing import List, Dict, Optional
from .data_preparation import DataPreparator


class JapaneseDataPreparator(DataPreparator):
    """日语数据准备工具"""
    
    def __init__(self):
        super().__init__()
        # 日语特有的停用词
        self.stop_words.update([
            'です', 'ます', 'である', 'だ', 'の', 'に', 'は', 'を', 'た', 'が', 'で',
            'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'する', 'です', 'ます',
            'カード', 'モンスター', '魔法', '罠', '効果', 'デュエル', 'プレイヤー'
        ])
    
    def clean_japanese_text(self, text: str) -> str:
        """清洗日语文本"""
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 保留日文、中文、英文和基本标点
        # text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\uff00-\uffef，。！？；：""''（）【】]', '', text)
        
        # 移除Ruby注音标记
        # text = re.sub(r'《[^》]*》', '', text)  # 移除振假名
        # text = re.sub(r'［[^］]*］', '', text)  # 移除注音
        
        # 统一标点符号
        # text = text.replace('（', '(').replace('）', ')')
        # text = text.replace('「', '"').replace('」', '"')
        # text = text.replace('』', '"').replace('『', '"')
        
        return text.strip()
    
    def extract_japanese_category(self, text: str, default: str = "general") -> str:
        """从日语文本中提取类别"""
        text_lower = text.lower()
        
        # 游戏王日文术语
        monster_keywords = ['モンスター', '怪兽', 'ドラゴン', '戦士', '魔法使い', '悪魔', '機械']
        spell_keywords = ['魔法', 'スペル', '呪文', '発動', '効果']
        trap_keywords = ['罠', 'トラップ', '反応', 'カウンター']
        action_keywords = ['召喚', '特殊召喚', '攻撃', '守備', '破壊', 'バトル']
        
        if any(keyword in text for keyword in monster_keywords):
            return "monster"
        elif any(keyword in text for keyword in spell_keywords):
            return "spell"
        elif any(keyword in text for keyword in trap_keywords):
            return "trap"
        elif any(keyword in text for keyword in action_keywords):
            return "action"
        
        return default
    
    def validate_ja_zh_pair(self, japanese: str, chinese: str) -> bool:
        """验证中日翻译对质量"""
        # 基本验证
        if not super().validate_pair(japanese, chinese):
            return False
        
        # 检查日文文本是否包含日文字符
        if not re.search(r'[\u3040-\u309f\u30a0-\u30ff]', japanese):
            # 如果没有日文字符，检查是否主要是英文
            english_chars = len(re.findall(r'[a-zA-Z]', japanese))
            if english_chars < len(japanese) * 0.5:  # 英文字符少于50%
                return False
        
        # 检查中文文本是否包含中文字符
        if not re.search(r'[\u4e00-\u9fff]', chinese):
            return False
        
        # 检查长度合理性
        ja_len = len(japanese)
        zh_len = len(chinese)
        
        # 长度比例检查（中文通常比日文短）
        if zh_len > ja_len * 2.5 or zh_len < ja_len * 0.3:
            return False
        
        return True
    
    def prepare_from_ja_txt(self, txt_path: str) -> List[Dict]:
        """从日语文本文件准备数据"""
        print(f"正在从日语文本文件加载数据: {txt_path}")
        
        prepared_data = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"日语文本读取失败: {e}")
            return []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析格式：日文||中文||元数据||类别||优先级
            parts = line.split('||')
            
            if len(parts) < 2:
                print(f"第{line_num}行格式错误，跳过: {line}")
                continue
            
            japanese = parts[0].strip()
            chinese = parts[1].strip()
            
            # 清洗文本
            japanese = self.clean_japanese_text(japanese)
            chinese = self.clean_text(chinese)
            
            # 验证质量
            if not self.validate_ja_zh_pair(japanese, chinese):
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
                category = self.extract_japanese_category(japanese)
            
            if len(parts) >= 5 and parts[4].strip():
                try:
                    priority = float(parts[4].strip())
                except:
                    priority = self.calculate_priority(japanese, chinese, category)
            else:
                priority = self.calculate_priority(japanese, chinese, category)
            
            # 添加语言标识
            metadata['source_lang'] = 'ja'
            metadata['target_lang'] = 'zh'
            
            prepared_data.append({
                'original': japanese,
                'translated': chinese,
                'metadata': metadata,
                'category': category,
                'priority': priority
            })
        
        print(f"✓ 从日语文本文件加载了 {len(prepared_data)} 条有效数据")
        return prepared_data


