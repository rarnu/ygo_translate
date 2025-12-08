"""
中文数据准备工具
专门处理中文到日文的翻译数据
"""

import re
import json
from typing import List, Dict, Optional
from .data_preparation import DataPreparator


class ChineseDataPreparator(DataPreparator):
    """中文数据准备工具"""
    
    def __init__(self):
        super().__init__()
        # 中文特有的停用词
        self.stop_words.update([
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那',
            '卡片', '怪兽', '魔法', '陷阱', '效果', '决斗', '玩家', '召唤', '攻击'
        ])
    
    def clean_chinese_text(self, text: str) -> str:
        """清洗中文文本"""
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 保留中文、英文和基本标点
        # text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】]', '', text)
        
        # 统一标点符号
        # text = text.replace('（', '(').replace('）', ')')
        # text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
    
    def extract_chinese_category(self, text: str, default: str = "general") -> str:
        """从中文文本中提取类别"""
        text_lower = text.lower()
        
        # 游戏王中文术语
        monster_keywords = ['怪兽', '龙', '战士', '魔法师', '恶魔', '机械', ' dragon']
        spell_keywords = ['魔法', '咒文', '发动', '效果', 'spell']
        trap_keywords = ['陷阱', '反应', '反击', 'trap']
        action_keywords = ['召唤', '特殊召唤', '攻击', '守备', '破坏', '战斗']
        
        if any(keyword in text for keyword in monster_keywords):
            return "monster"
        elif any(keyword in text for keyword in spell_keywords):
            return "spell"
        elif any(keyword in text for keyword in trap_keywords):
            return "trap"
        elif any(keyword in text for keyword in action_keywords):
            return "action"
        
        return default
    
    def validate_zh_ja_pair(self, chinese: str, japanese: str) -> bool:
        """验证中日翻译对质量"""
        # 基本验证
        if not super().validate_pair(chinese, japanese):
            return False
        
        # 检查中文文本是否包含中文字符
        if not re.search(r'[\u4e00-\u9fff]', chinese):
            return False
        
        # 检查日文文本是否包含日文字符
        if not re.search(r'[\u3040-\u309f\u30a0-\u30ff]', japanese):
            # 如果没有日文字符，检查是否主要是英文
            english_chars = len(re.findall(r'[a-zA-Z]', japanese))
            if english_chars < len(japanese) * 0.5:  # 英文字符少于50%
                return False
        
        # 检查长度合理性（日文通常比中文长）
        zh_len = len(chinese)
        ja_len = len(japanese)
        
        # 长度比例检查
        if ja_len > zh_len * 3.0 or ja_len < zh_len * 0.4:
            return False
        
        return True
    
    def prepare_from_zh_txt(self, txt_path: str) -> List[Dict]:
        """从中文文本文件准备数据"""
        print(f"正在从中文文本文件加载数据: {txt_path}")
        
        prepared_data = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"中文文本读取失败: {e}")
            return []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析格式：中文||日文||元数据||类别||优先级
            parts = line.split('||')
            
            if len(parts) < 2:
                print(f"第{line_num}行格式错误，跳过: {line}")
                continue
            
            chinese = parts[0].strip()
            japanese = parts[1].strip()
            
            # 清洗文本
            chinese = self.clean_chinese_text(chinese)
            japanese = self.clean_text(japanese)
            
            # 验证质量
            if not self.validate_zh_ja_pair(chinese, japanese):
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
                category = self.extract_chinese_category(chinese)
            
            if len(parts) >= 5 and parts[4].strip():
                try:
                    priority = float(parts[4].strip())
                except:
                    priority = self.calculate_priority(chinese, japanese, category)
            else:
                priority = self.calculate_priority(chinese, japanese, category)
            
            # 添加语言标识
            metadata['source_lang'] = 'zh'
            metadata['target_lang'] = 'ja'
            
            prepared_data.append({
                'original': chinese,
                'translated': japanese,
                'metadata': metadata,
                'category': category,
                'priority': priority
            })
        
        print(f"✓ 从中文文本文件加载了 {len(prepared_data)} 条有效数据")
        return prepared_data