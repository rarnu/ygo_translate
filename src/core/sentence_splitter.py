#!/usr/bin/env python3
"""
改进的句子拆分器 - 专门处理游戏王卡牌效果
确保日文和中文句子的对应关系
"""

import re
from typing import List, Tuple

class YugiohSentenceSplitter:
    """改进的游戏王句子拆分器"""
    
    def __init__(self):
        # 游戏王效果的特殊标记
        self.effect_markers = {
            'ja': [
                r'①：', r'②：', r'③：', r'④：', r'⑤：', r'⑥：', r'⑦：', r'⑧：', r'⑨：', r'⑩：',  # 序号标记
                r'Ｓ召喚', r'Ｘ召喚', r'リンク召喚', r'融合召喚', r'儀式召喚',  # 特殊召唤
                r'このカード', r'自分', r'相手', r'フィールド',  # 常见主语
                r'場合', r'時', r'ターン',  # 条件和时机
            ],
            'zh': [
                r'①：', r'②：', r'③：', r'④：', r'⑤：', r'⑥：', r'⑦：', r'⑧：', r'⑨：', r'⑩：',  # 序号标记
                r'同调召唤', r'超量召唤', r'链接召唤', r'融合召唤', r'仪式召唤',  # 特殊召唤
                r'这张卡', r'自己', r'对方', r'场上',  # 常见主语
                r'的场合', r'时', r'回合',  # 条件和时机
            ]
        }
        
        # 句子分隔符
        self.separators = {
            'ja': ['。', '；', '：'],
            'zh': ['。', '；', '：']
        }
    
    def extract_sentences_with_markers(self, text: str, lang: str) -> List[Tuple[str, str]]:
        """
        提取带标记的句子
        
        Returns:
            List[Tuple[sentence, marker]]: 句子和对应的标记
        """
        sentences = []
        
        # 首先处理换行符，按行分割
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 首先按序号标记拆分当前行
            if lang == 'ja':
                pattern = r'(①：[^。]*。|②：[^。]*。|③：[^。]*。|④：[^。]*。|⑤：[^。]*。|⑥：[^。]*。|⑦：[^。]*。|⑧：[^。]*。|⑨：[^。]*。|⑩：[^。]*。)'
            else:
                pattern = r'(①：[^。]*。|②：[^。]*。|③：[^。]*。|④：[^。]*。|⑤：[^。]*。|⑥：[^。]*。|⑦：[^。]*。|⑧：[^。]*。|⑨：[^。]*。|⑩：[^。]*。)'
            
            # 查找有序号标记的句子
            numbered_sentences = re.findall(pattern, line)
            if numbered_sentences:
                for sent in numbered_sentences:
                    marker = self._extract_marker(sent, lang)
                    sentences.append((sent.strip(), marker))
                
                # 处理当前行中剩余的文本
                remaining_line = line
                for sent in numbered_sentences:
                    remaining_line = remaining_line.replace(sent, '', 1)
                
                if remaining_line.strip():
                    # 拆分剩余文本
                    remaining_sentences = self._split_remaining_text(remaining_line.strip(), lang)
                    for sent in remaining_sentences:
                        if sent.strip():
                            marker = self._extract_marker(sent, lang)
                            sentences.append((sent.strip(), marker))
            else:
                # 没有序号标记，按普通句子处理
                marker = self._extract_marker(line, lang)
                sentences.append((line, marker))
        
        # 处理剩余的文本（有序号标记的句子之外的）
        remaining_text = text
        for sent, _ in sentences:
            remaining_text = remaining_text.replace(sent, '', 1)
        
        # 拆分剩余文本
        if remaining_text.strip():
            remaining_sentences = self._split_remaining_text(remaining_text.strip(), lang)
            for sent in remaining_sentences:
                marker = self._extract_marker(sent, lang)
                if sent.strip():  # 确保句子不为空
                    sentences.append((sent.strip(), marker))
        
        return sentences
    
    def _extract_marker(self, sentence: str, lang: str) -> str:
        """提取句子标记"""
        if lang == 'ja':
            # 提取序号标记
            if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence):
                return re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence).group()
            
            # 提取其他关键标记
            for marker in ['Ｓ召喚', 'Ｘ召喚', 'リンク召喚', '融合召喚', '儀式召喚']:
                if marker in sentence:
                    return marker
            
            # 提取条件标记
            if '場合のみ' in sentence:
                return '場合のみ'
            elif '場合' in sentence:
                return '場合'
            elif '時' in sentence:
                return '時'
        else:
            # 中文标记提取
            if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence):
                return re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence).group()
            
            for marker in ['同调召唤', '超量召唤', '链接召唤', '融合召唤', '仪式召唤']:
                if marker in sentence:
                    return marker
            
            if '的场合才能' in sentence:
                return '的场合才能'
            elif '的场合' in sentence:
                return '的场合'
            elif '时' in sentence:
                return '时'
        
        return 'general'  # 通用标记
    
    def _split_remaining_text(self, text: str, lang: str) -> List[str]:
        """拆分剩余文本"""
        sentences = []
        separators = self.separators[lang]
        
        # 按分隔符拆分
        parts = [text]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        for part in parts:
            part = part.strip()
            if part:
                # 加上分隔符
                if lang == 'ja':
                    part += '。'
                else:
                    part += '。'
                sentences.append(part)
        
        return sentences
    
    def _extract_marker(self, sentence: str, lang: str) -> str:
        """提取句子标记"""
        if lang == 'ja':
            # 提取序号标记
            if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence):
                return re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence).group()
            
            # 提取其他关键标记
            for marker in ['Ｓ召喚', 'Ｘ召喚', 'リンク召喚', '融合召喚', '儀式召喚']:
                if marker in sentence:
                    return marker
            
            # 提取条件标记
            if '場合のみ' in sentence:
                return '場合のみ'
            elif '場合' in sentence:
                return '場合'
            elif '時' in sentence:
                return '時'
        else:
            # 中文标记提取
            if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence):
                return re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]：', sentence).group()
            
            for marker in ['同调召唤', '超量召唤', '链接召唤', '融合召唤', '仪式召唤']:
                if marker in sentence:
                    return marker
            
            if '的场合才能' in sentence:
                return '的场合才能'
            elif '的场合' in sentence:
                return '的场合'
            elif '时' in sentence:
                return '时'
        
        return 'general'  # 通用标记
    
    def extract_card_names(self, text: str) -> List[str]:
        """提取卡名"""
        # 匹配「卡名」格式
        pattern1 = r'「([^」]+)」'
        card_names = re.findall(pattern1, text)
        
        # 匹配"卡名"格式
        if not card_names:
            pattern2 = r'"([^"]+)"'
            card_names = re.findall(pattern2, text)
        
        return card_names
    
    def replace_card_names(self, text: str, replacement: str = '「卡名」') -> str:
        """替换卡名"""
        # 替换「卡名」格式
        pattern1 = r'「[^」]+」'
        text = re.sub(pattern1, replacement, text)
        
        # 替换"卡名"格式
        pattern2 = r'"[^"]+"'
        text = re.sub(pattern2, replacement, text)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """标准化数字"""
        # 全角转半角
        text = re.sub(r'[０-９]', lambda m: str(ord(m.group()) - ord('０')), text)
        return text
    
    def create_pattern_key(self, sentence: str, marker: str, lang: str) -> str:
        """创建模式匹配的键"""
        # 移除卡名
        pattern = self.replace_card_names(sentence)
        
        # 标准化数字
        pattern = self.normalize_numbers(pattern)
        
        # 标准化常用表达式
        if lang == 'ja':
            pattern = re.sub(r'\d+枚', 'N枚', pattern)
            pattern = re.sub(r'\d+体', 'N体', pattern)
            pattern = re.sub(r'\d+ターン', 'Nターン', pattern)
            pattern = re.sub(r'レベル\d+', 'レベルN', pattern)
        else:
            pattern = re.sub(r'\d+张', 'N张', pattern)
            pattern = re.sub(r'\d+只', 'N只', pattern)
            pattern = re.sub(r'\d+次', 'N次', pattern)
            pattern = re.sub(r'\d+回合', 'N回合', pattern)
            pattern = re.sub(r'等级\d+', '等级N', pattern)
        
        # 清理多余空格
        pattern = re.sub(r'\s+', '', pattern)
        
        return f"{marker}:{pattern}"