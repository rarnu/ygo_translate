#!/usr/bin/env python3
"""
修复版增强翻译器 - 使用改进的范式匹配系统
"""

import time
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import re

from .sentence_splitter import YugiohSentenceSplitter
from .pattern_vector_db import JapaneseToChinesePatternDB, ChineseToJapanesePatternDB, PatternMatch
from .faiss_knowledge_base import FAISSKnowledgeBase

@dataclass
class EnhancedTranslationRequest:
    """增强翻译请求"""
    source_text: str
    source_lang: str = 'ja'
    target_lang: str = 'zh'
    max_context: int = 4096
    use_pattern_matching: bool = True
    use_card_name_translation: bool = True

@dataclass
class EnhancedTranslationResult:
    """增强翻译结果"""
    translated_text: str
    source_lang: str
    target_lang: str
    processing_time: float
    pattern_matches: List[Dict]
    card_name_translations: Dict[str, str]
    sentence_breakdown: List[Dict]
    confidence: float

class EnhancedYugiohTranslator:
    """修复版游戏王翻译器"""
    
    def __init__(self, 
                 model_name: str = "alibayram/Qwen3-30B-A3B-Instruct-2507:latest",
                 ollama_url: str = "http://localhost:11434",
                 use_pattern_db: bool = True,
                 pattern_db_path: str = "./data/pattern_vector_db"):
        """
        初始化修复版翻译器
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_pattern_db = use_pattern_db
        
        # 初始化组件
        self.splitter = YugiohSentenceSplitter()
        self.cardname_registry: Dict[str, str] = {}
        
        # 加载范式数据库
        self.pattern_dbs = {}
        self._load_pattern_databases()
        
        # 加载卡名知识库
        self.knowledge_bases: Dict[str, FAISSKnowledgeBase] = {}
        self._load_knowledge_bases()
        
        # 移除硬编码范式，完全依赖范式库
    
    def _load_pattern_databases(self):
        """加载范式数据库"""
        try:
            # 加载日译中数据库
            self.pattern_dbs['ja_to_zh'] = JapaneseToChinesePatternDB()
            self.pattern_dbs['ja_to_zh'].load_from_file("./data/pattern_vector_db_ja_to_zh")
            print("✓ 已加载日译中范式数据库")
        except Exception as e:
            print(f"⚠️  加载日译中范式数据库失败: {e}")
            self.pattern_dbs['ja_to_zh'] = JapaneseToChinesePatternDB()
        
        try:
            # 加载中译日数据库
            self.pattern_dbs['zh_to_ja'] = ChineseToJapanesePatternDB()
            self.pattern_dbs['zh_to_ja'].load_from_file("./data/pattern_vector_db_zh_to_ja")
            print("✓ 已加载中译日范式数据库")
        except Exception as e:
            print(f"⚠️  加载中译日范式数据库失败: {e}")
            self.pattern_dbs['zh_to_ja'] = ChineseToJapanesePatternDB()
    

    
    def translate(self, request: EnhancedTranslationRequest) -> EnhancedTranslationResult:
        """执行RAG增强翻译"""
        start_time = time.time()
        
        # 提取句子和标记
        sentences = self.splitter.extract_sentences_with_markers(request.source_text, request.source_lang)
        
        # 翻译每个句子
        translated_sentences = []
        all_pattern_matches = []
        all_card_translations = {}
        sentence_breakdown = []
        
        for i, (source_sent, marker) in enumerate(sentences):
            # 提取卡名
            card_names = self.splitter.extract_card_names(source_sent)
            
            # 翻译卡名
            card_translations = {}
            if card_names and request.use_card_name_translation:
                card_translations = self._translate_card_names(card_names, request.source_lang, request.target_lang)
                all_card_translations.update(card_translations)
            
            # 查找相关范式用于RAG
            relevant_patterns = []
            if request.use_pattern_matching:
                relevant_patterns = self._find_relevant_patterns(
                    source_sent, request.source_lang, request.target_lang
                )
            
            # 使用RAG调用大模型
            final_translation = None
            llm_prompt = ""
            llm_response = ""
            
            if relevant_patterns or not request.use_pattern_matching:
                llm_prompt, llm_response = self._call_llm_with_rag(
                    source_sent, request, relevant_patterns, card_translations
                )
                if llm_response:
                    final_translation = llm_response
            else:
                # 如果没有相关范式且要求使用范式匹配，回退
                final_translation = self._fallback_translation(source_sent, request)
            
            # 记录所有匹配的范式
            all_pattern_matches.extend(relevant_patterns)
            
            translated_sentences.append(final_translation)
            
            # 记录句子处理信息
            sentence_breakdown.append({
                'index': i,
                'source': source_sent,
                'translation': final_translation,
                'card_names': card_names,
                'card_translations': card_translations,
                'pattern_match': 1 if relevant_patterns else 0,
                'match_type': 'rag_llm' if llm_response else 'fallback',
                'patterns_found': len(relevant_patterns),
                'llm_prompt': llm_prompt,
                'llm_response': llm_response
            })
        
        # 重建完整翻译，保留序号标记
        final_text = self._reconstruct_translation_with_markers(sentences, translated_sentences)
        
        # 后处理
        final_text = self._post_process_translation(final_text, request.source_lang)
        
        # 计算置信度
        confidence = self._calculate_overall_confidence(all_pattern_matches, sentence_breakdown)
        
        processing_time = time.time() - start_time
        
        return EnhancedTranslationResult(
            translated_text=final_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            processing_time=processing_time,
            pattern_matches=[{
                'source_pattern': m.source_pattern,
                'target_pattern': m.target_pattern,
                'similarity': m.similarity,
                'confidence': m.confidence
            } for m in all_pattern_matches],
            card_name_translations=all_card_translations,
            sentence_breakdown=sentence_breakdown,
            confidence=confidence
        )
    
    def _translate_card_names(self, card_names: List[str], source_lang: str, target_lang: str) -> Dict[str, str]:
        """翻译卡名：优先使用知识库，失败时回退到范式库扫描"""
        translations: Dict[str, str] = {}
        kb = self.knowledge_bases.get(source_lang)
        for card_name in card_names:
            translated = None
            # 1) 知识库精确/相似搜索（带「」优先）
            if kb:
                wrapped = f"「{card_name}」"
                candidates = kb.smart_search(wrapped, max_tokens=200, diversity=True)
                for pair, sim in candidates:
                    if wrapped in pair.original:
                        m = re.search(r'「([^」]+)」', pair.translated)
                        if m:
                            translated = m.group(1)
                            break
                if not translated:
                    candidates = kb.smart_search(card_name, max_tokens=100, diversity=True)
                    for pair, sim in candidates:
                        # 简单结构相似判断
                        if self._char_jaccard(card_name, pair.original) > 0.3:
                            translated = pair.translated
                            break
            # 2) 失败时使用范式库扫描
            if not translated:
                try:
                    db = self.pattern_dbs.get('ja_to_zh' if source_lang == 'ja' else 'zh_to_ja')
                    if db and db.index.ntotal > 0:
                        for _, pdata in db.patterns.items():
                            sp = pdata.get('source_pattern','')
                            tp = pdata.get('target_pattern','')
                            if f"「{card_name}」" in sp:
                                m = re.search(r'「([^」]+)」', tp)
                                if m:
                                    translated = m.group(1)
                                    break
                            if not translated and f"「{card_name}」" in tp:
                                m = re.search(r'「([^」]+)」', sp)
                                if m:
                                    translated = m.group(1)
                                    break
                except Exception:
                    pass
            translations[card_name] = translated or card_name
        return translations

    def _load_knowledge_bases(self):
        try:
            kb_ja = FAISSKnowledgeBase(index_type="flat")
            kb_ja.load_from_file("data/ja/knowledge_base")
            self.knowledge_bases['ja'] = kb_ja
        except Exception:
            self.knowledge_bases['ja'] = FAISSKnowledgeBase(index_type="flat")
        try:
            kb_zh = FAISSKnowledgeBase(index_type="flat")
            kb_zh.load_from_file("data/zh/knowledge_base")
            self.knowledge_bases['zh'] = kb_zh
        except Exception:
            self.knowledge_bases['zh'] = FAISSKnowledgeBase(index_type="flat")
    
    def _fallback_translation(self, source_text: str, request: EnhancedTranslationRequest) -> str:
        """回退翻译方法"""
        # 暂时禁用网络请求，直接返回标记
        return f"[回退翻译: {source_text[:20]}...]"
    

    def _reconstruct_translation_with_markers(self, original_sentences: List[Tuple[str, str]], translated_sentences: List[str]) -> str:
        """
        重建翻译结果，保留序号标记
        
        Args:
            original_sentences: 原始句子和标记的列表 [(sentence, marker), ...]
            translated_sentences: 翻译后的句子列表
            
        Returns:
            重建的完整翻译文本
        """
        reconstructed_parts = []
        
        for i, (original_sent, marker) in enumerate(original_sentences):
            translated_sent = translated_sentences[i] if i < len(translated_sentences) else ""
            
            # 如果原句有序号标记，确保翻译结果也有序号
            if marker and marker in ['①：', '②：', '③：', '④：', '⑤：', '⑥：', '⑦：', '⑧：', '⑨：', '⑩：']:
                # 检查翻译结果是否已经有序号
                if not translated_sent.startswith(marker):
                    # 检查翻译结果是否以其他序号开头
                    has_number_prefix = False
                    for num_marker in ['①：', '②：', '③：', '④：', '⑤：', '⑥：', '⑦：', '⑧：', '⑨：', '⑩：']:
                        if translated_sent.startswith(num_marker):
                            has_number_prefix = True
                            break
                    
                    # 如果没有序号前缀，添加原序号
                    if not has_number_prefix:
                        translated_sent = marker + translated_sent
            
            reconstructed_parts.append(translated_sent)
        
        # 合并所有部分
        return ''.join(reconstructed_parts)
    
    def _post_process_translation(self, translation: str, source_lang: str) -> str:
        """翻译后处理"""
        # 确保句末有正确的标点
        if not translation.endswith(('。', '！', '？')):
            if translation.endswith('.'):
                translation = translation[:-1] + '。'
            elif not translation.endswith(('。', '！', '？', '.', '!', '?')):
                translation += '。'
        
        return translation.strip()

    def add_cardname(self, ja_name: str, zh_name: str) -> tuple[str, str] | None:
        if not ja_name or not zh_name:
            return None
        self.cardname_registry[ja_name] = zh_name
        self.cardname_registry[zh_name] = ja_name
        return ja_name, zh_name

    def delete_cardname(self, ja_name: str, zh_name: str) -> bool:
        ok = False
        if ja_name in self.cardname_registry:
            del self.cardname_registry[ja_name]
            ok = True
        if zh_name in self.cardname_registry:
            del self.cardname_registry[zh_name]
            ok = True
        return ok

    def cardname_exists(self, name: str) -> tuple[list[str], list[str]] | None:
        if not name:
            return None
        ja_list: list[str] = []
        zh_list: list[str] = []
        if name in self.cardname_registry:
            other = self.cardname_registry[name]
            # 简单判断字符集来归类
            if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', name) and re.search(r'[\u4e00-\u9fff]', other):
                ja_list.append(name)
                zh_list.append(other)
            else:
                # 双向返回
                ja_list.append(name)
                zh_list.append(other)
        return ja_list, zh_list

    def _calculate_overall_confidence(self, pattern_matches: List[PatternMatch], sentence_breakdown: List[Dict]) -> float:
        """计算整体置信度"""
        if not sentence_breakdown:
            return 0.0
        
        # 基于范式匹配的置信度
        pattern_confidence = 0.0
        if pattern_matches:
            avg_similarity = sum(m.similarity for m in pattern_matches) / len(pattern_matches)
            avg_confidence = sum(m.confidence for m in pattern_matches) / len(pattern_matches)
            pattern_confidence = (avg_similarity + avg_confidence) / 2
        
        # 基于句子处理成功率的置信度
        successful_sentences = sum(1 for s in sentence_breakdown if s['pattern_match'] > 0 or s['card_translations'])
        sentence_confidence = successful_sentences / len(sentence_breakdown)
        
        # 加权平均
        return pattern_confidence * 0.7 + sentence_confidence * 0.3
    
    def _find_relevant_patterns(self, source_sentence: str, source_lang: str, target_lang: str) -> List[PatternMatch]:
        """
        查找与源句子相关的所有范式
        支持部分句子匹配和子句匹配
        """
        # print(f"查找与句子相关的范式：{source_sentence}")
        
        relevant_patterns = []
        
        try:
            # 1. 整体句子匹配
            whole_match = self._search_vector_database(source_sentence, source_lang, target_lang)
            if whole_match and whole_match.similarity > 0.3:
                relevant_patterns.append(whole_match)
                # print(f"✓ 整体匹配：{whole_match.source_pattern[:50]}... (相似度: {whole_match.similarity:.3f})")
            
            # 2. 子句匹配 - 按顿号和标点分割
            segments = self._split_sentence_for_matching(source_sentence, source_lang)
            # print(f"分割得到 {len(segments)} 个子句：{segments}")
            
            for segment in segments:
                segment_match = self._search_vector_database(segment, source_lang, target_lang)
                if segment_match and segment_match.similarity > 0.3:
                    relevant_patterns.append(segment_match)
                    # print(f"✓ 子句匹配：{segment_match.source_pattern[:50]}... (相似度: {segment_match.similarity:.3f})")
            
            # 3. 部分句子匹配 - 滑动窗口匹配
            partial_matches = self._find_partial_matches(source_sentence, source_lang, target_lang)
            relevant_patterns.extend(partial_matches)
            
            # 4. 关键术语匹配
            key_term_matches = self._find_key_term_matches(source_sentence, source_lang, target_lang)
            relevant_patterns.extend(key_term_matches)
            
            # 去重（基于相似度高的模式）
            relevant_patterns = self._deduplicate_patterns(relevant_patterns)
            
            # print(f"总共找到 {len(relevant_patterns)} 个相关范式")
            
        except Exception as e:
            print(f"查找相关范式时出错：{e}")
        
        return relevant_patterns
    
    def _split_sentence_for_matching(self, sentence: str, lang: str) -> List[str]:
        """
        为匹配目的分割句子
        """
        if lang == 'ja':
            # 日文分割规则：按顿号、句末标点、以及特定动词分割
            import re
            # 使用正则表达式分割，保留标点符号
            segments = re.findall(r'[^、。！？…]+[、。！？…]?', sentence)
            
            # 额外的关键词分割
            keyword_patterns = [
                r'(.+?時、)',  # ...时、
                r'(.+?場合、)', # ...场合、
                r'(.+？)(?=。)', # ...吗？（在句号前）
            ]
            
            for pattern in keyword_patterns:
                matches = re.findall(pattern, sentence)
                segments.extend(matches)
        
        else:
            # 中文分割规则
            import re
            segments = re.findall(r'[^、。！？…]+[、。！？…]?', sentence)
        
        # 过滤和清理
        valid_segments = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) >= 3:  # 至少3个字符
                valid_segments.append(seg)
        
        return list(set(valid_segments))  # 去重
    
    def _find_partial_matches(self, sentence: str, source_lang: str, target_lang: str) -> List[PatternMatch]:
        """
        查找部分句子匹配
        使用滑动窗口方法
        """
        partial_matches = []
        
        if len(sentence) < 8:
            return partial_matches
        
        # 滑动窗口长度
        window_sizes = [len(sentence) - 4, len(sentence) - 6, len(sentence) - 8]
        
        db = self.pattern_dbs.get('ja_to_zh' if source_lang == 'ja' else 'zh_to_ja')
        if not db or db.index.ntotal == 0:
            return partial_matches
        
        for window_size in window_sizes:
            if window_size < 6:
                continue
                
            for i in range(len(sentence) - window_size + 1):
                substring = sentence[i:i + window_size]
                
                # 确保以标点符号或完整词汇结尾
                if not substring.endswith(('、', '。', '！', '？', '…')):
                    continue
                
                match = self._search_vector_database(substring, source_lang, target_lang)
                if match and match.similarity > 0.4:
                    # 避免重复
                    if not any(pm.source_pattern == match.source_pattern for pm in partial_matches):
                        partial_matches.append(match)
                        # print(f"✓ 部分匹配：{match.source_pattern[:50]}... (相似度: {match.similarity:.3f})")
        
        return partial_matches[:5]  # 最多返回5个部分匹配
    
    def _find_key_term_matches(self, sentence: str, source_lang: str, target_lang: str) -> List[PatternMatch]:
        """
        基于关键术语查找相关范式
        """
        key_term_matches = []
        key_terms = self._extract_key_terms(sentence, source_lang)
        
        if not key_terms:
            return key_term_matches
        
        db = self.pattern_dbs.get('ja_to_zh' if source_lang == 'ja' else 'zh_to_ja')
        if not db or db.index.ntotal == 0:
            return key_term_matches
        
        # 为每个关键术语查找相关范式
        for term in key_terms[:3]:  # 最多取3个关键术语
            # 查找包含该术语的范式
            for pattern_id, pattern_data in db.patterns.items():
                source_pattern = pattern_data.get('source_pattern', '')
                
                if term in source_pattern:
                    # 计算相似度
                    similarity = self._keyword_similarity(sentence, source_pattern)
                    
                    if similarity > 0.3:
                        match = PatternMatch(
                            pattern_id=pattern_id,
                            source_pattern=source_pattern,
                            target_pattern=pattern_data.get('target_pattern', ''),
                            category=pattern_data.get('category', ''),
                            confidence=similarity,
                            similarity=similarity,
                            examples=pattern_data.get('examples', []),
                            match_type='key_term'
                        )
                        
                        # 避免重复
                        if not any(pm.source_pattern == match.source_pattern for pm in key_term_matches):
                            key_term_matches.append(match)
        
        # 按相似度排序，取前5个
        key_term_matches.sort(key=lambda m: m.similarity, reverse=True)
        return key_term_matches[:5]
    
    def _deduplicate_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """
        去除重复的范式，保留相似度最高的
        """
        if not patterns:
            return patterns
        
        # 按相似度排序
        patterns.sort(key=lambda m: m.similarity, reverse=True)
        
        deduplicated = []
        seen_sources = set()
        seen_targets = set()
        
        for pattern in patterns:
            # 基于源模式和目标模式的相似度去重
            source_key = pattern.source_pattern[:30]  # 前30个字符作为key
            target_key = pattern.target_pattern[:30]
            
            if (source_key not in seen_sources and 
                target_key not in seen_targets):
                deduplicated.append(pattern)
                seen_sources.add(source_key)
                seen_targets.add(target_key)
        
        return deduplicated
    
    def _call_llm_with_rag(self, source_text: str, request: EnhancedTranslationRequest, 
                          relevant_patterns: List[PatternMatch], card_translations: Dict[str, str]) -> Tuple[str, str]:
        """
        使用RAG调用大模型进行翻译
        """
        try:
            # 构建RAG提示词
            prompt = self._build_rag_prompt(source_text, request, relevant_patterns, card_translations)

            # 调用大模型
            response = self._call_ollama_api(prompt)

            # 清理响应
            cleaned_response = self._clean_llm_response(response)
            
            return prompt, cleaned_response
            
        except Exception as e:
            print(f"调用大模型时出错：{e}")
            return "", ""
    
    def _build_rag_prompt(self, source_text: str, request: EnhancedTranslationRequest,
                         relevant_patterns: List[PatternMatch], card_translations: Dict[str, str]) -> str:
        """
        构建RAG提示词
        """
        # 基础翻译指令
        if request.source_lang == 'ja' and request.target_lang == 'zh':
            base_instruction = """你是一个专业的游戏王日译中翻译专家，必须严格遵循游戏王官方翻译范式。"""
            
            constraints = """
【严格禁止】
- 不得改变游戏王的固定表达方式
- 不得添加解释性内容

【翻译要求】
- 严格翻译输入的文本内容
- 保持术语的一致性和专业性
- 遵循游戏王内容的表达习惯
- 只输出翻译结果，不要解释或添加其他内容
- 句末需要句号"。"，不需要额外句号"""
        
        elif request.source_lang == 'zh' and request.target_lang == 'ja':
            base_instruction = """你是一个专业的中译日翻译专家，必须严格遵循游戏王官方翻译范式。"""
            
            constraints = """
【严格禁止】
- 不得改变游戏王的固定表达方式
- 不得添加解释性内容

【翻译要求】
1. 保持术语的一致性和专业性
2. 遵循游戏王内容的表达习惯
3. 使用准确的日语语法和用词
4. 只输出翻译结果，不要解释或添加其他内容"""
        else:
            base_instruction = f"你是一个专业的翻译专家，专门将{request.source_lang}翻译为{request.target_lang}。"
            constraints = "请准确翻译以下文本，保持专业性和一致性。"
        
        # RAG相关范式信息
        rag_context = ""
        if relevant_patterns:
            rag_context = "\n\n【参考翻译范式】\n请参考以下已有的翻译范式，保持术语和表达的一致性：\n\n"
            for i, pattern in enumerate(relevant_patterns[:5], 1):  # 最多显示5个
                rag_context += f"范式{i}：\n"
                rag_context += f"原文：{pattern.source_pattern}\n"
                rag_context += f"译文：{pattern.target_pattern}\n"
                rag_context += f"相似度：{pattern.similarity:.2f}\n\n"
        
        # 卡名翻译信息
        card_context = ""
        if card_translations:
            card_context = "\n【卡名翻译】\n以下卡名已确定翻译，请使用：\n"
            for original, translated in card_translations.items():
                card_context += f"{original} → {translated}\n"
        
        # 组合完整提示词
        full_prompt = f"""{base_instruction}
{constraints}
{rag_context}
{card_context}

【待翻译文本】
{source_text}

【翻译输出】
"""
        
        return full_prompt.strip()
    
    def _call_ollama_api(self, prompt: str) -> str:
        """
        调用Ollama API
        """
        import json
        
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # 低温度确保一致性
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama API请求失败：{e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"解析API响应失败：{e}")
            return ""
        except Exception as e:
            print(f"调用API时出现未知错误：{e}")
            return ""
    
    def _clean_llm_response(self, response: str) -> str:
        """
        清理大模型响应
        """
        if not response:
            return response
        
        # 移除可能的多余内容
        lines = response.strip().split('\n')
        
        # 查找包含实际翻译内容的行
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('翻译：', '译文：', '输出：', '【', '译：')):
                return line
        
        # 如果没找到，返回第一行
        return lines[0].strip() if lines else response.strip()
    
    def _segment_based_translation(self, source_sentence: str, source_lang: str, target_lang: str, card_translations: Dict[str, str]) -> Tuple[str, List[PatternMatch]]:
        """
        改进的子句匹配翻译方法
        1. 优先尝试带标点的子句匹配
        2. 如果失败，按顿号分割后重新匹配
        """
        try:
            print(f"开始子句匹配翻译：{source_sentence}")
            
            # 第一步：尝试带标点的子句匹配
            translation_with_punctuation, matches_with_punctuation = self._try_punctuation_based_translation(
                source_sentence, source_lang, target_lang, card_translations
            )
            
            if translation_with_punctuation:
                print("✓ 带标点子句匹配成功")
                return translation_with_punctuation, matches_with_punctuation
            
            print("✗ 带标点子句匹配失败，尝试按顿号分割...")
            
            # 第二步：按顿号分割后匹配
            translation_split_by_comma, matches_split_by_comma = self._try_comma_split_translation(
                source_sentence, source_lang, target_lang, card_translations
            )
            
            if translation_split_by_comma:
                print("✓ 顿号分割匹配成功")
                return translation_split_by_comma, matches_split_by_comma
            
            print("✗ 所有子句匹配方法都失败")
            return "", []
            
        except Exception as e:
            print(f"子句匹配翻译出错: {e}")
            return "", []
    
    def _try_punctuation_based_translation(self, source_sentence: str, source_lang: str, target_lang: str, card_translations: Dict[str, str]) -> Tuple[str, List[PatternMatch]]:
        """
        尝试基于标点符号的子句匹配
        """
        if source_lang == 'ja':
            # 日文分割规则：按句末标点和顿号分割，保留标点
            # 特别注意：对于带顿号的子句，要包含顿号
            import re
            # 使用正则表达式分割，保留标点符号
            segments = re.findall(r'[^、。！？…]+[、。！？…]?', source_sentence)
        else:
            # 中文分割规则
            import re
            segments = re.findall(r'[^、。！？…]+[、。！？…]?', source_sentence)
        
        # 过滤空段和过短的段
        valid_segments = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) >= 2:  # 至少2个字符
                valid_segments.append(seg)
        
        if len(valid_segments) <= 1:
            return "", []
        
        print(f"带标点分割得到 {len(valid_segments)} 个子句：{valid_segments}")
        
        # 为每个段寻找最佳匹配
        segment_matches = []
        unmatched_segments = []
        
        for segment in valid_segments:
            print(f"尝试匹配子句：{segment}")
            best_match = self._search_vector_database(segment, source_lang, target_lang)
            if best_match and best_match.similarity > 0.3:  # 设置相似度阈值
                print(f"✓ 匹配成功：{best_match.target_pattern[:30]}... (相似度: {best_match.similarity:.3f})")
                segment_matches.append((segment, best_match))
            else:
                print(f"✗ 匹配失败")
                unmatched_segments.append(segment)
        
        # 如果所有段都能匹配，合并翻译结果
        if len(unmatched_segments) == 0 and len(segment_matches) == len(valid_segments):
            translated_segments = []
            for segment, match in segment_matches:
                translated_segment = match.target_pattern
                
                # 替换数字占位符
                if 'N' in translated_segment:
                    nums = self._extract_numbers(segment)
                    for n in nums:
                        translated_segment = translated_segment.replace('N', n, 1)
                
                # 替换卡名
                if '「卡名」' in translated_segment and card_translations:
                    first_translated = next(iter(card_translations.values()))
                    translated_segment = translated_segment.replace('「卡名」', f'「{first_translated}」')
                for original, translated in card_translations.items():
                    translated_segment = translated_segment.replace(f"「{original}」", f"「{translated}」")
                
                translated_segments.append(translated_segment)
            
            # 合并翻译结果
            final_translation = ''.join(translated_segments)
            matches_list = [match for _, match in segment_matches]
            
            return final_translation, matches_list
        
        return "", []
    
    def _try_comma_split_translation(self, source_sentence: str, source_lang: str, target_lang: str, card_translations: Dict[str, str]) -> Tuple[str, List[PatternMatch]]:
        """
        按顿号分割后的翻译匹配
        """
        # 按顿号分割，不保留顿号
        if '、' in source_sentence:
            segments = source_sentence.split('、')
            # 重新添加顿号到除最后一段外的所有段
            for i in range(len(segments) - 1):
                segments[i] += '、'
        else:
            # 如果没有顿号，按其他标点分割
            import re
            segments = re.split(r'([。！？…])', source_sentence)
            # 重组分割结果，保留标点
            segments = [''.join(segments[i:i+2]) for i in range(0, len(segments), 2)]
        
        # 过滤空段和过短的段
        valid_segments = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) >= 2:  # 至少2个字符
                valid_segments.append(seg)
        
        if len(valid_segments) <= 1:
            return "", []
        
        print(f"按顿号分割得到 {len(valid_segments)} 个子句：{valid_segments}")
        
        # 为每个段寻找最佳匹配
        segment_matches = []
        unmatched_segments = []
        
        for segment in valid_segments:
            print(f"尝试匹配顿号分割子句：{segment}")
            best_match = self._search_vector_database(segment, source_lang, target_lang)
            if best_match and best_match.similarity > 0.3:  # 设置相似度阈值
                print(f"✓ 匹配成功：{best_match.target_pattern[:30]}... (相似度: {best_match.similarity:.3f})")
                segment_matches.append((segment, best_match))
            else:
                print(f"✗ 匹配失败")
                unmatched_segments.append(segment)
        
        # 如果所有段都能匹配，合并翻译结果
        if len(unmatched_segments) == 0 and len(segment_matches) == len(valid_segments):
            translated_segments = []
            for segment, match in segment_matches:
                translated_segment = match.target_pattern
                
                # 替换数字占位符
                if 'N' in translated_segment:
                    nums = self._extract_numbers(segment)
                    for n in nums:
                        translated_segment = translated_segment.replace('N', n, 1)
                
                # 替换卡名
                if '「卡名」' in translated_segment and card_translations:
                    first_translated = next(iter(card_translations.values()))
                    translated_segment = translated_segment.replace('「卡名」', f'「{first_translated}」')
                for original, translated in card_translations.items():
                    translated_segment = translated_segment.replace(f"「{original}」", f"「{translated}」")
                
                translated_segments.append(translated_segment)
            
            # 合并翻译结果
            final_translation = ''.join(translated_segments)
            matches_list = [match for _, match in segment_matches]
            
            return final_translation, matches_list
        
        return "", []

    def _search_vector_database(self, text: str, source_lang: str, target_lang: str) -> Optional[PatternMatch]:
        """改进的范式库检索：支持子串匹配和语义相似性"""
        try:
            query_text = self._normalize_text(text)
            
            if source_lang == 'ja' and target_lang == 'zh':
                db = self.pattern_dbs.get('ja_to_zh')
                if db and db.index.ntotal > 0:
                    return self._enhanced_search(db, query_text, text, 'ja')
            
            elif source_lang == 'zh' and target_lang == 'ja':
                db = self.pattern_dbs.get('zh_to_ja')
                if db and db.index.ntotal > 0:
                    return self._enhanced_search(db, query_text, text, 'zh')
        
        except Exception as e:
            print(f"数据库搜索出错: {e}")
        
        return None

    def _enhanced_search(self, db, query_text: str, original_text: str, lang: str) -> Optional[PatternMatch]:
        """
        增强的搜索方法，支持多种匹配策略
        """
        # 1. 精确匹配
        for pattern_id, pattern_data in db.patterns.items():
            source_pattern = pattern_data.get('source_pattern', '')
            if query_text == self._normalize_text(source_pattern):
                return PatternMatch(
                    pattern_id=pattern_id,
                    source_pattern=source_pattern,
                    target_pattern=pattern_data.get('target_pattern', ''),
                    category=pattern_data.get('category', ''),
                    confidence=1.0,
                    similarity=1.0,
                    examples=pattern_data.get('examples', []),
                    match_type='exact'
                )
        
        # 2. 子串匹配 - 检查查询文本是否是某个模式的一部分
        best_substring_match = None
        best_substring_sim = 0.0
        
        for pattern_id, pattern_data in db.patterns.items():
            source_pattern = pattern_data.get('source_pattern', '')
            normalized_pattern = self._normalize_text(source_pattern)
            
            # 检查查询是否是模式的子串
            if query_text in normalized_pattern:
                # 计算覆盖度
                coverage = len(query_text) / len(normalized_pattern)
                char_sim = self._char_jaccard(query_text, normalized_pattern)
                keyword_sim = self._keyword_similarity(original_text, source_pattern)
                
                # 综合相似度：覆盖度 + 字符相似度 + 关键词相似度
                sim = (coverage * 0.3 + char_sim * 0.4 + keyword_sim * 0.3)
                
                if sim > best_substring_sim:
                    best_substring_sim = sim
                    best_substring_match = PatternMatch(
                        pattern_id=pattern_id,
                        source_pattern=source_pattern,
                        target_pattern=pattern_data.get('target_pattern', ''),
                        category=pattern_data.get('category', ''),
                        confidence=sim,
                        similarity=sim,
                        examples=pattern_data.get('examples', []),
                        match_type='substring'
                    )
            
            # 检查模式是否是查询的子串
            elif normalized_pattern in query_text:
                coverage = len(normalized_pattern) / len(query_text)
                char_sim = self._char_jaccard(query_text, normalized_pattern)
                keyword_sim = self._keyword_similarity(original_text, source_pattern)
                
                sim = (coverage * 0.3 + char_sim * 0.4 + keyword_sim * 0.3)
                
                if sim > best_substring_sim:
                    best_substring_sim = sim
                    best_substring_match = PatternMatch(
                        pattern_id=pattern_id,
                        source_pattern=source_pattern,
                        target_pattern=pattern_data.get('target_pattern', ''),
                        category=pattern_data.get('category', ''),
                        confidence=sim,
                        similarity=sim,
                        examples=pattern_data.get('examples', []),
                        match_type='superstring'
                    )
        
        # 如果找到好的子串匹配（相似度>0.6），返回
        if best_substring_match and best_substring_sim > 0.6:
            return best_substring_match
        
        # 3. 基于关键术语的智能匹配
        key_terms = self._extract_key_terms(original_text, lang)
        if key_terms:
            term_matches = []
            for pattern_id, pattern_data in db.patterns.items():
                source_pattern = pattern_data.get('source_pattern', '')
                
                # 计算关键术语匹配度
                term_coverage = sum(1 for term in key_terms if term in source_pattern) / len(key_terms)
                
                if term_coverage > 0.5:  # 至少一半关键术语匹配
                    char_sim = self._char_jaccard(query_text, self._normalize_text(source_pattern))
                    keyword_sim = self._keyword_similarity(original_text, source_pattern)
                    
                    # 综合相似度
                    sim = (term_coverage * 0.4 + char_sim * 0.3 + keyword_sim * 0.3)
                    
                    term_matches.append((sim, pattern_id, pattern_data))
            
            # 选择最佳的术语匹配
            if term_matches:
                term_matches.sort(key=lambda x: x[0], reverse=True)
                best_sim, best_id, best_data = term_matches[0]
                
                if best_sim > 0.4:  # 术语匹配阈值
                    return PatternMatch(
                        pattern_id=best_id,
                        source_pattern=best_data.get('source_pattern', ''),
                        target_pattern=best_data.get('target_pattern', ''),
                        category=best_data.get('category', ''),
                        confidence=best_sim,
                        similarity=best_sim,
                        examples=best_data.get('examples', []),
                        match_type='term_based'
                    )
        
        # 4. 常规相似度匹配（作为最后的选择）
        best = None
        best_sim = 0.0
        for pid, pdata in db.patterns.items():
            sp = pdata.get('source_pattern', '')
            cj = self._char_jaccard(self._normalize_text(sp), query_text)
            ks = self._keyword_similarity(sp, original_text)
            sim = (cj + 2*ks) / 3.0
            
            if sim > best_sim:
                best_sim = sim
                best = PatternMatch(
                    pattern_id=pid,
                    source_pattern=sp,
                    target_pattern=pdata.get('target_pattern',''),
                    category=pdata.get('category',''),
                    confidence=pdata.get('confidence', sim),
                    similarity=sim,
                    examples=pdata.get('examples', []),
                    match_type='similarity'
                )
        
        # 只有当相似度足够高时才返回
        if best and best_sim > 0.3:
            return best
        
        return None

    def _extract_key_terms(self, text: str, lang: str) -> List[str]:
        """提取关键术语"""
        if lang == 'ja':
            # 日文关键术语
            terms = [
                '召喚', '特殊召喚', '反転召喚', '蘇生', '墓地', '手札', 'デッキ', '除外', 
                'フィールド', 'モンスター', '魔法カード', '罠カード', '効果', '発動',
                '攻撃力', '守備力', 'レベル', '属性', '種族', '破壊', '無効',
                'コントロール', '装備', '融合', '儀式', ' Spirit', 'ユニオン',
                'トゥーン', 'デュアル', 'チェーン', 'カウンター', 'ライフポイント'
            ]
        else:
            # 中文关键术语
            terms = [
                '召唤', '特殊召唤', '反转召唤', '苏生', '墓地', '手卡', '卡组', '除外',
                '场上', '怪兽', '魔法卡', '陷阱卡', '效果', '发动',
                '攻击力', '守备力', '等级', '属性', '种族', '破坏', '无效',
                '控制', '装备', '融合', '仪式', '同盟',
                '卡通', '二重', '连锁', '指示物', '生命点数'
            ]
        
        # 返回文本中包含的关键术语
        return [term for term in terms if term in text]

    def _filter_by_keywords(self, query_text: str, matches: List[PatternMatch], lang: str) -> List[PatternMatch]:
        """基于关键术语过滤候选，避免误匹配"""
        qt = self._normalize_text(query_text)
        if lang == 'ja':
            keywords = [
                'Ｓ召喚', 'チューナー以外のモンスター', '攻撃力', 'レベル', '通常召喚できない',
                '手札に加える', '墓地', '除外', '発動', '無効にし破壊する'
            ]
        else:
            keywords = [
                '同调召唤', '调整以外的怪兽', '攻击力', '等级', '不能通常召唤',
                '加入手卡', '墓地', '除外', '发动', '无效并破坏'
            ]
        present = [kw for kw in keywords if kw in qt]
        if not present:
            return matches
        filtered = [m for m in matches if all(kw in m.source_pattern for kw in present)]
        # 如果一个都没有匹配到，退化到包含至少一个关键字
        if not filtered:
            filtered = [m for m in matches if any(kw in m.source_pattern for kw in present)]
        # 如果仍为空，直接在库里基于关键字扫描候选
        if not filtered:
            db = self.pattern_dbs.get('ja_to_zh' if lang == 'ja' else 'zh_to_ja')
            if db:
                candidates: List[PatternMatch] = []
                for pid, pdata in db.patterns.items():
                    sp = pdata.get('source_pattern', '')
                    if all(kw in sp for kw in present):
                        sim = self._char_jaccard(sp, query_text)
                        candidates.append(PatternMatch(
                            pattern_id=pid,
                            source_pattern=sp,
                            target_pattern=pdata.get('target_pattern',''),
                            category=pdata.get('category',''),
                            confidence=pdata.get('confidence', sim),
                            similarity=sim,
                            examples=pdata.get('examples', []),
                            match_type='similarity'
                        ))
                candidates.sort(key=lambda x: x.similarity, reverse=True)
                filtered = candidates[:5]
        filtered.sort(key=lambda x: x.similarity, reverse=True)
        return filtered

    def _char_jaccard(self, a: str, b: str) -> float:
        sa = set(a)
        sb = set(b)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        uni = len(sa | sb)
        return inter / uni

    def _select_best_by_keywords(self, db, query_text: str, required_keywords: List[str]) -> Optional[PatternMatch]:
        candidates: List[PatternMatch] = []
        for pid, pdata in db.patterns.items():
            sp = pdata.get('source_pattern', '')
            if all(kw in sp for kw in required_keywords):
                sim = self._char_jaccard(sp, query_text)
                candidates.append(PatternMatch(
                    pattern_id=pid,
                    source_pattern=sp,
                    target_pattern=pdata.get('target_pattern',''),
                    category=pdata.get('category',''),
                    confidence=pdata.get('confidence', sim),
                    similarity=sim,
                    examples=pdata.get('examples', []),
                    match_type='similarity'
                ))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.similarity, reverse=True)
        return candidates[0]

    def _tokenize(self, text: str) -> List[str]:
        t = self._normalize_text(text)
        tokens = re.findall(r'「[^」]+」|[\u3040-\u30ff]+|[\u4e00-\u9fff]+|[A-Za-z]+|[０-９]+|\d+', t)
        # 统一全角数字
        norm = []
        for tk in tokens:
            if re.fullmatch(r'[０-９]+', tk):
                hw = ''.join(chr(ord(c) - 0xFF10 + 0x30) for c in tk)
                norm.append(hw)
            else:
                norm.append(tk)
        return norm

    def _keyword_similarity(self, a: str, b: str) -> float:
        ta = set(self._tokenize(a))
        tb = set(self._tokenize(b))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        uni = len(ta | tb)
        return inter / uni
    
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """创建查询向量，与pattern_vector_db保持一致特征空间"""
        text = self._normalize_text(text)
        vector = np.zeros(768)
        features = [
            len(text),
            text.count('「卡名」'),
            text.count('N'),
            hash(text) % 1000 / 1000,
        ]
        for i, feature in enumerate(features):
            if i < len(vector):
                vector[i] = feature
        char_features = [ord(c) / 65536.0 for c in text[:100]]
        for i, cf in enumerate(char_features):
            idx = len(features) + i
            if idx < len(vector):
                vector[idx] = cf
        return vector.astype('float32')

    def _normalize_text(self, text: str) -> str:
        """规范化文本以提高匹配成功率"""
        t = text.replace('\r', '')
        t = t.replace(':', '：')
        t = t.replace('^^', '\n')
        t = re.sub(r'[\s\u3000]+', ' ', t)
        t = t.strip()
        return t

    def _extract_numbers(self, text: str) -> List[str]:
        s = self._normalize_text(text)
        def fw_to_hw(ch: str) -> str:
            code = ord(ch)
            if 0xFF10 <= code <= 0xFF19:
                return chr(code - 0xFF10 + 0x30)
            return ch
        nums: List[str] = []
        for m in re.findall(r'[０-９]+|\d+', s):
            hw = ''.join(fw_to_hw(c) for c in m)
            nums.append(hw)
        return nums