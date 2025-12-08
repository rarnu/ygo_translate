#!/usr/bin/env python3
"""
多语言RAG翻译器CLI工具
支持英语、日语到中文的翻译
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.faiss_knowledge_base import FAISSKnowledgeBase
from src.data.japanese_preparator import JapaneseDataPreparator
from src.data.chinese_preparator import ChineseDataPreparator


class MultiLanguageRAGTranslator:
    """多语言RAG翻译器"""

    def __init__(self,
                 model_name: str = "alibayram/Qwen3-30B-A3B-Instruct-2507:latest",
                 ollama_url: str = "http://localhost:11434",
                 use_faiss: bool = True,
                 faiss_index_type: str = "flat"):
        """
        初始化多语言RAG翻译器

        Args:
            model_name: 模型名称
            ollama_url: Ollama服务URL
            use_faiss: 是否使用FAISS
            faiss_index_type: FAISS索引类型, 可选值: flat, hnsw, ivf
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_faiss = use_faiss

        # 初始化知识库
        self.knowledge_bases = {}
        self._load_knowledge_bases(faiss_index_type)

        # 语言配置
        self.language_config = {
            'ja': {
                'name': '日语',
                'kb_prefix': 'data/ja/knowledge_base'
            },
            'zh': {
                'name': '中文',
                'kb_prefix': 'data/zh/knowledge_base'
            }
        }

    def _load_knowledge_bases(self, index_type: str):
        """加载所有语言的知识库"""
        for lang in ['ja', 'zh']:
            kb_path = f"data/{lang}/knowledge_base"
            try:
                kb = FAISSKnowledgeBase(index_type=index_type)
                kb.load_from_file(kb_path)
                self.knowledge_bases[lang] = kb
                print(f"✓ 已加载{lang}知识库，数据量: {len(kb.pairs)}")
            except Exception as e:
                print(f"⚠️  无法加载{lang}知识库: {e}")
                # 创建空知识库
                kb = FAISSKnowledgeBase(index_type=index_type)
                self.knowledge_bases[lang] = kb
                print(f"✓ 创建了空的{lang}知识库")

    def add_translation_pair(self,
                             source_text: str,
                             translated_text: str,
                             source_lang: str,
                             category: str = "",
                             priority: float = 1.0,
                             metadata: Dict = None,
                             auto_save: bool = True) -> str:
        """添加翻译对到对应语言知识库"""
        if source_lang not in self.knowledge_bases:
            raise ValueError(f"不支持的语言: {source_lang}")

        kb = self.knowledge_bases[source_lang]
        pair_id = kb.add_translation_pair(
            original=source_text,
            translated=translated_text,
            category=category,
            priority=priority,
            metadata=metadata or {}
        )

        if auto_save:
            # 重新构建索引
            kb.build_index()

            # 保存知识库
            kb_path = f"data/{source_lang}/knowledge_base"
            kb.save_to_file(kb_path)

        return pair_id

    def batch_add_translation_pairs(self,
                                    pairs_data: List[Dict],
                                    source_lang: str,
                                    auto_save: bool = True) -> int:
        """批量添加翻译对（性能优化版本）"""
        if source_lang not in self.knowledge_bases:
            raise ValueError(f"不支持的语言: {source_lang}")

        kb = self.knowledge_bases[source_lang]
        kb.batch_add_pairs(pairs_data)

        if auto_save:
            # 一次性构建索引
            kb.build_index()

            # 保存知识库
            kb_path = f"data/{source_lang}/knowledge_base"
            kb.save_to_file(kb_path)

        return len(pairs_data)

    def get_stats(self) -> Dict:
        """获取翻译器统计信息"""
        stats = {
            'supported_languages': list(self.language_config.keys()),
            'knowledge_bases': {},
            'model_name': self.model_name
        }

        for lang, kb in self.knowledge_bases.items():
            kb_stats = kb.get_stats()
            stats['knowledge_bases'][lang] = kb_stats

        return stats


def add_translation_data(translator: MultiLanguageRAGTranslator, lang: str, file_path: str):
    """添加翻译数据（批量优化版本）"""
    print(f"正在添加{lang}翻译数据: {file_path}")
    
    try:
        if lang == 'ja':
            preparator = JapaneseDataPreparator()
            data = preparator.prepare_from_ja_txt(file_path)
        elif lang == 'zh':
            preparator = ChineseDataPreparator()
            data = preparator.prepare_from_zh_txt(file_path)
        else:
            from src.data.data_preparation import DataPreparator
            preparator = DataPreparator()
            data = preparator.prepare_from_txt(file_path)
        
        if not data:
            print("没有找到有效数据")
            return
        
        # 使用批量API添加数据
        print(f"开始批量添加 {len(data)} 条翻译数据...")
        added_count = translator.batch_add_translation_pairs(data, lang)
        
        print(f"✓ 成功批量添加 {added_count} 条{lang}翻译数据")
        
    except Exception as e:
        print(f"添加数据失败: {e}")


def show_stats(translator: MultiLanguageRAGTranslator):
    """显示统计信息"""
    stats = translator.get_stats()
    
    print("=== 翻译器统计信息 ===")
    print(f"支持语言: {', '.join(stats['supported_languages'])}")
    print(f"使用模型: {stats['model_name']}")
    
    print("\n=== 知识库统计 ===")
    for lang, kb_stats in stats['knowledge_bases'].items():
        print(f"\n{lang.upper()} 知识库:")
        print(f"  总数据量: {kb_stats['total_pairs']}")
        print(f"  索引类型: {kb_stats['index_type']}")
        print(f"  向量维度: {kb_stats['embedding_dim']}")
        print(f"  索引已构建: {kb_stats['is_built']}")
        
        if kb_stats['categories']:
            print("  类别分布:")
            for category, count in kb_stats['categories'].items():
                print(f"    {category}: {count}")



def add_translation_data_batch(translator: MultiLanguageRAGTranslator, lang: str, file_pattern: str):
    """批量添加多个文件的翻译数据"""
    import glob
    import time
    
    files = glob.glob(file_pattern)
    if not files:
        print(f"没有找到匹配的文件: {file_pattern}")
        return
    
    print(f"找到 {len(files)} 个文件，开始批量处理...")
    
    # 获取对应语言的知识库
    kb = translator.knowledge_bases[lang]
    total_added = 0
    start_time = time.time()
    
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 处理文件: {file_path}")
        
        try:
            # 准备数据
            if lang == 'ja':
                preparator = JapaneseDataPreparator()
                data = preparator.prepare_from_ja_txt(file_path)
            elif lang == 'zh':
                preparator = ChineseDataPreparator()
                data = preparator.prepare_from_zh_txt(file_path)
            else:
                from src.data.data_preparation import DataPreparator
                preparator = DataPreparator()
                data = preparator.prepare_from_txt(file_path)
            
            if not data:
                print("  -> 没有找到有效数据，跳过")
                continue
            
            # 批量添加到知识库（先不构建索引）
            kb.batch_add_pairs(data)
            total_added += len(data)
            print(f"  -> 添加了 {len(data)} 条数据")
            
        except Exception as e:
            print(f"  -> 处理失败: {e}")
    
    # 一次性构建索引
    if total_added > 0:
        print(f"\n开始构建FAISS索引（{total_added} 条数据）...")
        kb.build_index()
        
        # 保存知识库
        kb_path = f"data/{lang}/knowledge_base"
        kb.save_to_file(kb_path)
        
        end_time = time.time()
        print(f"✓ 批量处理完成！")
        print(f"  - 处理文件数: {len(files)}")
        print(f"  - 添加数据量: {total_added}")
        print(f"  - 总耗时: {end_time - start_time:.2f} 秒")
        print(f"  - 平均速度: {total_added/(end_time - start_time):.1f} 条/秒")
    else:
        print("没有添加任何数据")


def main():
    parser = argparse.ArgumentParser(description='游戏王中日翻译器')
    parser.add_argument('command', choices=['add_data', 'add_batch', 'stats'], help='执行的命令', nargs='?')
    parser.add_argument('lang', nargs='?', choices=['ja', 'zh'], help='语言 (ja/zh)')
    parser.add_argument('input_file', nargs='?', help='输入文件或文件模式（支持通配符）')
    
    args = parser.parse_args()
    
    # 初始化翻译器
    try:
        translator = MultiLanguageRAGTranslator()
    except Exception as e:
        print(f"❌ 初始化翻译器失败: {e}")
        return
    
    # 执行命令
    if args.command == 'add_data':
        if not args.lang or not args.input_file:
            print("❌ 请指定语言和输入文件")
            print("用法: python ygo_translate.py add_data <lang> <input_file>")
            return
        add_translation_data(translator, args.lang, args.input_file)
    elif args.command == 'add_batch':
        if not args.lang or not args.input_file:
            print("❌ 请指定语言和文件模式")
            print("用法: python ygo_translate.py add_batch <lang> <file_pattern>")
            print("示例: python ygo_translate.py add_batch ja 'data/ja/*.txt'")
            return
        add_translation_data_batch(translator, args.lang, args.input_file)
    elif args.command == 'stats':
        show_stats(translator)

if __name__ == '__main__':
    main()