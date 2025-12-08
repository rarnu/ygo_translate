#!/usr/bin/env python3
"""
构建拆分的范式数据库
分别创建日译中和中译日两个独立的范式数据库
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.pattern_vector_db import JapaneseToChinesePatternDB, ChineseToJapanesePatternDB, create_pattern_embeddings
from src.core.pattern_extractor import DirectionalPatternExtractor

def build_split_pattern_dbs():
    """构建拆分的范式数据库"""
    print("=== 构建拆分的范式数据库 ===\n")
    
    # 大模型配置
    model_name = "alibayram/Qwen3-30B-A3B-Instruct-2507:latest"
    ollama_url = "http://localhost:11434"
    
    # 构建日译中数据库
    print("1. 构建日译中数据库...")
    try:
        # 提取日文到中文范式
        print("   提取日文到中文范式...")
        ja_to_zh_extractor = DirectionalPatternExtractor.extract_ja_to_zh(
            'data/ja/knowledge_base_metadata.json', model_name, ollama_url
        )
        ja_to_zh_patterns_data = ja_to_zh_extractor.export_for_vector_db()
        print(f"   提取到 {len(ja_to_zh_patterns_data)} 个日文到中文范式")
        ja_to_zh_extractor.save_patterns('data/ja_to_zh_patterns.json')
        
        # 创建向量
        print("   创建日文到中文范式向量...")
        ja_to_zh_embeddings = create_pattern_embeddings(ja_to_zh_patterns_data)
        
        # 构建日译中数据库
        print("   构建日译中向量数据库...")
        ja_to_zh_db = JapaneseToChinesePatternDB(index_type="flat")
        ja_to_zh_db.add_patterns(ja_to_zh_patterns_data, ja_to_zh_embeddings)
        ja_to_zh_db.save_to_file("data/pattern_vector_db_ja_to_zh")
        print(f"   日译中数据库构建完成: {len(ja_to_zh_db.patterns)} 个范式")
        
    except FileNotFoundError:
        print("   错误: 未找到日文知识库文件 data/ja/knowledge_base_metadata.json")
        return
    except Exception as e:
        print(f"   构建日译中数据库时出错: {e}")
        return
    
    print()
    
    # 构建中译日数据库
    print("2. 构建中译日数据库...")
    try:
        # 提取中文到日文范式
        print("   提取中文到日文范式...")
        zh_to_ja_extractor = DirectionalPatternExtractor.extract_zh_to_ja(
            'data/zh/knowledge_base_metadata.json', model_name, ollama_url
        )
        zh_to_ja_patterns_data = zh_to_ja_extractor.export_for_vector_db()
        print(f"   提取到 {len(zh_to_ja_patterns_data)} 个中文到日文范式")
        zh_to_ja_extractor.save_patterns('data/zh_to_ja_patterns.json')
        
        # 创建向量
        print("   创建中文到日文范式向量...")
        zh_to_ja_embeddings = create_pattern_embeddings(zh_to_ja_patterns_data)
        
        # 构建中译日数据库
        print("   构建中译日向量数据库...")
        zh_to_ja_db = ChineseToJapanesePatternDB(index_type="flat")
        zh_to_ja_db.add_patterns(zh_to_ja_patterns_data, zh_to_ja_embeddings)
        zh_to_ja_db.save_to_file("data/pattern_vector_db_zh_to_ja")
        print(f"   中译日数据库构建完成: {len(zh_to_ja_db.patterns)} 个范式")
        
    except FileNotFoundError:
        print("   警告: 未找到中文知识库文件 data/zh/knowledge_base_metadata.json，跳过中译日数据库构建")
    except Exception as e:
        print(f"   构建中译日数据库时出错: {e}")
    
    print("\n=== 数据库构建完成 ===")
    print("数据库文件:")
    print("  - data/pattern_vector_db_ja_to_zh.* (日译中数据库)")
    print("  - data/pattern_vector_db_zh_to_ja.* (中译日数据库)")
    print("  - data/ja_to_zh_patterns.json (日译中范式)")
    print("  - data/zh_to_ja_patterns.json (中译日范式)")

if __name__ == "__main__":
    build_split_pattern_dbs()