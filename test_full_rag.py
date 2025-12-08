#!/usr/bin/env python3
"""
完整测试RAG翻译功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.enhanced_translator import EnhancedYugiohTranslator, EnhancedTranslationRequest

def test_full_rag_translation():
    """完整测试RAG翻译"""

    translator = EnhancedYugiohTranslator()
    
    test_text = "このカードが召喚に成功した時、デッキから魔法カード１枚を手札に加える。"
    
    print("=" * 60)
    print("完整测试RAG翻译")
    print("=" * 60)
    print(f"测试句子：{test_text}")
    
    # 创建翻译请求
    request = EnhancedTranslationRequest(
        source_text=test_text,
        source_lang='ja',
        target_lang='zh',
        use_pattern_matching=True,
        use_card_name_translation=True
    )
    
    try:
        # 执行翻译（这将调用LLM）
        print("\n开始执行RAG翻译...")
        result = translator.translate(request)
        
        print(f"\n✓ 翻译完成！")
        print(f"最终译文：{result.translated_text}")
        print(f"处理时间：{result.processing_time:.3f}秒")
        print(f"置信度：{result.confidence:.3f}")
        
        # 显示句子分解详情
        for breakdown in result.sentence_breakdown:
            print(f"\n=== 句子处理详情 ===")
            print(f"原文：{breakdown['source']}")
            print(f"译文：{breakdown['translation']}")
            print(f"匹配类型：{breakdown.get('match_type', 'unknown')}")
            print(f"找到范式数：{breakdown.get('patterns_found', 0)}")
            
            # 如果有LLM提示词和响应，显示它们
            if breakdown.get('llm_prompt'):
                print(f"\n--- LLM提示词 ---")
                print(breakdown['llm_prompt'])
                print("--- 提示词结束 ---")
            
            if breakdown.get('llm_response'):
                print(f"\n--- LLM响应 ---")
                print(breakdown['llm_response'])
                print("--- 响应结束 ---")
        
    except Exception as e:
        print(f"✗ 翻译过程中出错：{e}")
        import traceback
        traceback.print_exc()

def test_multiple_cases():
    """测试多个用例"""
    
    translator = EnhancedYugiohTranslator()
    
    test_cases = [
        "このカードが召喚に成功した時、デッキから魔法カード１枚を手札に加える。",
        "１ターンに１度、手札から魔法カード１枚を捨てて発動できる。",
        "相手がモンスターを召喚した時、このカードを破壊する。"
    ]
    
    print("=" * 60)
    print("测试多个RAG翻译用例")
    print("=" * 60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n【用例 {i}】{test_text}")
        
        request = EnhancedTranslationRequest(
            source_text=test_text,
            source_lang='ja',
            target_lang='zh',
            use_pattern_matching=True,
            use_card_name_translation=True
        )
        
        try:
            result = translator.translate(request)
            print(f"译文：{result.translated_text}")
            print(f"处理时间：{result.processing_time:.3f}秒")
            
            # 显示匹配类型
            if result.sentence_breakdown:
                match_type = result.sentence_breakdown[0].get('match_type', 'unknown')
                patterns_found = result.sentence_breakdown[0].get('patterns_found', 0)
                print(f"匹配类型：{match_type}, 找到范式：{patterns_found}个")
                
        except Exception as e:
            print(f"✗ 翻译失败：{e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_full_rag_translation()
    print("\n\n")
    test_multiple_cases()