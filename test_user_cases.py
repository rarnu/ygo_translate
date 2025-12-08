#!/usr/bin/env python3
"""
测试用户提供的测试用例
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_translator import EnhancedYugiohTranslator, EnhancedTranslationRequest

def test_user_cases():
    """测试用户提供的测试用例"""
    print("=== 测试用户提供的测试用例 ===\n")
    
    # 初始化翻译器
    try:
        translator = EnhancedYugiohTranslator()
        print("✓ 翻译器初始化成功\n")
    except Exception as e:
        print(f"✗ 翻译器初始化失败: {e}\n")
        return
    
    # 用户提供的测试用例
    test_cases = [
        {
            "name": "特殊召唤条件",
            "source_lang": "ja",
            "target_lang": "zh",
            "expected": "把自己墓地3张「废品究极龙」卡除外的场合才能特殊召唤。",
            "text": "自分の墓地の「ジャンクの究極竜」カード３枚を除外した場合のみ特殊召喚できる。"
        },
        {
            "name": "同调召唤+效果",
            "source_lang": "ja",
            "target_lang": "zh",
            "expected": "「废品同调士」＋调整以外的怪兽1只以上①：这张卡同调召唤的场合发动。这张卡的攻击力上升自己场上的2星以下的怪兽的攻击力的合计数值。",
            "text": "「ジャンク・シンクロン」＋チューナー以外のモンスター１体以上\n①：このカードがＳ召喚した場合に発動する。このカードの攻撃力は、自分フィールドのレベル２以下のモンスターの攻撃力の合計分アップする。"
        },
        {
            "name": "通常召唤限制",
            "source_lang": "ja",
            "target_lang": "zh",
            "expected": "这张卡不能通常召唤。",
            "text": "このカードは通常召喚できない。"
        },
        {
            "name": "召唤成功效果",
            "source_lang": "ja",
            "target_lang": "zh",
            "expected": "这张卡召唤成功时，从卡组将1张魔法卡加入手卡。",
            "text": "このカードが召喚に成功した時、デッキから魔法カード１枚を手札に加える。"
        },
        {
            "name": "复杂效果测试",
            "source_lang": "ja",
            "target_lang": "zh",
            "expected": "「彩宝龙」＋调整以外的怪兽1只以上①：这张卡同调召唤的场合发动。这张卡的攻击力上升自己场上的12星以下的怪兽的攻击力的合计数值。",
            "text": "「彩宝龍」＋チューナー以外のモンスター１体以上\n①：このカードがＳ召喚した場合に発動する。このカードの攻撃力は、自分フィールドのレベル１２以下のモンスターの攻撃力の合計分アップする。"
        },
        {
            "name": "中译日测试1",
            "source_lang": "zh",
            "target_lang": "ja",
            "expected": "「彩宝龍」＋チューナー以外のモンスター1体以上①：このカードがＳ召喚した場合に発動する。このカードの攻撃力は、自分フィールド上に存在するレベル12以下のモンスターの攻撃力の合計分アップする。",
            "text": "「彩宝龙」＋调整以外的怪兽1只以上\n①：这张卡同调召唤的场合发动。这张卡的攻击力上升自己场上等级12以下的怪兽的攻击力合计数值。"
        },
        {
            "name": "中译日测试2", 
            "source_lang": "zh",
            "target_lang": "ja",
            "expected": "このカードが召喚に成功した時、デッキから魔法カード１枚を手札に加える。",
            "text": "这张卡召唤成功时，从卡组将1张魔法卡加入手卡。"
        }
    ]
    
    # 运行测试
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}/{total_tests}: {test_case['name']}")
        print("-" * 50)
        
        try:
            # 创建翻译请求
            request = EnhancedTranslationRequest(
                source_text=test_case['text'],
                source_lang=test_case['source_lang'],
                target_lang=test_case['target_lang'],
                use_pattern_matching=True,
                use_card_name_translation=True
            )
            
            # 执行翻译
            result = translator.translate(request)
            
            # 输出结果
            print(f"原文: {test_case['text']}")
            print(f"预期: {test_case['expected']}")
            print(f"结果: {result.translated_text}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"处理时间: {result.processing_time:.3f}秒")

        except Exception as e:
            print(f"✗ 测试出错: {e}")
        
        print("\n")

    print("=" * 60)

if __name__ == "__main__":
    test_user_cases()