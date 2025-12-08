import sqlite3
import requests
from dataclasses import dataclass
from typing import List

extra_dict: dict[str, str] = {
    "リミットレギュレーション": "禁限卡表",
    "デュエルモード": "决斗模式",
    "詰めデュエル": "残局决斗",
    "じゃんけん": "猜拳",
    "先攻": "先攻",
    "後攻": "后攻",
    "デッキ": "卡组",
    "メインデッキ": "主卡组",
    "ＥＸデッキ": "额外卡组",
    "サイドデッキ": "副卡组",
    "カード種類": "卡片种类",
    "属性": "属性",
    "魔法・罠の種類": "魔法·陷阱的种类",
    "種族": "种族",
    "新規カード": "先行卡",
    "リンクマーカー": "连接箭头",
    "不明カード": "未知卡片",
    "ペンデュラム効果": "灵摆效果",
    "モンスター効果": "怪兽效果",
    "モンスターテキスト": "怪兽描述",
    "召喚": "召唤",
    "特殊召喚": "特殊召唤",
    "反転召喚": "反转召唤",
    "ペンデュラム召喚": "灵摆召唤",
    "ペンデュラム効果発動": "灵摆发动",
    "効果発動": "发动效果",
    "攻撃表示になる": "变为攻击表示",
    "守備表示になる": "变为守备表示",
    "コイントス表": "硬币正面",
    "コイントス裏": "硬币反面",
    "正規通常召喚": "通常召唤登场",
    "不正規召喚": "未正规登场",
    "検索": "搜索",
    "ドローフェイズ": "抽卡阶段",
    "スタンバイフェイズ": "准备阶段",
    "メインフェイズ": "主要阶段",
    "バトルフェイズ": "战斗阶段",
    "エンドフェイズ": "结束阶段",
    "ドロー": "抽卡",
    "送る": "送至",
    "加える": "加入",
    "戻す": "回到",
    "回収": "回收",
    "発動": "发动",
    "公開": "公开",
    "手札公開": "公开手卡",
    "セットカード公開": "公开盖卡",
    "デッキ公開": "公开卡组",
    "コスト": "代价",
    "リリース": "解放",
    "破壊": "破坏",
    "セット": "盖放",
    "置く": "放置",
    "儀式召喚": "仪式召唤",
    "融合召喚": "融合召唤",
    "シンクロ召喚": "同调召唤",
    "エクシーズ召喚": "超量召唤",
    "リンク召喚": "连接召唤",
    "チェイン": "连锁",
    "チェイン終了": "连锁结束",
    "効果処理": "效果处理",
    "処理完了": "处理结束",
    "発動無効": "发动无效",
    "効果無効": "效果无效",
    "ペンデュラム召喚終了": "灵摆召唤结束",
    "対象": "对象",
    "表示形式変更": "更改表示形式",
    "移動": "移动",
    "コントロール変更": "转移控制权",
    "魔法カード": "魔法卡",
    "罠カード": "陷阱卡",
    "通常": "通常",
    "効果": "效果",
    "儀式": "仪式",
    "融合": "融合",
    "シンクロ": "同步",
    "エクシーズ": "超量",
    "トゥーン": "卡通",
    "スピリット": "灵魂",
    "ユニオン": "同盟",
    "デュアル": "二重",
    "チューナー": "协调",
    "リバース": "反转",
    "ペンデュラム": "灵摆",
    "リンク": "连接",
    "魔法使い族": "魔法师族",
    "ドラゴン族": "龙族",
    "アンデット族": "不死族",
    "戦士族": "战士族",
    "獣戦士族": "兽战士族",
    "獣族": "兽族",
    "鳥獣族": "鸟兽族",
    "悪魔族": "恶魔族",
    "天使族": "天使族",
    "昆虫族": "昆虫族",
    "恐竜族": "恐龙族",
    "爬虫類族": "爬虫类族",
    "魚族": "鱼族",
    "海竜族": "海龙族",
    "水族": "水族",
    "炎族": "炎族",
    "雷族": "雷族",
    "岩石族": "岩石族",
    "植物族": "植物族",
    "機械族": "机械族",
    "サイキック族": "念动力族",
    "幻神獣族": "幻神兽族",
    "創造神族": "创造神族",
    "幻竜族": "幻龙族",
    "サイバース族": "电子界族",
    "サイボーグ族": "电子人族",
    "魔導騎士族": "魔导骑士族",
    "ハイドラゴン族": "多头龙族",
    "天界戦士族": "天界骑士族",
    "オメガサイキック族": "欧米茄念动力族",
    "ギャラクシー族": "银河族",
    "幻想魔族": "幻想魔族"
}


@dataclass
class CardTranslateData:
    ja_name: str
    ja_desc: str
    zh_name: str
    zh_desc: str


def download_database(url: str, local_path: str) -> None:
    """下载SQLite数据库文件"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def query_database(db_path: str) -> List[CardTranslateData]:
    """查询数据库并返回卡片翻译数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
            select j.name, j.desc, z.name, z.desc
            from datas d
                     left join ja_texts j on d.id = j.id
                     left join zhcn_texts z on d.id = z.id
            where j.name is not null
              and z.name is not null; \
            """

    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    card_data_list = []
    for row in results:
        ja_name, ja_desc, zh_name, zh_desc = row
        card_data = CardTranslateData(
            ja_name=ja_name or "",
            ja_desc=ja_desc or "",
            zh_name=zh_name or "",
            zh_desc=zh_desc or ""
        )
        card_data_list.append(card_data)

    return card_data_list

def save_prepared_data(extra: dict[str, str], cl: list[CardTranslateData]):
    # 青眼の白龍||青眼白龙
    s = ""
    for k, v in extra.items():
        s += f"{k}||{v}\n"
    for card in cl:
        s += f"{card.ja_name}||{card.zh_name}\n"
    for card in cl:
        s += f"{card.ja_desc.replace("\n", "^^").replace("\r", "")}||{card.zh_desc.replace("\n", "^^").replace("\r", "")}\n"

    with open("ja_knowledge.txt", "w") as f:
        f.write(s)

    s2 = ""

    for k, v in extra.items():
        s2 += f"{v}||{k}\n"
    for card in cl:
        s2 += f"{card.zh_name}||{card.ja_name}\n"
    for card in cl:
        s2 += f"{card.zh_desc.replace("\n", "^^").replace("\r", "")}||{card.ja_desc.replace("\n", "^^").replace("\r", "")}\n"

    with open("zh_knowledge.txt", "w") as f:
        f.write(s2)


def main():
    """主函数"""
    db_url = "https://duelistsunite.org/omega/OmegaDB.cdb"
    local_db_path = "OmegaDB.cdb"

    # 下载数据库
    download_database(db_url, local_db_path)

    # 查询数据
    card_data_list = query_database(local_db_path)

    save_prepared_data(extra_dict, card_data_list)


if __name__ == "__main__":
    main()
