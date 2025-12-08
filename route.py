from dataclasses import dataclass, asdict
from typing import Callable, Any
from aiohttp import web
from src.core.enhanced_translator import EnhancedYugiohTranslator, EnhancedTranslationRequest

_translator: EnhancedYugiohTranslator | None = None

def get_translator() -> EnhancedYugiohTranslator:
    global _translator
    if _translator is None:
        _translator = EnhancedYugiohTranslator()
    return _translator

@dataclass
class TranslationRequest:
    """翻译请求"""
    text: str

@dataclass
class TranslationResponse:
    translated_text: str
    processing_time: float
    confidence: float


async def api_ja2zh(request: web.Request, dumps: Callable[[Any], str]) -> web.Response:
    return await _translate(request, dumps, 'ja', 'zh')


async def api_zh2ja(request: web.Request, dumps: Callable[[Any], str]) -> web.Response:
    return await _translate(request, dumps, 'zh', 'ja')


async def _translate(request: web.Request, dumps: Callable[[Any], str], source_lang: str, target_lang: str) -> web.Response:
    try:
        data = await request.json()
        req = TranslationRequest(**data)
        req.text = req.text.replace('\r', '').strip()

        # 使用增强翻译
        enhanced_result = get_translator().translate(EnhancedTranslationRequest(
            source_text=req.text,
            source_lang=source_lang,
            target_lang=target_lang
        ))

        wrapper = TranslationResponse(
            translated_text=enhanced_result.translated_text,
            processing_time=enhanced_result.processing_time,
            confidence=float(enhanced_result.confidence)
        )

        return web.json_response(data={'code': 0, 'message': 'success', 'data': asdict(wrapper)}, dumps=dumps)

    except Exception as e:
        return web.json_response(data={'code': 500, 'message': str(e)}, dumps=dumps)

