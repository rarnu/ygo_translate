#!/usr/bin/env python3
"""
基于aiohttp的翻译服务端程序
提供中译日和日译中两个接口
支持传统翻译和增强翻译两种模式
"""
from typing import Callable, Any
import json
from aiohttp import web
from aiohttp.typedefs import Handler
from route import api_ja2zh, api_zh2ja, get_translator

CORS_HEADERS: dict[str, str] = {
    "Access-Control-Allow-Origin": "*",  # 允许所有来源
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, X-CSRF-Token",
    "Access-Control-Allow-Credentials": "true"
}


@web.middleware
async def cors_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    # 处理OPTIONS预检请求
    if request.method == "OPTIONS":
        # 创建预检请求响应
        response = web.Response(status=200, headers=CORS_HEADERS)
        return response

    # 处理正常请求
    try:
        response = await handler(request)
        # 为正常响应添加CORS头
        response.headers.update(CORS_HEADERS)
        return response
    except web.HTTPException as ex:
        # 为异常响应也添加CORS头
        ex.headers.update(CORS_HEADERS)
        raise


class TranslationServer:
    """翻译服务端"""
    
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self.app: web.Application = web.Application(middlewares=[cors_middleware])
        self.setup_routes()

    def setup_routes(self):
        dumps: Callable[[Any], str] = lambda x: json.dumps(obj=x, ensure_ascii=False)

        @self.routes.post('/api/yugioh/translate/j2z')
        async def j2z(request: web.Request) -> web.Response:
            return await api_ja2zh(request, dumps)

        @self.routes.post('/api/yugioh/translate/z2j')
        async def z2j(request: web.Request) -> web.Response:
            return await api_zh2ja(request, dumps)

        @self.routes.post('/api/yugioh/cardname/add')
        async def add_cardname(request: web.Request) -> web.Response:
            return await api_cardname_add(request, dumps)

        @self.routes.post('/api/yugioh/cardname/delete')
        async def delete_cardname(request: web.Request) -> web.Response:
            return await api_cardname_delete(request, dumps)

        @self.routes.post('/api/yugioh/cardname/exists')
        async def cardname_exists(request: web.Request) -> web.Response:
            return await api_cardname_exists(request, dumps)

        # 添加路由到应用
        self.app.add_routes(self.routes)


if __name__ == "__main__":
    # 初始化翻译器
    get_translator()
    
    # 启动服务器
    server = TranslationServer()
    print("🚀 翻译服务启动在 http://0.0.0.0:8082")
    print("📡 API接口:")
    print("  POST /api/yugioh/translate/j2z - 日译中")
    print("  POST /api/yugioh/translate/z2j - 中译日")

    web.run_app(app=server.app, host='0.0.0.0', port=8082)