from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from dotenv import load_dotenv
from .routers import generate, train
import os

# 加载 .env 文件
load_dotenv()

# 创建模型保存目录
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

app = FastAPI(
    title="Neural Network Builder API",
    description="可视化神经网络搭建平台后端 API",
    version="1.0.0"
)


# 自定义 CORS 中间件
@app.middleware("http")
async def cors_handler(request: Request, call_next):
    # 处理预检请求
    if request.method == "OPTIONS":
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "600"
        return response

    # 处理其他请求
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# 注册路由
app.include_router(generate.router, prefix="/api", tags=["generate"])
app.include_router(train.router, prefix="/api", tags=["train"])


@app.get("/")
async def root():
    return {
        "message": "Neural Network Builder API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
