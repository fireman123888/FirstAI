from fastapi import APIRouter, HTTPException
from ..models.schemas import NetworkStructure, GenerateResponse
from ..services.ai_generator import generate_pytorch_code

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate_code(structure: NetworkStructure):
    """根据网络结构生成 PyTorch 代码"""
    try:
        if not structure.nodes:
            raise HTTPException(status_code=400, detail="请先添加网络层")

        code = await generate_pytorch_code(structure.nodes, structure.edges)
        return GenerateResponse(code=code)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
