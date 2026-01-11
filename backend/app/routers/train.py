from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from ..models.schemas import TrainRequest, TrainStartResponse, TrainStatusResponse
from ..services.trainer import start_training_task, get_training_status, get_sample_predictions
import asyncio
import json

router = APIRouter()


@router.post("/train/start", response_model=TrainStartResponse)
async def start_training(request: TrainRequest):
    """启动训练任务"""
    try:
        if not request.nodes:
            raise HTTPException(status_code=400, detail="请先添加网络层")

        task_id = await start_training_task(
            request.nodes,
            request.edges,
            request.config
        )

        return TrainStartResponse(
            task_id=task_id,
            message="训练任务已启动"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/{task_id}", response_model=TrainStatusResponse)
async def get_training_progress(task_id: str):
    """获取训练进度"""
    status = get_training_status(task_id)

    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    return TrainStatusResponse(
        status=status["status"],
        current_epoch=status["current_epoch"],
        total_epochs=status["total_epochs"],
        loss=status["loss"],
        accuracy=status["accuracy"],
        history=status["history"]
    )


@router.websocket("/ws/train/{task_id}")
async def training_websocket(websocket: WebSocket, task_id: str):
    """WebSocket 实时推送训练状态"""
    await websocket.accept()

    try:
        last_epoch = 0
        while True:
            status = get_training_status(task_id)

            if status is None:
                await websocket.send_json({"error": "任务不存在"})
                break

            # 发送状态更新
            if status["current_epoch"] != last_epoch or status["status"] in ["completed", "failed"]:
                await websocket.send_json(status)
                last_epoch = status["current_epoch"]

            if status["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


@router.get("/train/{task_id}/predictions")
async def get_predictions(task_id: str, count: int = 6):
    """获取模型预测示例"""
    status = get_training_status(task_id)

    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="训练尚未完成")

    try:
        predictions = get_sample_predictions(task_id, count)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
