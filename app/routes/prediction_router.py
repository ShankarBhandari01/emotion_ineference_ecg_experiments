from fastapi import APIRouter
from app.models.schema import PredictionRequest, PredictionResponse
from app.services.services import predict
from app.logger import logger

router = APIRouter(prefix="/model")

@router.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    result = predict([request.feature1, request.feature2])
    return PredictionResponse(result=result)

@router.get("/")
def root():
    logger.info("predict_success", output={"prediction": '2'})
    return {"message": "Hello World"}
