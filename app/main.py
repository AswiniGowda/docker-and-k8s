
import hashlib
import json
import os

import joblib
import numpy as np
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Iris ML Prediction API")

model = joblib.load("app/model.pkl")

class_names = ["setosa", "versicolor", "virginica"]

# Connect to Redis using the Docker Compose service name
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: IrisInput):
    try:
        # Create a unique cache key for the input
        features = [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]

        cache_key = "prediction:" + hashlib.sha256(
            json.dumps(features).encode()
        ).hexdigest()

        # Check Redis cache
        cached_result = redis_client.get(cache_key)

        if cached_result:
            result = json.loads(cached_result)
            result["cached"] = True
            return result

        # Cache miss: run model inference
        input_data = np.array([features])
        prediction = model.predict(input_data)

        result = {
            "prediction": class_names[int(prediction[0])],
            "class_id": int(prediction[0])
        }

        # Store result in Redis for 300 seconds
        redis_client.setex(
            cache_key,
            300,
            json.dumps(result)
        )

        result["cached"] = False
        return result

    except redis.RedisError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Redis is unavailable: {e!s}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
