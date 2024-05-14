from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()
MODEL = tf.keras.models.load_model("C:/Users/Varun/OneDrive/Desktop/ml project/model_1.h5")
CLASS_NAMES = ["Early Blight","Late Blight", "Healthy"]


@app.get("/ping")
def ping():
    return {"message": "hello, I am alive"}


def read_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
def predict(file: UploadFile):
    image = read_image(file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return {
        "class": CLASS_NAMES[int(predicted_class)],
        "confidence": float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)