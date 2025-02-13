# Databricks notebook source
def parse_contents(contents):
    """Parse uploaded image contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))

# COMMAND ----------

from PIL import Image

# COMMAND ----------

img = Image.open("/Workspace/Users/zach.jacobson@databricks.com/databricks_apps/databricks_app_omniverse/web-app-demos/image-annotation-app/assets/driving.jpg")

# COMMAND ----------

content_type, content_string = img.split(',')

# COMMAND ----------

import base64
from PIL import Image
import io

image_path = "/Workspace/Users/zach.jacobson@databricks.com/databricks_apps/databricks_app_omniverse/web-app-demos/image-annotation-app/assets/driving.jpg"
img = Image.open(image_path)
img_array = np.array(img)

# COMMAND ----------

img_array

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import torch
import mlflow
from mlflow.pyfunc import PythonModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os
from mlflow.models import infer_signature

os.environ['DATABRICKS_TOKEN'] = 'dapif374ba9190f8780936b3eb4addfc14c4'

class FasterRCNNModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self,model_input):
        with torch.no_grad():
            predictions = self.model(model_input)
        return predictions

with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Load pre-trained model
    image = Image.open("/Workspace/Users/zach.jacobson@databricks.com/databricks_apps/databricks_app_omniverse/web-app-demos/image-annotation-app/assets/driving.jpg")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_deploy = model.to(device)


    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    # Convert to tensor and normalize
    img_tensor = F.to_tensor(image_pil)

    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    #Add signature
    model = FasterRCNNModel(model_deploy)
    model_output = model.predict(img_tensor)
    signature = infer_signature(img_tensor.numpy(), np.array(model_output))
    # Log the model
    mlflow.pyfunc.log_model(
        "model",
        python_model=model,
        signature= signature,
        pip_requirements=["torch", "torchvision", "pillow", "numpy"]
    )

# COMMAND ----------

img_tensor.shape

# COMMAND ----------

model = FasterRCNNModel(model_deploy)
model_output = model.predict(img_tensor)

# COMMAND ----------

predictions = FasterRCNNModel(model_deploy).predict(torch.tensor(img_tensor.numpy()))

# COMMAND ----------

predictions

# COMMAND ----------

img_tensor.numpy().shape

# COMMAND ----------

signature = infer_signature(img_tensor.numpy(), np.array(model_output))
signature

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np


catalog = "mfg_mid_central_sa"
schema = "zach_jacobson"
model_name = "cv_car_detection_model"

_register = mlflow.register_model(f"runs:/{run_id}/model", f"{catalog}.{schema}.{model_name}") # Register the model to our schema

client = MlflowClient()
client.set_registered_model_alias(f"{catalog}.{schema}.{model_name}", "Production", int(_register.version))

# COMMAND ----------

input_data

# COMMAND ----------

input_data

# COMMAND ----------

input_data.shape

# COMMAND ----------


model_uri = 'runs:/2dbe4b50e0024e1583b2c5b802445915/model'

# Replace INPUT_EXAMPLE with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
input_data = img_tensor.numpy()

# Load the model
model = mlflow.pyfunc.load_model(model_uri)

# Predict using the loaded model
predictions = model.predict(input_data)

# Display the predictions
display(predictions)

# COMMAND ----------

model_instance = FasterRCNNModel(model_deploy)

# Log the model using this instance
mlflow.pyfunc.log_model(
    "model",
    python_model=model_instance,  # use the instance here
    signature=signature,
    pip_requirements=["torch", "torchvision", "pillow", "numpy"]
)

# COMMAND ----------

model_uri = 'runs:/a8fc7dede674414d97b544f39725205e/model'

# Load the model
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Prepare input as a DataFrame or a suitable format for `mlflow.pyfunc`
import pandas as pd

# Assuming input_data is numpy array, convert it to DataFrame
input_dataframe = pd.DataFrame(input_data.reshape(1, 3, 445, 800))  # Adjust shape if necessary based on the model input format

# Predict using the loaded model
predictions = loaded_model.predict(input_dataframe)

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col, base64
logged_model = f'models:/{catalog}.{schema}.{model_name}'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
# pd_loaded_model = mlflow.pyfunc.load_model(logged_model)
df = spark.table(f"mfg_mid_central_sa.zach_jacobson.image_annotation_raw").limit(1).withColumn("image_base64", base64("content"))
# Predict on a Spark DataFrame.
# pd_loaded_model.predict(df.toPandas())
df.withColumn('predictions', loaded_model("image_base64")).select("predictions").display()