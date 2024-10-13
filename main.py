from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import numpy as np
import joblib
import logging
from typing import Union
from datetime import datetime, timedelta, timezone
import jwt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the default prediction model
model_path = "xgb_classifier_model.pkl"  # Update with your actual model file path
try:
    model = joblib.load(model_path)
    logging.info("Default prediction model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading default prediction model: {e}")
    raise RuntimeError(f"Error loading model: {e}")

# JWT settings
SECRET_KEY = "1b09eae62fffc38eac635a40e90f68652ef34161ade629343429538b53a67344"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Define user database
users_db = {
    "admin": {"username": "admin", "password": "adminpass", "role": "admin"},
    "user": {"username": "user", "password": "userpass", "role": "user"},
}

# Helper function to authenticate user
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user:
        logging.error(f"User '{username}' not found.")
        return False
    if user["password"] != password:
        logging.error(f"Password mismatch for user '{username}'.")
        return False
    logging.info(f"User '{username}' authenticated successfully.")
    return user

# Function to create a JWT token
def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Login endpoint to get the JWT token
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Pydantic model for the 23 input fields
class FraudPredictionInput(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

# Fraud prediction endpoint using individual inputs
@app.post("/predict-fraud/")
async def predict_fraud(data: FraudPredictionInput, token: str = Depends(oauth2_scheme)):
    # Convert input to numpy array
    input_data = np.array([[
        data.LIMIT_BAL, data.SEX, data.EDUCATION, data.MARRIAGE, data.AGE, 
        data.PAY_0, data.PAY_2, data.PAY_3, data.PAY_4, data.PAY_5, data.PAY_6,
        data.BILL_AMT1, data.BILL_AMT2, data.BILL_AMT3, data.BILL_AMT4, data.BILL_AMT5, data.BILL_AMT6,
        data.PAY_AMT1, data.PAY_AMT2, data.PAY_AMT3, data.PAY_AMT4, data.PAY_AMT5, data.PAY_AMT6
    ]])

    logging.info(f"Received input data: {input_data}")

    try:
        # Make prediction using the model
        prediction = model.predict(input_data)
        logging.info(f"Model prediction: {prediction}")

        # Check if the prediction output is valid
        if prediction is None or len(prediction) == 0:
            raise ValueError("Prediction returned an empty or invalid value.")

        # Return a meaningful result based on the prediction
        if prediction[0] == 0:
            return {"prediction": "The instance is predicted as not fraudulent."}
        else:
            return {"prediction": "The instance is predicted as fraudulent."}

    except ValueError as ve:
        logging.error(f"ValueError during prediction: {ve}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {ve}")

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction.")

@app.get("/")
def root():
    return {"message": "Default prediction API is running"}
