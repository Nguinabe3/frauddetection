from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
import logging
import io
from datetime import datetime, timedelta, timezone
from typing import Union
from passlib.context import CryptContext
import jwt
from jwt import PyJWTError
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the default prediction model
model_path = "xgb_classifier_model.pkl"  # Update to the correct model file name if needed
if not os.path.exists(model_path):
    logging.error(f"Model file not found: {model_path}")
    raise RuntimeError(f"Model file not found: {model_path}")

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

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User database
users_db = {
    "admin": {"username": "admin", "password": pwd_context.hash("adminpass"), "role": "admin"},
    "user": {"username": "user", "password": pwd_context.hash("userpass"), "role": "user"},
}

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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

# Fraud prediction endpoint from CSV file with authentication
@app.post("/predict-fraud-csv/")
async def predict_fraud_csv(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File type not supported. Please upload a CSV file.")

    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))  # Read the CSV

        # Ensure the CSV has the expected columns
        expected_columns = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        
        if not all(column in df.columns for column in expected_columns):
            raise HTTPException(status_code=400, detail="CSV must contain the required columns.")

        # Apply the default prediction model on the DataFrame
        predictions = model.predict(df[expected_columns])

        # Attach predictions to DataFrame and return as response
        df['prediction'] = predictions
        results = df.to_dict(orient="records")

        return JSONResponse(content=results)

    except Exception as e:
        logging.error(f"Error processing the CSV file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error while processing the file")


# Data model for individual inputs
class FraudPredictionInput(BaseModel):
    LIMIT_BAL: float
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
    # Convert input data to a DataFrame
    df = pd.DataFrame([data.dict()])

    try:
        # Apply the default prediction model
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        logging.error(f"Error processing the input data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error while processing the input data")


@app.get("/")
def root():
    return {"message": "Default prediction API is running"}
