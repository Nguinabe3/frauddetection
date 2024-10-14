# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000 8501

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run uvicorn server
CMD ["sh", "-c", "streamlit run app.py --server.port 8501 --server.address 0.0.0.0 & sleep 5 && uvicorn main:app --host 0.0.0.0 --port 8000"]