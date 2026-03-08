# Base image
FROM python:3.11-slim

# set working directory
WORKDIR /app

# copy requirements & install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit joblib pyyaml

# copy everything else
COPY . .

# run preprocessing and train model at build time so the container is ready to serve
RUN python data_preprocessing.py \
    && python model_building.py

# expose streamlit default port
EXPOSE 8501

# default command
CMD ["streamlit", "run", "front.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
