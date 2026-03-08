# Builder stage: install dependencies and build artifacts
FROM python:3.11-slim AS builder

# set working directory in builder
WORKDIR /app

# copy requirements & install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit joblib pyyaml

# copy application source code
COPY . .

# preprocess data and train model during build
RUN python data_preprocessing.py \
    && python model_building.py


# Runtime stage: minimal final image
FROM python:3.11-slim AS runtime

# use non-root user for security (optional)
# RUN useradd -m appuser && chown -R appuser /app

WORKDIR /app

# copy only necessary Python packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# copy application code and generated artifacts from builder
COPY --from=builder /app/front.py ./
COPY --from=builder /app/models ./models
COPY --from=builder /app/preprocessed_data ./preprocessed_data

# expose streamlit default port
EXPOSE 8501

# default command
CMD ["streamlit", "run", "front.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
