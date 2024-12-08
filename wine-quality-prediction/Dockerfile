# Use Amazon Corretto as base image
FROM amazoncorretto:11

# Install required packages
RUN yum install -y wget tar gzip

# Set Spark version
ENV SPARK_VERSION=3.5.3
ENV SPARK_HOME=/opt/spark

# Download and set up Spark
RUN wget https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop3 /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop3.tgz

# Copy application files
COPY target/wine-quality-prediction-1.0-SNAPSHOT.jar /app/wine-predictor.jar
COPY data/ /app/data/

# Set working directory
WORKDIR /app

# Create models directory
RUN mkdir -p /app/models

# Add Spark to PATH
ENV PATH=$PATH:$SPARK_HOME/bin

# Command to run the application
ENTRYPOINT ["spark-submit", \
            "--class", "com.wine.prediction.WineQualityPredictor", \
            "--master", "spark://54.173.237.74:7077", \
            "/app/wine-predictor.jar", \
            "/app/data/ValidationDataset.csv"]
