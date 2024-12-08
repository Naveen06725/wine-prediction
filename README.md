
# Wine Quality Prediction

This project aims to predict the quality of wine based on various features using machine learning. The application is designed to be run in a containerized environment with Apache Spark for data processing.

## Project Structure

```
wine-quality-prediction/
├── src/                # Source code (currently empty)
├── data/               # Data files for training and validation
│   ├── TrainingDataset.csv
│   ├── ValidationDataset.csv
├── models/             # Placeholder for trained models (currently empty)
├── pom.xml             # Maven configuration file for project dependencies
├── Dockerfile          # Docker configuration file for containerization
```

## Prerequisites

Before running the project, ensure the following software is installed on your system:
- Docker
- Maven (for local builds)
- Java 11+

## Running the Project

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t wine-quality-prediction .
   ```

2. Run the Docker container:
   ```bash
   docker run -it --rm wine-quality-prediction
   ```

### Using Maven

1. Build the project using Maven:
   ```bash
   mvn clean install
   ```

2. Run the application:
   ```bash
   java -jar target/wine-quality-prediction-1.0-SNAPSHOT.jar
   ```

## Data Description

- **TrainingDataset.csv**: Used for training the machine learning model.
- **ValidationDataset.csv**: Used for validating the model's performance.

## Notes

- Ensure that the data files are in the correct format and properly pre-processed before running the model.
- The `models/` directory is a placeholder for storing trained models, which will be generated during execution.

## Future Enhancements

- Add source code for data preprocessing and model training.
- Include model evaluation metrics and a pipeline for deploying trained models.
- Provide sample predictions and an API endpoint for accessing the model.

---

Developed with Apache Spark and Maven.

