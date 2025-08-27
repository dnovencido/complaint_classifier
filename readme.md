# Complaint Classifier API

This project provides a FastAPI-based API to classify complaints into categories using a trained machine learning model. The application is containerized using Docker for easy deployment.

---

## **Docker Setup**

### Build the Docker Image

```bash
docker build -t complaint_classifier .

docker run --name complaint_classifier -p 8001:8001 complaint_classifier