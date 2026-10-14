ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt .
RUN python -c "import deap, fastapi, httpx, langchain_core, langchain_openai, numpy, pydantic, uvicorn" \
    || pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 5001
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "5001"]
