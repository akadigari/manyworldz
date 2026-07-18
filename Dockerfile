# Run the manyworldz engine anywhere with one command.
# Build:  docker build -t manyworldz .
# Run:    docker run -e ANTHROPIC_API_KEY=your-key manyworldz
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Which crowd model to use: haiku (default), sonnet, opus, or fable.
ENV MANYWORLDZ_MODEL=haiku

CMD ["python", "run.py"]
