# ChromaDB Docker Setup

This directory contains Docker configuration for running ChromaDB locally.

## Quick Start

```bash
# Start ChromaDB
docker-compose up -d

# Stop ChromaDB
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Using the Helper Script

From the repository root:

```bash
# Start ChromaDB
make docker-up-chroma

# Check health
make docker-health-chroma

# Stop ChromaDB
make docker-down-chroma

# Clean everything
make docker-clean-chroma
```

## Accessing ChromaDB

- **HTTP API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/heartbeat

## Configuration

### Environment Variables

- `DOCKER_VOLUME_DIRECTORY`: Base directory for volumes (default: current directory)
- `ANONYMIZED_TELEMETRY`: Send anonymous usage statistics (default: TRUE)
- `CHROMA_SERVER_AUTH_CREDENTIALS_FILE`: Path to authentication file
- `CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER`: Authentication provider class

### Persistent Storage

Data is persisted in `./volumes/chroma` by default. Change this by setting:

```bash
DOCKER_VOLUME_DIRECTORY=/path/to/data docker-compose up -d
```

## Using ChromaDB in Python

```python
from rag_toolkit.infra.vectorstores.factory import create_chroma_service

# Connect to Docker instance
service = create_chroma_service(
    host="localhost",
    port=8000
)

# Create collection
service.create_collection("my_collection")

# Add documents
service.add(
    collection_name="my_collection",
    documents=["Hello world", "ChromaDB is great"],
    ids=["id1", "id2"],
    metadatas=[{"source": "test"}, {"source": "test"}],
)

# Query
results = service.query(
    collection_name="my_collection",
    query_texts=["greeting"],
    n_results=2,
)
```

## Authentication (Optional)

To enable basic authentication:

1. Create `auth.json` file:
```json
{
  "credentials": {
    "username": "your_password_hash"
  }
}
```

2. Update `docker-compose.yml`:
```yaml
environment:
  - CHROMA_SERVER_AUTH_CREDENTIALS_FILE=/chroma/auth.json
  - CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.basic.BasicAuthCredentialsProvider
volumes:
  - ./auth.json:/chroma/auth.json:ro
```

## In-Memory vs Persistent Mode

ChromaDB can run in different modes:

### In-Memory (No Docker needed)
```python
service = create_chroma_service()  # No host/port = in-memory
```

### Persistent Local (No Docker needed)
```python
service = create_chroma_service(path="./my_chroma_db")
```

### Remote Server (Docker)
```python
service = create_chroma_service(host="localhost", port=8000)
```

## Monitoring

Check container health:
```bash
docker exec chromadb curl http://localhost:8000/api/v1/heartbeat
```

View logs:
```bash
docker-compose logs -f chromadb
```

## Troubleshooting

### Connection Refused
- Ensure container is running: `docker ps`
- Check health: `docker-compose ps`
- View logs: `docker-compose logs chromadb`

### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./volumes/chroma
```

### Port Already in Use
Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

## Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)
- [Docker Image](https://github.com/chroma-core/chroma/pkgs/container/chroma)
