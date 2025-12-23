# Unified Vector Store Testing

This directory contains a unified testing suite for all vector store implementations in RAG-Toolkit.

## Architecture

### Base Test Classes (`base_tests.py`)
Abstract test classes that define the standard test interface:
- `BaseVectorStoreTest` - Base class with common fixtures
- `VectorStoreCollectionTests` - Collection management tests
- `VectorStoreDataTests` - Data operation tests
- `VectorStoreFilterTests` - Filtering capability tests
- `VectorStoreBatchTests` - Batch operation tests
- `VectorStoreHealthTests` - Health check tests

### Parameterized Tests (`test_unified.py`)
Pytest parameterized tests that run the same tests across all vector stores:
- `TestUnifiedVectorStoreCollections` - Collection operations
- `TestUnifiedVectorStoreData` - Data operations
- `TestUnifiedVectorStoreHealth` - Health checks

### Shared Fixtures (`conftest.py`)
Common fixtures for all vector stores:
- `milvus_service` - Milvus service instance
- `qdrant_service` - Qdrant service instance
- `chroma_service` - ChromaDB service instance

## Running Tests

### Run all unified tests
```bash
# All vector stores
pytest tests/test_infra/test_vectorstores/test_unified.py

# Specific vector store
pytest tests/test_infra/test_vectorstores/test_unified.py -k milvus
pytest tests/test_infra/test_vectorstores/test_unified.py -k qdrant
pytest tests/test_infra/test_vectorstores/test_unified.py -k chroma
```

### Run with markers
```bash
# Only unified tests
pytest -m unified

# Only integration tests (requires Docker)
pytest -m integration

# Specific vector store
pytest -m milvus
pytest -m qdrant
pytest -m chroma
```

### Run specific test classes
```bash
# Collection tests only
pytest tests/test_infra/test_vectorstores/test_unified.py::TestUnifiedVectorStoreCollections

# Data tests only
pytest tests/test_infra/test_vectorstores/test_unified.py::TestUnifiedVectorStoreData
```

## Prerequisites

### Docker Services
Start the required vector stores:
```bash
# All services
make docker-up

# Individual services
make docker-up-milvus
make docker-up-qdrant
make docker-up-chroma
```

### Python Dependencies
```bash
# Install all vector store clients
uv pip install -e ".[qdrant,chroma]"

# Or install individually
uv pip install -e ".[qdrant]"
uv pip install -e ".[chroma]"
```

## Test Matrix

| Test | Milvus | Qdrant | ChromaDB |
|------|--------|--------|----------|
| Create Collection | ✅ | ✅ | ✅ |
| Drop Collection | ✅ | ✅ | ✅ |
| List Collections | ✅ | ✅ | ✅ |
| Add Vectors | ✅ | ✅ | ✅ |
| Search Vectors | ✅ | ✅ | ✅ |
| Count Vectors | ✅ | ✅ | ✅ |
| Delete Vectors | ✅ | ✅ | ✅ |
| Metadata Filtering | ✅ | ✅ | ✅ |
| Health Check | ✅ | ✅ | ✅ |

## Adding New Vector Stores

To add a new vector store to the unified test suite:

1. **Implement the service** following the pattern:
   ```python
   class NewVectorStoreService:
       def __init__(self, config):
           self.connection = ConnectionManager(config)
           self.collections = CollectionManager(self.connection)
           self.data = DataManager(self.connection)
       
       def health_check(self) -> bool:
           ...
       
       def close(self) -> None:
           ...
   ```

2. **Add fixture** in `conftest.py`:
   ```python
   @pytest.fixture(scope="session")
   def newstore_service() -> Generator[NewStoreService, None, None]:
       config = NewStoreConfig(...)
       service = NewStoreService(config)
       try:
           yield service
       finally:
           service.close()
   ```

3. **Add to parameterization** in `test_unified.py`:
   ```python
   @pytest.mark.parametrize("vector_store_name", ["milvus", "qdrant", "chroma", "newstore"])
   ```

4. **Add implementation logic** in test methods:
   ```python
   elif vector_store_name == "newstore":
       service.create_collection(collection_name)
   ```

## Environment Variables

Configure connection endpoints:
```bash
# Milvus
export MILVUS_URI=http://localhost:19530
export MILVUS_USER=username
export MILVUS_PASSWORD=password

# Qdrant
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=your-api-key

# ChromaDB
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
```

## CI/CD Integration

Example GitHub Actions workflow:
```yaml
name: Vector Store Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      milvus:
        image: milvusdb/milvus:latest
        ports:
          - 19530:19530
      
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      
      chromadb:
        image: ghcr.io/chroma-core/chroma:latest
        ports:
          - 8000:8000
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev,qdrant,chroma]"
      
      - name: Run unified tests
        run: pytest tests/test_infra/test_vectorstores/test_unified.py -v
```

## Benefits

1. **Consistency** - All vector stores tested with identical tests
2. **Maintainability** - Add one test, it runs on all implementations
3. **Reliability** - Catch backend-specific issues early
4. **Documentation** - Tests serve as usage examples
5. **Extensibility** - Easy to add new vector stores

## Troubleshooting

### Tests skipped for ChromaDB
```bash
# Install ChromaDB
uv pip install -e ".[chroma]"
```

### Connection refused errors
```bash
# Check services are running
make docker-ps
make docker-health

# Start services
make docker-up
```

### Tests fail on specific vector store
```bash
# Run only that vector store's tests
pytest tests/test_infra/test_vectorstores/test_unified.py -k qdrant -v

# Check service logs
docker logs qdrant
```
