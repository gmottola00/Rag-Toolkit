# Unified Vector Store Testing Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Unified Testing Framework                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          base_tests.py (Abstract)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BaseVectorStoreTest (ABC)                                                  │
│  ├── @abstractmethod vector_store_service()                                │
│  ├── @abstractmethod collection_name()                                     │
│  ├── @abstractmethod vector_size()                                         │
│  └── sample_vectors(), sample_ids(), sample_metadata() [concrete]         │
│                                                                              │
│  VectorStoreCollectionTests(BaseVectorStoreTest)                           │
│  ├── test_create_collection()                                              │
│  ├── test_collection_exists()                                              │
│  ├── test_drop_collection()                                                │
│  └── test_list_collections()                                               │
│                                                                              │
│  VectorStoreDataTests(BaseVectorStoreTest)                                 │
│  ├── test_add_vectors()                                                    │
│  ├── test_upsert_vectors()                                                 │
│  ├── test_search_vectors()                                                 │
│  ├── test_get_by_ids()                                                     │
│  ├── test_delete_vectors()                                                 │
│  └── test_count_vectors()                                                  │
│                                                                              │
│  VectorStoreHealthTests(BaseVectorStoreTest)                               │
│  ├── test_health_check()                                                   │
│  └── test_connection()                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ inherits
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Concrete Test Implementations                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  test_milvus/                      test_qdrant/         test_chroma/       │
│  ├── test_connection.py            ├── test_connection  ├── test_connection│
│  ├── test_collection.py            ├── test_collection  ├── test_collection│
│  ├── test_data.py                  ├── test_data        ├── test_data      │
│  └── test_service.py               └── test_service     └── test_service   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ uses
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     test_unified.py (Parameterized)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  @pytest.mark.parametrize("vector_store_name", ["milvus", "qdrant", "chro │
│                                                                              │
│  TestUnifiedVectorStoreCollections                                          │
│  ├── test_create_and_drop_collection(vector_store_name)                   │
│  │   ├── if milvus: service.ensure_collection(...)                        │
│  │   ├── elif qdrant: service.ensure_collection(...)                      │
│  │   └── elif chroma: service.create_collection(...)                      │
│  │                                                                          │
│  └── test_list_collections(vector_store_name)                             │
│                                                                              │
│  TestUnifiedVectorStoreData                                                 │
│  ├── test_add_and_search_vectors(vector_store_name)                       │
│  │   ├── if milvus: service.data.insert(...)                              │
│  │   ├── elif qdrant: service.data.upsert(...)                            │
│  │   └── elif chroma: service.add(...)                                    │
│  │                                                                          │
│  └── test_count_vectors(vector_store_name)                                │
│                                                                              │
│  TestUnifiedVectorStoreHealth                                               │
│  └── test_health_check(vector_store_name)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ uses fixtures from
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       conftest.py (Shared Fixtures)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  @pytest.fixture(scope="session")                                           │
│  milvus_service() ──┐                                                       │
│                     │                                                        │
│  @pytest.fixture(scope="session")                                           │
│  qdrant_service() ──┼─── Connects to Docker services                       │
│                     │                                                        │
│  @pytest.fixture(scope="session")                                           │
│  chroma_service() ──┘                                                       │
│                                                                              │
│  pytest_collection_modifyitems() ─── Skip unavailable vector stores        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          Docker Services Layer                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │   Milvus    │    │   Qdrant    │    │  ChromaDB   │                   │
│  │ :19530      │    │ :6333       │    │ :8000       │                   │
│  │ :9091 (UI)  │    │ :6334 (gRPC)│    │ :8000/docs  │                   │
│  └─────────────┘    └─────────────┘    └─────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


Test Execution Flow:
════════════════════

1. Start Docker Services
   └─> make docker-up

2. Pytest Discovers Tests
   ├─> Reads conftest.py
   ├─> Registers markers (vectorstore, milvus, qdrant, chroma, unified)
   └─> Creates session-scoped fixtures (milvus_service, qdrant_service, chroma_service)

3. Run Parameterized Tests
   ├─> test_unified.py::TestUnifiedVectorStoreCollections::test_create_and_drop_collection
   │   ├─> [milvus] - Uses milvus_service fixture
   │   ├─> [qdrant] - Uses qdrant_service fixture
   │   └─> [chroma] - Uses chroma_service fixture (or skip if not installed)
   │
   └─> test_unified.py::TestUnifiedVectorStoreData::test_add_and_search_vectors
       ├─> [milvus] - Tests Milvus implementation
       ├─> [qdrant] - Tests Qdrant implementation
       └─> [chroma] - Tests ChromaDB implementation

4. Cleanup
   └─> Fixtures call service.close()


Test Selection Examples:
═══════════════════════

# All unified tests across all vector stores (9 tests = 3 stores × 3 test classes)
pytest tests/test_infra/test_vectorstores/test_unified.py

# Only Milvus tests (3 tests)
pytest tests/test_infra/test_vectorstores/test_unified.py -k milvus

# Only collection tests across all stores (3 tests = 3 stores × 1 test class)
pytest tests/test_infra/test_vectorstores/test_unified.py::TestUnifiedVectorStoreCollections

# Specific test for specific store (1 test)
pytest tests/test_infra/test_vectorstores/test_unified.py::TestUnifiedVectorStoreData::test_count_vectors[qdrant]

# All vector store tests (unified + implementation-specific)
pytest tests/test_infra/test_vectorstores/ -v


Benefits:
════════

✅ Write test once, run on all vector stores
✅ Consistent behavior across implementations
✅ Easy to add new vector stores
✅ Catches implementation-specific bugs
✅ Tests serve as cross-platform documentation
✅ CI/CD friendly with Docker services
```
