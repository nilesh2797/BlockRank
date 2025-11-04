# BlockRank Data Format

This guide explains the data format required for training and evaluating BlockRank models.

## Table of Contents

- [JSONL Format](#jsonl-format)
- [Field Descriptions](#field-descriptions)
- [Qrels Format](#qrels-format)

## JSONL Format

BlockRank uses **JSONL** (JSON Lines) format for training and evaluation data. Each line is a valid JSON object representing one query-documents pair.

### Basic Structure

```json
{
  "query": "what is machine learning",
  "query_id": "q1",
  "documents": [
    {
      "doc_id": "0",
      "title": "Machine Learning Overview",
      "text": "Machine learning is a subset of artificial intelligence..."
    },
    {
      "doc_id": "1",
      "title": "Deep Learning",
      "text": "Deep learning uses neural networks with multiple layers..."
    }
  ],
  "answer_ids": ["0"]
}
```

## Field Descriptions

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The search query or question |
| `documents` | list | List of candidate documents (see below) |
| `answer_ids` | list | List of relevant document IDs (can be empty for inference) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | string | Unique identifier for the query (auto-generated if missing) |

### Document Structure

Each document in the `documents` list should have:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `doc_id` | string | Yes | Unique identifier for the document |
| `title` | string | No | Document title (optional, can be empty) |
| `text` | string | Yes | Document content |

**Note**: If `title` is provided, it will be prepended to `text` (separated by space).

### Answer IDs

- `answer_ids`: List of `doc_id` values that are relevant to the query
- Can contain multiple IDs for queries with multiple relevant documents
- Should be empty list `[]` for inference mode (evaluation without ground truth)
- IDs must match the `doc_id` values in the `documents` list

## Document Format Variants

BlockRank supports two document formats:

### Format 1: Dict with Fields (Recommended)

```json
{
  "query": "capital of france",
  "documents": [
    {"doc_id": "0", "title": "Paris", "text": "Paris is the capital of France."},
    {"doc_id": "1", "title": "Berlin", "text": "Berlin is the capital of Germany."}
  ],
  "answer_ids": ["0"]
}
```

**Advantages:**
- More structured
- Supports titles
- Clearer document boundaries
- Better for complex datasets

### Format 2: List of Strings (Simple)

```json
{
  "query": "capital of france",
  "documents": [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany."
  ],
  "answer_ids": [0]
}
```

**Note**: In this format:
- Documents are indexed 0, 1, 2, ...
- `answer_ids` contains integer indices
- No title field
- Simpler but less flexible

**Recommendation**: Use Format 1 (dict with fields) for production.

## Qrels Format

For evaluation metrics (nDCG, MRR), provide a **qrels** file in TSV format.

### Format

```
query-id\tdoc-id\trelevance
q1\t0\t2
q1\t1\t1
q2\t5\t1
```

- **Tab-separated** values (TSV)
- Optional header row: `query-id\tdoc-id\trelevance`
- **Relevance scores**: 0 (not relevant), 1 (relevant), 2 (highly relevant), etc.

### Example

```tsv
query-id	doc-id	relevance
q1	doc123	2
q1	doc456	1
q2	doc789	1
```

**Note**: TREC format (4 columns) is also supported:
```
query_id iteration doc_id relevance
q1 0 doc123 2
```

---

For more help, open an issue on [GitHub](https://github.com/nilesh2797/BlockRank/issues).