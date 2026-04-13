# Assignment 4 — Architectural Analysis of RAGFlow

---

## 1. Deep document understanding vs naive chunking

Deep document understanding improves retrieval fidelity because it preserves structural semantics that fixed-size chunking destroys. In enterprise documents such as financial reports or PDFs with tables, meaning is often encoded in layout rather than linear text. A table row, for example, may only make sense when read alongside its column headers. Naive chunking breaks this relationship, causing the system to retrieve incomplete or misleading context. From an index design perspective, layout-aware parsing produces section titles, table structure, and document hierarchy, which enables more precise filtering and ranking. This increases signal density in the index. Fixed-size chunks, by contrast, produce context-poor entries that are harder to distinguish during retrieval.

The main tradeoff is preprocessing cost. Layout-aware parsing is significantly more expensive in terms of compute and engineering complexity. This increases ingestion latency and operational overhead. Additionally, parsing errors such as misreading a table can propagate downstream and produce subtle retrieval failures that are harder to debug than errors from naive chunking.

---

## 2. Chunking strategy: template vs semantic

Template-based chunking relies on predefined structural boundaries such as headings and sections, while semantic chunking uses embeddings to segment text based on shifts in meaning. The core tradeoff is predictability versus adaptability. Template-based chunking is deterministic and easy to debug. Semantic chunking is flexible but less controllable, more computationally expensive, and harder to reason about when it fails.

Template-based chunking performs well on highly structured documents like financial reports, where document structure aligns closely with logical meaning. It fails, however, when documents are inconsistently formatted or when important information spans multiple sections. Semantic chunking performs better on loosely structured corpora such as chat logs, where meaning is not tied to any explicit structure and topics shift organically. It fails on structured documents because it can split logically connected content. For instance, it can separate a table header from its rows, or a premise from its conclusion.

---

## 3. Hybrid retrieval architecture

Lexical-only retrieval fails when queries use synonyms or paraphrases. For example, a query for "revenue growth" will miss documents that use "top-line increase." Vector-only retrieval fails on rare tokens, numbers, or exact identifiers such as product codes or legal clause references, where semantic similarity is insufficient to surface the right result.

Hybrid retrieval mitigates both failure modes by merging candidate sets from both methods before reranking, increasing recall through broader candidate coverage and precision through better final ordering. An edge case failure for hybrid systems occurs when the two signals conflict. Lexical retrieval can find keyword-matching but topically irrelevant documents, while vector retrieval surfaces semantically relevant ones. If the reranking step is poorly optimised, the system may still prioritise the wrong results despite having both signals available.

---

## 4. Multi-stage retrieval pipeline

A multi-stage pipeline is better than single-pass ANN search because it separates recall optimisation from precision optimisation. The first stage prioritises high recall, retrieving a broad candidate set quickly using approximate nearest neighbour search. Later stages apply more expensive models to improve precision over that candidate set. This avoids running a slow, high-precision model over the entire index for every query.

The key tradeoff is recall versus latency. Larger candidate sets increase recall but slow down re-ranking. Smaller candidate sets are faster but risk missing relevant documents entirely. A significant problem is cascading error propagation. If the first stage fails to retrieve a relevant document, no later stage can recover it, making early-stage recall critical. Equally, errors in re-ranking can misorder results even when the candidate set contains the correct document.

---

## 5. Indexing strategy and storage backends

The choice of storage backend should be driven by a small set of design criteria: the dominant query pattern (keyword lookup, semantic search, or multi-hop reasoning), the need for structured filtering by metadata such as date or author, update frequency, and the degree of relationship complexity between documents.

An Elasticsearch-like hybrid store is the best choice when queries combine keyword and semantic signals, strong metadata filtering is required, and updates are moderately frequent. It supports hybrid retrieval natively and handles structured filtering efficiently, making it the default for enterprise RAG systems where users issue mixed queries such as "latest financial report mentioning X."

A vector-native database is better when queries are primarily semantic, the dataset is large-scale and embedding-heavy, and there is not much need for filtering. Vector databases are optimised for approximate nearest neighbour search and scale well with embedding dimensionality. However, they perform poorly when queries require exact matching or structured constraints.

A graph-augmented store is best when queries require multi-hop reasoning — for example, finding all suppliers connected to a given entity through an intermediate relationship — and when the data has explicit, well-defined relationships. Graphs require higher modelling and maintenance overhead and, since they are computationally expensive, work best on corpora with relatively low update frequency.

---

## 6. Query understanding and reformulation

Static queries assume that user input is already sufficient for retrieval. In practice there is often a semantic gap between how a user phrases a question and how the relevant knowledge is represented in the index. Query transformation addresses this by aligning the query with the representation space before retrieval. To increase recall, expansion adds related terms. For example, expanding "AI safety" into associated concepts surfaces documents that use different vocabulary. Decomposition breaks a complex query into simpler sub-queries that can each be answered independently and then combined.

To recover from a poorly phrased initial query, the system detects ambiguity, issues follow-up queries based on retrieval results, and refines its approach before generating a final response. The tradeoff is latency and complexity. Static queries are fast but brittle. Iterative refinement improves robustness but increases response time and introduces the risk of loops or instability in agent behaviour if termination conditions are poorly designed.

---

## 7. Knowledge representation layer

Dense vector space enables fast semantic retrieval but lacks interpretability. Relationships between concepts are implied by the geometry of the embedding space, which makes compositional reasoning difficult. The system cannot reliably answer queries that require chaining multiple logical steps.

Relational schemas provide structured, queryable data with strong consistency guarantees. They support compositional queries through joins and constraints, but require a predefined schema, which makes them inflexible for unstructured or heterogeneous data.

Knowledge graphs explicitly encode relationships as edges between entities, enabling multi-hop reasoning and clear provenance. They are the strongest representation for compositional reasoning and retrieval explainability, since every answer can be traced to a path through the graph. However, at large scale they are expensive to construct, maintain, and query.

As it is hard to interpret why a result is retrieved from a nearest-neighbour match, structured representations (relational or graph) are better for retrieval explainability, while vector representations are relatively opaque.

---

## 8. Data ingestion pipeline architecture

A robust ingestion system should ideally be a separate, event-driven pipeline that converts heterogeneous data into a unified, retrieval-ready format.

For schema normalisation, I would define a document schema containing a document ID, content, metadata fields (source, timestamp, section, and author), structural information (headers and table hierarchy), and a precomputed embedding. Each data source is handled by a source-specific adapter. For example, a PDF adapter performs layout parsing and section extraction, and a log adapter groups messages by timestamp. These adapters isolate source-specific complexity and ensure that downstream retrieval operates on a consistent representation.

For incremental indexing, I would implement event-driven ingestion. New or updated documents trigger events that a delta processing service handles by reprocessing only the affected documents. Each document is updated by ID, and index updates are applied incrementally rather than through a full rebuild. To avoid inconsistencies, writes should be idempotent and a change log should be maintained to allow replay in the event of failures.

On the consistency versus throughput tradeoff, I would choose eventual consistency with bounded staleness for a production RAG system. Ingestion runs asynchronously from the serving path, so retrieval may briefly return slightly stale results. Strict consistency would require blocking queries during indexing, which is unacceptable for latency. Generally, slightly stale data is preferable to degraded responsiveness.

---

## 9. Memory design in RAG systems

Vector memory enables semantic recall of past interactions by storing embeddings of prior turns and retrieving the most semantically similar ones at query time. It is flexible and requires no schema, but may retrieve approximate or irrelevant matches and does not support precise structured queries. Structured memory using SQL or graph representations supports precise retrieval and explicit reasoning over stored facts, but requires schema design and ongoing maintenance. It is best suited to domains where past information has a predictable structure.

Episodic logs store temporal sequences of interactions, which is useful for maintaining conversational context over time. However, logs can become large and noisy and are difficult to query efficiently without additional indexing. The overall tradeoff is between precision, flexibility, and scalability. Vector memory is flexible but imprecise. Structured memory is precise but rigid. Episodic memory is informationally rich but hard to query at scale.

---

## 10. End-to-end system decomposition

**Stateless vs stateful services.** Stateless services (the API service, Admin service, and ingestion orchestrator) hold no persistent data and can be freely replaced or scaled. Stateful services (the Meta service, retrieval index, storage service, and message queue) own data and require replication and careful consistency management. Since the Meta service isolates all system configuration, document metadata, and processing status into a dedicated relational store shared by both the Admin and API paths, RAGFlow ensures that administrative writes never contend with live query traffic.

**Scaling strategy.** Each component scales according to its workload characteristics. The API service scales horizontally behind NGINX since it is stateless and CPU-bound. The ingestion service scales through parallel workers consuming from the message queue. Because documents are processed independently, throughput scales linearly with worker count, and the queue absorbs upload bursts without propagating pressure downstream. DeepDoc scales independently of the ingestion service because parsing is the most compute-intensive step in the offline pipeline and benefits from being tuned separately. The retrieval service scales through index sharding and read replicas, while the model provider is an external dependency whose scaling is managed outside RAGFlow.

**Failure isolation.** Isolation is enforced at each inter-service boundary. The message queue is the most critical boundary: if ingestion crashes, queued jobs are retained and replayed on recovery without the API service being affected. DeepDoc's isolation means a parsing failure on a malformed document does not bring down the broader ingestion pipeline. On the query path, model provider calls are timeout-bounded, with a fallback to returning retrieved context directly if the provider is unavailable. Each layer can therefore fail or scale independently, which is the architecture's primary design goal.
