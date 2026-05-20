-- PostgreSQL schema for the RAG pipeline.
-- Requires the pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;
-- The first statement below installs it if not already present.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
	doc_id TEXT PRIMARY KEY,
	filename TEXT NOT NULL,
	source_path TEXT NOT NULL,
	source_type TEXT NOT NULL DEFAULT 'pdf_book',
	num_pages INTEGER NOT NULL,
	metadata_json TEXT NOT NULL,
	ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS artifacts (
	id BIGSERIAL PRIMARY KEY,
	doc_id TEXT NOT NULL,
	artifact_type TEXT NOT NULL,
	artifact_path TEXT NOT NULL,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	UNIQUE(doc_id, artifact_type),
	FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
	chunk_id TEXT PRIMARY KEY,
	doc_id TEXT NOT NULL,
	-- collection_id: user-defined grouping of documents (e.g. "ML Books", "CS7646").
	-- NULL means the document has not been assigned to a collection.
	collection_id TEXT,
	source_name TEXT,
	document_title TEXT,
	document_path TEXT,
	section_id TEXT,
	path_text TEXT,
	title TEXT,
	level INTEGER,
	page_start INTEGER,
	page_end INTEGER,
	has_table INTEGER NOT NULL DEFAULT 0,
	token_count_est INTEGER,
	source_type TEXT NOT NULL DEFAULT 'pdf_book',
	structural_role TEXT NOT NULL DEFAULT 'body',
	text TEXT NOT NULL,
	embedding vector(4096),
	FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_path_text ON chunks(path_text);
CREATE INDEX IF NOT EXISTS idx_chunks_collection_id ON chunks(collection_id);
-- NOTE: vector(4096) exceeds pgvector's 2000-dim index limit.
-- Create the ANN index manually after ingesting data:
--   CREATE INDEX idx_chunks_embedding ON chunks
--     USING ivfflat ((embedding::halfvec(4096)) halfvec_cosine_ops) WITH (lists = 100);


CREATE TABLE IF NOT EXISTS conversations (
	conversation_id TEXT PRIMARY KEY,
	title           TEXT,
	created_at      TIMESTAMPTZ DEFAULT NOW(),
	updated_at      TIMESTAMPTZ DEFAULT NOW(),
	archived        INTEGER DEFAULT 0,
	summary         TEXT
);

CREATE TABLE IF NOT EXISTS conversation_messages (
	message_id      TEXT PRIMARY KEY,
	conversation_id TEXT NOT NULL,
	role            TEXT NOT NULL,
	content         TEXT NOT NULL,
	mode            TEXT,
	sequence        INTEGER NOT NULL,
	created_at      TIMESTAMPTZ DEFAULT NOW(),
	FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conv_messages ON conversation_messages(conversation_id, sequence);

-- Named, user-defined document collections.
-- collection_id is a short slug chosen by the user (e.g. "CS7646", "CS7646/notes").
-- parent_id allows a two-level hierarchy: parent collections act as folders,
-- sub-collections hold the actual documents. Querying a parent automatically
-- includes all its sub-collections.
CREATE TABLE IF NOT EXISTS collections (
	collection_id TEXT PRIMARY KEY,
	name TEXT NOT NULL,
	description TEXT,
	parent_id TEXT REFERENCES collections(collection_id) ON DELETE SET NULL,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tracks which idempotent column migrations (_CHUNK_MIGRATIONS list in db/client.py)
-- have been applied to this database, keyed by their 0-based index in that list.
CREATE TABLE IF NOT EXISTS schema_migrations (
	id         INTEGER PRIMARY KEY,
	applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hypothetical questions generated per chunk at index time.
-- Each row stores one question and its embedding vector so that
-- question-to-question ANN search can surface relevant chunks even when
-- the user's phrasing does not match the chunk text directly.
CREATE TABLE IF NOT EXISTS chunk_questions (
	id          BIGSERIAL PRIMARY KEY,
	chunk_id    TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
	question    TEXT NOT NULL,
	embedding   vector(4096),
	created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_questions_chunk_id ON chunk_questions(chunk_id);
-- ANN index must be created manually after populating the table:
--   CREATE INDEX idx_chunk_questions_embedding ON chunk_questions
--     USING ivfflat ((embedding::halfvec(4096)) halfvec_cosine_ops) WITH (lists = 50);

-- Job queue — powers the background job runner (session 79)
CREATE TABLE IF NOT EXISTS jobs (
	job_id       TEXT PRIMARY KEY,
	job_type     TEXT NOT NULL,
	params_json  TEXT NOT NULL DEFAULT '{}',
	status       TEXT NOT NULL DEFAULT 'pending',
	attempts     INTEGER NOT NULL DEFAULT 0,
	max_attempts INTEGER NOT NULL DEFAULT 3,
	error        TEXT,
	created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	started_at   TIMESTAMPTZ,
	finished_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);

