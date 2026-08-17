# Production CPU and synchronization settings

The embedding model is loaded lazily once per process. PyTorch defaults to two
intra-op threads and one inter-op thread, and catalogue embeddings use batches
of 16. Configure these without changing code:

```env
AI_TORCH_NUM_THREADS=2
AI_TORCH_INTEROP_THREADS=1
AI_EMBEDDING_BATCH_SIZE=16
AI_INDEX_BUILD_BATCH_SIZE=500
AI_BUILD_LOCAL_INDEX_ON_STARTUP=false
ENABLE_BACKUP_SCHEDULER=false
BACKUP_SYNC_INTERVAL_MINUTES=15
```

`AI_EMBEDDING_BATCH_SIZE` controls transformer inference. The separate
`AI_INDEX_BUILD_BATCH_SIZE` bounds staging text/vector memory while avoiding
thousands of tiny FAISS reallocations. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` default to the PyTorch thread limit when they are not
already defined. Explicit process-level values take precedence. FAISS is also
limited to `AI_TORCH_NUM_THREADS`.

The backup scheduler is disabled by default. Enable it in exactly one API
process after an explicit successful `POST /admin/sync-catalogue`. A backup sync
without initialized state is skipped; it never requests history from year 2000.
The current `Procfile` starts one Uvicorn worker, so enabling the scheduler in
that single process is safe. If deployment is changed to multiple workers, only
one worker may receive `ENABLE_BACKUP_SCHEDULER=true`; this service keeps its
FAISS index in process memory, so a multi-worker deployment also creates one
independent index/model per worker and is not recommended.

Local JSON indexing on startup is disabled by default. Set
`AI_BUILD_LOCAL_INDEX_ON_STARTUP=true` only when the bundled local catalogue is
the intended source. Otherwise, start the API and run one controlled explicit
full sync after checking available memory. Full rebuild correctness remains
atomic, so its unavoidable peak contains both the old and staging indexes; its
embedding and pagination temporaries are bounded.

The repository production command is exactly:

```bash
uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1
```

It explicitly has one worker and no development reloader. Do not increase the
worker count or add `--reload`: every process would own another transformer model, FAISS index, and
catalogue mapping. No Docker, Compose, systemd, Supervisor, PM2, or restart
policy is defined in this repository; inspect the server or platform service
definition before enabling the process after an OOM event.

On an 8 GB shared host, use one AI process and initially cap it around 2.5 GB
with a 2 GB soft/high watermark, plus a two-core CPU quota. These values leave
headroom over the measured approximately 1.2 GB model/10k-product peak but must
be revisited for larger catalogues. Prefer `Restart=on-failure` with start-rate
limiting over an unconditional tight restart loop. Do not run package installs,
Node builds, mock stacks, and a full AI sync concurrently on that host.

Calls from this service to ChedMed use `X-API-Key` with `CHEDMED_API_KEY`.
Inbound webhooks use the separate `AI_WEBHOOK_SECRET` and require
`X-ChedMed-Event-Id`, `X-ChedMed-Timestamp`, and `X-ChedMed-Signature`. The
signature input is `timestamp + "." + raw_body`, with a five-minute tolerance.

For a local performance check (using the real cached model when available):

```bash
python scripts/benchmark_indexing.py --products 100
```

Use `--synthetic-model` when the model is not cached. Its timings are only a
development baseline, but its encode-call counts remain deterministic.

For RSS, native peak, and Python-allocation checkpoints:

```bash
python scripts/profile_memory.py --products 10000 --repetitions 5
python scripts/profile_memory.py --products 1000 --real-model
```

The default synthetic model isolates catalogue/FAISS behavior. `--real-model`
requires the model to be cached or downloadable. The profiler is CLI-only and
does not print environment variables or product contents.
