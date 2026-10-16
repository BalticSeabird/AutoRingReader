# AutoRingReader

AutoRingReader runs a three-stage pipeline that detects bird rings, extracts ring tags, and recognizes the tag text.

## Run the pipeline

1. Create a Python environment and install the dependencies:

   ```bash
   ./setupvenv.sh
   ```

2. Edit a batch configuration in [`config/`](config/) to point to the input
   videos, model files, output directory, and available GPUs.

3. Run the orchestrator from the repository root:

   ```bash
   venv/bin/python -m pipeline.pipeline_orchestrator config/batch_pipeline_261016.yaml
   ```

The pipeline writes processing logs and its SQLite progress database to
`log/`. The database contains per-step tables and a `final_summary` table with
one combined result row per video. Results are written to the `output_root_path`
configured in the batch YAML file; no separate diagnostics CSV or JSON files
are generated.

## Rebuild a database from existing outputs

If the inference outputs already exist, the database can be rebuilt without
rerunning any pipeline step:

```bash
venv/bin/python utils/rebuild_database.py \
  --step1 temp/step1_ring_crops \
  --step2 temp/step2_tag_crops \
  --step3 temp/step3_ocr_results \
  --database out/rebuilt.db
```

Each `--stepN` argument may also point to the output root containing the
corresponding `stepN_*` directory. Step 1 `_metadata.json` files and Step 3
YAML summaries are used when present. Values that cannot be recovered from
saved inference output, such as processing time and detection confidences for
Steps 1-2, are stored as `NULL` rather than estimated.

The `archive/` directory contains historical documentation, tests, debugging
utilities, and generated artifacts retained for reference but not required to
run the pipeline.
