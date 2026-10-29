# Make targets wrap the module entry points so the pipeline is discoverable.
# Everything is also runnable directly as `python -m issue_rag.<stage>`.

PRESET ?= vite
# Second grader used by `make regrade` / `make sensitivity`.
VARIANT ?= reasoner
VARIANT_MODEL ?= deepseek-reasoner
JUDGE_TOKENS ?= 3000

.PHONY: help install test harvest corpus evalset review models eval eval-retrieval \
        regrade sensitivity serve clean

help:
	@echo "make install       install the package with model extras"
	@echo "make test          run the hermetic unit tests (no GPU, no network)"
	@echo "make harvest       harvest issues then per-issue comments"
	@echo "make corpus        build the retrieval corpus"
	@echo "make evalset       mine question/answer candidates"
	@echo "make review        apply the review sheet and finalize the eval set"
	@echo "make models        download weights via ModelScope"
	@echo "make eval          run the full matrix (generator == judge)"
	@echo "make eval-retrieval  retrieval metrics only (no generator needed)"
	@echo "make regrade       re-grade the stored answers with a second model"
	@echo "make sensitivity   compare the two graders and report the interval"
	@echo "make serve         serve the demo UI"
	@echo ""
	@echo "override with: make <target> PRESET=ollama VARIANT_MODEL=deepseek-reasoner"

install:
	pip install -e ".[models,dev]"

test:
	pytest

harvest:
	python -m issue_rag.fetch issues   --preset $(PRESET) --months 12 --allow-wait
	python -m issue_rag.fetch comments --preset $(PRESET) --mode per-issue --max-comment-issues 60

corpus:
	python -m issue_rag.corpus --preset $(PRESET)

evalset:
	python -m issue_rag.evalset --preset $(PRESET)

review:
	python -m issue_rag.evalset --preset $(PRESET) --finalize

models:
	python -m issue_rag.models

eval:
	python -m issue_rag.evaluate --preset $(PRESET) \
		--systems closed_book,bm25,dense,hybrid,hybrid_rerank

eval-retrieval:
	python -m issue_rag.evaluate --preset $(PRESET) \
		--systems bm25,dense,hybrid,hybrid_rerank --retrieval-only

# Re-grades the answers already on disk, so only the judge changes.
regrade:
	python -m issue_rag.evaluate --preset $(PRESET) --regrade \
		--judge-model $(VARIANT_MODEL) --judge-max-tokens $(JUDGE_TOKENS) \
		--run-tag $(VARIANT) \
		--systems closed_book,bm25,dense,hybrid,hybrid_rerank

sensitivity:
	python -m issue_rag.sensitivity --preset $(PRESET) --variant $(VARIANT)

serve:
	python -m issue_rag.serve --preset $(PRESET)

clean:
	rm -rf data/cache data/raw
	@echo "kept data/processed (the eval set is a deliverable, not a build artifact)"
	@echo "kept data/results   (the measured evidence behind the README)"
