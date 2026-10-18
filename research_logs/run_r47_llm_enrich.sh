#!/bin/bash
# R47 (task #6): full DeepSeek LLM item semantic-tag enrichment for lastfm-2k and
# ml-1m, then rebuild the frozen semantic item features with the LLM tags merged.
# LLM is inference-only (no fine-tuning); resumable JSON caches in dataset/_raw/.
# After enrichment, util/build_semantic_features.py regenerates dataset/<ds>/item_sem.npy
# (TF-IDF over native tags + LLM tags -> frozen seeded Gaussian projection -> L2-norm).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch

echo "[r47-llm] BEGIN $(date)"
$PY util/llm_enrich_items.py --dataset lastfm-2k --batch 25 --cache dataset/_raw/llm_tags_lastfm-2k.json
$PY util/llm_enrich_items.py --dataset ml-1m --batch 25 --cache dataset/_raw/llm_tags_ml-1m.json

echo "[r47-llm] rebuild semantic features (native + LLM tags)"
$PY util/build_semantic_features.py --datasets ml-1m,lastfm-2k --embed_size 256 \
    --llm_json dataset/_raw/llm_tags_ml-1m.json  # note: build merges one json per run
# merge both: rebuild each dataset with its own llm json
$PY util/build_semantic_features.py --datasets ml-1m --embed_size 256 --llm_json dataset/_raw/llm_tags_ml-1m.json
$PY util/build_semantic_features.py --datasets lastfm-2k --embed_size 256 --llm_json dataset/_raw/llm_tags_lastfm-2k.json
echo "[r47-llm] ALLDONE $(date)"
