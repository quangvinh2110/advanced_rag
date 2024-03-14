#!/bin/bash  
set -ex

python src/crawl_data/crawl_wiki_stats.py \
    --links_path /home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs/xad.txt \
    --output_path /home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs/wiki_stats.jsonl

python src/crawl_data/crawl_wiki_stats.py \
    --links_path /home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs/xae.txt \
    --output_path /home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs/wiki_stats.jsonl

python src/crawl_data/crawl_wiki_stats.py \
    --links_path /home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs/xaf.txt \
    --output_path /home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs/wiki_stats.jsonl