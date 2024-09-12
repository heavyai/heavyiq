model=BAAI/bge-large-en-v1.5
revision=refs/pr/5
volume=$PWD/embeddings_data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.2 --model-id $model --revision $revision
