# ENV Requirements
## python3.10 (Installed)

# Create New Virtual Environment
python3.10 -m venv venv
. venv/bin/activate

# Install Requirements
pip install -r requirements.txt
pip freeze -l > requirements-lockfile.txt
pip install -r requirements-dev.txt

# Create Obfuscated Build
pyarmor reg pyarmor-regfile-5130.zip
pyarmor gen ./heavyiq
cp requirements-lockfile.txt ./dist/requirements.txt
cp requirements-linux.txt ./dist/requirements-linux.txt
rm requirements-lockfile.txt
cp -r heavyiq/langchain/tokenizer_models/ ./dist/heavyiq/langchain/tokenizer_models/

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Compress Build Dir
tar -czf dist.tgz dist
