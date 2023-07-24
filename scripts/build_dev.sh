# ENV Requirements

# Copy to dist
rm -rf dist
mkdir -p dist
cp -r heavynl dist/heavynl
cp requirements.txt ./dist/requirements.txt

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Compress Build Dir
tar -czf dist.tgz dist
