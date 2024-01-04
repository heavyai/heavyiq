# Run Unit Tests:


```bash
pytest tests --disable-warnings -rs --config-path="./config.test.toml"
```

## Run Tests Inside Container

1. Build test container.

```bash
docker-compose build heavyiq-test
```

2. Run tests.

```bash
docker-compose up --abort-on-container-exit heavyiq-test
```
