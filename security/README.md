# Run Trivy security scan

In repo root, run command after replacing `<PATH_TO_REPO>`:

```
trivy fs ./ --format template --template "<PATH_TO_REPO>/experiments/html.tpl" -o security/report.html
```
