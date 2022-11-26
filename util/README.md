# Utility Scripts

## `collect_metrics.py`
Utility script to collect tensorboard files and store them in single directory
for a single run (might be separated into different directories when training
is interrupted).

Example usage:
```bash
python collect_metrics.py /path/to/my/eval/dir
```
