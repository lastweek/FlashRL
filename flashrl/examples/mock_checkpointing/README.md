# FlashRL Mock-Checkpointing Example

This example is fully offline. It does not launch model backends or touch the
platform path. It exists to exercise the managed checkpoint file layout and
resume flow in a small, deterministic script.

Run it from the repository root:

```bash
python3 -m flashrl.examples.mock_checkpointing.train \
  --log-dir /tmp/flashrl-mock-logs \
  --checkpoint-dir /tmp/flashrl-mock-checkpoints \
  --save-every-steps 2 \
  --prompt-count 4 \
  --save-on-run-end
```

Resume from the latest managed checkpoint:

```bash
python3 -m flashrl.examples.mock_checkpointing.train \
  --log-dir /tmp/flashrl-mock-logs \
  --checkpoint-dir /tmp/flashrl-mock-checkpoints \
  --save-every-steps 2 \
  --prompt-count 4 \
  --resume-from latest
```
