# Mock Checkpointing Example

This example demonstrates the production checkpointing flow without downloading
real models. It uses local fake backends, but it exercises the real
`FlashRL.train()` runtime, managed checkpoint saving, `latest.json`, and
append-resume logging.

## Initial Run

```bash
python3 flashrl/framework/examples/mock-checkpointing/train.py \
  --log-dir /tmp/flashrl-mock-logs \
  --checkpoint-dir /tmp/flashrl-mock-checkpoints \
  --save-every-steps 2
```

That creates:

- interval checkpoints like `step-00000002.pt`
- a managed latest manifest at `/tmp/flashrl-mock-checkpoints/latest.json`
- one FlashRL run directory under `/tmp/flashrl-mock-logs`

## Resume From Latest

```bash
python3 flashrl/framework/examples/mock-checkpointing/train.py \
  --log-dir /tmp/flashrl-mock-logs \
  --checkpoint-dir /tmp/flashrl-mock-checkpoints \
  --save-every-steps 2 \
  --resume-from latest
```

The resumed invocation:

- loads the checkpoint pointed to by `latest.json`
- appends to the original run directory instead of allocating a new one
- continues managed interval checkpointing from the resumed global step count

## Optional Final Checkpoint

```bash
python3 flashrl/framework/examples/mock-checkpointing/train.py \
  --log-dir /tmp/flashrl-mock-logs \
  --checkpoint-dir /tmp/flashrl-mock-checkpoints \
  --save-every-steps 2 \
  --save-on-run-end
```

That also writes `/tmp/flashrl-mock-checkpoints/final.pt` and updates
`latest.json` to point at the final checkpoint.
