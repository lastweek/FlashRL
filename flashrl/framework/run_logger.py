"""Simple run logging for FlashRL training."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from flashrl.framework.config import LoggingConfig


class RunLogger:
    """Simple logger for FlashRL training runs.

    Provides basic stdout logging and optional JSON file output.
    Designed for simplicity over complex observability features.
    """

    def __init__(self, config: LoggingConfig, model_name: str) -> None:
        """Initialize the run logger.

        Args:
            config: Logging configuration.
            model_name: Name of the model being trained.
        """
        self.config = config
        self.model_name = model_name
        self.run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics: dict[str, Any] = {}
        self._log_file = None
        self._metrics_file = None

        if config.file:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / f"{self.run_id}.log"
            self._metrics_file = log_dir / f"{self.run_id}_metrics.json"

    def start_run(
        self,
        dataset_size: int,
        batch_size: int,
        max_epochs: int,
        total_batches: int,
        device: str,
        dtype: str,
        cpu_threads: int,
        runtime_shape: str,
        reference_enabled: bool,
        reference_device: str,
    ) -> None:
        """Log the start of a training run."""
        print(f"\n{'=' * 60}")
        print(f"FlashRL Training Run")
        print(f"{'=' * 60}")
        print(f"Model: {self.model_name}")
        print(f"Run ID: {self.run_id}")
        print(f"Dataset size: {dataset_size}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {max_epochs}")
        print(f"Device: {device}")
        print(f"CPU threads: {cpu_threads}")
        print(f"Reference model: {'enabled' if reference_enabled else 'disabled'}")
        print(f"{'=' * 60}\n")

        self._log_to_file(f"Training run started: {self.run_id}")

    def log_epoch_start(self, epoch: int, total_epochs: int, num_batches: int) -> None:
        """Log the start of an epoch."""
        print(f"Epoch {epoch}/{total_epochs} ({num_batches} batches)")

    def log_step(
        self,
        step: int,
        epoch: int,
        metrics: dict[str, float],
        phase_timings: dict[str, float] | None = None,
    ) -> None:
        """Log a training step."""
        if step % self.config.log_every_steps == 0:
            loss = metrics.get("loss", 0)
            policy_loss = metrics.get("policy_loss", 0)
            kl_div = metrics.get("kl_divergence", 0)
            print(f"  Step {step} (epoch {epoch}): loss={loss:.4f}, policy={policy_loss:.4f}, kl={kl_div:.4f}")

            # Save metrics for JSON output
            self.metrics[f"step_{step}"] = {
                "epoch": epoch,
                "metrics": metrics,
                "timings": phase_timings or {},
            }

    def log_phase_timing(self, phase: str, duration: float) -> None:
        """Log timing for a training phase (optional, for detailed profiling)."""
        pass  # Simplified out - no detailed timing tracking

    def log_model_load(
        self,
        component: str,
        status: str,
        metadata: dict[str, Any],
    ) -> None:
        """Log model loading events."""
        if self.config.console:
            device = metadata.get("device", "unknown")
            if status == "completed":
                print(f"✓ {component} loaded on {device}")

    def log_exception(self, exc: Exception, context: dict[str, Any]) -> None:
        """Log an exception during training."""
        print(f"\n✗ Error: {exc}")
        if context:
            print(f"  Context: {context}")

    def log_checkpoint(
        self,
        action: str,
        path: str,
        epoch: int,
        step: int,
        duration_seconds: float,
    ) -> None:
        """Log checkpoint save/load."""
        action_verb = "Saved" if action == "save" else "Loaded"
        print(f"✓ {action_verb} checkpoint from epoch {epoch}, step {step} ({path})")

    def close(self) -> None:
        """Close the run and save metrics."""
        if self._metrics_file and self.metrics:
            with open(self._metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
            print(f"\n✓ Metrics saved to {self._metrics_file}")

        if self._log_file:
            self._log_file.close()
