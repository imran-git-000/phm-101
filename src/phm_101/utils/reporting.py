import json
from dataclasses import asdict, fields
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from phm_101.data_types.enums import ImsChannel
from phm_101.data_types.models import Metrics

if TYPE_CHECKING:
    from pathlib import Path

    from phm_101.data_types.models import EvalResult, RunResult, TrainResult

mpl.use('Agg')

# the metric names, in the order a table should show them
METRIC_COLUMNS = tuple(field.name for field in fields(Metrics))


def _channel_name(channel: ImsChannel | str) -> str:
    """The plain name, whichever form the caller holds.

    An ImsChannel is not JSON serialisable and renders as 'ImsChannel.T2B1'
    in a title or a CSV cell, so it never reaches a file as-is.
    """
    return channel.value if isinstance(channel, ImsChannel) else channel


def write_metrics(
    channel: ImsChannel | str, result: EvalResult, out_dir: Path
) -> None:
    """Write one channel's metrics as JSON and as a single-row CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _row(channel, result)
    (out_dir / 'metrics.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    _metrics_frame([(channel, result)]).to_csv(
        out_dir / 'metrics.csv', index=False
    )
    logger.info(
        'Wrote metrics for {channel} to {dir}', channel=channel, dir=out_dir
    )


def write_summary(results: list[RunResult], path: Path) -> None:
    """One row per channel, so a sweep can be compared at a glance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _metrics_frame([(r.channel, r.eval_result) for r in results])
    frame.to_csv(path, index=False)
    logger.info(
        'Wrote summary of {n} channels to {path}', n=len(results), path=path
    )


def plot_loss_curves(result: TrainResult, path: Path) -> None:
    """Training and validation reconstruction loss per epoch."""
    figure, axes = plt.subplots(figsize=(9, 4))
    axes.plot(result.train_losses, label='train')
    axes.plot(result.val_losses, label='val')
    axes.set_xlabel('epoch')
    axes.set_ylabel('reconstruction MSE')
    axes.set_yscale('log')
    axes.set_title('trained on healthy windows only')
    axes.legend()
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_scores(
    channel: ImsChannel | str,
    result: EvalResult,
    timestamps: np.ndarray,
    onset: str | None,
    path: Path,
) -> None:
    """Anomaly score over the test period, with threshold and labelled onset."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].scatter(
        timestamps, result.scores, c=result.labels, cmap='coolwarm', s=8
    )
    axes[0].axhline(result.threshold, ls='--', c='black', label='threshold')
    if onset is not None:
        axes[0].axvline(
            np.datetime64(onset), ls=':', c='green', label='labelled onset'
        )
    axes[0].set_yscale('log')
    axes[0].set_ylabel('reconstruction error')
    axes[0].set_title(
        f'{_channel_name(channel)} - anomaly score over the test period'
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        timestamps, (result.scores > result.threshold).astype(int), lw=1
    )
    axes[1].plot(timestamps, result.labels, lw=1, ls='--')
    axes[1].set_yticks([0, 1], ['healthy', 'faulty'])
    axes[1].legend(['predicted', 'labelled'])
    axes[1].grid(alpha=0.3)

    _save(figure, path)


def plot_score_histogram(result: EvalResult, path: Path) -> None:
    """How far apart healthy and faulty snapshots score."""
    healthy = result.scores[result.labels == 0]
    faulty = result.scores[result.labels == 1]
    figure, axes = plt.subplots(figsize=(9, 4))
    bins = np.logspace(
        np.log10(result.scores.min()), np.log10(result.scores.max()), 60
    )
    axes.hist(
        healthy, bins=bins, alpha=0.6, label=f'healthy (n={len(healthy)})'
    )
    if faulty.size:
        axes.hist(
            faulty, bins=bins, alpha=0.6, label=f'faulty (n={len(faulty)})'
        )
    axes.axvline(result.threshold, ls='--', c='black', label='threshold')
    axes.set_xscale('log')
    axes.set_xlabel('reconstruction error')
    axes.set_title('score separation on the test split')
    axes.legend()
    axes.grid(alpha=0.3)
    _save(figure, path)


def _row(channel: ImsChannel | str, result: EvalResult) -> dict[str, object]:
    """One flat record: who, at what threshold, scoring how well."""
    return {
        'channel': _channel_name(channel),
        'threshold': result.threshold,
        **asdict(result.metrics),
    }


def _metrics_frame(
    results: list[tuple[ImsChannel | str, EvalResult]],
) -> pd.DataFrame:
    """Tabulate metrics, one row per channel.

    auroc is nan for a channel that never fails, since it needs both classes
    present; reindex fixes the column order.
    """
    rows = [_row(channel, result) for channel, result in results]
    return pd.DataFrame(rows).reindex(
        columns=['channel', 'threshold', *METRIC_COLUMNS]
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    # a run draws one figure per channel; closing keeps a sweep from
    # accumulating all of them in memory
    plt.close(figure)
