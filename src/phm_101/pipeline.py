from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from phm_101.data_pipeline.data_extractor import DataExtractor
from phm_101.data_pipeline.data_loader import DataLoader
from phm_101.data_pipeline.data_transformer import DataTransformer
from phm_101.data_types.enums import ImsChannel, Paradigm
from phm_101.data_types.models import Checkpoint, RunResult
from phm_101.detectors.forecasting import ForecastingDetector
from phm_101.detectors.reconstruction import ReconstructionDetector
from phm_101.ml_pipeline.eval_pipeline import Evaluator
from phm_101.utils import reporting
from phm_101.utils.ims import CHANNELS
from phm_101.utils.utils import load_checkpoint, save_checkpoint, set_seed

if TYPE_CHECKING:
    import numpy as np

    from phm_101.config.configs import Config
    from phm_101.data_types.models import (
        EvalResult,
        TrainResult,
    )


class Pipeline:
    """Take one channel from parquet to scored snapshots and artifacts."""

    def __init__(
        self,
        config: Config,
        raw_signals_dir: Path,
        artifacts_dir: Path,
        checkpoints_dir: Path,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.artifacts_dir = artifacts_dir
        self.checkpoints_dir = checkpoints_dir
        self.raw_signals_dir = raw_signals_dir

        self.all_channels = [
            channel for group in CHANNELS.values() for channel in group
        ]
        self.config = config

        self.data_extractor = DataExtractor(raw_signals_dir=raw_signals_dir)

    def run(
        self,
        channel: ImsChannel | None = None,
        checkpoint_path: Path | None = None,
    ) -> RunResult:
        """
        Run the fault detection pipeline.

        - scenario 1: no checkpoint -> everything from the config
        - scenario 2: checkpoint -> it owns the architecture and the window
                     geometry; the channel stays the caller's
        """
        channel = ImsChannel(channel or self.config.data_config.channel)
        model_config = self.config.model_config
        data_config = replace(self.config.data_config, channel=channel)
        self.logger.info(
            'Running Fault Detection pipeline on {channel}{how}',
            channel=channel.value,
            how='' if checkpoint_path is None else f' from {checkpoint_path}',
        )
        checkpoint: Checkpoint | None = None
        if checkpoint_path is not None:
            self.logger.warning(
                'A checkpoint has been passed, overriding config.yaml'
            )
            checkpoint = load_checkpoint(path=checkpoint_path)
            model_config = checkpoint.model_config
            data_config = replace(
                data_config,
                window_size=checkpoint.data_config.window_size,
                in_channels=checkpoint.data_config.in_channels,
            )
        if self.config.training_config.seed is not None:
            set_seed(self.config.training_config.seed)

        # Data Pipeline
        self.logger.info('Starting Data Pipeline')
        self.data_transformer = DataTransformer(data_config=data_config)
        self.data_loaders = DataLoader(
            window_size=data_config.window_size,
            hop=data_config.hop,
            num_workers=data_config.num_workers,
            pin_memory=True,
            seed=data_config.seed,
        )
        channel_data = self.data_extractor.load(channel=data_config.channel)

        data_splitted = self.data_transformer.train_test_split(
            data=channel_data
        )
        self.data_transformer.fit(data=data_splitted.train)

        normalized_train_data = self.data_transformer.transform(
            data=data_splitted.train
        )
        normalized_val_data = self.data_transformer.transform(
            data=data_splitted.val
        )
        normalized_test_data = self.data_transformer.transform(
            data=data_splitted.test
        )

        train_loader = self.data_loaders.get_dataloader(
            data=normalized_train_data,
            batch_size=data_config.batch_size,
            train=True,
        )
        val_loader = self.data_loaders.get_dataloader(
            data=normalized_val_data,
            batch_size=data_config.batch_size,
            train=False,
        )
        test_loader = self.data_loaders.get_dataloader(
            data=normalized_test_data,
            batch_size=data_config.batch_size,
            train=False,
        )
        self.logger.info('Terminated Data Pipeline')

        if model_config.paradigm is Paradigm.FORECASTING:
            detector = ForecastingDetector(
                model_config=model_config,
                data_config=data_config,
                train_config=self.config.training_config,
            )
        else:
            detector = ReconstructionDetector(
                model_config=model_config,
                data_config=data_config,
                train_config=self.config.training_config,
            )

        train_result: TrainResult | None = None
        if checkpoint is None:
            self.logger.info('Starting training pipeline')
            train_result = detector.fit(
                train_loader=train_loader, val_loader=val_loader
            )
            save_checkpoint(
                state=detector.state(),
                model_config=model_config,
                data_config=data_config,
                channel=channel,
                path=self._checkpoint_path(channel=channel),
            )
            self.logger.info(
                'Terminated training pipeline, saving checkpoint.'
            )
        else:
            detector.load_state(state=checkpoint.state_dict)

        self.logger.info('Starting evaluation pipeline')
        eval_result = Evaluator(
            eval_config=self.config.evaluation_config,
            detector=detector,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ).evaluate()

        output_dir = self._output_dir(channel=channel)
        self._write_artifacts(
            channel=channel,
            eval_result=eval_result,
            train_result=train_result,
            timestamps=data_splitted.test.timestamps,
            output_dir=output_dir,
        )
        self.logger.info(
            'Terminated evaluation pipeline. Saving artifacts in {output}',
            output=output_dir,
        )

        return RunResult(
            channel=channel,
            eval_result=eval_result,
            train_result=train_result,
        )

    def train_all(self, summary_path: Path) -> list[RunResult]:
        """Train and evaluate one model per channel, each on its own data."""
        results: list[RunResult] = []
        for channel in self.all_channels:
            try:
                results.append(self.run(channel=channel, checkpoint_path=None))
            except Exception:
                self.logger.exception(
                    'Skipping {channel}', channel=channel.value
                )
        if results:
            reporting.write_summary(results=results, path=summary_path)
        return results

    def run_all(
        self, checkpoint_path: Path, summary_path: Path
    ) -> list[RunResult]:
        """Run every IMS channel and write a table comparing them."""
        results: list[RunResult] = []
        for channel in self.all_channels:
            try:
                results.append(
                    self.run(channel=channel, checkpoint_path=checkpoint_path)
                )
            except Exception:
                self.logger.exception(
                    'Skipping {channel}', channel=channel.value
                )
        if results:
            reporting.write_summary(results=results, path=summary_path)
        return results

    def _write_artifacts(
        self,
        channel: ImsChannel,
        eval_result: EvalResult,
        train_result: TrainResult | None,
        timestamps: np.ndarray,
        output_dir: Path,
    ) -> None:
        reporting.write_metrics(channel, eval_result, output_dir)
        if train_result is not None:
            reporting.plot_loss_curves(
                train_result, output_dir / 'loss_curves.png'
            )
        reporting.plot_scores(
            channel=channel,
            result=eval_result,
            timestamps=timestamps,
            onset=self.data_extractor.fault_onset.get(channel),
            path=output_dir / 'scores_over_time.png',
        )
        reporting.plot_score_histogram(
            eval_result, output_dir / 'score_histogram.png'
        )

    def _checkpoint_path(self, channel: ImsChannel) -> Path:
        """One file per channel: train_all writes sixteen, not one."""
        stem = Path(self.config.training_config.checkpoint_name).stem
        return self.checkpoints_dir / f'{stem}_{channel.value}.pt'

    def _output_dir(self, channel: ImsChannel) -> Path:
        """Keep a transfer run's artifacts apart from the channel's own run."""
        return (
            self.artifacts_dir
            / self.config.evaluation_config.output_artifacts
            / channel.value
        )
