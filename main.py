import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from phm_101.config.configs import ConfigLoader
from phm_101.data_types.enums import ImsChannel
from phm_101.pipeline import Pipeline
from phm_101.utils.utils import build_parser


def _require_checkpoint(checkpoint: Path | None, mode: str) -> Path:
    """The two scoring modes have no weights without one."""
    if checkpoint is None:
        raise SystemExit(f'--mode {mode} needs --checkpoint')
    if not checkpoint.is_file():
        raise SystemExit(f'no checkpoint at {checkpoint}')
    return checkpoint


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()

    config_path = Path(os.environ.get('CONFIG_PATH'))
    raw_signals_dir = Path(os.environ.get('RAW_SIGNALS_DIR'))
    artifacts_dir = Path(os.environ.get('ARTIFACTS_DIR'))
    checkpoints_dir = Path(os.environ.get('CHECKPOINTS_DIR'))

    config = ConfigLoader.load_config(path=config_path)
    pipeline = Pipeline(
        config=config,
        raw_signals_dir=raw_signals_dir,
        artifacts_dir=artifacts_dir,
        checkpoints_dir=checkpoints_dir,
    )
    channel = ImsChannel(args.channel) if args.channel else None

    match args.mode:
        case 'train':
            # fit and evaluate one channel, everything from the config
            result = pipeline.run(channel=channel, checkpoint_path=None)
        case 'test':
            # evaluate one channel with weights that already exist
            result = pipeline.run(
                channel=channel,
                checkpoint_path=_require_checkpoint(
                    args.checkpoint, args.mode
                ),
            )
        case 'train-all':
            # one model per channel, each fitted on its own data
            pipeline.train_all()
            return
        case 'test-all':
            # one checkpoint scored against every channel
            pipeline.run_all(
                checkpoint_path=_require_checkpoint(args.checkpoint, args.mode)
            )
            return

    logger.info(
        'Finished {channel}: {metrics}',
        channel=result.channel.value,
        metrics=result.eval_result.metrics,
    )


if __name__ == '__main__':
    main()
