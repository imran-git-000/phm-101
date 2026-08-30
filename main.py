import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from loguru import logger

from phm_101.config.configs import ConfigLoader
from phm_101.pipeline import Pipeline
from phm_101.utils.utils import build_parser

if TYPE_CHECKING:
    from phm_101.data_types.models import RunResult


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

    channel = args.channel or config.data_config.channel
    if channel == 'all':
        if args.model_path is not None:
            raise SystemExit(
                '--model-path takes a single checkpoint, so it cannot be '
                'combined with --channel all'
            )
        results = pipeline.run_all()
        logger.info('Finished {n} channels', n=len(results))
        return

    result: RunResult = pipeline.run(channel=channel, model_path=args.model_path)
    logger.info(
        'Finished {channel}: {metrics}',
        channel=channel,
        metrics=result.eval_result.metrics,
    )


if __name__ == '__main__':
    main()
