from loguru import logger

from phm_101.data_loader import DataLoader


def main() -> None:
    logger.info('Starting IMS raw signals to parquet conversion')
    DataLoader().save()


if __name__ == '__main__':
    main()
