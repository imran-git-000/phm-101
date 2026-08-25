import argparse

from phm_101.data_loader import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()

    loader = DataLoader()
    for test in args.tests:
        for path in loader.save(loader.convert(test)):
            print(path.name)


if __name__ == '__main__':
    main()
