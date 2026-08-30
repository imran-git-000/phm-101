Repository to learn solve phm tasks: starting from EDA to train/evaluate ML models on phm datasets. 

Requirements:
- uv
- Just

To setup the repository run:
    just setup

How to use the codebase:

- uv run python main.py                                              # 1. train + test, config channel
- uv run python main.py --mode test --channel t3b3 --checkpoint P    # 2. test one channel
- uv run python main.py --mode test-all --checkpoint P               # 3. test all
- uv run python main.py --mode train-all                             # 4. train all
