"""QTRAN-base command-line entry point."""
import sys
from algorithms.common.value_decomposition.train import main

if __name__ == "__main__":
    sys.exit(main("qtran"))
