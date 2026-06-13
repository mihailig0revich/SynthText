# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images.

The implementation lives in the synthtext package; this file is kept as the
stable command users already run: `python gen.py`.
"""

from synthtext.cli import main


if __name__ == "__main__":
    main()
