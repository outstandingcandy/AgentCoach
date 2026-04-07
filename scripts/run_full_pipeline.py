#!/usr/bin/env python3
"""Run full GoalInsight pipeline. See goalinsight.cli for implementation."""

import sys

from goalinsight.cli import main

if __name__ == "__main__":
    sys.exit(main())
