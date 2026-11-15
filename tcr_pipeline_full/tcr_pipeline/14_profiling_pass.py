#!/usr/bin/env python3
"""Per-action profiling at the frozen calibration rates, against an explicitly supplied vLLM URL."""
import sys

from profiling_pass import main

if __name__ == '__main__':
    sys.exit(main())
