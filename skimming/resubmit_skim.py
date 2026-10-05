#!/usr/bin/env python3
"""Submit only missing skim outputs using the shared submission workflow."""

from submit_all import main


if __name__ == "__main__":
    main(only_missing=True)
