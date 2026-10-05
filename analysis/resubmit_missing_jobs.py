#!/usr/bin/env python3
#Resubmit only missing/incomplete jobs using the shared submission machinery.Keep this file beside submit_all.py.All modes and defaults match submit_all.py. Pass --mode to select the samemode used for the original submission.
from submit_all import main

if __name__ == "__main__":
    main(only_missing=True)
