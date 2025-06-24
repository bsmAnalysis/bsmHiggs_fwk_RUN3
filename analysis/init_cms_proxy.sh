#!/bin/bash

voms-proxy-init -voms cms -rfc --out x509up
export X509_USER_PROXY=$(realpath x509up)
