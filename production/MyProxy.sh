#!/bin/bash
voms-proxy-init --voms cms 
cp $(voms-proxy-info --path) MyProxy
