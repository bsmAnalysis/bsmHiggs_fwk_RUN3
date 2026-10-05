#!/bin/bash

#determine CMSSW config
SCRIPT=$(readlink -f $1)
SCRIPTPATH=`dirname $SCRIPT`
ARCH=${SCRIPTPATH##/*/}
CMSSW=${SCRIPTPATH}/../../src
source /cvmfs/cms.cern.ch/cmsset_default.sh

#configure environment
cd $CMSSW
export SCRAM_ARCH=$ARCH
eval `scram r -sh`
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk/

#run with the arguments passed
echo $1 + $2
$1 $2
