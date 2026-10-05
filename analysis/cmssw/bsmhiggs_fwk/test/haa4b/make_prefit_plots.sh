#!/usr/bin/env bash

#----------------------------------------------------------- $1 ZH, WH ---------------------------------------------------
#---------------------------- all variables except bdt -------------------------------
shapes=("bdt")
#shapes=("H_mass" "H_pt" "HT" "dr_bb" "met" "n_jets")

blind="--SRblind" #" "
flds=("PrefitPlots_$1ZH_VR_T_new_2" "PrefitPlots_$1ZH_VR_T_new_2")
REGIONS=("resolved") 


PLOTTER_ZH=total_plotter_ZH_2024_2024_03_04.root
SAMPLES=samples$1.json

if [[ $1 =~ "2024" ]]; then
    INTLUMI=109820.0
else
    echo "Please specify the year to calculate int. luminosity!"
fi

## ZH
if [[ -z "$2" || "$2" == "ZH" ]]; then
    fld="${flds[1]}"
    mkdir -p "${fld}" && cd "${fld}"
    count=1
    for shape in "${shapes[@]}"; do
      for reg in "${REGIONS[@]}"; do
        echo "${count}. **********************************${shape} (${reg})***********************************"
        dir="TEST_${shape}_${reg}/"
        rm -rf "${dir}"; mkdir -p "${dir}"; cd "${dir}"

        HNAME="${shape}_shapes_${reg}"
	computeLimit_vr   --lumi "$INTLUMI" --verbose  --runZh --signalScale 1 --m 20 --histo "$HNAME" --shape --subFake --index 1 --shapeMin -9999 --shapeMax 9999 --bins 3b --in $CMSSW_BASE/src/UserCode/bsmhiggs_fwk/test/haa4b/$PLOTTER_ZH --json "$CMSSW_BASE/src/UserCode/bsmhiggs_fwk/test/haa4b/$SAMPLES" --key haa_mcbased --systpostfix _13p6TeV --rebin 2 --dropBckgBelow 0.00015 --modeDD  
	cd ..
	count=$((count+1))
      done
    done
    cd ..
    exit 0
fi

