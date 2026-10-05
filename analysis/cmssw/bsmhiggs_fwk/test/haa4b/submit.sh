#!/usr/bin/env bash

#--------------------------------------------------
# Global Code 
#--------------------------------------------------

if [[ $# -eq 0 ]]; then 
    printf "NAME\n\tsubmit.sh - Main driver to submit jobs\n"
    printf "\nSYNOPSIS\n"
    printf "\n\t%-5s\n" "./submit.sh [OPTION]" 
    printf "\nOPTIONS\n" 
## Run Analysis over samples
    printf "\n\t%-5s  %-40s\n"  "0"  "completely clean up the directory" 

## Merge Results
    printf "\n\t%-5s  %-40s\n"  "2"  "compute integrated luminosity from processed samples" 
    printf "\n\t%-5s  %-40s\n"  "3.0"  "make plots and combine root files" 

## Make plots in mcbased(_blind), datadriven(_blind) cases
    printf "\n\t%-5s  %-40s\n"  "3.1"  "make plots for mcbased analysis"  
    printf "\n\t%-5s  %-40s\n"  "3.01"  "make root file for input to the Limits"  
    printf "\n\t%-5s  %-40s\n"  "3.02"  "make plots for QCD analysis"
    printf "\n\t%-5s  %-40s\n"  "3.2"  "make plots with data-driven bkgs"
fi

step=$1   #variable that store the analysis step to run

#Additional arguments to take into account
arguments=''; for var in "${@:2}"; do arguments=$arguments" "$var; done
#arguments='crab3'; for var in "${@:2}"; do arguments=$arguments" "$var; done
if [[ $# -ge 4 ]]; then echo "Additional arguments will be considered: "$arguments ;fi 

#--------------------------------------------------
# Global Variables
#--------------------------------------------------

YEAR=2024
CHANNEL=ZH

do_syst=True # Always run with Systematics, unless its QCD mode


if [[ $CHANNEL == "ZH" ]]; then doZH=True ; 
else doZH=False ; fi

if [[ $YEAR == "2016" ]]; then SUFFIX=_2024_01_15 
else SUFFIX=_2024_03_04 ; fi

MAINDIR=$CMSSW_BASE/src/UserCode/bsmhiggs_fwk/test/haa4b

# Json and python template for all years
JSON=$MAINDIR/samples$YEAR.json
NTPL_JSON=$MAINDIR/samples$YEAR.json
FULLANALYSISCFG=$MAINDIR/../fullAnalysis_cfg_$YEAR.py.templ
RUNNTPLANALYSISCFG=$MAINDIR/../runNtplAnalysis_cfg_$YEAR.py.templ
    
#SUFFIX=$(date +"_%Y_%m_%d") 
GOLDENJSON=$CMSSW_BASE/src/UserCode/bsmhiggs_fwk/data/json/

RESULTSDIR=$MAINDIR/results_$YEAR$SUFFIX 

if [[ $arguments == *"crab3"* ]]; then STORAGEDIR='';
else STORAGEDIR=/eos/user/a/ataxeidi/results$SUFFIX ; fi

PLOTSDIR=$MAINDIR/plots_${CHANNEL}_${YEAR}${SUFFIX}
PLOTTER=$MAINDIR/plotter_${CHANNEL}_${YEAR}${SUFFIX}
 
####################### Settings for Ntuple Analysis ##################
if [[ $YEAR == "2016" ]]; then
    NTPL_INPUT=/eos/user/a/ataxeidi/results_2016$SUFFIX
elif [[ $YEAR == "2017" ]]; then
    NTPL_INPUT=/eos/user/a/ataxeidi/results_2017$SUFFIX 
elif [[ $YEAR == "2018" ]]; then
    NTPL_INPUT=/eos/user/a/ataxeidi/results_2018$SUFFIX   
elif [[ $YEAR == "2018" ]]; then
    NTPL_INPUT=/eos/user/a/ataxeidi/results_2024$SUFFIX
fi

ZPtSF_OUT=$MAINDIR/VPtSF_${CHANNEL}_$YEAR$SUFFIX
BTAG_NTPL_OUTDIR=$MAINDIR/btag_SFs/$YEAR/btag_Ntpl$SUFFIX
NTPL_OUTDIR=$MAINDIR/results_Ntpl_Zh_2024_2024_03_04
#NTPL_OUTDIR=$EOSDIR/results_Ntpl_${CHANNEL}_$YEAR$SUFFIX

RUNLOG=$NTPL_OUTDIR/LOGFILES/runSelection.log
queue='workday'   

TopSF_INPUT=$MAINDIR/PrefitPlots_${YEAR}${CHANNEL}_noSoftb/TEST_ht
TopSF_OUT=$MAINDIR/TopPtSF_$YEAR$CHANNEL
if [[ $CHANNEL == "ZH" ]] ; then
    vh_tag="zh" # wh or zh channel when computing Top Pt weights
else vh_tag="zh" ; fi

#IF CRAB3 is provided in argument, use crab submission instead of condor/lsf 
if [[ $arguments == *"crab3"* ]]; then queue='crab3' ;fi  

################################################# STEPS between 0 and 1
if [[ $step == 0 ]]; then   
        #analysis cleanup
    echo "Really delete directory "$RESULTSDIR" ?" 
    echo "ALL DATA WILL BE LOST! [N/y]?"
    read answer
    if [[ $answer == "y" ]];
    then
	echo "CLEANING UP..."
	rm -rdf $RESULTSDIR $PLOTSDIR LSFJOB_* core.* *.sh.e* *.sh.o*
    fi
fi #end of step0
if [[ $step == 0.1 ]]; then
    echo "Really delete directory "$NTPL_OUTDIR" ?"
    echo "ALL DATA WILL BE LOST! [N/y]?"
    read answer
    if [[ $answer == "y" ]];
    then
	echo "CLEANING UP..."
	rm -rdf $NTPL_OUTDIR LSFJOB_* core.* *.sh.e* *.sh.o*
    fi
fi

###  ############################################## STEPS between 1 and 2

###  ############################################## STEPS between 2 and 3
if [[ $step > 1.999 && $step < 3 ]]; then
    if [[ $step == 2 ]]; then    #extract integrated luminosity of the processed lumi blocks

	echo "WARNING: There must be no directories other than 'Data' in results_ directory."

        # Automatically run crab report for all available CRAB jobs                                                                                                                                         
        echo "Running crab report to extract processed lumis..."

        # Base path where CRAB reports are stored                                                                                                                                                           
        CRAB_BASE_DIR="$RESULTSDIR/FARM/inputs"

        # Find all CRAB job directories dynamically                                                                                                                                                         
        CRAB_DIRS=$(find "$CRAB_BASE_DIR" -maxdepth 1 -type d -name "crab_*")

        for CRAB_DIR in $CRAB_DIRS; do
            echo "Generating CRAB report for: $CRAB_DIR"
            crab report -d "$CRAB_DIR"

            # Copy the JSON files from the CRAB results directory to $RESULTSDIR                                                                                                                            
            CRAB_RESULTS_DIR="${CRAB_DIR}/results"

            if [ -d "$CRAB_RESULTS_DIR" ]; then
                JSON_NAME=$(basename "$CRAB_DIR")  # Use the CRAB directory name as identifier                                                                                                              
                echo "Copying processed luminosity JSON from $CRAB_RESULTS_DIR to $RESULTSDIR..."
                cp "$CRAB_RESULTS_DIR/processedLumis.json" "$RESULTSDIR/Data_processed_${JSON_NAME}.json" 2>/dev/null
            else
                echo "WARNING: Results directory not found in $CRAB_DIR"
            fi
        done

        jq -s 'reduce .[] as $item ({}; . * $item)' $RESULTSDIR/Data_*.json > $RESULTSDIR/json_all.json
        echo "COMPUTE INTEGRATED LUMINOSITY"

        # Full RUN 2                                                                                                                                                                                        
        export PATH=$HOME/.local/bin:/cvmfs/cms-bril.cern.ch/brilconda310/bin:$PATH
        pip install --user --upgrade brilws

        brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /fb -i $RESULTSDIR/json_all.json -o $RESULTSDIR/LUMI.csv

        tail -n 3 $RESULTSDIR/LUMI.csv
     fi
  fi
###  ############################################## STEPS between 3 and 4
if [[ $step > 2.999 && $step < 4 ]]; then
    if [ -f $NTPL_OUTDIR/LUMI.txt ]; then
      INTLUMI=`tail -n 3 $RESULTSDIR/LUMI.txt | cut -d ',' -f 6`
    else
	if [[ $JSON =~ "2016" ]]; then  
	    INTLUMI=36330.00 #35866.932
            echo "Please run step==2 above to calculate int. luminosity for 2016 data!" 
	else
            if [[ $JSON =~ "2017" ]]; then
		INTLUMI=41529.152
	        echo "Please run step==2 above to calculate int. luminosity for 2017 data!"
            else
		if [[ $JSON =~ "2018" ]]; then
		    INTLUMI=59740.565
		    echo "Please run step==2 above to calculate int. luminosity for 2018 data!"
		else
	            echo "Please run step==2 above to calculate int. luminosity!"
		fi
            fi                                                                                                                   
	fi
	echo "WARNING: $RESULTSDIR/LUMI.txt file is missing so use fixed integrated luminosity value, this might be different than the dataset you ran on"
    fi
    
    if [[ $step == 3 || $step == 3.0 ]]; then  # make plots and combined root files
	echo "MAKE SUMMARY ROOT FILE, BASED ON AN INTEGRATED LUMINOSITY OF $INTLUMI"
	echo "Input DIR = "$NTPL_OUTDIR
        runPlotter --iEcm 13 --iLumi $INTLUMI --inDir $NTPL_OUTDIR/ --outFile ${PLOTTER}.root  --json $JSON --noPlot --fileOption RECREATE --key haa_mcbased $arguments        
   fi        

    if [[ $step == 3 || $step == 3.01 ]]; then  # make plots and combined root files for limits only
	echo "MAKE SUMMARY ROOT FILE FOR LIMITS, BASED ON AN INTEGRATED LUMINOSITY OF $INTLUMI" 
	echo "Input DIR = "$NTPL_OUTDIR  
	runPlotter --iEcm 13.6 --iLumi 109820  --inDir ${NTPL_OUTDIR}/ --outFile ${PLOTTER}_forLimits.root  --json $JSON --noPlot --fileOption RECREATE --key haa_mcbased --only '(all_optim_systs|all_optim_cut|(ee|mumu|emu)_(A)_(SR|CR)_(3b)_(bdt_shapes|bdt)_(boosted|resolved))' $arguments
	#runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outFile ${PLOTTER}_forLimits.root  --json $JSON --noPlot --fileOption RECREATE --key haa_mcbased --only '(all_optim_systs|all_optim_cut|(veto|)_(A|B|C|D)_(SR)_(3b)_(bdt_shapes|H_mass_shapes|H_pt_shapes|HT_shapes|dr_bb_shapes|met_shapes|n_jets_shapes|bdt)_(boosted|resolved)|(lep1)_A_CR_3b_(bdt_shapes|H_mass_shapes|H_pt_shapes|HT_shapes|dr_bb_shapes|met_shapes|bdt)_(boosted|resolved))' $arguments
    #|(lep1)_A_CR_3b_(bdt_shapes|H_mass_shapes|H_pt_shapes|HT_shapes|dr_bb_shapes|met_shapes|bdt)_(boosted|resolved))' $arguments
	#runPlotter --iEcm 13.6 --iLumi 109000 --inDir ${NTPL_OUTDIR}/ --outFile ${PLOTTER}_forLimits.root  --json $JSON --noPlot --fileOption RECREATE --key haa_mcbased --only '(all_optim_systs|all_optim_cut|(veto)_(Astar|Bstar|C|D)_(SR)_(3b)_(bdt_shapes|H_mass_shapes|H_pt_shapes|HT_shapes|dr_bb_shapes|met_shapes|bdt)_(boosted|resolved))' $arguments  
    fi

    if [[ $step == 3.02 ]]; then # make plots for data-driven QCD bkg
	echo "MAKE SUMMARY ROOT FILE, for data-driven QCD estimate"
	runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outFile ${PLOTTER}_qcd.root  --json $JSON --noPlot --fileOption RECREATE --key haa_mcbased --only '(all_optim_systs|all_optim_cut|(veto)_(Astar|Bstar|C|D)_(SR)_(3b)_(bdt_shapes|H_mass_shapes|H_pt_shapes|HT_shapes|dr_bb_shapes|met_shapes|n_jets_shapes|bdt)_(boosted|resolved))' $arguments
	#runPlotter --iEcm 13.6 --iLumi 109000  --inDir ${NTPL_OUTDIR}/ --outDir $PLOTSDIR/mcbased_qcd/ --outFile ${PLOTTER}_qcd.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ  --only "(veto|lep1)_(A|B|C|D)_(SR|CR)_(3b)_bdt" $arguments
    fi

#(ht|pfmet|ptw|mtw|higgsPt|higgsMass|dRave|dmmin|dphijmet|dphiWh|
    if [[ $step == 3 || $step == 3.1 ]]; then  # make plots and combine root files for mcbased study    
	runPlotter --iEcm 13.6 --iLumi 109820  --inDir ${NTPL_OUTDIR}/ --outFile ${PLOTTER}.root  --json $JSON --noPlot --fileOption RECREATE  --key haa_mcbased --only $arguments
	
	#runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outDir $PLOTSDIR/mcbased_0L_TMML/  --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ --noLog  --signalScale 1 $arguments
	
	#runPlotter --iEcm 13.6 --iLumi 109000 --inDir ${NTPL_OUTDIR}/ --outDir  $PLOTSDIR/mcbased_2lep/  --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ --noLog --removeRatioPlot  --only '(ee_SR|mumu_SR|emu_TTCR)_(event_btagSF_multiWP_resolved|ge3j_n(B|C|Light)Truth_(nobtagSF|rawMultiWPSF|calibratedMultiWPSF)_resolved)'$arguments
 
	#runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outDir $PLOTSDIR/mcbased_0L_log_TMML/ --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased  --fileOption READ
	#runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outDir  $PLOTSDIR/mcbased_0lep_res/  --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ   --blind 0.86  --only '(veto_A_SR_3b_bdt_(resolved))'$arguments
	runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outDir  $PLOTSDIR/mcbased_0lep_boo/  --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ  --rebin 2  --only '((veto)_(A|Astar|B|Bstar|C|D)_(SR)_(3b)_(bdt)_(boosted|resolved))'$arguments
	runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outDir  $PLOTSDIR/mcbased_0lep_boo/  --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ   --blind 0.54 --rebin 2  --only '((veto)_(A)_(SR)_(3b)_(bdt)_(boosted))'$arguments
	 runPlotter --iEcm 13.6 --iLumi 109820 --inDir ${NTPL_OUTDIR}/ --outDir  $PLOTSDIR/mcbased_0lep_res/  --outFile ${PLOTTER}.root  --json $JSON --plotExt .pdf --key haa_mcbased --fileOption READ   --blind 0.86  --rebin 2 --only '(veto_A_SR_3b_bdt_(resolved))'$arguments
	#runPlotter --iEcm 13 --iLumi $INTLUMI --inDir ${NTPL_OUTDIR}/ --outDir $PLOTSDIR/mcbased_0lep/ --outFile ${PLOTTER}.root  --json $JSON  --plotExt .pdf --key haa_mcbased --fileOption READ --noLog  --signalScale 1 $arguments 
	#runPlotter --iEcm 13 --iLumi $INTLUMI --inDir ${NTPL_OUTDIR}/ --outDir $PLOTSDIR/mcbased_0lep_log/ --outFile ${PLOTTER}.root  --json $JSON  --plotExt .pdf --key haa_mcbased  --fileOption READ

    fi

    if [[ $step == 3 || $step == 3.2 ]]; then # make plots and combine root files for data-driven backgrounds 
	runPlotter --iEcm 13 --iLumi $INTLUMI --inDir $RESULTSDIR/ --outFile ${PLOTTER}.root  --json $JSON --noPlot --fileOption UPDATE   --key haa_datadriven $arguments
        runPlotter --iEcm 13 --iLumi $INTLUMI --inDir $RESULTSDIR/ --outDir $PLOTSDIR/datadriven/  --outFile ${PLOTTER}.root  --json $JSON --no2D --plotExt .png --plotExt .pdf  --key haa_datadriven --fileOption READ $arguments
    fi	
fi

