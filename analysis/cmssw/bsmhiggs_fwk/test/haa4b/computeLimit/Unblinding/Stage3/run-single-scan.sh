#dirs=("cards_SB13TeV_SM_Wh_2018_noSoftb")
dirs=("cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_2017_noSoftb" "cards_SB13TeV_SM_2018_noSoftb" "cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")

rm -rf single-scans; mkdir single-scans

for dir in ${dirs[@]}; do

    echo " **** Entering directory $dir ****"

    if [[ $dir == *"2016"* ]]; then
	year="2016"
    elif [[ $dir == *"2017"* ]]; then 
	year="2017"
    elif [[ $dir == *"2018"* ]]; then  
	year="2018"
    else
	year="all"
    fi

    if [[ $dir == *"Wh"* ]]; then
	ch="Wh"
    elif [[ $dir == *"Zh"* ]]; then
	ch="Zh"
    else
	ch=""
    fi

    cd $dir 
    masspoint=60

    #make single likelihood scan:
    cd 00${masspoint}
    combine -M MultiDimFit workspace.root -n .part3E -m 60 --rMin -1 --rMax 5 --algo grid --points 30    
    plot1DScan.py higgsCombine.part3E.MultiDimFit.mH60.root -o single_scan
    cp single_scan.pdf ../../single-scans/single_scan_m${masspoint}_${ch}_${year}.pdf

    cd ../../
done
