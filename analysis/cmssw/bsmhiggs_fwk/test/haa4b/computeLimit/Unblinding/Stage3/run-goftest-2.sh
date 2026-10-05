#dirs=("SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_2017_noSoftb" "SB13TeV_SM_2018_noSoftb") # "SB13TeV_SM_Wh_2016_noSoftb" "SB13TeV_SM_Wh_2017_noSoftb" "SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_2017_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb")
#dirs=("SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_all_noSoftb_Combined") 
#dirs=("SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_all_noSoftb_Combined")   
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")      
#dirs=("SB13TeV_SM_2018_noSoftb" "SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb") 
#dirs=("SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")
dirs=("cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")  
# "SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_all_noSoftb_Combined")

for dir in ${dirs[@]}; do

#    cp plotGOF.py $dir/.

    cd $dir
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

<<EOF

    tempfile1="logfirst_${ch}_${year}.txt"
    tempfile2="log_${ch}_${year}.txt"         
    result1=`grep "Unable to determine uncertainties on all fit parameters" 0060/log-first.txt`
    result2=`grep "Unable to determine uncertainties on all fit parameters" 0060/log.txt` 
    echo -e "$result1 \n" >> $tempfile1
    echo -e "$result2 \n" >> $tempfile2   
    cp $tempfile1 $tempfile2 ../.

EOF

    # make GOF plot:
    rm GOF.pdf; python ../plotGOF.py
    cp GOF.pdf ../GOF_${ch}_${year}.pdf

    cd ../
done
