#dirs=("cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_2017_noSoftb" "cards_SB13TeV_SM_2018_noSoftb" "cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb")
dirs=("cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_2017_noSoftb" "cards_SB13TeV_SM_2018_noSoftb" "cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")
#dirs=("cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_all_noSoftb_Combined") 
#dirs=("cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined")
#dirs=("cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined")   
#dirs=("cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")      
#dirs=("cards_SB13TeV_SM_2018_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb") 
#dirs=("cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")
#dirs=("cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb")  
#dirs=("cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_2017_noSoftb" "cards_SB13TeV_SM_2018_noSoftb")     
#dirs=("cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_2018_noSoftb") 
#dirs=("cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")  
# "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined")

for dir in ${dirs[@]}; do

#    cp run-impacts-batch.sh $dir/.      

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


    # make impacts plot:
    masspoint=60

    sh ../run-impacts-batch.sh 3 
    cp impacts_00${masspoint}/impacts_00${masspoint}.pdf ../impacts_m${masspoint}_${ch}_${year}.pdf 

    cd ../
done
