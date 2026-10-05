#dirs=("SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_2017_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb")
#dirs=("SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_2017_noSoftb" "SB13TeV_SM_2018_noSoftb") 
#dirs=("SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined") 
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined") # "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_2017_noSoftb" "SB13TeV_SM_2018_noSoftb" 
#dirs=("SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined")
dirs=("cards_SB13TeV_SM_all_noSoftb_Combined")  
#dirs=("cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined") 

# find cards_SB13TeV_SM_*/0060/. -type f -name '*GoodnessOfFit*' -delete

for dir in ${dirs[@]}; do

    cp goodnessfit.sh $dir/

    cd $dir
    echo "***** We are in directory $dir *****"

    # run the GOF tests:
    sh goodnessfit.sh 200

    cd ../
done
