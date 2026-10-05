dirs=("cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_2017_noSoftb" "cards_SB13TeV_SM_2018_noSoftb" "cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined") 
#dirs=("cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined")
#dirs=("cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb") 
#dirs=("cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined") # "cards_SB13TeV_SM_all_noSoftb_Combined")
#dirs=("cards_SB13TeV_SM_2016_noSoftb" "cards_SB13TeV_SM_2017_noSoftb" "cards_SB13TeV_SM_2018_noSoftb")
#dirs=("cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_2018_noSoftb")
#dirs=("cards_SB13TeV_SM_Wh_2016_noSoftb" "cards_SB13TeV_SM_Wh_2017_noSoftb" "cards_SB13TeV_SM_Wh_2018_noSoftb" "cards_SB13TeV_SM_Wh_all_noSoftb_Combined" "cards_SB13TeV_SM_Zh_2016_noSoftb" "cards_SB13TeV_SM_Zh_2017_noSoftb" "cards_SB13TeV_SM_Zh_2018_noSoftb" "cards_SB13TeV_SM_Zh_all_noSoftb_Combined" "cards_SB13TeV_SM_all_noSoftb_Combined") 

for dir in ${dirs[@]}; do

    cp run-impacts-batch.sh $dir/.

    cd $dir
    echo "***** We are in directory $dir *****"

    # run the impacts:
    sh run-impacts-batch.sh 1
#    sh run-impacts-batch.sh 2
    cd ../
done
