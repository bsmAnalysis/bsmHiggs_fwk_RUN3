#dirs=("SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_2017_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb")
#dirs=("SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_2017_noSoftb" "SB13TeV_SM_2018_noSoftb") 
dirs=("SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined") 
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined") # "SB13TeV_SM_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_2017_noSoftb" "SB13TeV_SM_2018_noSoftb" 
#dirs=("SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined")
#dirs=("SB13TeV_SM_Wh_2016_noSoftb" "SB13TeV_SM_Wh_2017_noSoftb" "SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_2017_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined") 

for dir in ${dirs[@]}; do

    cp goodnessfit.sh run-impacts-batch.sh $dir/

    cd $dir
    echo "***** We are in directory $dir *****"

    rm -rf out_60/

    sh script_mass_60.sh
    # run the GOF tests:
    sh goodnessfit.sh 200

    rm -rf impacts/    
    # run the impacts:
    sh run-impacts-batch.sh 1
    sh run-impacts-batch.sh 2
    cd ../
done
