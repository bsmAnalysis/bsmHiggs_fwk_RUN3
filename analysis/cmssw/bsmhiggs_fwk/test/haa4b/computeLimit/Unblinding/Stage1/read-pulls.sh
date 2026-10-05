dirs=("SB13TeV_SM_2016_noSoftb" "SB13TeV_SM_2017_noSoftb" "SB13TeV_SM_2018_noSoftb" "SB13TeV_SM_Wh_2016_noSoftb" "SB13TeV_SM_Wh_2017_noSoftb" "SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Zh_2016_noSoftb" "SB13TeV_SM_Zh_2017_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb")
#dirs=("SB13TeV_SM_Wh_all_noSoftb_Combined" "SB13TeV_SM_Zh_all_noSoftb_Combined" "SB13TeV_SM_all_noSoftb_Combined")      
#dirs=("SB13TeV_SM_2018_noSoftb" "SB13TeV_SM_Wh_2018_noSoftb" "SB13TeV_SM_Zh_2018_noSoftb")   

for dir in ${dirs[@]}; do

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

    file=impacts_${ch}_${year}.json 
    rm $file ; cp $dir/impacts/impacts.json $file

    outfile=pulls_${ch}_${year}.txt; rm $outfile       
    
    echo " ***** Reading impacts json file: $file" >>$outfile

    # read impacts json:
    while read -r line;
    do
	if [[ $line == *"fit"* && $line != *"prefit"* ]]; then 
	    read line1
	    read line2
	    read line3
#	    echo -e "first line: $line and second line: $line1 and third line: $line2"
	    pull=${line2//,/}
	    unc1=${line1//,/}; unc2=${line3//,/};
	fi
	if [[ $line == *"name"* ]]; then
	    echo -e "${line//,/}" >>$outfile #|  awk '{print $2}' >>$outfile

	    sumsq=$(echo "scale=3; $unc1*$unc1+$unc2*$unc2" | bc)
	    square_root=`echo "scale=4; sqrt($sumsq)" | bc`

	    dpull=$(echo "scale=4; ${pull#-}" | bc)
	    dpullsumsq=$(echo "scale=4; $dpull/$square_root" | bc)
	    printf  " with pull = %.2f , dpull/sumsq = %.2f \n\n"  $pull $dpullsumsq >>$outfile
	    #unc1 =  %.2f and unc2 =  %.2f\n\n" $pull $unc1 $unc2  >>$outfile
	fi    

    done <$file	

done
