#!/bin/bash
###https://www.baeldung.com/linux/read-lines-two-input-files

two_lines_operation ()
{
    echo "Doing something with lines from two files:"
    printf 'fileA.txt line: %s\n' "${1}" 
    printf 'fileB.txt line: %s\n' "${2}" 
    printf '\n'
}

## Here we have all the txt files with the pulls for each channel
years=("2016" "2017" "2018")
#years=("all")

for year in ${years[@]}; do

    echo -e "\n\n"
    echo -e " ***** Loop through year $year  *****\n"

    file1=pulls__${year}.txt ; file2=pulls_Wh_${year}.txt ; file3=pulls_Zh_${year}.txt 
 
    i=1 ## COUNT the number of Columns

    while read lineA
    do
	if [[ "$lineA" == *"name"* ]]; then
	    pullnameA=`echo -e $lineA |awk '{print $2}' | tr -d '"'`
	    
	    #pulls that moved significantly in the combined WH+ZH fit:
	    read lineA1
	    pullA=`echo -e $lineA1 |awk '{print $4}'`; low_limit=-999.
	    dpullA=`echo -e $lineA1 |awk '{print $8}'`;
	    
	    result=$(echo "${pullA#-}<${low_limit}" | bc)

	    if [ $result = 1 ]; then
		continue;
	    fi

	    # rename parameter for bin-wise uncertainties: 
	    if [[ $pullnameA != *"JES"* && $pullnameA != *"res_j"* && $pullnameA != *"eff"* && $pullnameA != *"toppt"*  ]]; then 
		if [[ $pullnameA =~ "ch1_" ]]; then
		    pullnameA=${pullnameA/ch1_}
		fi
		if [[ $pullnameA =~ "ch2_" ]]; then  
		    pullnameA=${pullnameA/ch2_}    
		fi
	    fi

	    while read lineB   
	    do
		if [[ "$lineB" == *"name"* ]]; then
		    pullnameB=`echo -e $lineB |awk '{print $2}' | tr -d '"'`

		    if [[ "$pullnameB" == "$pullnameA" ]]; then
			read lineB1
			pullB=`echo -e $lineB1 |awk '{print $4}'`;
			break
		    else
			pullB=-999.
		    fi
		fi
            done < $file2

	    while read lineC
	    do
		if [[ "$lineC" == *"name"* ]]; then      
		    pullnameC=`echo -e $lineC |awk '{print $2}' | tr -d '"'`      
		    
		    if [[ "$pullnameC" == "$pullnameA" ]]; then
			read lineC1
			pullC=`echo -e $lineC1 |awk '{print $4}'`;  
			break
		    else
			pullC=-999.
		    fi
		fi
	    done < $file3

	    declare -A Array2D

	    ## Make an array and sort in dpullA:
	    Array2D[0,$i]=$pullnameA
	    Array2D[1,$i]=$pullA
	    Array2D[2,$i]=$dpullA
	    Array2D[3,$i]=$pullB
	    Array2D[4,$i]=$pullC

	    let i=i+1

	    #optional case:
	    <<EOF
	    id=-1
	    if (( $(bc <<<"$pullA < 0.") )); then
		result=$(echo "${pullB}>0 || ${pullC}>0" | bc)                                                                                                 
		if [ $result = 1 ]; then
		    printf "$format" $pullnameA $pullA $pullB $pullC
		fi
	    else
		result1=$(echo "${pullB}<0 && ${pullB}!=-999." | bc) 
		result2=$(echo "${pullC}<0 && ${pullC}!=-999." | bc)  
		if [ $result1 = 1 ]; then 
		    printf "$format" $pullnameA $pullA $pullB $pullC        
		    id=1
		fi
		if [ $result2 = 1 ]; then
		    if [ $id -lt 0 ]; then
			printf "$format" $pullnameA $pullA $pullB $pullC    
		    fi
		fi
	    fi
EOF

	fi
    done < $file1
    RowNum=$i
    printf "RowNum = %d \n\n" $i

    tempfile=temp_${year}.txt ; rm -rf $tempfile    
    if [[ $low_limit == 2.0 ]]; then
	outfile=largepulls_${year}.txt ; rm -rf $outfile
    else 
	outfile=sortedpulls_${year}.txt ; rm -rf $outfile  
    fi

    echo -e "|--------------------------------------------------------------------------------------------------------------|" 
    format="%45s%12s%12s%18s%18s\n" 
    printf "$format" "Parameter " "|  Pull (WH+ZH) " "| D(pull)/unc "  "|  Pull (WH) " "|  Pull (ZH) " 
    echo -e "|--------------------------------------------------------------------------------------------------------------|" 

    # sort array in dpullA and print:
    for ((j=1;j<=RowNum;j++)) do
       printf "$format" ${Array2D[0,$j]} ${Array2D[1,$j]} ${Array2D[2,$j]} ${Array2D[3,$j]} ${Array2D[4,$j]} >>$tempfile
    done

    sort -r -k3 $tempfile >> $outfile
#    sort -n -k3 $tempfile # ascending order
    sed -i '1i |--------------------------------------------------------------------------------------------------------------|' $outfile
    sed -i '2i "                                    Parameter |  Pull (WH+ZH) | D(pull)/unc  |  Pull (WH)   |  Pull (ZH)     ' $outfile
    sed -i '3i |--------------------------------------------------------------------------------------------------------------|' $outfile
    echo -e "|--------------------------------------------------------------------------------------------------------------|"    >>$outfile        
    echo -r "Sorted pulls for pulls with |pull|>$low_limit " >>$outfile
done
