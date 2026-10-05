

  printf "\n\n Number of arguments: %d\n\n" $#

  if [ $# -ne 1 ]
  then
     printf "\n\n Usage:\n"
     printf "  sh run-impacts-batch.sh <cards-dir> <step>\n"
     printf "\n"
     printf "    step:\n"
     printf "     1 : do initial fit\n"
     printf "     2 : submit batch jobs\n"
     printf "     3 : make the plots pdf file\n"
     printf "\n"
     printf "\n"
     printf "   Note: If your fits take longer than the time limit for the default queue in condor\n"
     printf "         you may need to set the queue to workday.  To do that, edit this file\n"
     printf "         CombineHarvester/CombineTools/python/combine/CombineToolBase.py\n"
     printf "         and add the following line\n\n"
     printf "+JobFlavour = \"workday\" \n\n"
     printf "         in the CONDOR_TEMPLATE section right after the line that sets log.\n\n"
     printf "         If step3 fails for you and you don't get the pdf file, this is probably the reason.\n\n\n\n\n"
     exit
  fi


  cards_dir=`pwd`

  cd out_60
 
  if [[ $cards_dir == *"Wh_all"* ]]; then 
      printf "\n Found Wh_all\n\n"
      tt_e_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_e'  | awk '{print $4;}'`;                                                                         
      tt_mu_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                        
      v_e_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_e'   | awk '{print $4;}'`;                                                                          
      v_mu_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_mu'  | awk '{print $4;}'`;  
      tt_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_e'  | awk '{print $4;}'`;                                                                         
      tt_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                        
      v_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_e'   | awk '{print $4;}'`;                                                                          
      v_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_mu'  | awk '{print $4;}'`; 
      tt_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_e'  | awk '{print $4;}'`;                                                                         
      tt_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                        
      v_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_e'   | awk '{print $4;}'`;                                                                          
      v_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_mu'  | awk '{print $4;}'`; 
      
      parametersString="tt_norm_e_2016=$tt_e_2016,w_norm_e_2016=$v_e_2016,tt_norm_mu_2016=$tt_mu_2016,w_norm_mu_2016=$v_mu_2016,tt_norm_e_2017=$tt_e_2017,w_norm_e_2017=$v_e_2017,tt_norm_mu_2017=$tt_mu_2017,w_norm_mu_2017=$v_mu_2017,tt_norm_e_2018=$tt_e_2018,w_norm_e_2018=$v_e_2018,tt_norm_mu_2018=$tt_mu_2018,w_norm_mu_2018=$v_mu_2018"

      rangeString="tt_norm_e_2016=0.1,4.0:tt_norm_mu_2016=0.1,4.0:w_norm_e_2016=0.1,4:w_norm_mu_2016=0.1,4:tt_norm_e_2017=0.1,4.0:tt_norm_mu_2017=0.1,4.0:w_norm_e_2017=0.1,4:w_norm_mu_2017=0.1,4:tt_norm_e_2018=0.1,4.0:tt_norm_mu_2018=0.1,4.0:w_norm_e_2018=0.1,4:w_norm_mu_2018=0.1,4" 
      
  elif [[ $cards_dir == *"Zh_all"* ]]; then
      printf "\n Found Zh_all\n\n"
      
      v_3b_e_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                    
      v_4b_e_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`; 
      v_3b_mu_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                 
      v_4b_mu_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`;
      v_3b_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                    
      v_4b_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`;  
      v_3b_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                 
      v_4b_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`; 
      v_3b_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                    
      v_4b_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`;
      v_3b_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                 
      v_4b_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`;
      
      parametersString="z_norm_3b_e_2016=$v_3b_e_2016,z_norm_4b_e_2016=$v_4b_e_2016,z_norm_3b_mu_2016=$v_3b_mu_2016,z_norm_4b_mu_2016=$v_4b_mu_2016,z_norm_3b_e_2017=$v_3b_e_2017,z_norm_4b_e_2017=$v_4b_e_2017,z_norm_3b_mu_2017=$v_3b_mu_2017,z_norm_4b_mu_2017=$v_4b_mu_2017,z_norm_3b_e_2018=$v_3b_e_2018,z_norm_4b_e_2018=$v_4b_e_2018,z_norm_3b_mu_2018=$v_3b_mu_2018,z_norm_4b_mu_2018=$v_4b_mu_2018"

      rangeString="z_norm_3b_e_2016=0.1,4.0:z_norm_3b_mu_2016=0.1,4.0:z_norm_4b_e_2016=0.1,4.0:z_norm_4b_mu_2016=0.1,4.0:z_norm_3b_e_2017=0.1,4.0:z_norm_3b_mu_2017=0.1,4.0:z_norm_4b_e_2017=0.1,4.0:z_norm_4b_mu_2017=0.1,4.0:z_norm_3b_e_2018=0.1,4.0:z_norm_3b_mu_2018=0.1,4.0:z_norm_4b_e_2018=0.1,4.0:z_norm_4b_mu_2018=0.1,4.0"
      
  elif [[ $cards_dir == *"Wh_"* ]]; then
      printf "\n Found Wh_\n\n"
      
      tt_e=`cat simfit_m60.txt | grep 'tt_norm_e' | awk '{print $4;}'`;                                                                                           
      tt_mu=`cat simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                                         
      v_e=`cat simfit_m60.txt | grep 'w_norm_e' | awk '{print $4;}'`;                                                                                             
      v_mu=`cat simfit_m60.txt | grep 'w_norm_mu' | awk '{print $4;}'`; 
      
      parametersString=" tt_norm_e=$tt_e,w_norm_e=$v_e,tt_norm_mu=$tt_mu,w_norm_mu=$v_mu"
   
      rangeString="tt_norm_e=0.1,4.0:tt_norm_mu=0.1,4.0:w_norm_e=0.1,4:w_norm_mu=0.1,4"
   
  elif [[ $cards_dir == *"Zh_"* ]]; then
      printf "\n Found Zh_\n\n"
      
      v_3b_e=`cat simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                                     
      v_4b_e=`cat simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`;                                                                                     
      v_3b_mu=`cat simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                                  
      v_4b_mu=`cat simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`;    
      
      parametersString="z_norm_3b_e=$v_3b_e,z_norm_4b_e=$v_4b_e,z_norm_3b_mu=$v_3b_mu,z_norm_4b_mu=$v_4b_mu"
      
      rangeString="z_norm_3b_e=0.1,4.0:z_norm_3b_mu=0.1,4.0:z_norm_4b_e=0.1,4.0:z_norm_4b_mu=0.1,4.0:tt_norm_e=0.1,4.0:tt_norm_mu=0.1,4.0"

  elif [[ $cards_dir == *"SM_2"* ]]; then
      printf "\n Found SM_2\n\n"
      
      tt_e=`cat simfit_m60_e.txt | grep 'tt_norm_e' | awk '{print $4;}'`;
      z_3b_e=`cat simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;
      z_4b_e=`cat simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`;
      w_e=`cat simfit_m60_e.txt | grep 'w_norm_e' | awk '{print $4;}'`;
      tt_mu=`cat simfit_m60_mu.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;
      z_3b_mu=`cat simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;
      z_4b_mu=`cat simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`;
      w_mu=`cat simfit_m60_mu.txt | grep 'w_norm_mu' | awk '{print $4;}'`;
      
      parametersString="tt_norm_e=$tt_e,z_norm_3b_e=$z_3b_e,z_norm_4b_e=$z_4b_e,w_norm_e=$w_e,tt_norm_mu=$tt_mu,z_norm_3b_mu=$z_3b_mu,z_norm_4b_mu=$z_4b_mu,w_norm_mu=$w_mu"
      
      rangeString="tt_norm_e=0.1,4.0:tt_norm_mu=0.1,4.0:w_norm_e=0.1,4:w_norm_mu=0.1,4:z_norm_3b_e=0.1,4.0:z_norm_3b_mu=0.1,4.0:z_norm_4b_e=0.1,4.0:z_norm_4b_mu=0.1,4.0"

  elif [[ $cards_dir == *"SM_all"* ]]; then
      printf "\n Found SM_all\n\n"
      
      tt_e_2016=`cat  /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_e'  | awk '{print $4;}'`;                                                                         
      tt_mu_2016=`cat  /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                        
      w_e_2016=`cat  /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_e'   | awk '{print $4;}'`;                                                                          
      w_mu_2016=`cat  /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2016_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_mu'  | awk '{print $4;}'`;  
      tt_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_e'  | awk '{print $4;}'`;                                                                         
      tt_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                        
      w_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_e'   | awk '{print $4;}'`;                                                                          
      w_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2017_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_mu'  | awk '{print $4;}'`; 
      tt_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_e'  | awk '{print $4;}'`;                                                                         
      tt_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'tt_norm_mu' | awk '{print $4;}'`;                                                                        
      w_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_e'   | awk '{print $4;}'`;                                                                          
      w_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Wh_2018_noSoftb/out_60/simfit_m60.txt | grep 'w_norm_mu'  | awk '{print $4;}'`; 

      z_3b_e_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                    
      z_4b_e_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`; 
      z_3b_mu_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                 
      z_4b_mu_2016=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2016_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`;
      z_3b_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                    
      z_4b_e_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`;  
      z_3b_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                 
      z_4b_mu_2017=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2017_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`; 
      z_3b_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_3b_e' | awk '{print $4;}'`;                                                                    
      z_4b_e_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_e.txt | grep 'z_norm_4b_e' | awk '{print $4;}'`;
      z_3b_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_3b_mu' | awk '{print $4;}'`;                                                                 
      z_4b_mu_2018=`cat /afs/cern.ch/work/g/georgia/BSMAnalysis/limits-combine-v8.1.0/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/JOBS_TEST/SB13TeV_SM_Zh_2018_noSoftb/out_60/simfit_m60_mu.txt | grep 'z_norm_4b_mu' | awk '{print $4;}'`;
      
      parametersString="tt_norm_e_2016=$tt_e_2016,z_norm_3b_e_2016=$z_3b_e_2016,z_norm_4b_e_2016=$z_4b_e_2016,w_norm_e_2016=$w_e_2016,tt_norm_mu_2016=$tt_mu_2016,z_norm_3b_mu_2016=$z_3b_mu_2016,z_norm_4b_mu_2016=$z_4b_mu_2016,w_norm_mu_2016=$w_mu_2016,tt_norm_e_2017=$tt_e_2017,z_norm_3b_e_2017=$z_3b_e_2017,z_norm_4b_e_2017=$z_4b_e_2017,w_norm_e_2017=$w_e_2017,tt_norm_mu_2017=$tt_mu_2017,z_norm_3b_mu_2017=$z_3b_mu_2017,z_norm_4b_mu_2017=$z_4b_mu_2017,w_norm_mu_2017=$w_mu_2017,tt_norm_e_2018=$tt_e_2018,z_norm_3b_e_2018=$z_3b_e_2018,z_norm_4b_e_2018=$z_4b_e_2018,w_norm_e_2018=$w_e_2018,tt_norm_mu_2018=$tt_mu_2018,z_norm_3b_mu_2018=$z_3b_mu_2018,z_norm_4b_mu_2018=$z_4b_mu_2018,w_norm_mu_2018=$w_mu_2018"
      
      rangeString="tt_norm_e_2016=0.1,4.0:tt_norm_mu_2016=0.1,4.0:w_norm_e_2016=0.1,4:w_norm_mu_2016=0.1,4:z_norm_3b_e_2016=0.1,4.0:z_norm_3b_mu_2016=0.1,4.0:z_norm_4b_e_2016=0.1,4.0:z_norm_4b_mu_2016=0.1,4.0:tt_norm_e_2017=0.1,4.0:tt_norm_mu_2017=0.1,4.0:w_norm_e_2017=0.1,4:w_norm_mu_2017=0.1,4:z_norm_3b_e_2017=0.1,4.0:z_norm_3b_mu_2017=0.1,4.0:z_norm_4b_e_2017=0.1,4.0:z_norm_4b_mu_2017=0.1,4.0:tt_norm_e_2018=0.1,4.0:tt_norm_mu_2018=0.1,4.0:w_norm_e_2018=0.1,4:w_norm_mu_2018=0.1,4:z_norm_3b_e_2018=0.1,4.0:z_norm_3b_mu_2018=0.1,4.0:z_norm_4b_e_2018=0.1,4.0:z_norm_4b_mu_2018=0.1,4.0"
else
    printf "\n\n *** can't figure out what kind of cards directory this is:  %s\n\n" $cards_dir
    exit -1
fi

printf "\n\n parametersString = %s\n\n" $parametersString
  cd ../

  step=$1
  printf "\n step = %d\n\n" $step


  impacts_dir=impacts
#`echo $cards_dir | sed s/cards/impacts/`

  printf "\n\n impacts output directory: %s\n\n" $impacts_dir



# if [ -d "$impacts_dir" ]
# then
#    datestring=`date +%0G-%0m-%0d`
#    mv_dir=`printf "old-%s-%s" $impacts_dir $datestring`
#    printf "  impacts dir exists.  moving it to %s \n\n" $mv_dir
#    mv $impacts_dir $mv_dir
# fi

#  rangeString="tt_norm_e=0.1,4.0:tt_norm_mu=0.1,4.0:w_norm_e=0.1,4:w_norm_mu=0.1,4"
  #rangeString="z_norm_3b_e=0.1,4.0:z_norm_3b_mu=0.1,4.0:z_norm_4b_e=0.1,4.0:z_norm_4b_mu=0.1,4.0:tt_norm_e=0.1,4.0:tt_norm_mu=0.1,4.0"

  printf "\n\n rangeString = %s\n\n" $rangeString


  mkdir -p $impacts_dir

  cd $impacts_dir

  this_dir=`pwd`
  printf "\n\n this directory: %s\n\n" $this_dir

  f_name=$impacts_dir


  if [ "$step" = "1" ]
  then
     printf "\n will run initial fit\n\n"
     combineTool.py -M Impacts -d ../out_60/workspace.root -v 3 -m 60 --doInitialFit --robustFit 1  --rMin -5.0 --rMax 5.0 --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.1 --setParameters $parametersString --setParameterRanges $rangeString  |& tee step1.log
  fi


  if [ "$step" = "2" ]
  then
     printf "\n will submit condor jobs\n\n"
     combineTool.py -M Impacts -d ../out_60/workspace.root -v 3 -m 60 --doFits  --robustFit 1   --rMin -5.0 --rMax 5.0 --cminDefaultMinimizerStrategy 0  --cminDefaultMinimizerTolerance 0.1  --setParameters $parametersString --setParameterRanges $rangeString  --job-mode condor |& tee step2.log
  fi


  if [ "$step" = "3" ]
  then
     printf "\n will make the plots pdf file\n\n"
     combineTool.py -M Impacts -d ../out_60/workspace.root -m 60 -o impacts.json  |& tee step3.log
     plotImpacts.py -i impacts.json -o $f_name --blind |& tee step4.log
  fi

  printf "\n\n Done.\n\n"

  cd -

