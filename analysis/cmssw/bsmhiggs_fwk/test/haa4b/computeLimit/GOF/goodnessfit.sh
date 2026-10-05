# comment out Wh or Zh part
###################### ZH #####################
#cd ../cards_SB13TeV_SM_$1$2_noSoftb/0060/    
#cd ../cards_SB13TeV_SM_$1$2_noSoftb/
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

    rangeString="z_norm_3b_e=0.1,4.0:z_norm_3b_mu=0.1,4.0:z_norm_4b_e=0.1,4.0:z_norm_4b_mu=0.1,4.0"

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
printf "\n\n rangeString = %s\n\n" $rangeString

# run distribution for ee channel
combine -M GoodnessOfFit -d workspace.root --algo=saturated -n _result_sb --rMin -5.0 --rMax 5.0 --setParameterRanges $rangeString --setParameters $parametersString  --cminDefaultMinimizerStrategy 0
#--freezeParameters r --setParameters r=0 
#--setParametersForFit mask_ee_A_SR_3b=1,mask_ee_A_SR_4b=1 --setParametersForEval mask_ee_A_SR_3b=1,mask_ee_A_SR_4b=1 --freezeParameters r --setParameters r=0
combine -M GoodnessOfFit -d workspace.root --algo=saturated -n result_toy_sb -t $1 --toysFrequentist   --rMin -5.0 --rMax 5.0 --setParameterRanges $rangeString --setParameters $parametersString  --cminDefaultMinimizerStrategy 0
#--freezeParameters r --setParameters r=0 
#--setParametersForFit mask_ee_A_SR_3b=1,mask_ee_A_SR_4b=1 --setParametersForEval mask_ee_A_SR_3b=1,mask_ee_A_SR_4b=1 --freezeParameters r --setParameters r=0,mask_ee_A_SR_3b=1,mask_ee_A_SR_4b=1 -t 500 --toysFrequentist

# run distributioni for mumu channel
#combine -M GoodnessOfFit -d workspace_mu.root --algo=saturated --setParametersForFit mask_mumu_A_SR_3b=1,mask_mumu_A_SR_4b=1 --setParametersForEval mask_mumu_A_SR_3b=1,mask_mumu_A_SR_4b=1 --freezeParameters r --setParameters r=0
#combine -M GoodnessOfFit -d workspace_mu.root --algo=saturated --setParametersForFit mask_mumu_A_SR_3b=1,mask_mumu_A_SR_4b=1 --setParametersForEval mask_mumu_A_SR_3b=1,mask_mumu_A_SR_4b=1 --freezeParameters r --setParameters r=0,mask_ee_A_SR_3b=1,mask_ee_A_SR_4b=1 -t 500 --toysFrequentist


##################### WH ######################
#cd /afs/cern.ch/work/y/yuanc/Analysis/H2a4b/CMSSW_10_2_13/src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/cards_SB13TeV_SM_Wh_2018_noSoftb/0060/

#combine -M GoodnessOfFit -d workspace.root --algo=saturated --setParametersForFit mask_e_A_SR_3b=1,mask_mu_A_SR_3b=1,mask_e_A_SR_4b=1,mask_mu_A_SR_4b=1 --setParametersForEval mask_e_A_SR_3b=1,mask_mu_A_SR_3b=1,mask_e_A_SR_4b=1,mask_mu_A_SR_4b=1 --freezeParameters r --setParameters r=0
#combine -M GoodnessOfFit -d workspace.root --algo=saturated --setParametersForFit mask_e_A_SR_3b=1,mask_mu_A_SR_3b=1,mask_e_A_SR_4b=1,mask_mu_A_SR_4b=1 --setParametersForEval mask_e_A_SR_3b=1,mask_mu_A_SR_3b=1,mask_e_A_SR_4b=1,mask_mu_A_SR_4b=1 --freezeParameters r --setParameters r=0,mask_e_A_SR_3b=1,mask_mu_A_SR_3b=1,mask_e_A_SR_4b=1,mask_mu_A_SR_4b=1 -t 500 --toysFrequentist

#printf "\n\n parametersString = %s\n\n" $parametersString

