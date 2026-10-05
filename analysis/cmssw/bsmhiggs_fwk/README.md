# Installation for CMSSW_10_6_30 (latest UL/2016,2017,2018/twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun2LegacyAnalysis )
```bash
scram arch   -- see what architecture you have
export SCRAM_ARCH=slc7_amd64_gcc700

cmsrel CMSSW_10_6_30
cd CMSSW_10_6_30/src
cmsenv
git cms-init

#clone repository
git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git UserCode/bsmhiggs_fwk
#git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git -b ul_run2_branch UserCode/bsmhiggs_fwk

cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
#git checkout -b <new branch name>   #copy the branch to a new one to host future modifications (ease pull request and code merging)

#Switch CMSSW_80X to CMSSW_10X by running: (without this compile fails)
cd test/haa4b
sh ./converter.sh # input 1 when you are prompted to select

cd $CMSSW_BASE/src

#twiki: https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018#Scale_and_smearing_corrections_f
#Recipe for running scales and smearings using EgammaPostRecoTools

#WARNING In case you are checking out multiple packages, please always do git cms-init before checking out any packages - otherwise it can complain and create problems in checking out from git

git cms-init  
git cms-addpkg RecoEgamma/EgammaTools  #essentially just checkout the package from CMSSW
git clone https://github.com/cms-egamma/EgammaPostRecoTools.git
mv EgammaPostRecoTools/python/EgammaPostRecoTools.py RecoEgamma/EgammaTools/python/.
git clone -b ULSSfiles_correctScaleSysMC https://github.com/jainshilpi/EgammaAnalysis-ElectronTools.git EgammaAnalysis/ElectronTools/data/
git cms-addpkg EgammaAnalysis/ElectronTools

#And compile
scram b -j 8
```

# Run limits
```bash
export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_10_2_13
cd CMSSW_10_2_13/src
cmsenv

# download the Higgs Combine tool

# download the Higgs Combine tool
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
# Update to a reccomended tag - currently the reccomended tag is v8.1.0
git fetch origin
git checkout v8.1.0
scramv1 b clean; scramv1 b # always make a clean build
cd $CMSSW_BASE/src

# download Haa4b analysis code
git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git UserCode/bsmhiggs_fwk
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
git checkout -b modified
#Switching from CMSSW8_X version to CMSSW10_X version by running:
cd test/haa4b
sh ./converter.sh # input 1 when you are prompted to select
cd $CMSSW_BASE/src

scram b -j 4
```

# Installation for 102X (2018) 
```bash
setenv SCRAM_ARCH slc7_amd64_gcc700
cmsrel CMSSW_10_2_15
cd CMSSW_10_2_15/src
cmsenv
git cms-init

# https://twiki.cern.ch/twiki/bin/view/CMS/EgammaPostRecoRecipes#102X
git cms-merge-topic cms-egamma:EgammaPostRecoTools
# OR:
git cms-merge-topic cms-egamma:PhotonIDValueMapSpeedup1029 #optional but speeds up the photon ID value module so things run faster
#now build everything
scram b -j 8
#now add in E/gamma Post reco tools
git clone git@github.com:cms-egamma/EgammaPostRecoTools.git  EgammaUser/EgammaPostRecoTools
cd  EgammaUser/EgammaPostRecoTools
git checkout master
cd -
echo $CMSSW_BASE
cd $CMSSW_BASE/src
scram b -j 8

# get code to run ecalBadCalibReducedMINIAODFilter on MiniAOD
# https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
git cms-addpkg RecoMET/METFilters
scram b

git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git UserCode/bsmhiggs_fwk
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
git checkout -b modified
#Switching from 2016 version to 2017 (compatible with 2018) by running:
cd test/haa4b
sh ./converter.sh # input 1 when you are prompted to select
cd $CMSSW_BASE/src

git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v8.0.1
scramv1 b clean; scramv1 b # always make a clean build

#And compile
scram b -j 4
```

# Installation for 94X (2017) 
```bash
export SCRAM_ARCH=slc6_amd64_gcc630
#or
setenv SCRAM_ARCH slc6_amd64_gcc630
cmsrel CMSSW_9_4_9
cd CMSSW_9_4_9/src
cmsenv
git cms-init

#Packages to rerun electrons energy correction
#https://twiki.cern.ch/twiki/bin/view/CMS/EgammaMiniAODV2#2017%20MiniAOD%20V2
git cms-merge-topic cms-egamma:EgammaPostRecoTools
scram b -j 4

#Calculating corrections and uncertainties on MET
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETUncertaintyPrescription#Instructions%20for%20%209_4_X,%20X%20%3E=0%20f
git cms-merge-topic cms-met:METFixEE2017_949_v2
scram b -j 4

#Add Fall17_94X_V2 ID for electrons
#https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
git cms-merge-topic cms-egamma:EgammaID_949
scram b -j 4

#Add Fall17_94X_V2 ID for photons
#https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedPhotonIdentificationRun2
git cms-merge-topic lsoffi:CMSSW_9_4_0_pre3_TnP
scram b -j 10
# Add the area containing the MVA weights (from cms-data, to appear in "external").
# Note: the "external" area appears after "scram build" is run at least once, as above
cd $CMSSW_BASE/external/slc6_amd64_gcc630/
git clone https://github.com/lsoffi/RecoEgamma-PhotonIdentification.git data/RecoEgamma/PhotonIdentification/data
cd data/RecoEgamma/PhotonIdentification/data
git checkout CMSSW_9_4_0_pre3_TnP
cd $CMSSW_BASE/external/slc6_amd64_gcc630/
git clone https://github.com/lsoffi/RecoEgamma-ElectronIdentification.git data/RecoEgamma/ElectronIdentification/data
cd data/RecoEgamma/ElectronIdentification/data
git checkout CMSSW_9_4_0_pre3_TnP
cd $CMSSW_BASE/src

# get code to run ecalBadCalibReducedMINIAODFilter on MiniAOD
# https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
git cms-addpkg RecoMET/METFilters
scram b

git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git UserCode/bsmhiggs_fwk
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
git checkout -b modified #copy the branch to a new one to host future modifications (ease pull request and code merging)
#Switching from 2016 version to 2017 by running:
cd test/haa4b
sh ./converter.sh # input 1 when you are prompted to select
cd $CMSSW_BASE/src

#And compile
scram b -j 4
```
# Installation for 94X (2016 Legacy)
```bash
export SCRAM_ARCH=slc6_amd64_gcc630
#or
setenv SCRAM_ARCH slc6_amd64_gcc630
cmsrel CMSSW_9_4_9
cd CMSSW_9_4_9/src
cmsenv
git cms-init

git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git UserCode/bsmhiggs_fwk
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
git checkout -b modified #copy the branch to a new one to host future modifications (ease pull request and code merging)
#Switching from 2016 version to 2017 by running:
cd test/haa4b
sh ./converter.sh # input 1 when you are prompted to select
cd $CMSSW_BASE/src

#And compile
scram b -j 4
```

# Installation for 80X (2016) 

```bash
export SCRAM_ARCH=slc6_amd64_gcc530
#or:
setenv SCRAM_ARCH slc6_amd64_gcc530
cmsrel CMSSW_8_0_26_patch1
cd CMSSW_8_0_26_patch1/src
cmsenv
 
#Checkout Some Packages from Egamma 
# ( https://twiki.cern.ch/twiki/bin/viewauth/CMS/EGMRegression#Consistent_EGMSmearer )
git cms-init 
git cms-merge-topic cms-egamma:EGM_gain_v1
cd $CMSSW_BASE/src/EgammaAnalysis/ElectronTools/data
git clone -b Moriond17_gainSwitch_unc https://github.com/ECALELFS/ScalesSmearings.git
cd $CMSSW_BASE/src
scram b -j 8

#Adding DeepCSV
git cms-merge-topic -u mverzett:DeepFlavour-from-CMSSW_8_0_21 
mkdir $CMSSW_BASE/src/RecoBTag/DeepFlavour/data/
cd $CMSSW_BASE/src/RecoBTag/DeepFlavour/data/  
wget http://home.fnal.gov/~verzetti//DeepFlavour/training/DeepFlavourNoSL.json
cd $CMSSW_BASE/src/ 
scram b -j 8

# Check out packages for Hbb tagging (https://twiki.cern.ch/twiki/bin/viewauth/CMS/Hbbtagging#8_0_X)
setenv CMSSW_GIT_REFERENCE /cvmfs/cms.cern.ch/cmssw.git.daily
#or
export CMSSW_GIT_REFERENCE=/cvmfs/cms.cern.ch/cmssw.git.daily
git cms-init
git remote add btv-cmssw https://github.com/cms-btv-pog/cmssw.git
git fetch --tags btv-cmssw
git cms-merge-topic -u cms-btv-pog:BoostedDoubleSVTaggerV4-WithWeightFiles-v1_from-CMSSW_8_0_21
scram b -j 8

#git clone -b svFit_2015Apr03 https://github.com/veelken/SVfit_standalone.git TauAnalysis/SVfitStandalone

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETUncertaintyPrescription#Instructions_for_8_0_X_X_26_patc
git cms-merge-topic cms-met:METRecipe_8020 -u
git cms-merge-topic cms-met:METRecipe_80X_part2 -u


git clone https://github.com/bsmAnalysis/BSMHiggs_fwk.git UserCode/bsmhiggs_fwk
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
git checkout -b modified #copy the branch to a new one to host future modifications (ease pull request and code merging)
cd $CMSSW_BASE/src

## HiggsCombine in 80X 
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
##Update to a reccomended tag - currently the reccomended tag is v7.0.13
# http://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
git checkout v7.0.13
scramv1 b clean; scramv1 b # always make a clean build
cd ../../

#JetToolBox: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetToolbox
git clone git@github.com:cms-jet/JetToolbox.git JMEAnalysis/JetToolbox -b jetToolbox_80X_V3

#And compile
scramv1 b -j 16 
```

# For developers

We have decided to use pull-request mode for the master development.

- Fork the code with your personal github ID. See [details](https://help.github.com/articles/fork-a-repo/)
- Make a clean git clone in the UserCode directory
```
cd $CMSSW_BASE/src/UserCode 
git clone git@github.com:yourgithubid/BSMHiggs_fwk.git bsmhiggs_fwk
cd bsmhiggs_fwk
git remote add upstream git@github.com:bsmAnalysis/BSMHiggs_fwk.git

- Update your repository and start your changes:
git remote update
git merge upstream/master
```
- Make your own change and commit
```
git add .
git commit -a -m "Added feature A, B, C"
git push
```
And also switch back to 2016 from 2017 if you work in 2017 before commiting by running:
```
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk/test/haa4b
sh ./converter # input 2 when you are prompted to select
cd $CMSSW_BASE/src/UserCode/bsmhiggs_fwk
```
<!--- If work under 94X, be sure that do not commit src/PatUtils.cc, use the below instead:
 ```
 git add -u
 git reset src/PatUtils.cc
 git commit -m "Added feature A, B, C"
 git push
 ``` -->
- Make a pull request against the bsmAnalysis. See [details](https://help.github.com/articles/using-pull-requests/)


# For creating private samples


# Analysing
See example files in UserCode/bsmhiggs_fwk/test/haa4b/ Executing the
file 'submit.sh' will run the analysis on all samples defined in
'samples.json'.

Please see the full usage of 'submit.sh' with

> ./submit.sh

# Analysis specific
If you need set your stuff under test/analysis_subdir

In general to run an executable or script you can do:
```
runLocalAnalysisOverSamples.py -e my_exe -j data/my_samples.json -d my_input_dir -o my_output_dir -c test/runAnalysis_cfg.py.templ -p "@par1=val1 @par2=val2" -s queue
```

 The easiest is however to create a submit.sh script in the directory
 of your analysis.  The executable can also be added in the
 UserCode/bsmhiggs_fwk/test/haa4b/bin directory (also add it to the
 Buildfile there)

# Ntuple locations (July 2022 Updated)
- 2016 DATA  and  MC Samples: ```/eos/user/g/georgia/results_2016_2020_06_19```
- 2017 DATA  and  MC Samples: ```/eos/cms/store/user/georgia/results_2017_2020_02_05```
- 2018 DATA  and  MC Samples: ```/eos/user/g/georgia/results_2018_2020_02_05```
