#include <iostream>

#include <boost/shared_ptr.hpp>

#include "Math/GenVector/Boost.h"

#include "UserCode/bsmhiggs_fwk/interface/tdrstyle.h"

#include "UserCode/bsmhiggs_fwk/interface/JSONWrapper.h"

#include "UserCode/bsmhiggs_fwk/interface/RootUtils.h"

#include "UserCode/bsmhiggs_fwk/interface/MacroUtils.h"

//#include "UserCode/bsmhiggs_fwk/interface/HxswgUtils.h"

#include "UserCode/bsmhiggs_fwk/interface/th1fmorph.h"

#include "TLine.h"

#include "TSystem.h"

#include "TFile.h"

#include "TTree.h"

#include "TCanvas.h"

#include "TH1F.h"

#include "TH2F.h"

#include "TProfile.h"

#include "TROOT.h"

#include "TString.h"

#include "TList.h"

#include "TGraph.h"

#include "TCanvas.h"

#include "TPad.h"

#include "TLegend.h"

#include "TLegendEntry.h"

#include "TPaveText.h"

#include "TObjArray.h"

#include "THStack.h"

#include "TGraphErrors.h"

#include "TLatex.h"

#include "Math/QuantFuncMathCore.h"

#include "TMath.h"

#include "TGraphAsymmErrors.h"

#include "RooFitResult.h"

#include "RooRealVar.h"

#include<iostream>

#include<fstream>

#include<map>

#include<algorithm>

#include<vector>

#include<set>

#include <regex>

using namespace std;

bool verbose = false ;

float minErrOverSqrtNBGForBinByBin = 0.35 ;

bool autoMCStats = false ;

enum StatUncMode_t {
  kStatNone,
  kStatCorrelated,
  kStatHybrid
};

// Keep the same explicit interface as the two-lepton executable.
// autoMCStats is mutually exclusive with all custom MC-stat shapes.
StatUncMode_t statUncMode = kStatCorrelated;
TString statUncModeName("correlated");

TString signalSufix="";

TString histo(""), histoVBF("");

int rebinVal = 1;

double MCRescale = 1.0;

double SignalRescale = 1.0;

double datadriven_qcd_Syst = 0.50;    

// Add externally produced systematic shape variations: 

//bool addsyst=false;

// Use postfit normalizations in W/DY/Top bkg components:

bool postfit=false;

double rfr_tt_norm ;

int mass;

bool shape = true;

TString postfix="";

TString systpostfix="";

bool runSystematics = true; 

bool runZh = true;

bool modeDD = true;

bool simfit = false;

bool doDDQCDValidation = false;

TString vh_tag("");

TString year("");

std::vector<TString> Channels;

std::vector<string> AnalysisBins;

double DDRescale = 1.0;

TString DYFile ="";

TString FREFile="";

string signalTag="";

bool BackExtrapol  = false;

bool subNRB        = false;

bool MCclosureTest = false;

bool scaleVBF      = false;

bool mergeWWandZZ = false;

bool skipWW = true;

bool skipGGH = false;

bool skipQQH = false;

bool subDY = false;

bool subWZ = false;

bool subFake = false;

bool blindData = false;

bool blindWithSignal = false; 

bool replaceHighSensitivityBinsWithBG = false ;

bool noCorrelatedStatUnc = false ;

bool correlatedLumi = false;

bool plotsOnly = false ;

TString showOneUncertainty("");

TString inFileUrl(""),jsonFile("");

TString sumFileUrl("") ;

TString fdInputFile("") ;

TString rfrInputFile("") ;

TString inFileUrl17("/afs/cern.ch/user/l/lrouseli/Analysis/CMSSW_10_6_30/src/UserCode/bsmhiggs_fwk/test/haa4b/plotter_WH_2017_2025_05_23_forLimits.root");    

//= std::string(std::getenv("CMSSW_BASE"))+"src/UserCode/bsmhiggs_fwk/test/haa4b/plotter_WH_2017_2020_02_05_forLimits.root";

double wscale17=(59.7/41.5);

double shapeMin =-9999;

double shapeMax = 9999;

double shapeMinVBF =-9999;

double shapeMaxVBF = 9999;

bool doInterf = false;

double minSignalYield = 0;

float statBinByBin = -1;

bool useLogy = true;

bool blindSR = false;

double lumi = 108960;

int signalScale = 1;

double sstyCut = -1;

bool docut = false;

bool dirtyFix1 = false;

bool dirtyFix2 = false;

std::vector<int> shapeBinToConsider;

std::vector<int> indexcutV;

std::vector<int> indexcutVL;

std::vector<int> indexcutVR;

std::map<string, int> indexcutM;

std::map<string, int> indexcutML;

std::map<string, int> indexcutMR;

std::vector<string> keywords;


int indexvbf = -1;

int massL=-1, massR=-1;

double dropBckgBelow=0.001; //0.001; 

/*

 *Case Sensitive Implementation of startsWith()

 *It checks if the string 'mainStr' starts with given string 'toMatch'

 */

bool startsWith(std::string mainStr, std::string toMatch){

  // std::string::find returns 0 if toMatch is found at starting

  if(mainStr.find(toMatch) == 0)

    return true;

  else

    return false;

}

bool matchKeyword(JSONWrapper::Object& process, std::vector<string>& keywords){

  if(keywords.size()<=0)return true;

  if(process.isTag("keys")){

    std::vector<JSONWrapper::Object> dsetkeywords = process["keys"].daughters();

    for(size_t ikey=0; ikey<dsetkeywords.size(); ikey++){

      for(unsigned int i=0;i<keywords.size();i++){if(std::regex_match(dsetkeywords[ikey].toString(),std::regex(keywords[i])))return true;}

    }

  }else{

    return true;

  }

  return false;

}



//--------------

void resetNegativeBinsAndErrors( TH1* hp, double val_for_reset = 1., double min_error = 1. ) {

   if ( hp == 0x0 ) { printf("\n\n *** resetNegativeBinsAndErrors: null pointer.\n\n") ; gSystem -> Exit(-1) ; }

   for ( int hbi=1; hbi<= hp->GetNbinsX(); hbi++ ) {

      double val, err ;

      val = hp -> GetBinContent( hbi ) ;

      err = hp -> GetBinError( hbi ) ;

      if ( err < min_error ) {

         if ( verbose ) { printf("  resetNegativeBinsAndErrors : hist %s, bin %d, err = %.1f, reset err to %.1f\n", hp->GetName(), hbi, err, min_error ) ; }

         hp -> SetBinError( hbi, min_error ) ;

         err = min_error ;

      }

      if ( val <= 0 ) {

         if ( verbose ) {

            printf("  resetNegativeBinsAndErrors : hist %s, bin %d, val = %.1f, reset val to %.1f and err to %.1f.\n",

             hp->GetName(), hbi, val, val_for_reset, sqrt( pow( err, 2. ) + pow( val, 2. ) ) ) ;

         }

	 hp -> SetBinContent( hbi, val_for_reset ) ;

	 hp -> SetBinError( hbi, sqrt( pow( err, 2. ) + pow( val, 2. ) ) ) ;

	 if(fabs(val)>10.) {

           hp -> SetBinError( hbi, 1.0 ) ; 

	   printf("  resetNegativeBinsAndErrors : hist %s, bin %d, val = %.1f, reset val to %.1f and err to %.1f.\n", 

		  hp->GetName(), hbi, val, val_for_reset,1.0);

         }   

	 double max_error=0.5*fabs(val);   

	 if(val_for_reset==0.) { // for region B (QCD template) reset errors if weights too large

	   if ( (fabs(val)>=1) && (err > max_error) ) {   

	     printf(" REL ERR > 1: resetNegativeBinsAndErrors : hist %s, bin %d, val = %.1f, reset val to %.1f with err_old = %.1f. to err = %.1f.\n", 

		    hp->GetName(), hbi, val, val_for_reset, err, 1.0 ); //max_error );  

	     hp -> SetBinError( hbi, 1.0 ) ; 

	   }

	 }

      }

   }

} // resetNegativeBinsAndErrors






void filterBinContent(TH1* histo){

  if(shapeBinToConsider.size()<=0)return;

  for(int i=0;i<=histo->GetNbinsX()+1;i++){

    bool toBeConsidered=false;  for(unsigned int j=0;j<shapeBinToConsider.size();j++){if(shapeBinToConsider[j]==i){toBeConsidered=true;break;}}

    if(!toBeConsidered){histo->SetBinContent(i,0); histo->SetBinError(i,0);}

  }

}


//wrapper for a projected shape for a given proc

class ShapeData_t

{

  public:

        std::map<string, double> uncScale;

        std::map<string, TH1*  > uncShape;

        TH1* fit;

        ShapeData_t(){

          fit=NULL;

        }

        ~ShapeData_t(){}

        TH1* histo(){

        if(uncShape.find("")==uncShape.end())return NULL;

        return uncShape[""];

        }

        void clearSyst(){

        TH1* nominal = histo();

        uncScale.clear();

        uncShape.clear();

        uncShape[""] = nominal;

        }


        void removeStatUnc(){

        // OLD (wrong): erasing 'unc' and then doing unc-- uses an invalidated
        // iterator and can crash once the custom statistical shapes are active.
        // for(auto unc = uncShape.begin(); unc!= uncShape.end(); unc++){
        for(auto unc = uncShape.begin(); unc!= uncShape.end(); ){

        TString name = unc->first.c_str();

	//	if(name.Contains("stat") && (name.Contains("Up") || name.Contains("Down"))){  

	if(name.Contains("stat") && (name.EndsWith("Up") || name.EndsWith("Down"))){

	  // OLD (wrong):
	  // uncShape.erase(unc);
	  // unc--;
	  TH1* statShapeToDelete = unc->second;
	  auto statShapeEntryToErase = unc++;
	  uncShape.erase(statShapeEntryToErase);
	  delete statShapeToDelete;

	} else {
	  ++unc;
	}

        }

        }





      //----------------------------------------------------------------------------------

       void makeStatUnc(string prefix="", string suffix="", string suffix2="", bool noBinByBin=false, TH1* total_hist = 0x0 ){

	 if ( verbose ) { printf("\n --- verbose : makeStatUnc : begin.  prefix = %s, suffix = %s, suffix2 = %s\n", prefix.c_str(), suffix.c_str(), suffix2.c_str() ) ; fflush(stdout) ; }

       // Do not create custom template-statistics shapes in none mode or
       // when Combine autoMCStats is used.
       if (statUncMode == kStatNone || autoMCStats) {
	   if (verbose) {
	     printf(" --- verbose : makeStatUnc : mode %s; skipping manual statistical shapes.\n",
	            statUncModeName.Data());
	     fflush(stdout);
	   }
	   return;
	 }

	 if(!histo() || histo()->Integral()<=0)return;

	 string delimiter = "_";

	 unsigned firstDelimiter = suffix.find(delimiter);

	 unsigned lastDelimiter = suffix.find_last_of(delimiter);

	 unsigned endPosOfFirstDelimiter = firstDelimiter + delimiter.length();

	 string channel_and_bin = suffix.substr(endPosOfFirstDelimiter, lastDelimiter-endPosOfFirstDelimiter);

	 TH1* h = (TH1*) histo()->Clone("TMPFORSTAT");

	 //bin by bin stat uncertainty

	 if(statUncMode == kStatHybrid && statBinByBin>0 && shape==true && !noBinByBin){

	   // OLD (wrong): this counter labels the first selected physical bin as b0,
	   // which hides which histogram bin the nuisance belongs to.
	   // int BIN=0;

	   for(int ibin=1; ibin<=h->GetXaxis()->GetNbins(); ibin++){           

	     /////////////if( !(h->GetBinContent(ibin)<=0 && h->GetBinError(ibin)>0) &&  (h->GetBinContent(ibin)<=0 || h->GetBinContent(ibin)/h->Integral()<0.01 || h->GetBinError(ibin)/h->GetBinContent(ibin)<statBinByBin))continue;

	     /*if ( h->GetBinError(ibin)/h->GetBinContent(ibin)<statBinByBin ) continue;

	       if ( h->GetBinContent(ibin) <= 0 ) continue ;*/

	     const double content = h->GetBinContent(ibin);

	     const double error   = h->GetBinError(ibin);

	     if (content <= 0.0 || error <= 0.0) continue;

	     if (error / content < statBinByBin) continue;

	     if ( total_hist == 0x0 ) {

	       if ( verbose ) { printf("  *** verbose: makeStatUnc :  statBinByBin requested bu no total_hist.  Skipping statBinByBin.\n") ; fflush(stdout) ; }

	       continue ;

	     }

	     if ( total_hist -> GetNbinsX() != h -> GetNbinsX() ) {

	       printf("\n\n *** makeStatUnc : inconsistent histogram binnings:  %d for this hist, %d for total_hist.\n\n", h -> GetNbinsX(), total_hist -> GetNbinsX() ) ;

	       gSystem -> Exit(-1) ;

	     }

	     double Nbg = total_hist -> GetBinContent( ibin ) ;

	     double err = h->GetBinError( ibin ) ;

	     if ( verbose ) { printf("  --- verbose: makeStatUnc :  bin %d,  Nbg = %9.1f, err = %.1f.  ", ibin, Nbg, err ) ; fflush(stdout) ; }

	     if ( Nbg <= 0 ) {

	       if ( verbose ) { printf("\n") ; fflush(stdout) ; }

	       continue ;

	     }

	     if ( verbose ) { printf("  err / sqrt(Nbg) = %7.3f , threshold = %7.3f\n", (err / sqrt(Nbg) ), minErrOverSqrtNBGForBinByBin ) ; fflush(stdout) ; }

	     if ( (err / sqrt(Nbg) ) < minErrOverSqrtNBGForBinByBin ) continue ;



	     // char ibintxt[255]; sprintf(ibintxt, "_b%i", BIN);BIN++;

	     char ibintxt[255];

	     sprintf(ibintxt, "_b%i", ibin);

	     if ( verbose ) {

	       TString hname( TString(h->GetName())+"StatU"+ibintxt ) ;

	       printf(" --- verbose : makeStatUnc : making clones 3.  name = %s\n", hname.Data() ) ; fflush(stdout) ;

	     }

	     TH1* statU=(TH1 *)h->Clone(TString(h->GetName())+"StatU"+ibintxt);//  statU->Reset();

	     TH1* statD=(TH1 *)h->Clone(TString(h->GetName())+"StatD"+ibintxt);//  statD->Reset();           

	     if(h->GetBinContent(ibin)>0){

	       statU->SetBinContent(ibin,std::min(2*h->GetBinContent(ibin), std::max(0.01*h->GetBinContent(ibin), h->GetBinContent(ibin) + h->GetBinError(ibin))));   statU->SetBinError(ibin, 0.0);

	       statD->SetBinContent(ibin,std::min(2*h->GetBinContent(ibin), std::max(0.01*h->GetBinContent(ibin), h->GetBinContent(ibin) - h->GetBinError(ibin))));   statD->SetBinError(ibin, 0.0);

	     }else{

	       statU->SetBinContent(ibin,std::max(0.0, statU->GetBinContent(ibin) + statU->GetBinError(ibin)));

	       statD->SetBinContent(ibin,std::max(0.0, statD->GetBinContent(ibin) - statD->GetBinError(ibin)));

	     }

	     if ( verbose ) { printf(" --- verbose: makeStatUnc : setting uncShape[%s] to hist with name %s\n", (prefix+"stat"+suffix+ibintxt+suffix2+"Up").c_str(), statU -> GetName() ) ; fflush(stdout) ; }

	     if ( verbose ) { printf(" --- verbose: makeStatUnc : setting uncShape[%s] to hist with name %s\n", (prefix+"stat"+suffix+ibintxt+suffix2+"Down").c_str(), statD -> GetName() ) ; fflush(stdout) ; }

	     if ( verbose ) { printf(" --- verbose : makeStatUnc : 3 adding to uncShape with key %s\n", (prefix+"stat"+suffix+ibintxt+suffix2+"Up").c_str() ) ; fflush(stdout) ; }

	     uncShape[prefix+"stat"+suffix+ibintxt+suffix2+"Up"  ] = statU;

	     uncShape[prefix+"stat"+suffix+ibintxt+suffix2+"Down"] = statD;

	     /*h->SetBinContent(ibin, 0);*/  h->SetBinError(ibin, 0);  //remove this bin from shape variation for the other ones

	     //printf("%s --> %f - %f - %f\n", (prefix+"stat"+suffix+ibintxt+suffix2+"Up").c_str(), statD->Integral(), h->GetBinContent(ibin), statU->Integral() );

	   }

	 }

	 //after this line, all bins with large stat uncertainty have been considered separately

	 //so now it remains to consider all the other bins for which we assume a total correlation bin by bin

	 // OLD (wrong): the integral stays positive after per-bin nuisances are made,
	 // because only the bin errors are cleared, not the bin contents.
	 // if(h->Integral()<=0)return;

	 // Only make the residual process-correlated statistical nuisance when at
	 // least one bin still has an uncertainty that was not split per bin.
	 bool hasResidualStatUncertainty = false;
	 for (int ibin=1; ibin<=h->GetXaxis()->GetNbins(); ++ibin) {
	   if (h->GetBinContent(ibin) > 0.0 && h->GetBinError(ibin) > 0.0) {
	     hasResidualStatUncertainty = true;
	     break;
	   }
	 }
	 if (!hasResidualStatUncertainty) {
	   delete h;
	   return;
	 }


	 if ( verbose ) {

	   TString hname( TString(h->GetName())+"StatU" ) ;

	   printf(" --- verbose : makeStatUnc : making clones 4.  name = %s\n", hname.Data() ) ; fflush(stdout) ;

	 }

	 TH1* statU=(TH1 *)h->Clone(TString(h->GetName())+"StatU");

	 TH1* statD=(TH1 *)h->Clone(TString(h->GetName())+"StatD");

	 for(int ibin=1; ibin<=statU->GetXaxis()->GetNbins(); ibin++){

	   /*

	   if(h->GetBinContent(ibin)<0.1){

	     statU->SetBinContent(ibin,0.1); statU->SetBinError(ibin, 2.); //hmap_wgt[hname.Data()]);

	     statD->SetBinContent(ibin,0.1); statD->SetBinError(ibin, 0.1);

	   } 

	   */  

	   if(h->GetBinContent(ibin)>0){

	     statU->SetBinContent(ibin,std::min(2*h->GetBinContent(ibin), std::max(0.01*h->GetBinContent(ibin), statU->GetBinContent(ibin) + statU->GetBinError(ibin))));

	     statD->SetBinContent(ibin,std::min(2*h->GetBinContent(ibin), std::max(0.01*h->GetBinContent(ibin), statD->GetBinContent(ibin) - statD->GetBinError(ibin))));

	   }else{

	     //statU->SetBinContent(ibin,              statU->GetBinContent(ibin) + statU->GetBinError(ibin));

	     statU->SetBinContent(ibin,std::min(0.0, statU->GetBinContent(ibin) + statU->GetBinError(ibin)));

	     statD->SetBinContent(ibin,std::min(0.0, statD->GetBinContent(ibin) - statD->GetBinError(ibin)));

	   }

	 }


	 // Keep suffix2 in the residual key, exactly as is already done for the
	 // per-bin keys.  This prevents collisions between different years/postfixes.
	 const string residualStatKey = prefix + "stat" + suffix + suffix2;


	 if ( verbose ) { printf(" --- verbose : makeStatUnc : 4 adding to uncShape with key %s\n", (residualStatKey+"Up").c_str() ) ; fflush(stdout) ; }

           //-- owen: first check if this is already set.  If so, delete the existing one to avoid memory leak.


	 if ( uncShape.find( residualStatKey+"Up" ) != uncShape.end() ) {


	   if ( uncShape[residualStatKey+"Up"] != 0x0 ) {

	     if ( verbose ) { printf(" --- verbose : makeStatUnc : 4 deleting existing hist with name %s before assignment.\n", uncShape[residualStatKey+"Up"] -> GetName() ) ; fflush(stdout) ; }


	     delete uncShape[residualStatKey+"Up"];

	   }

	 }


	 if ( uncShape.find( residualStatKey+"Down" ) != uncShape.end() ) {

	 
	   if ( uncShape[residualStatKey+"Down"] != 0x0 ) {

	     if ( verbose ) { printf(" --- verbose : makeStatUnc : 4 deleting existing hist with name %s before assignment.\n", uncShape[residualStatKey+"Down"] -> GetName() ) ; fflush(stdout) ; }

	  
	     delete uncShape[residualStatKey+"Down"];

	   }

	 }
	 uncShape[residualStatKey+"Up"]   = statU;
	 uncShape[residualStatKey+"Down"] = statD;

	 delete h; //all done with this copy

	    //}

       } // makeStatUnc



      //----------------------------------------------------------------------------------





        double getScaleUncertainty(){

        double Total=0;

        TH1* h = NULL;

        double integral = 0;

        int start_bin = 1;

        if (this->histo()!=NULL){

          h = (TH1*)(this->histo()->Clone("nominal"));

          integral = h->Integral();

          if(docut) start_bin = h->FindBin(sstyCut);

        }

        for(std::map<string, double>::iterator unc=uncScale.begin();unc!=uncScale.end();unc++){

        if(unc->second<0)continue;

        double unc_val = unc->second;

        if(docut) {

          if(h!=NULL && integral>0) unc_val = unc_val/integral*h->Integral(start_bin, h->GetXaxis()->GetNbins());

          else unc_val = 0;

        }

        Total+=pow(unc_val,2);

//      std::cout << "scale Unc: " << unc->first << ", value: " << unc->second << std::endl;

        }

        return Total>0?sqrt(Total):-1;

        }

        double getIntegratedShapeUncertainty(string name, string upORdown){

        double Total=0;

        //this = ch->second.shapes[histoName.Data()]

        for(std::map<string, TH1*>::iterator var = uncShape.begin(); var!=uncShape.end(); var++){

        TString systName = var->first.c_str();

        if(var->first=="")continue; //Skip Nominal shape

        //if(!systName.Contains(upORdown))continue; //only look for syst up or down at a time (upORdown should be either "Up" or "Down", buggy code

        if(!systName.EndsWith(upORdown.c_str()))continue; //only look for syst up or down at a time (upORdown should be either "Up" or "Down"

        TH1* hvar = (TH1*)(var->second->Clone((name+var->first).c_str()));

        double varYield = 0.;

        int start_bin = 1;

        if(hvar) {

          if(docut) start_bin = hvar->FindBin(sstyCut);

          varYield = hvar->Integral(start_bin, hvar->GetXaxis()->GetNbins());

        }

        TH1* h = NULL;

        if (this->histo()!=NULL) h = (TH1*)(this->histo()->Clone((name+"Nominal").c_str()));

        double yield = 0.; 

        if (h!=NULL) {

          if(docut) start_bin = h->FindBin(sstyCut);

          yield = h->Integral(start_bin, h->GetXaxis()->GetNbins());

        }

        Total+=pow(varYield-yield,2); //the total shape unc is the sqrt of the quadratical sum of the difference between the nominal and the variated yields.

        }     

        return Total>0?sqrt(Total):-1;

        }

      //------------------------------------

        double getBinShapeUncertainty(string name, int bin, string upORdown){

	  double Total=0;

	  //this = ch->second.shapes[histoName.Data()]

	  for(std::map<string, TH1*>::iterator var = uncShape.begin(); var!=uncShape.end(); var++){

	    TString systName = var->first.c_str();

	    if(var->first=="") continue; //Skip Nominal shape

	    //if(!systName.Contains(upORdown))continue; //only look for syst up or down at a time (upORdown should be either "Up" or "Down"

	    if(!systName.EndsWith(upORdown.c_str()))continue; //only look for syst up or down at a time (upORdown should be either "Up" or "Down"


            //--- no cloning

	    double varYield = var->second->GetBinContent(bin);

	    double yield = this->histo()->GetBinContent(bin);

	    //	      if(systName.Contains("dydR") && name.find("ee_A_CR_3b")) {

	    if ( verbose ) {

	      printf("--- verbose : getBinShapeUncertainty : name = %s , bin = %d , upORdown = %s , hvar clone name = %s , h clone name = %s, varYield = %.1f, yield = %.1f, diff = %.1f \n",

		     name.c_str(), bin, upORdown.c_str(),

		     (name+var->first).c_str(),

		     (name+"Nominal").c_str(),

		     varYield, yield, (varYield-yield)

		     );

	      fflush(stdout) ;

	    }

	    Total+=pow(varYield-yield,2); 

	    //the total shape unc is the sqrt of the quadratical sum of the difference between the nominal and the variated yields.

	  } // var loop     

	  return Total>0?sqrt(Total):-1;

        } // getBinShapeUncertainty

      //------------------------------------

        void rescaleScaleUncertainties(double StartIntegral, double EndIntegral){

        for(std::map<string, double>::iterator unc=uncScale.begin();unc!=uncScale.end();unc++){

        printf("%E/%E = %E = %E/%E\n", unc->second, StartIntegral, unc->second/StartIntegral, EndIntegral * (unc->second/StartIntegral), EndIntegral); 

        if(StartIntegral!=0){unc->second = EndIntegral * (unc->second/StartIntegral);}else{unc->second = unc->second * EndIntegral;}

        }     

        }
};

class ChannelInfo_t

{

  public:

    string bin;

    string channel;

  std::map<string, ShapeData_t> shapes;

  ChannelInfo_t(){}

  ~ChannelInfo_t(){}

};

class ProcessInfo_t

{

  public:

        bool isData;

        bool isBckg;

        bool isSign;

        double xsec;

        double br;

        double mass;

        string shortName;

  //std::map<string, double> hmap_wgt; // map with (average) event weights for each process

        std::map<string, ChannelInfo_t> channels;

        JSONWrapper::Object jsonObj;

        ProcessInfo_t(){xsec=0;isSign=false;

	}

        ~ProcessInfo_t(){}

        void printProcess() {

           for ( std::map<string, ChannelInfo_t>::iterator ic = channels.begin(); ic!= channels.end(); ic++ ) {

              string chan_key = ic -> first ;

              ChannelInfo_t chan = ic -> second ;

              // printf("       proc key = %s , chan key = %s ,  bin = %s , channel = %s\n", proc_key.c_str(), chan_key.c_str(), chan.bin.c_str(), chan.channel.c_str() ) ;

              for ( std::map<string, ShapeData_t>::iterator is = chan.shapes.begin(); is!= chan.shapes.end(); is++ ) {

                 string shape_key = is -> first ;

                 ShapeData_t shape = is -> second ;

                 //printf("            proc key = %s , chan key = %s , shape key = %s , hist pointer = %p, uncScale has %lu entries, uncShape has %lu entries\n",

                 //  proc_key.c_str(), chan_key.c_str(), shape_key.c_str(), shape.histo(), shape.uncScale.size(), shape.uncShape.size() ) ;

                 int nshapesyst = shape.uncShape.size()-1 ;

                 if ( nshapesyst < 0 ) nshapesyst = 0 ;

                 printf("    printProcess:  %15s :  %18s  : N syst, scale = %2lu, shape = %2d : hist ",  shortName.c_str(), chan_key.c_str(), shape.uncScale.size(), nshapesyst ) ;

                 TH1* hp = shape.histo() ;

                 if ( hp != 0x0 ) {

                    //printf("                  hist name = %s, n bins = %d\n", hp -> GetName(), hp -> GetNbinsX() ) ;

                    printf(" %2d bins | ", hp -> GetNbinsX() ) ;

                    bool is_data_sr = false ;

                    if ( shortName == "data" && chan_key.find("SR")!=string::npos && chan_key.find("emu")==string::npos ) {

                       is_data_sr = true ;

                    }

                    if ( !is_data_sr ) {

                       printf(" entries = %9.1f integral = %9.1f | ", hp -> GetEntries(), hp -> Integral() ) ;

                    } else {

                       printf(" entries = *******.* integral = *******.* | " ) ;

                    }

                    if ( hp -> GetNbinsX() < 10 ) {

                       for ( int bi=1; bi<= hp -> GetNbinsX(); bi++ ) {

                          if ( !(is_data_sr && bi >= 4) ) {

			    printf(" b%d %9.1f +/- %6.1f |", bi, hp -> GetBinContent( bi ), hp -> GetBinError ( bi) ) ;

                          } else {

                             printf(" b%d *******.* |", bi ) ;

                          }

                       } // bi

                    }

                 } else {

                    printf(" *** no histogram ***" ) ;

                 }

                 printf("\n") ;

              } // is

           } // ic

        } // printProcess

};

class AllInfo_t

{

  public:

    std::map<string, ProcessInfo_t> procs;

    std::vector<string> sorted_procs;

                AllInfo_t(){};

                ~AllInfo_t(){};

    // reorder the procs to get the backgrounds; total bckg, signal, data 

    void sortProc();

    // Sum up all background processes and add this as a total process

    //////void addChannel(ChannelInfo_t& dest, ChannelInfo_t& src, bool computeSyst = false, bool addDiffProcs = true);

    void addChannel(ChannelInfo_t& dest, ChannelInfo_t& src, bool computeSyst = false, bool addDiffProcs = true, double scale_factor = 1.);

    // Sum up all background processes and add this as a total process

    void addProc(ProcessInfo_t& dest, ProcessInfo_t& src, bool computeSyst = false);

    void addProc(ProcessInfo_t& dest, ProcessInfo_t& src, bool computeSyst, double scale_factor);

   // void addProc(ProcessInfo_t& dest, ProcessInfo_t& src, bool computeSyst = false, double scale_factor_e = 1., double scale_factor_mu = 1. );

    //void AllInfo_t::addProc(ProcessInfo_t& dest, ProcessInfo_t& src, bool computeSyst, double scale_factor=1 );

    // Sum up all background processes and add this as a total process

    void computeTotalBackground();

    // Replace the Data process by TotalBackground

    void blind();

    // Replace high sensitivity SR bins for data with total BG.

    void replaceHighSensitivityBinsWithBG();

    // Print the Yield table

    void getYieldsFromShape(FILE* pFile, std::vector<TString>& selCh, string histoName, FILE* pFileInc=NULL);

    // Dump efficiencies

    void getEffFromShape(FILE* pFile, std::vector<TString>& selCh, string histoName);

    // drop background process that have a negligible yield

    void dropSmallBckgProc(std::vector<TString>& selCh, string histoName, double threshold);

    // drop control channels

    void dropCtrlChannels(std::vector<TString>& selCh);

   // Subtract nonQCD MC processes from A,C,D regions in data

    void doBackgroundSubtraction(FILE* pFile, std::vector<TString>& selCh,TString mainHisto, AllInfo_t* sumAllInfo=0x0 );

    // Make a summary plot

    void showShape(std::vector<TString>& selCh , TString histoName, TString SaveName);

    //void showDDQCDValidation(std::vector<TString>& selCh, TString histoName, TString SaveName);

    // Make a summary plot of the uncertainties

    void showUncertainty(std::vector<TString>& selCh , TString histoName, TString SaveName);

    // Turn to cut&count (rebin all histo to 1 bin only)

    void turnToCC(string histoName);

    // Make a summary plot

    void saveHistoForLimit(string histoName, TFile* fout);

    // Add hardcoded uncertainties 

    void addHardCodedUncertainties(string histoName);

    // produce the datacards 

    void buildDataCards(string histoName, TString url);

    // Load histograms from root file and json to memory

  void getShapeFromFile(TFile* inF, std::vector<string> channelsAndShapes, int cutBin, JSONWrapper::Object &Root,  double minCut=0, double maxCut=9999, bool onlyData=false);

    // Rebin histograms to make sure that high mt/met region have no empty bins

    void rebinMainHisto(string histoName);

    //Merge bins together

    void mergeBins(std::vector<string>& binsToMerge, string NewName);

    // Handle empty bins

    void HandleEmptyBins(string histoName);

    // Dump to understand data organization

    void printInventory();

};

// ============================================================================

// Standalone 0-lepton DD-QCD validation.

// Does NOT modify procs and does NOT build datacards.

//

// It computes:

//   SR:     A_pred     = B * C / D

//   Val:    Astar_pred = Bstar * C / D

//   MC:     QCD closure in A and Astar

//   syst:   closureRelUnc = max(|1 - MC_A_pred/MC_A_obs|,

//                               |1 - Astar_pred/Astar_obs|)

//   syst:   nonQCDsubRelUnc from varying subtracted non-QCD MC

//

// It also plots:

//   A SR data vs nonQCD + DDQCD

//   Astar data vs nonQCD + DDQCD prediction

// ============================================================================

struct DDResult_t {

  double B = 0.0;

  double C = 0.0;

  double D = 0.0;

  double Berr = 0.0;

  double Cerr = 0.0;

  double Derr = 0.0;

  double alpha = 0.0;

  double alphaErr = 0.0;

  double pred = 0.0;

  double predErr = 0.0;

  TH1* hPred = 0x0;

};

TH1* getHistFromProc(

  ProcessInfo_t& proc,

  TString channelKey,

  TString histoName

) {

  auto ic = proc.channels.find(channelKey.Data());

  if (ic == proc.channels.end()) return 0x0;

  auto is = ic->second.shapes.find(histoName.Data());

  if (is == ic->second.shapes.end()) return 0x0;

  return is->second.histo();

}

TH1* getHistFromAllInfo(

  AllInfo_t& info,

  TString procName,

  TString channelKey,

  TString histoName

) {

  auto ip = info.procs.find(procName.Data());

  if (ip == info.procs.end()) return 0x0;

  return getHistFromProc(ip->second, channelKey, histoName);

}

double nonQCDScaleUncForProcess(TString procName, TString shortName) {

  procName.ToLower();

  shortName.ToLower();

  if (procName.Contains("t#bar{t} + b#bar{b}") || shortName.Contains("ttbarbba")) return 0.0;

  if (procName.Contains("t#bar{t} + c#bar{c}") || shortName.Contains("ttbarcba")) return 0.50;

  if (procName.Contains("t#bar{t} + light")    || shortName.Contains("ttbarlig")) return 0.06;

  if (procName.Contains("z#rightarrow") || shortName.Contains("znunu"))  return 0.02;

      //if (procName.Contains("w#rightarrow") || shortName.Contains("wlnu")) return 0.20;

  if (procName.Contains("other") || shortName.Contains("otherbkg")) return 0.50;

  return 0.30;

}

bool isNonQCDProcess(TString procName, TString shortName) {

  TString p = procName;

  TString s = shortName;

  p.ToLower();

  s.ToLower();

  if (p.Contains("qcd") || s.Contains("qcd")) return false;

  if (p.Contains("other")) return true;

  if (p.Contains("z#rightarrow")) return true;

  //if (p.Contains("w#rightarrow")) return true;

  if (p.Contains("t#bar{t}")) return true;

  return false;

}

bool isQCDProcess(TString procName, TString shortName) {

  TString p = procName;

  TString s = shortName;

  p.ToLower();

  s.ToLower();

  return (p.Contains("qcd") || s.Contains("qcd"));

}

ProcessInfo_t buildNonQCDProc(

  AllInfo_t& info,

  int variationSign

) {

  ProcessInfo_t out;

  out.shortName = "nonqcd";

  out.isData = true;

  out.isSign = false;

  out.isBckg = true;

  out.xsec = 0.0;

  out.br = 1.0;

  for (auto it = info.procs.begin(); it != info.procs.end(); ++it) {

    if (it->second.isData) continue;

    if (it->second.isSign) continue;

    if (it->first == "total") continue;

    TString procName = it->first.c_str();

    TString shortName = it->second.shortName.c_str();

    if (!isNonQCDProcess(procName, shortName)) continue;

    double scale = 1.0;

    if (variationSign != 0) {

      double unc = nonQCDScaleUncForProcess(procName, shortName);

      scale = 1.0 + variationSign * unc;

      if (scale < 0.0) scale = 0.0;

    }

    printf("NonQCD sum: %-15s  scale = %.3f  process = %s\n",

           shortName.Data(), scale, procName.Data());

    info.addProc(out, it->second, false, scale);

  }

  return out;

}

ProcessInfo_t buildQCDMCProc(AllInfo_t& info) {

  ProcessInfo_t out;

  out.shortName = "qcdmc";

  out.isData = false;

  out.isSign = false;

  out.isBckg = true;

  out.xsec = 0.0;

  out.br = 1.0;

  for (auto it = info.procs.begin(); it != info.procs.end(); ++it) {

    if (it->second.isData) continue;

    if (it->second.isSign) continue;

    if (it->first == "total") continue;

    TString procName = it->first.c_str();

    TString shortName = it->second.shortName.c_str();

    if (!isQCDProcess(procName, shortName)) continue;

    printf("QCD MC sum: %-15s  process = %s\n",

           shortName.Data(), procName.Data());

    info.addProc(out, it->second, false);

  }

  return out;

}

ProcessInfo_t buildSignalProc(AllInfo_t& info) {

  ProcessInfo_t out;

  out.shortName = "signal";

  out.isData = false;

  out.isSign = true;

  out.isBckg = false;

  out.xsec = 0.0;

  out.br = 1.0;

  for (auto it = info.procs.begin(); it != info.procs.end(); ++it) {

    if (!it->second.isSign) continue;

    info.addProc(out, it->second, false);

  }

  return out;

}

TH1* makeDataMinusNonQCD(

  AllInfo_t& info,

  ProcessInfo_t& nonQCD,

  TString channelKey,

  TString histoName,

  double resetValue

) {

  TH1* hData = getHistFromAllInfo(info, "data", channelKey, histoName);

  if (!hData) {

    printf("\n *** Missing data histogram: %s / %s\n",

           channelKey.Data(), histoName.Data());

    return 0x0;

  }

  TH1* hOut = (TH1*) hData->Clone("dataMinusNonQCD_" + channelKey);

  hOut->SetDirectory(0);

  TH1* hNonQCD = getHistFromProc(nonQCD, channelKey, histoName);

  if (hNonQCD) {

    hOut->Add(hNonQCD, -1.0);

  } else {

    printf("Warning: no NonQCD hist for %s. Using raw data.\n",

           channelKey.Data());

  }

  resetNegativeBinsAndErrors(hOut, resetValue);

  return hOut;

}

double integralAndError(TH1* h, double& err) {

  err = 0.0;

  if (!h) return 0.0;

  return h->IntegralAndError(1, h->GetNbinsX(), err);

}

DDResult_t computeABCDPrediction(

  TH1* hB,

  TH1* hC,

  TH1* hD,

  TString predName

) {

  DDResult_t r;

  if (!hB || !hC || !hD) {

    printf("\n *** computeABCDPrediction: missing B/C/D histogram.\n");

    return r;

  }

  // if (r.D <= 0.0 || r.C <= 0.0 || r.B <= 0.0) {

  //printf("Invalid ABCD input: B=%.3f C=%.3f D=%.3f\n", r.B, r.C, r.D);

  //return r;

  //}

  r.B = integralAndError(hB, r.Berr);

  r.C = integralAndError(hC, r.Cerr);

  r.D = integralAndError(hD, r.Derr);

  if (r.D <= 0.0 || r.C <= 0.0 || r.B <= 0.0) {

    printf("Invalid ABCD input: B=%.3f C=%.3f D=%.3f\n", r.B, r.C, r.D);

    return r;

  }

  if (r.C > 0.0 && r.D > 0.0) {

    r.alpha = r.C / r.D;

    r.alphaErr = r.alpha * sqrt(

      pow(r.Cerr / r.C, 2) +

      pow(r.Derr / r.D, 2)

    );

  }

  r.hPred = (TH1*) hB->Clone(predName);

  r.hPred->SetDirectory(0);

  r.hPred->Scale(r.alpha);

  r.pred = integralAndError(r.hPred, r.predErr);

  if (r.B > 0.0 && r.C > 0.0 && r.D > 0.0) {

    r.predErr = r.pred * sqrt(

      pow(r.Berr / r.B, 2) +

      pow(r.Cerr / r.C, 2) +

      pow(r.Derr / r.D, 2)

    );

  }

  return r;

}

void printDDResult(TString label, DDResult_t& r) {

  printf("\n------------------------------------------------------------\n");

  printf("%s\n", label.Data());

  printf("B          = %.4f +/- %.4f\n", r.B, r.Berr);

  printf("C          = %.4f +/- %.4f\n", r.C, r.Cerr);

  printf("D          = %.4f +/- %.4f\n", r.D, r.Derr);

  printf("alpha C/D  = %.6f +/- %.6f\n", r.alpha, r.alphaErr);

  printf("prediction = %.4f +/- %.4f\n", r.pred, r.predErr);

  printf("------------------------------------------------------------\n");

}

void plotDataVsPrediction0lep(

  TH1* hData,

  TH1* hNonQCD,

  TH1* hDDQCD,

  TString outName,

  TString title

) {

  if (!hData || !hNonQCD || !hDDQCD) {

    printf("\n *** plotDataVsPrediction0lep: missing input for %s\n",

           outName.Data());

    printf("     hData   = %p\n", hData);

    printf("     hNonQCD = %p\n", hNonQCD);

    printf("     hDDQCD  = %p\n", hDDQCD);

    return;

  }

  TH1* hTotal = (TH1*) hNonQCD->Clone("totalPred_" + outName);

  hTotal->SetDirectory(0);

  hTotal->Add(hDDQCD);

  TH1* hRatio = (TH1*) hData->Clone("ratio_" + outName);

  hRatio->SetDirectory(0);

  hRatio->Divide(hTotal);

  hData->SetMarkerStyle(20);

  hData->SetMarkerSize(0.8);

  hData->SetLineColor(kBlack);

  hNonQCD->SetFillColor(kAzure - 9);

  hNonQCD->SetLineColor(kBlack);

  hDDQCD->SetFillColor(kOrange - 2);

  hDDQCD->SetLineColor(kBlack);

  TCanvas* c = new TCanvas("c_" + outName, "c_" + outName, 800, 800);

  TPad* pad1 = new TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0);

  TPad* pad2 = new TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30);

  pad1->SetBottomMargin(0.03);

  pad2->SetTopMargin(0.04);

  pad2->SetBottomMargin(0.30);

  pad1->Draw();

  pad2->Draw();

  pad1->cd();

  pad1->SetLogy(true);

  THStack* st = new THStack("stack_" + outName, title);

  st->Add(hDDQCD);

  st->Add(hNonQCD);

  st->Draw("hist");

  st->GetYaxis()->SetTitle("Events");

  double ymax = std::max(st->GetMaximum(), hData->GetMaximum());

  st->SetMaximum(std::max(1.0, ymax) * 20.0);

  st->SetMinimum(1e-2);

  hData->Draw("E same");

  TLegend* leg = new TLegend(0.55, 0.70, 0.88, 0.88);

  leg->SetBorderSize(0);

  leg->SetFillStyle(0);

  leg->AddEntry(hData, "Data", "lep");

  leg->AddEntry(hNonQCD, "Non-QCD MC", "f");

  leg->AddEntry(hDDQCD, "DD QCD", "f");

  leg->Draw();

  pad2->cd();

  hRatio->SetTitle("");

  hRatio->GetYaxis()->SetTitle("Data / pred.");

  hRatio->GetYaxis()->SetRangeUser(0.0, 2.0);

  hRatio->GetYaxis()->SetNdivisions(505);

  hRatio->GetYaxis()->SetTitleSize(0.09);

  hRatio->GetYaxis()->SetTitleOffset(0.55);

  hRatio->GetYaxis()->SetLabelSize(0.08);

  hRatio->GetXaxis()->SetTitleSize(0.10);

  hRatio->GetXaxis()->SetLabelSize(0.08);

  hRatio->Draw("E");

  TLine* line = new TLine(

    hRatio->GetXaxis()->GetXmin(),

    1.0,

    hRatio->GetXaxis()->GetXmax(),

    1.0

  );

  line->SetLineColor(kRed);

  line->SetLineWidth(2);

  line->Draw("same");

  c->SaveAs(outName + ".pdf");

  c->SaveAs(outName + ".root");

  delete c;

}

double computeNonQCDSubtractionUncertainty(

  AllInfo_t& info,

  TString histoName,

  TString keyB,

  TString keyC,

  TString keyD,

  double nominalPred

) {

  if (nominalPred <= 0.0) return 0.0;

  ProcessInfo_t nonQCDUp   = buildNonQCDProc(info, +1);

  ProcessInfo_t nonQCDDown = buildNonQCDProc(info, -1);

  TH1* hB_up = makeDataMinusNonQCD(info, nonQCDUp, keyB, histoName, 0.0);

  TH1* hC_up = makeDataMinusNonQCD(info, nonQCDUp, keyC, histoName, 1.0);

  TH1* hD_up = makeDataMinusNonQCD(info, nonQCDUp, keyD, histoName, 1.0);

  TH1* hB_dn = makeDataMinusNonQCD(info, nonQCDDown, keyB, histoName, 0.0);

  TH1* hC_dn = makeDataMinusNonQCD(info, nonQCDDown, keyC, histoName, 1.0);

  TH1* hD_dn = makeDataMinusNonQCD(info, nonQCDDown, keyD, histoName, 1.0);

  DDResult_t up = computeABCDPrediction(hB_up, hC_up, hD_up, "nonQCDsubUp");

  DDResult_t dn = computeABCDPrediction(hB_dn, hC_dn, hD_dn, "nonQCDsubDown");

  double diffUp = fabs(up.pred - nominalPred);

  double diffDn = fabs(dn.pred - nominalPred);

  double rel = std::max(diffUp, diffDn) / nominalPred;

  printf("\n------------------------------------------------------------\n");

  printf("NonQCD subtraction uncertainty\n");

  printf("Nominal A pred = %.4f\n", nominalPred);

  printf("NonQCD up pred = %.4f, diff = %.4f\n", up.pred, diffUp);

  printf("NonQCD dn pred = %.4f, diff = %.4f\n", dn.pred, diffDn);

  printf("Relative uncertainty = %.4f\n", rel);

  printf("------------------------------------------------------------\n");

  return rel;

}

void printSignalContamination(

  ProcessInfo_t& signalProc,

  TString histoName,

  TString keyB,

  TString keyC,

  TString keyD,

  DDResult_t& nominal

) {

  TH1* hSB = getHistFromProc(signalProc, keyB, histoName);

  TH1* hSC = getHistFromProc(signalProc, keyC, histoName);

  TH1* hSD = getHistFromProc(signalProc, keyD, histoName);

  double eSB = 0.0;

  double eSC = 0.0;

  double eSD = 0.0;

  double SB = integralAndError(hSB, eSB);

  double SC = integralAndError(hSC, eSC);

  double SD = integralAndError(hSD, eSD);

  double predWithSig = 0.0;

  if ((nominal.D + SD) > 0.0) {

    predWithSig = (nominal.B + SB) * (nominal.C + SC) / (nominal.D + SD);

  }

  double relBias = 0.0;

  if (nominal.pred > 0.0) {

    relBias = (predWithSig - nominal.pred) / nominal.pred;

  }

  printf("\n------------------------------------------------------------\n");

  printf("Signal contamination check in B,C,D\n");

  printf("S_B = %.4f, S_C = %.4f, S_D = %.4f\n", SB, SC, SD);

  printf("Nominal pred       = %.4f\n", nominal.pred);

  printf("Pred with signal CR = %.4f\n", predWithSig);

  printf("Relative bias       = %.4f\n", relBias);

  printf("------------------------------------------------------------\n");

}

void runDDQCDValidation0lep(

  AllInfo_t& info,

  TString histoName,

  TString binName = "3b",

  TString outDir = "DDQCD_validation"

) {

  gSystem->mkdir(outDir, true);

  TString y = year;

  TString keyA     = "veto_A_SR_"     + binName + y;

  TString keyB     = "veto_B_SR_"     + binName + y;

  TString keyC     = "veto_C_SR_"     + binName + y;

  TString keyD     = "veto_D_SR_"     + binName + y;

  TString keyAstar = "veto_Astar_SR_" + binName + y;

  TString keyBstar = "veto_Bstar_SR_" + binName + y;

  printf("\n\n============================================================\n");

  printf("Running standalone 0-lepton DD-QCD validation\n");

  printf("histo = %s\n", histoName.Data());

  printf("bin   = %s\n", binName.Data());

  printf("year  = %s\n", y.Data());

  printf("A     = %s\n", keyA.Data());

  printf("B     = %s\n", keyB.Data());

  printf("C     = %s\n", keyC.Data());

  printf("D     = %s\n", keyD.Data());

  printf("Astar = %s\n", keyAstar.Data());

  printf("Bstar = %s\n", keyBstar.Data());

  printf("============================================================\n\n");

  ProcessInfo_t nonQCD = buildNonQCDProc(info, 0);

  ProcessInfo_t qcdMC  = buildQCDMCProc(info);

  ProcessInfo_t signal = buildSignalProc(info);

  // --------------------------------------------------------------------------

  // SR prediction: A = B*C/D in data after non-QCD subtraction

  // --------------------------------------------------------------------------

  TH1* hB = makeDataMinusNonQCD(info, nonQCD, keyB, histoName, 0.0);

  TH1* hC = makeDataMinusNonQCD(info, nonQCD, keyC, histoName, 0.0);

  TH1* hD = makeDataMinusNonQCD(info, nonQCD, keyD, histoName, 0.0);

  DDResult_t sr = computeABCDPrediction(hB, hC, hD, "ddqcd_A_SR_pred");

  printDDResult("SR DD-QCD prediction: A = B*C/D", sr);

  // --------------------------------------------------------------------------

  // Astar validation in data: Astar_pred = Bstar*C/D

  // --------------------------------------------------------------------------

  TH1* hAstarObs = makeDataMinusNonQCD(info, nonQCD, keyAstar, histoName, 0.0);

  TH1* hBstar    = makeDataMinusNonQCD(info, nonQCD, keyBstar, histoName, 0.0);

  DDResult_t astar;

  double AstarObs = 0.0;

  double AstarObsErr = 0.0;

  double dataClosure = 0.0;

  double dataClosureErr = 0.0;

  double dataClosureRelUnc = 0.0;

  if (hAstarObs && hBstar && hC && hD) {

    astar = computeABCDPrediction(hBstar, hC, hD, "ddqcd_Astar_pred");

    printDDResult("Validation prediction: Astar = Bstar*C/D", astar);

    AstarObs = integralAndError(hAstarObs, AstarObsErr);

    if (AstarObs > 0.0 && astar.pred > 0.0) {

      dataClosure = astar.pred / AstarObs;

      dataClosureErr = dataClosure * sqrt(

        pow(astar.predErr / astar.pred, 2) +

        pow(AstarObsErr / AstarObs, 2)

      );

      dataClosureRelUnc = fabs(1.0 - dataClosure);

    }

    printf("\n------------------------------------------------------------\n");

    printf("Astar data closure\n");

    printf("Astar pred      = %.4f +/- %.4f\n", astar.pred, astar.predErr);

    printf("Astar obs       = %.4f +/- %.4f\n", AstarObs, AstarObsErr);

    printf("pred/obs        = %.4f +/- %.4f\n", dataClosure, dataClosureErr);

    printf("|1 - pred/obs|  = %.4f\n", dataClosureRelUnc);

    printf("------------------------------------------------------------\n");

  } else {

    printf("\n *** Missing Astar/Bstar inputs. Skipping Astar validation.\n");

  }

  // --------------------------------------------------------------------------

  // QCD MC closure in SR A

  // --------------------------------------------------------------------------

  TH1* hQCD_A = getHistFromProc(qcdMC, keyA, histoName);

  TH1* hQCD_B = getHistFromProc(qcdMC, keyB, histoName);

  TH1* hQCD_C = getHistFromProc(qcdMC, keyC, histoName);

  TH1* hQCD_D = getHistFromProc(qcdMC, keyD, histoName);

  DDResult_t mcSR;

  double qcdAObs = 0.0;

  double qcdAObsErr = 0.0;

  double mcClosureSR = 0.0;

  double mcClosureSRErr = 0.0;

  double mcClosureSRRelUnc = 0.0;

  if (hQCD_A && hQCD_B && hQCD_C && hQCD_D) {

    mcSR = computeABCDPrediction(hQCD_B, hQCD_C, hQCD_D, "qcdMC_A_pred");

    printDDResult("QCD MC SR closure prediction: A_MC = B_MC*C_MC/D_MC", mcSR);

    qcdAObs = integralAndError(hQCD_A, qcdAObsErr);

    if (qcdAObs > 0.0 && mcSR.pred > 0.0) {

      mcClosureSR = mcSR.pred / qcdAObs;

      mcClosureSRErr = mcClosureSR * sqrt(

        pow(mcSR.predErr / mcSR.pred, 2) +

        pow(qcdAObsErr / qcdAObs, 2)

      );

      mcClosureSRRelUnc = fabs(1.0 - mcClosureSR);

    }

    printf("\n------------------------------------------------------------\n");

    printf("QCD MC SR closure\n");

    printf("QCD A pred      = %.4f +/- %.4f\n", mcSR.pred, mcSR.predErr);

    printf("QCD A obs       = %.4f +/- %.4f\n", qcdAObs, qcdAObsErr);

    printf("pred/obs        = %.4f +/- %.4f\n", mcClosureSR, mcClosureSRErr);

    printf("|1 - pred/obs|  = %.4f\n", mcClosureSRRelUnc);

    printf("------------------------------------------------------------\n");

  } else {

    printf("\n *** Missing QCD MC A/B/C/D histograms. Skipping MC SR closure.\n");

  }

  // --------------------------------------------------------------------------

  // QCD MC Astar closure

  // --------------------------------------------------------------------------

  TH1* hQCD_Astar = getHistFromProc(qcdMC, keyAstar, histoName);

  TH1* hQCD_Bstar = getHistFromProc(qcdMC, keyBstar, histoName);

  double mcClosureAstarRelUnc = 0.0;

  if (hQCD_Astar && hQCD_Bstar && hQCD_C && hQCD_D) {

    DDResult_t mcAstar = computeABCDPrediction(

      hQCD_Bstar,

      hQCD_C,

      hQCD_D,

      "qcdMC_Astar_pred"

    );

    double qcdAstarObsErr = 0.0;

    double qcdAstarObs = integralAndError(hQCD_Astar, qcdAstarObsErr);

    double mcClosureAstar = 0.0;

    double mcClosureAstarErr = 0.0;

    if (qcdAstarObs > 0.0 && mcAstar.pred > 0.0) {

      mcClosureAstar = mcAstar.pred / qcdAstarObs;

      mcClosureAstarErr = mcClosureAstar * sqrt(

        pow(mcAstar.predErr / mcAstar.pred, 2) +

        pow(qcdAstarObsErr / qcdAstarObs, 2)

      );

      mcClosureAstarRelUnc = fabs(1.0 - mcClosureAstar);

    }

    printf("\n------------------------------------------------------------\n");

    printf("QCD MC Astar closure\n");

    printf("QCD Astar pred  = %.4f +/- %.4f\n", mcAstar.pred, mcAstar.predErr);

    printf("QCD Astar obs   = %.4f +/- %.4f\n", qcdAstarObs, qcdAstarObsErr);

    printf("pred/obs        = %.4f +/- %.4f\n", mcClosureAstar, mcClosureAstarErr);

    printf("|1 - pred/obs|  = %.4f\n", mcClosureAstarRelUnc);

    printf("------------------------------------------------------------\n");

  }

  // --------------------------------------------------------------------------

  // Non-QCD subtraction uncertainty

  // --------------------------------------------------------------------------

  double nonQCDSubRelUnc = computeNonQCDSubtractionUncertainty(

    info,

    histoName,

    keyB,

    keyC,

    keyD,

    sr.pred

  );

  // --------------------------------------------------------------------------

  // Signal contamination diagnostic

  // --------------------------------------------------------------------------

  printSignalContamination(signal, histoName, keyB, keyC, keyD, sr);

  // --------------------------------------------------------------------------

  // Recommended closure uncertainty to apply to SR ddqcd

  // --------------------------------------------------------------------------

  double closureRelUnc = std::max(mcClosureSRRelUnc, dataClosureRelUnc);

  printf("\n============================================================\n");

  printf("Recommended DD-QCD uncertainties for SR datacard\n");

  printf("SR ddqcd yield                    = %.4f\n", sr.pred);

  printf("TF stat relative uncertainty       = %.4f\n",

         (sr.pred > 0.0 ? sr.predErr / sr.pred : 0.0));

  printf("MC SR closure relative uncertainty = %.4f\n", mcClosureSRRelUnc);

  printf("Data Astar closure rel. unc.       = %.4f\n", dataClosureRelUnc);

  printf("Chosen closureRelUnc=max(MC_A,data_Astar) = %.4f\n", closureRelUnc);

  printf("NonQCD subtraction rel. unc.       = %.4f\n", nonQCDSubRelUnc);

  printf("\nFor datacard implementation:\n");

  printf("CMS_haa4b_sys_ddqcd_closure   = valDD * %.4f\n", closureRelUnc);

  printf("CMS_haa4b_sys_ddqcd_nonQCDsub = valDD * %.4f\n", nonQCDSubRelUnc);

  printf("============================================================\n\n");

  // --------------------------------------------------------------------------

  // Plots

  // --------------------------------------------------------------------------

  TH1* hDataA = getHistFromAllInfo(info, "data", keyA, histoName);

  TH1* hNonQCDA = getHistFromProc(nonQCD, keyA, histoName);

  if (hDataA && hNonQCDA && sr.hPred) {

    plotDataVsPrediction0lep(

      hDataA,

      hNonQCDA,

      sr.hPred,

      outDir + "/A_SR_after_DDQCD_" + binName,

      "A SR after DD-QCD"

    );

  }

  TH1* hDataAstar = getHistFromAllInfo(info, "data", keyAstar, histoName);

  TH1* hNonQCDAstar = getHistFromProc(nonQCD, keyAstar, histoName);

  if (hDataAstar && hNonQCDAstar && astar.hPred) {

    plotDataVsPrediction0lep(

      hDataAstar,

      hNonQCDAstar,

      astar.hPred,

      outDir + "/Astar_validation_" + binName,

      "A* validation after DD-QCD"

    );

  }

}

void printHelp();

void printHelp()

{

  printf("Options\n");

  printf("--verbose   --> turn on a lot of extra printing\n");

  printf("--autoMCStats   --> use Combine implementation of bin-by-bin stat errors on background histograms.  Will turn of statBinByBin.\n");

  printf("--statUncMode <none|correlated|hybrid> --> custom template-statistics model (default: correlated)\n");

  printf("--replaceHighSensitivityBinsWithBG  --> replace high-sensitivity histogram bins in SR with total BG.\n") ;

  printf("--showOneUncertaintyOnly --> draw only one type of uncertainty") ;

  printf("--in        --> input file with from plotter\n");

  printf("--json      --> json file with the sample descriptor\n");

  printf("--histoVBF  --> name of histogram to be used for VBF\n");

  printf("--histo     --> name of histogram to be used\n");

  printf("--shapeMin  --> left cut to apply on the shape histogram\n");

  printf("--shapeMax  --> right cut to apply on the shape histogram\n");

  printf("--shapeMinVBF  --> left cut to apply on the shape histogram for Vbf bin\n");

  printf("--shapeMaxVBF  --> right cut to apply on the shape histogram for Vbf bin\n");

  printf("--indexvbf  --> index of selection to be used for the vbf bin (if unspecified same as --index)\n");

  printf("--index     --> index of selection to be used (Xbin in histogram to be used); different comma separated values can be given for each analysis bin\n");

  printf("--indexL    --> index of selection to be used (Xbin in histogram to be used) used for interpolation;  different comma separated values can be given for each analysis bin\n");

  printf("--indexR    --> index of selection to be used (Xbin in histogram to be used) used for interpolation;  different comma separated values can be given for each analysis bin\n");

  printf("--m         --> higgs mass to be considered\n");

  printf("--mL        --> higgs mass on the left  of the mass to be considered (used for interpollation\n");

  printf("--mR        --> higgs mass on the right of the mass to be considered (used for interpollation\n");

  printf("--syst      --> use this flag if you want to run systematics, default is no systematics\n");

  printf("--shape     --> use this flag if you want to run shapeBased analysis, default is cut&count\n");

  printf("--subNRB    --> use this flag if you want to subtract non-resonant-backgounds similarly to what was done in 2011 (will also remove H->WW)\n");

  printf("--subNRB12  --> use this flag if you want to subtract non-resonant-backgounds using a new technique that keep H->WW\n");

  printf("--subDY     --> histogram that contains the Z+Jets background estimated from Gamma+Jets)\n");

  printf("--subWZ     --> use this flag if you want to subtract WZ background by the 3rd lepton SB)\n");

  printf("--DDRescale --> factor to be used in order to multiply/rescale datadriven estimations\n");

  printf("--closure   --> use this flag if you want to perform a MC closure test (use only MC simulation)\n");

  printf("--bins      --> list of bins to be used (they must be comma separated without space)\n");

  printf("--HWW       --> use this flag to consider HWW signal)\n");

  printf("--skipGGH   --> use this flag to skip GGH signal)\n");

  printf("--skipQQH   --> use this flag to skip GGH signal)\n");

  printf("--blind     --> use this flag to replace observed data by total predicted background)\n");

  printf("--blindWithSignal --> use this flag to replace observed data by total predicted background+signal)\n");

  printf("--postfix    --> use this to specify a postfix that will be added to the process names)\n");

  printf("--systpostfix    --> use this to specify a syst postfix that will be added to the process names)\n");

  printf("--MCRescale    --> use this to rescale the cross-section of all MC processes by a given factor)\n");

  printf("--postfit  ---> use this to apply postfit Normalization values for W and Top processes in the Signal + Control regions \n");

  //  printf("--addsyst ---> add more systematics than what was entered in syst with runhaaAnalysis, produced externally \n");

  printf("--signalRescale    --> use this to rescale signal cross-section by a given factor)\n");

  printf("--interf     --> use this to rescale xsection according to WW interferences)\n");

  printf("--minSignalYield   --> use this to specify the minimum Signal yield you want in each channel)\n");

  printf("--signalSufix --> use this flag to specify a suffix string that should be added to the signal 'histo' histogram\n");

  printf("--signalTag   --> use this flag to specify a tag that should be present in signal sample name\n");

  printf("--signalScale   --> use this flag to specify a Scale applied on signal\n");

  printf("--rebin         --> rebin the histogram\n");

  printf("--sstyCut         --> show event yields with bdt above sstyCut\n");

  // OLD (incomplete): printf("--statBinByBin --> make bin by bin statistical uncertainty\n");
  printf("--statBinByBin X --> split qualifying process statistical uncertainties per bin; keep a correlated residual for the other bins\n");

  printf("--noCorrelatedStatUnc --> legacy alias for --statUncMode none\n");

  printf("--inclusive  --> merge bins to make the analysis inclusive\n");

  printf("--dropBckgBelow --> drop all background processes that contributes for less than a threshold to the total background yields\n");

  printf("--scaleVBF    --> scale VBF signal by ggH/VBF\n");

  printf("--key        --> provide a key for sample filtering in the json\n");  

  printf("--noLogy        --> use this flag to make y-axis linear scale\n");  

  printf("--year        --> use this flag to indicate which year, useful when computing combined limits\n");  

  printf("--minErrOverSqrtNBGForBinByBin  --> Set minimum err / sqrt(NBG) for including a bin-by-bin stat error\n") ;

}

//

int main(int argc, char* argv[])

{

  setTDRStyle();

  gStyle->SetPadTopMargin   (0.06);

  gStyle->SetPadBottomMargin(0.12);

  gStyle->SetPadRightMargin (0.16);

  gStyle->SetPadLeftMargin  (0.14);

  gStyle->SetTitleSize(0.04, "XYZ");

  gStyle->SetTitleXOffset(1.1);

  gStyle->SetTitleYOffset(1.45);

  gStyle->SetPalette(1);

  gStyle->SetNdivisions(505);

  gStyle->SetOptStat(0);  

  gStyle->SetOptFit(0);

  if ( verbose ) { printf("  --- verbose : main :  processing %d arguments.\n", argc ) ; fflush(stdout) ; }

  //get input arguments

  for(int i=1;i<argc;i++){

    string arg(argv[i]);

    if(arg.find("--help")          !=string::npos) { printHelp(); return -1;} 

    else if(arg.find("--fitDiagnosticsInputFile") !=string::npos && i+1<argc) { fdInputFile = argv[i+1]; i++;  printf("fdInputFile = %s\n", fdInputFile.Data()); }

    else if(arg.find("--allRooFitResultsInputFile") !=string::npos && i+1<argc) { rfrInputFile = argv[i+1]; i++;  printf("rfrInputFile = %s\n", rfrInputFile.Data()); }

    else if(arg.find("--sumInputFile")       !=string::npos && i+1<argc)  { sumFileUrl = argv[i+1];  i++;  printf("sumFileUrl = %s\n", sumFileUrl.Data());  }

    // OLD (wrong): no i+1 bounds check and i was not advanced, so the value
    // was parsed again as if it were a command-line option.
    // else if(arg.find("--minErrOverSqrtNBGForBinByBin") !=string::npos) { sscanf(argv[i+1],"%f",&minErrOverSqrtNBGForBinByBin); printf("minErrOverSqrtNBGForBinByBin = %.3f\n", minErrOverSqrtNBGForBinByBin);}
    else if(arg.find("--minErrOverSqrtNBGForBinByBin") !=string::npos && i+1<argc) {
      sscanf(argv[++i], "%f", &minErrOverSqrtNBGForBinByBin);
      printf("minErrOverSqrtNBGForBinByBin = %.3f\n", minErrOverSqrtNBGForBinByBin);
    }

    else if(arg == "--statUncMode" && i+1<argc) {
      statUncModeName = argv[++i];
      if (statUncModeName == "none") {
        statUncMode = kStatNone;
      } else if (statUncModeName == "correlated") {
        statUncMode = kStatCorrelated;
      } else if (statUncModeName == "hybrid") {
        statUncMode = kStatHybrid;
      } else {
        printf("ERROR: invalid --statUncMode '%s'. Use none, correlated, or hybrid.\n",
               statUncModeName.Data());
        return -1;
      }
      printf("statUncMode = %s\n", statUncModeName.Data());
    }

    else if(arg.find("--replaceHighSensitivityBinsWithBG") !=string::npos) { replaceHighSensitivityBinsWithBG = true; printf("replaceHighSensitivityBinsWithBG = True\n");}

    else if(arg == "--noCorrelatedStatUnc") { noCorrelatedStatUnc = true; printf("noCorrelatedStatUnc = True (legacy alias for --statUncMode none)\n");}

    else if(arg.find("--correlatedLumi") !=string::npos) { correlatedLumi = true; printf("correlatedLumi = True\n");}

    else if(arg.find("--plotsOnly") !=string::npos) { plotsOnly = true ; printf("plotsOnly = True\n") ; }

    else if(arg.find("--autoMCStats")  !=string::npos) { autoMCStats=true; printf("autoMCStats = True\n");}

    else if(arg.find("--verbose")  !=string::npos) { verbose=true; printf("verbose = True\n");}

    else if(arg.find("--minSignalYield") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&minSignalYield ); i++; printf("minSignalYield = %f\n", minSignalYield);}

    else if(arg.find("--scaleVBF") !=string::npos) { scaleVBF=true; printf("scaleVBF = True\n");}

    else if(arg.find("--subNRB")   !=string::npos) { subNRB=true; skipWW=true; printf("subNRB = True\n");}

    else if(arg.find("--subDY")    !=string::npos) { subDY=true; DYFile=argv[i+1];  i++; printf("Z+Jets will be replaced by %s\n",DYFile.Data());}

    else if(arg.find("--subFake")  !=string::npos) { subFake=true; printf("Fake lepton QCD procs will be replaced by DD\n");}

    else if(arg.find("--subWZ")    !=string::npos) { subWZ=true; printf("WZ will be estimated from 3rd lepton SB\n");}

    else if(arg.find("--DDRescale")!=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&DDRescale); i++;}

    else if(arg.find("--MCRescale")!=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&MCRescale); i++;}

    else if(arg.find("--signalRescale")!=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&SignalRescale); i++;}

    else if(arg.find("--HWW")      !=string::npos) { skipWW=false; printf("HWW = True\n");}

    else if(arg.find("--skipGGH")  !=string::npos) { skipGGH=true; printf("skipGGH = True\n");}

    else if(arg.find("--skipQQH")  !=string::npos) { skipQQH=true; printf("skipQQH = True\n");}

    else if(arg.find("--blindWithSignal")    !=string::npos) { blindData=true; blindWithSignal=true; printf("blindData = True; blindWithSignal = True\n");}

    else if(arg.find("--blind")    !=string::npos) { blindData=true; printf("blindData = True\n");}

    else if(arg.find("--closure")  !=string::npos) { MCclosureTest=true; printf("MCclosureTest = True\n");}

    else if(arg.find("--shapeBinToConsider")    !=string::npos && i+1<argc)  { char* pch = strtok(argv[i+1],",");while (pch!=NULL){int C;  sscanf(pch,"%i",&C); shapeBinToConsider.push_back(C);  pch = strtok(NULL,",");} i++; printf("Only the following histo bins will be considered: "); for(unsigned int i=0;i<shapeBinToConsider.size();i++)printf(" %i ", shapeBinToConsider[i]);printf("\n");}

    else if(arg.find("--shapeMinVBF") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&shapeMinVBF); i++; printf("Min cut on shape for VBF = %f\n", shapeMinVBF);}

    else if(arg.find("--shapeMaxVBF") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&shapeMaxVBF); i++; printf("Max cut on shape for VBF = %f\n", shapeMaxVBF);}

    else if(arg.find("--shapeMin") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&shapeMin); i++; printf("Min cut on shape = %f\n", shapeMin);}

    else if(arg.find("--shapeMax") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&shapeMax); i++; printf("Max cut on shape = %f\n", shapeMax);}

    else if(arg.find("--interf")    !=string::npos) { doInterf=true; printf("doInterf = True\n");}

    else if(arg.find("--indexvbf") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%i",&indexvbf); i++; printf("indexVBF = %i\n", indexvbf);}

    else if(arg.find("--index" )   !=string::npos && i+1<argc)   { char* pch = strtok(argv[i+1],",");while (pch!=NULL){int C;  sscanf(pch,"%i",&C); indexcutV .push_back(C);  pch = strtok(NULL,",");} i++; printf("index  = "); for(unsigned int i=0;i<indexcutV .size();i++)printf(" %i ", indexcutV [i]);printf("\n");}

    else if(arg.find("--indexL")    !=string::npos && i+1<argc)  { char* pch = strtok(argv[i+1],",");while (pch!=NULL){int C;  sscanf(pch,"%i",&C); indexcutVL.push_back(C);  pch = strtok(NULL,",");} i++; printf("indexL = "); for(unsigned int i=0;i<indexcutVL.size();i++)printf(" %i ", indexcutVL[i]);printf("\n");}

    else if(arg.find("--indexR")    !=string::npos && i+1<argc)  { char* pch = strtok(argv[i+1],",");while (pch!=NULL){int C;  sscanf(pch,"%i",&C); indexcutVR.push_back(C);  pch = strtok(NULL,",");} i++; printf("indexR = "); for(unsigned int i=0;i<indexcutVR.size();i++)printf(" %i ", indexcutVR[i]);printf("\n");}

    else if(arg.find("--showOneUncertaintyOnly") !=string::npos && i+1<argc) { showOneUncertainty = argv[i+1]; printf("showOneUncertainty = %s\n", showOneUncertainty.Data()); }

    else if(arg.find("--in")       !=string::npos && i+1<argc)  { inFileUrl = argv[i+1];  i++;  printf("in = %s\n", inFileUrl.Data());  }

    else if(arg.find("--json")     !=string::npos && i+1<argc)  { jsonFile  = argv[i+1];  i++;  printf("json = %s\n", jsonFile.Data()); }

    else if(arg.find("--histoVBF") !=string::npos && i+1<argc)  { histoVBF  = argv[i+1];  i++;  printf("histoVBF = %s\n", histoVBF.Data()); }

    else if(arg.find("--histo")    !=string::npos && i+1<argc)  { histo     = argv[i+1];  i++;  printf("histo = %s\n", histo.Data()); }

    else if(arg.find("--year")     !=string::npos && i+1<argc)  { year      = argv[i+1];  i++;  printf("year postfix = %s\n", year.Data()); }

    else if(arg.find("--mL")       !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%i",&massL ); i++; printf("massL = %i\n", massL);}

    else if(arg.find("--mR")       !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%i",&massR ); i++; printf("massR = %i\n", massR);}

    else if(arg.find("--m")        !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%i",&mass ); i++; printf("mass = %i\n", mass);}

    else if(arg.find("--bins")     !=string::npos && i+1<argc)  { char* pch = strtok(argv[i+1],",");printf("bins are : ");while (pch!=NULL){printf(" %s ",pch); AnalysisBins.push_back(pch);  pch = strtok(NULL,",");}printf("\n"); i++; }

    else if(arg.find("--channels") !=string::npos && i+1<argc)  { char* pch = strtok(argv[i+1],",");printf("channels are : ");while (pch!=NULL){printf(" %s ",pch); Channels.push_back(pch);  pch = strtok(NULL,",");}printf("\n"); i++; }

    else if(arg.find("--postfix")   !=string::npos && i+1<argc)  { postfix = argv[i+1]; systpostfix = argv[i+1]; i++;  printf("postfix '%s' will be used\n", postfix.Data());  }

    else if(arg.find("--systpostfix")   !=string::npos && i+1<argc)  { systpostfix = argv[i+1];  i++;  printf("systpostfix '%s' will be used\n", systpostfix.Data());  }

    else if(arg.find("--shape")  !=string::npos) { shape=true; printf("shapeBased = True\n");}   

    else if(arg.find("--syst")  !=string::npos) { runSystematics=true; printf("syst = True\n");}     

    else if(arg.find("--simfit")  !=string::npos) { simfit=true; printf("simfit = True\n");}    

    else if(arg.find("--dirtyFix2")    !=string::npos) { dirtyFix2=true; printf("dirtyFix2 = True\n");}

    else if(arg.find("--dirtyFix1")    !=string::npos) { dirtyFix1=true; printf("dirtyFix1 = True\n");}

    else if(arg.find("--signalSufix") !=string::npos) { signalSufix = argv[i+1]; i++; printf("signalSufix '%s' will be used\n", signalSufix.Data()); }

    else if(arg.find("--signalTag") !=string::npos) { signalTag = argv[i+1]; i++; printf("signalTag '%s' will be used\n", signalTag.c_str()); }

    else if(arg.find("--signalScale") !=string::npos) { sscanf(argv[i+1],"%d",&signalScale); i++; printf("signalScale = %d\n", signalScale);}

    else if(arg.find("--rebin")    !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%i",&rebinVal); i++; printf("rebin = %i\n", rebinVal);}

    else if(arg.find("--sstyCut")    !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&sstyCut); i++; docut=true; printf("sstyCut = %f\n", sstyCut);}

    else if(arg.find("--BackExtrapol")    !=string::npos) { BackExtrapol=true; printf("BackExtrapol = True\n");}

    // OLD (wrong): argv[i+1] was accessed without checking that it exists.
    // else if(arg.find("--statBinByBin") !=string::npos) { sscanf(argv[i+1],"%f",&statBinByBin); i++; printf("statBinByBin = %f\n", statBinByBin);}
    else if(arg.find("--statBinByBin") !=string::npos && i+1<argc) {
      sscanf(argv[++i], "%f", &statBinByBin);
      printf("statBinByBin = %f\n", statBinByBin);
    }

    else if(arg.find("--dropBckgBelow")   !=string::npos) { sscanf(argv[i+1],"%lf",&dropBckgBelow); i++; printf("dropBckgBelow = %f\n", dropBckgBelow);}

    else if(arg.find("--key"          )   !=string::npos && i+1<argc){ keywords.push_back(argv[i+1]); printf("Only samples matching this (regex) expression '%s' are processed\n", argv[i+1]); i++;  }

    else if(arg.find("--noLogy")    !=string::npos) { useLogy=false; printf("useLogy = False\n");}

    else if(arg.find("--SRblind")    !=string::npos) { blindSR=true; printf("blindSR = True\n");}

    else if(arg.find("--lumi") !=string::npos && i+1<argc)  { sscanf(argv[i+1],"%lf",&lumi); i++; printf("Lumi = %lf\n", lumi);}

    if(arg.find("--runZh") !=string::npos) { runZh=true; printf("runZh = True\n");}

    if(arg.find("--modeDD") !=string::npos) { modeDD=true; printf("modeDD = True\n");}

    if(arg.find("--postfit")  !=string::npos) { postfit=true; printf("postfit = True\n");}

    else if(arg.find("--validateDDQCD") != string::npos) {

  doDDQCDValidation = true;

  printf("validateDDQCD = True\n");

}

    //    if(arg.find("--addsyst")  !=string::npos) { addsyst=true; printf("addsyst = True\n");}      

  }

  if (noCorrelatedStatUnc) {
    statUncMode = kStatNone;
    statUncModeName = "none";
  }

  if (autoMCStats) {
    if (statUncMode != kStatNone || statBinByBin > 0) {
      printf("\n\n *** WARNING: autoMCStats requested together with custom stat nuisances."
             "  Disabling the custom nuisances to avoid double counting.\n\n");
    }
    statUncMode = kStatNone;
    statUncModeName = "none";
    statBinByBin = -1;
  } else if (statUncMode == kStatHybrid) {
    if (statBinByBin <= 0) {
      printf("\n\n *** ERROR: --statUncMode hybrid requires a positive"
             " --statBinByBin threshold.\n\n");
      return -1;
    }
    if (minErrOverSqrtNBGForBinByBin < 0) {
      printf("\n\n *** ERROR: --minErrOverSqrtNBGForBinByBin must be non-negative.\n\n");
      return -1;
    }
  } else {
    statBinByBin = -1;
  }

  printf("Final template-statistics mode: %s\n", statUncModeName.Data());

  if ( postfit && rfrInputFile.Length() == 0 ) {

     printf("\n\n *** postfit set but no file given with --allRooFitResultsInputFile option.  Rerun with that set.\n\n") ;

     return -1 ; }

  if ( postfit && year.Length() == 0 ) {

     printf("\n\n *** postfit set but year not set.  Rerun with --year set.\n\n") ;

     return -1 ;

  }

  if(jsonFile.IsNull()) { printf("No Json file provided\nrun with '--help' for more details\n"); return -1; }

  if(inFileUrl.IsNull()){ printf("No Inputfile provided\nrun with '--help' for more details\n"); return -1; }

  if(histo.IsNull())    { printf("No Histogram provided\nrun with '--help' for more details\n"); return -1; }

  if(mass==-1)          { printf("No massPoint provided\nrun with '--help' for more details\n"); return -1; }

  if(indexcutV.size()<=0){printf("INDEX CUT SIZE IS NULL\n"); printHelp(); return -1; }

  if(AnalysisBins.size()==0)AnalysisBins.push_back("all");

  if(Channels.size()==0){ 

    if (modeDD) {

      Channels.push_back("veto_A_SR");

      Channels.push_back("veto_B_SR"); 

      Channels.push_back("veto_C_SR");  

      Channels.push_back("veto_D_SR");

      if (doDDQCDValidation) {

	Channels.push_back("veto_Astar_SR");

	Channels.push_back("veto_Bstar_SR");

      }

      if (simfit && runZh) {

	Channels.push_back("lep1_A_CR");

      }

    } else { 

    if(simfit && runZh){

	  Channels.push_back("lep1_A_CR");//Channels.push_back("mumu_A_CR");     

        }

    }

  }


  vh_tag = runZh ? "_zh" : "_wh";

  vh_tag = (year == "") ? vh_tag : TString("_") + year + vh_tag;

  year = (year == "") ? "" : TString("_") + year;

  //make sure that the index vector are well filled

  if(indexcutVL.size()==0) indexcutVL.push_back(indexcutV [0]);

  if(indexcutVR.size()==0) indexcutVR.push_back(indexcutV [0]);

  while(indexcutV .size()<AnalysisBins.size()){indexcutV .push_back(indexcutV [0]);}

  while(indexcutVL.size()<AnalysisBins.size()){indexcutVL.push_back(indexcutVL[0]);}

  while(indexcutVR.size()<AnalysisBins.size()){indexcutVR.push_back(indexcutVR[0]);}

  if(indexvbf>=0){for(unsigned int i=0;i<AnalysisBins.size();i++){if(AnalysisBins[i].find("vbf")!=string::npos){indexcutV[i]=indexvbf; indexcutVL[i]=indexvbf; indexcutVR[i]=indexvbf;} }}



  //handle merged bins

  std::vector<std::vector<string> > binsToMerge;

  for(unsigned int b=0;b<AnalysisBins.size();b++){

    if(AnalysisBins[b].find('+')!=std::string::npos){

      //std::cout << "Find the string: " << AnalysisBins[b] << std::endl;

      std::vector<string> subBins;

      std::istringstream iss(AnalysisBins[b]);

      std::string token;

      while (std::getline(iss, token, '+')){

      //char* pch = strtok(&AnalysisBins[b][0],"+"); 

      //while (pch!=NULL){

        //std::cout << "subBin pushed: " << token << std::endl;

        indexcutV.push_back(indexcutV[b]);

        indexcutVL.push_back(indexcutVL[b]);

        indexcutVR.push_back(indexcutVR[b]);

        AnalysisBins.push_back(token);

        subBins.push_back(token);

        //AnalysisBins.push_back(pch);

        //subBins.push_back(pch);

      //  pch = strtok(NULL,"+");

      //}

      }

      binsToMerge.push_back(subBins);

      AnalysisBins.erase(AnalysisBins.begin()+b);

      indexcutV .erase(indexcutV .begin()+b);

      indexcutVL.erase(indexcutVL.begin()+b);

      indexcutVR.erase(indexcutVR.begin()+b);

      b--;

    }

  }


  //fill the index map

  for(unsigned int i=0;i<AnalysisBins.size();i++){indexcutM[AnalysisBins[i]] = indexcutV[i]; indexcutML[AnalysisBins[i]] = indexcutVL[i]; indexcutMR[AnalysisBins[i]] = indexcutVR[i];}


  ///////////////////////////////////////////////

  //init the json wrapper

  JSONWrapper::Object Root(jsonFile.Data(), true);


  //init globalVariables

  TString massStr(""); if(mass>0)massStr += mass;

  std::vector<TString> allCh,allProcs;

  std::vector<TString> ch;

  if (modeDD) {

    ch.push_back("veto_A_SR");

    ch.push_back("veto_B_SR");

    ch.push_back("veto_C_SR");

    ch.push_back("veto_D_SR");

    if (doDDQCDValidation) {

      ch.push_back("veto_Astar_SR");

      ch.push_back("veto_Bstar_SR");

    }

    if (simfit && runZh) {

      ch.push_back("lep1_A_CR");

    }

  } else {

    if(runZh){// Zh

      ch.push_back("veto_A_SR") ;

      if (simfit) {

	ch.push_back("lep1_A_CR");
    

      }

    } else { // Wh

      ch.push_back("e_A_SR"); ch.push_back("mu_A_SR");

      if (simfit) { 

        ch.push_back("e_A_CR"); ch.push_back("mu_A_CR");

 

      }

    }

  }



  const size_t nch=ch.size(); //sizeof(ch)/sizeof(TString);

  std::vector<TString> sh;

  sh.push_back(histo);

  if(subNRB)sh.push_back(histo+"_NRBctrl");

  if(subWZ)sh.push_back(histo+"_3rdLepton");

  if ( verbose ) {

     printf("  --- verbose : main :  contents of ch vector:\n") ;

     for ( int i=0; i<ch.size();           i++ ) { printf("     --- verbose:  ch entry %2d : %s\n", i, ch[i].Data() ) ; }

     printf("  --- verbose : main :  contents of sh vector:\n") ;

     for ( int i=0; i<sh.size();           i++ ) { printf("     --- verbose:  sh entry %2d : %s\n", i, sh[i].Data() ) ; }

     printf("  --- verbose : main :  contents of AnalysisBins vector:\n") ;

     for ( int i=0; i<AnalysisBins.size(); i++ ) { printf("     --- verbose:  AnalysisBins entry %2d : %s\n", i, AnalysisBins[i].c_str() ) ; }

     printf("  --- verbose : main :  contents of Channels vector:\n") ;

     for ( int i=0; i<Channels.size();     i++ ) { printf("     --- verbose:  Channels entry %2d : %s\n", i, Channels[i].Data() ) ; }

     fflush(stdout) ;

  }

  AllInfo_t allInfo;

  AllInfo_t* allInfoSum(0x0) ;

  TFile* inF_sum(0x0) ;

  if ( !sumFileUrl.IsNull() ) {

     printf("\n\n  sumInputFile is set to %s.  Will read it in to a separate allInfo.\n\n", sumFileUrl.Data() ) ;

     allInfoSum = new AllInfo_t() ;

     inF_sum = TFile::Open(sumFileUrl);

     if( !inF_sum || inF_sum->IsZombie() ){ printf("Invalid file name : %s\n", sumFileUrl.Data()); gSystem -> Exit(-1); }

     gROOT->cd();  //THIS LINE IS NEEDED TO MAKE SURE THAT HISTOGRAM INTERNALLY PRODUCED IN LumiReWeighting ARE NOT DESTROYED WHEN CLOSING THE FILE

  }

  rfr_tt_norm = 1. ;


  if ( postfit ) {

     printf("\n\n  postfit it set.  Reading in fit normalizations from %s\n\n", rfrInputFile.Data() ) ;

     TFile tf_fd( rfrInputFile, "read" ) ;

     if ( !(tf_fd.IsOpen()) ) { printf("\n\n *** bad --allRooFitResultsInputFile file %s\n\n", rfrInputFile.Data() ) ; gSystem -> Exit(-1) ; }

     if ( runZh ) {

        char frname[100] ;

        RooFitResult* rfr(0x0) ;

        RooRealVar* rrv(0x0) ;

        char parname[100] ;

        sprintf( frname, "fit_b_zh%s", year.Data() ) ;

        rfr = (RooFitResult*) tf_fd.Get( frname ) ;

        if ( rfr == 0x0 ) {

           printf("\n\n *** postfit set but can't find RooFitResult %s \n\n", frname ) ;

           gSystem -> Exit(-1) ;

        }

        sprintf( parname, "tt_norm" ) ;

        rrv = (RooRealVar*)( rfr -> floatParsFinal()).find( parname ) ;

        if ( rrv == 0x0 ) { printf("\n\n *** postfit set but can't find %s in %s in %s\n\n", parname, frname, rfrInputFile.Data() ) ; gSystem -> Exit(-1) ; }

        rfr_tt_norm = rrv->getVal() ;

        rfr->Delete() ;

        printf("   postfit normalizations, Zh,  :  tt_norm  = %6.3f\n" , rfr_tt_norm ) ;

        fflush(stdout) ;

     }

     tf_fd.Close() ;

  }

  if ( verbose ) { printf("  --- verbose : main :  Opening input root file with name inFileUrl = %s\n", inFileUrl.Data() ) ; fflush(stdout) ; }

  //open input file

  TFile* inF = TFile::Open(inFileUrl);

  if( !inF || inF->IsZombie() ){ printf("Invalid file name : %s\n", inFileUrl.Data());}

  gROOT->cd();  //THIS LINE IS NEEDED TO MAKE SURE THAT HISTOGRAM INTERNALLY PRODUCED IN LumiReWeighting ARE NOT DESTROYED WHEN CLOSING THE FILE


  //LOAD shapes

  const size_t nsh=sh.size();

  for(size_t b=0; b<AnalysisBins.size(); b++){

    std::vector<string> channelsAndShapes;

    std::vector<string> channelsAndShapesSum;

    for(size_t i=0; i<nch; i++){

      for(size_t j=0; j<nsh; j++){

        channelsAndShapes.push_back((ch[i]+TString(";")+AnalysisBins[b]+TString(";")+sh[j]).Data());

        printf("allInfo   : Adding shape %s\n",(ch[i]+TString(";")+AnalysisBins[b]+TString(";")+sh[j]).Data());

        if ( !sumFileUrl.IsNull() && inF_sum!=0x0 ) {

          //--- only need SR 4b for sum (used in DD QCD).

	  if ( ch[i].Contains("SR") && strcmp( AnalysisBins[b].c_str(), "4b" ) == 0 ) {

              channelsAndShapesSum.push_back((ch[i]+TString(";")+AnalysisBins[b]+TString(";")+sh[j]).Data());

              printf("allInfoSum: Adding shape %s\n",(ch[i]+TString(";")+AnalysisBins[b]+TString(";")+sh[j]).Data());

           }

        }

      }

    }

    double cutMin=shapeMin; double cutMax=shapeMax;

    allInfo.getShapeFromFile(inF, channelsAndShapes, indexcutM[AnalysisBins[b]], Root, cutMin, cutMax );     

    if ( !sumFileUrl.IsNull() && inF_sum!=0x0 ) allInfoSum -> getShapeFromFile(inF_sum, channelsAndShapesSum, indexcutM[AnalysisBins[b]], Root, cutMin, cutMax );     

  }




  inF->Close();

  printf("Loading all shapes... Done\n");

  fflush(stdout) ;

  for(unsigned int B=0;B<binsToMerge.size();B++){

    std::string NewBinName = binsToMerge[B][0]; std::cout << "binsToMerge[B][0]: " << binsToMerge[B][0]; for(unsigned int b=1;b<binsToMerge[B].size();b++){NewBinName += "_"+binsToMerge[B][b];std::cout << "binsToMerge[B][b]: " << binsToMerge[B][b] << std::endl;;}

//    std::string NewBinName = string("["); binsToMerge[B][0];  for(unsigned int b=1;b<binsToMerge[B].size();b++){NewBinName += "+"+binsToMerge[B][b];} NewBinName+="]";

    allInfo.mergeBins(binsToMerge[B],NewBinName);

    if ( !sumFileUrl.IsNull() && allInfoSum != 0x0 ) allInfoSum -> mergeBins(binsToMerge[B],NewBinName);

  }


  if ( verbose ) {

     printf("\n\n --- verbose : main :  calling allInfo.printInventory for main allInfo\n\n") ;

     allInfo.printInventory() ;

     if ( !sumFileUrl.IsNull() && allInfoSum != 0x0 ) {

        printf("\n\n --- verbose : main :  calling allInfo.printInventory for sum allInfo\n\n") ;

        allInfoSum -> printInventory() ;

     }

     fflush(stdout) ;

  }


  if ( verbose ) { printf("\n --- verbose : main :  calling allInfo.computeTotalBackground() for first time.\n") ; fflush(stdout) ; }






  if ( verbose ) { if (shape && BackExtrapol ) printf("\n  --- verbose : main :  calling allInfo.rebinMainHisto(histo.Data()) where histo = %s\n", histo.Data() ) ; fflush(stdout) ; }



  //if ( verbose ) allInfo.printInventory() ;


  //allInfo.computeTotalBackground();

if (MCclosureTest) allInfo.blind();

if (shape && BackExtrapol) {

    allInfo.rebinMainHisto(histo.Data());

 }

if (shape && BackExtrapol && allInfoSum != 0x0) {

  allInfoSum->rebinMainHisto(histo.Data());

}

 allInfo.computeTotalBackground(); 

/*

if (doDDQCDValidation) {

  runDDQCDValidation0lep(allInfo, histo, AnalysisBins[0].c_str());

  return 0;

}

*/

if (verbose) allInfo.printInventory();


  FILE* pFile;

  //define vector for search

  std::vector<TString>& selCh = Channels;

  if(modeDD) {

    if ( allInfoSum != 0x0 ) {

       if ( verbose ) { printf("\n --- verbose : main :  calling doBackgroundSubtraction for sum allInfo first.\n") ; }

       pFile = fopen("datadriven_qcd"+year+"-sum.tex","w");

       if(subFake) 

        allInfoSum -> doBackgroundSubtraction(pFile,selCh,histo);

       fclose(pFile);

    }

    if ( verbose ) { printf("\n --- verbose : main :  calling doBackgroundSubtraction for main allInfo.\n") ; }

    pFile = fopen("datadriven_qcd"+year+".tex","w");

    if(subFake)

    {

      allInfo.doBackgroundSubtraction(pFile,selCh,histo, allInfoSum);

     }

    fclose(pFile);

  }

  //replace data by total MC background

  if(blindData)allInfo.blind();

  fflush(stdout) ;




  if ( verbose ) { printf("\n  --- verbose : main :  calling allInfo.dropSmallBckgProc(selCh, histo.Data(), dropBckgBelow) , \n") ; fflush(stdout) ; }

  //drop backgrounds with rate<1%

  allInfo.dropSmallBckgProc(selCh, histo.Data(), dropBckgBelow);


  if ( verbose ) { printf("\n  --- verbose : main :   calling allInfo.dropCtrlChannels(selCh);\n") ; fflush(stdout) ; }

  //drop control channels

  allInfo.dropCtrlChannels(selCh);

  if ( verbose && !shape ) { printf("\n  --- verbose : main :  calling allInfo.turnToCC(histo.Data()); \n") ; fflush(stdout) ; }

  //turn to CC analysis eventually

  if(!shape)allInfo.turnToCC(histo.Data());

  if ( verbose ) { printf("\n  --- verbose : main :  calling allInfo.HandleEmptyBins(histo.Data()); \n") ; fflush(stdout) ; }

  allInfo.HandleEmptyBins(histo.Data()); //needed for negative bin content --> May happens due to NLO interference for instance

  if ( verbose && blindData ) { printf("\n  --- verbose : main :  calling allInfo.blind(); \n") ; fflush(stdout) ; }

  // Blind data in Signal Regions only

  if(blindData) allInfo.blind();

  if (replaceHighSensitivityBinsWithBG) allInfo.replaceHighSensitivityBinsWithBG();

  if ( verbose ) { printf("\n  --- verbose : main :   calling       allInfo.getEffFromShape(pFile, selCh, histo.Data()); \n") ; fflush(stdout) ; }

  //print signal efficiency

  pFile = fopen(vh_tag+"Efficiency.tex","w");

  allInfo.getEffFromShape(pFile, selCh, histo.Data());

  fclose(pFile);

  if ( verbose ) { printf("\n  --- verbose : main :    calling   allInfo.addHardCodedUncertainties(histo.Data()); \n") ; fflush(stdout) ; }

  //add by hand the hard coded uncertainties

  // if( runSystematics ) -Penny

  allInfo.addHardCodedUncertainties(histo.Data());


  if ( verbose ) { printf("\n  --- verbose : main :    calling allInfo.getYieldsFromShape(pFile, selCh, histo.Data(), pFileInc); \n") ; fflush(stdout) ; }

  //print event yields from the histo shapes

  pFile = fopen(vh_tag+"Yields.tex","w");  FILE* pFileInc = fopen(vh_tag+"YieldsInc.tex","w");

  allInfo.getYieldsFromShape(pFile, selCh, histo.Data(), pFileInc);

  fclose(pFile); fclose(pFileInc);


  if ( verbose ) { printf("\n  --- verbose : main :    calling   allInfo.showShape(selCh,histo,\"plot\"); \n") ; fflush(stdout) ; }

  //produce a plot

  allInfo.showShape(selCh,histo,"plot"); //this produce the final global shape

  if ( plotsOnly ) {

     printf("\n\n plotsOnly is set to true.  Bailing out now.\n\n") ; fflush(stdout) ;

     return 0 ;

  }


  if ( verbose && runSystematics ) { printf("\n  --- verbose : main :    calling allInfo.showUncertainty(selCh,histo,\"plot\"); \n") ; fflush(stdout) ; }

  //produce a plot

  //if(runSystematics) allInfo.showUncertainty(selCh,histo,"plot"); //this produces all the plots with the syst  

  if(runSystematics && !(simfit)) allInfo.showUncertainty(selCh,histo,"plot");

  // georgia : now run reporting systematics only if simfit=false 

  // owen: temporarily turn this off.  Slows it down.

  if ( verbose ) allInfo.printInventory() ;

  //prepare the output

  string limitFile=("haa4b_"+massStr+systpostfix+vh_tag+".root").Data();

  TFile *fout=TFile::Open(limitFile.c_str(),"recreate");

  if ( verbose ) { printf("\n  --- verbose : main :   calling   allInfo.saveHistoForLimit(histo.Data(), fout);  \n") ; fflush(stdout) ; }

  allInfo.saveHistoForLimit(histo.Data(), fout);

  if ( verbose ) { printf("\n  --- verbose : main :    calling   allInfo.buildDataCards(histo.Data(), limitFile); \n") ; fflush(stdout) ; }

  allInfo.buildDataCards(histo.Data(), limitFile);

  //all done

  printf("\n\n calling fout->Close();\n\n") ; fflush(stdout) ;

  fout->Close();

  printf("\n\n At the end of main.\n\n") ; fflush(stdout) ;

 return 0;

}

// reorder the procs to get the backgrounds; total bckg, signal, data 

//

void AllInfo_t::sortProc(){

  std::vector<string>bckg_procs;

  std::vector<string>sign_procs;

  bool isTotal=false, isData=false;

  for(unsigned int p=0;p<sorted_procs.size();p++){

    string procName = sorted_procs[p];

    std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

    if(it==procs.end())continue;

    if(it->first=="total"){isTotal=true; continue;}

    if(it->first=="data"){isData=true; continue;}

    if(it->second.isSign)sign_procs.push_back(procName);

    if(it->second.isBckg)bckg_procs.push_back(procName);

  }

  sorted_procs.clear();

  sorted_procs.insert(sorted_procs.end(), bckg_procs.begin(), bckg_procs.end());

  if(isTotal)sorted_procs.push_back("total");

  if(isData)sorted_procs.push_back("data");

  sorted_procs.insert(sorted_procs.end(), sign_procs.begin(), sign_procs.end());

}

//

// Sum up all shapes from one src channel to a total shapes in the dest channel

//

/**/

void AllInfo_t::addChannel(ChannelInfo_t& dest, ChannelInfo_t& src, bool computeSyst, bool addDiffProcs, double scale_factor ){

  std::map<string, ShapeData_t>& shapesInfoDest = dest.shapes;

  std::map<string, ShapeData_t>& shapesInfoSrc  = src.shapes;

  if(!computeSyst){

    for(std::map<string, ShapeData_t>::iterator sh = shapesInfoSrc.begin(); sh!=shapesInfoSrc.end(); sh++){

      if(shapesInfoDest.find(sh->first)==shapesInfoDest.end())shapesInfoDest[sh->first] = ShapeData_t();

      //Loop on all shape systematics (including also the central value shape)

      for(std::map<string, TH1*>::iterator uncS = sh->second.uncShape.begin();uncS!= sh->second.uncShape.end();uncS++){

        if(uncS->first!="" || uncS->second == NULL) continue; //We only take nominal shapes

        if(shapesInfoDest[sh->first].uncShape.find(uncS->first)==shapesInfoDest[sh->first].uncShape.end()){

        shapesInfoDest[sh->first].uncShape[uncS->first] =

        (TH1*) uncS->second->Clone(TString(uncS->second->GetName() + dest.channel + dest.bin ) );

        shapesInfoDest[sh->first].uncShape[uncS->first]->Scale(scale_factor);

        }else{

          //////////shapesInfoDest[sh->first].uncShape[uncS->first]->Add(uncS->second);

          shapesInfoDest[sh->first].uncShape[uncS->first]->Add(uncS->second, scale_factor );

        }

      }

      //take care of the scale uncertainty 

      for(std::map<string, double>::iterator unc = sh->second.uncScale.begin();unc!= sh->second.uncScale.end();unc++){

        if(shapesInfoDest[sh->first].uncScale.find(unc->first)==shapesInfoDest[sh->first].uncScale.end()){

          shapesInfoDest[sh->first].uncScale[unc->first] = unc->second;

        }else{

          shapesInfoDest[sh->first].uncScale[unc->first] = sqrt( pow(shapesInfoDest[sh->first].uncScale[unc->first],2) + pow(unc->second,2) );

        }

      }

    }  

  }

  else {

    for(std::map<string, ShapeData_t>::iterator sh = shapesInfoSrc.begin(); sh!=shapesInfoSrc.end(); sh++){

      if(shapesInfoDest.find(sh->first)==shapesInfoDest.end())shapesInfoDest[sh->first] = ShapeData_t();

                //Loop on all shape systematics (including also the central value shape)

      for(std::map<string, TH1*>::iterator uncS = sh->second.uncShape.begin();uncS!= sh->second.uncShape.end();uncS++){

        if(uncS->first=="" || uncS->second == NULL) continue; //We only take systematic (i.e non-nominal) shapes

        if(shapesInfoSrc[sh->first].uncShape.find("")==shapesInfoSrc[sh->first].uncShape.end()) continue;

        if(addDiffProcs){ // add different procs

	    //1. Copy the nominal shape    

	  shapesInfoDest[sh->first].uncShape[uncS->first] = (TH1*) shapesInfoDest[sh->first].uncShape[""]->Clone(TString(uncS->second->GetName() + dest.channel + dest.bin ) );

          //2. we remove the nominal value of the process we are running on

	  shapesInfoDest[sh->first].uncShape[uncS->first]->Add(shapesInfoSrc[sh->first].uncShape[""], -1);

          //3. and add the variation up/down

          shapesInfoDest[sh->first].uncShape[uncS->first]->Add(uncS->second, scale_factor ); 

        }else{ // add same proc in different channels

          if(shapesInfoDest[sh->first].uncShape.find(uncS->first)==shapesInfoDest[sh->first].uncShape.end()){

            shapesInfoDest[sh->first].uncShape[uncS->first] = (TH1*) uncS->second->Clone(TString(uncS->second->GetName() + dest.channel + dest.bin ) );

          }else{

            ////////shapesInfoDest[sh->first].uncShape[uncS->first]->Add(uncS->second); 

            shapesInfoDest[sh->first].uncShape[uncS->first]->Add(uncS->second, scale_factor ); 

          }

        }

      }

    }

  } // if (computeSyst)

}

//

// Sum up all background processes and add this as a total process

//

void AllInfo_t::addProc(

    ProcessInfo_t& dest,

    ProcessInfo_t& src,

    bool computeSyst

) {

    addProc(dest, src, computeSyst, 1.0);

}

void AllInfo_t::addProc(

    ProcessInfo_t& dest,

    ProcessInfo_t& src,

    bool computeSyst,

    double scale_factor

) {

    dest.xsec = src.xsec * src.br;

    for (std::map<string, ChannelInfo_t>::iterator ch = src.channels.begin();

         ch != src.channels.end(); ch++) {

        if (dest.channels.find(ch->first) == dest.channels.end()) {

            dest.channels[ch->first] = ChannelInfo_t();

            dest.channels[ch->first].bin = ch->second.bin;

            dest.channels[ch->first].channel = ch->second.channel;

        }

        addChannel(

            dest.channels[ch->first],

            ch->second,

            computeSyst,

            true,

            scale_factor

        );

    }

}

//

  // Subtract nonQCD MC from B,C,D regions, then build DD QCD in A.

// Uses A = B * C/D..

// tt_norm, if provided, is applied only to ttbarbb subtraction.

void AllInfo_t::doBackgroundSubtraction(

  FILE* pFile,

  std::vector<TString>& selCh,

  TString mainHisto,

  AllInfo_t* sumAllInfo

) {

  if (verbose) printf("\n\n --- verbose : AllInfo_t::doBackgroundSubtraction : begin\n\n");

  char Lcol[1024]       = "|c";

  char Lchan[1024]      = "";

  char Lalph1[1024]     = "";

  char Lyield[1024]     = "";

  char LyieldMC[1024]   = "";

  char LalphMC[1024]    = "";

  char LyieldPred[1024] = "";

  char RatioMC[1024]    = "";

  char LBstat[1024]     = "";

  char LTF[1024]        = "";

 double ttbb_norm_val = 1.0;

double ttbb_norm_err = 0.0;

if (fdInputFile.Length() > 0) {

    TFile tf_fd(fdInputFile, "READ");

    if (!tf_fd.IsOpen()) {

        printf("\n\n *** problem opening fdInputFile %s\n\n",

               fdInputFile.Data());

        gSystem->Exit(-1);

    }

    RooFitResult* rfr = (RooFitResult*) tf_fd.Get("fit_b");

    if (!rfr) {

        printf("\n\n *** did not find fit_b RooFitResult in %s\n\n",

               fdInputFile.Data());

        gSystem->Exit(-1);

    }

    RooRealVar* rrv_ttbb =

        (RooRealVar*)(rfr->floatParsFinal()).find("ttbb_norm");

    if (!rrv_ttbb) {

        printf("\n\n *** did not find ttbb_norm in fit_b in %s\n\n",

               fdInputFile.Data());

        gSystem->Exit(-1);

    }

    ttbb_norm_val = rrv_ttbb->getVal();

    ttbb_norm_err = rrv_ttbb->getError();

    printf(

        "Preliminary CR ttbb_norm for DDQCD subtraction = %.3f +/- %.3f\n",

        ttbb_norm_val,

        ttbb_norm_err

    );

    rfr->Delete();

    tf_fd.Close();

}

  auto dataProcIt = procs.find("data");

  if (dataProcIt == procs.end()) {

    printf("The process 'data' was not found. Cannot do QCD background prediction.\n");

    return;

  }

  TString NRBProcName = "NonQCD";

  for (auto p = sorted_procs.begin(); p != sorted_procs.end(); ) {

    if ((*p) == NRBProcName.Data()) p = sorted_procs.erase(p);

    else ++p;

  }

  sorted_procs.push_back(NRBProcName.Data());

  procs[NRBProcName.Data()] = ProcessInfo_t();

  ProcessInfo_t& procInfo_NRB = procs[NRBProcName.Data()];

  procInfo_NRB.shortName = "nonqcd";

  procInfo_NRB.isData = true;

  procInfo_NRB.isSign = false;

  procInfo_NRB.isBckg = true;

  procInfo_NRB.xsec = 0.0;

  procInfo_NRB.br = 1.0;

  for (auto it = procs.begin(); it != procs.end(); ++it) {

    if (it->second.isData) continue;

    TString procName = it->first.c_str();

    if (!(procName.Contains("Other Bkgs") ||

          procName.Contains("Z#rightarrow  #nu #nu") ||

          procName.Contains("t#bar{t}"))) continue;

    bool isTTbb =

      procName.Contains("t#bar{t} + b#bar{b}") ||

      TString(it->second.shortName.c_str()).Contains("ttbarbba");

    printf("Subtracting nonQCD process from data: %s, long name %s\n",

           it->second.shortName.c_str(), procName.Data());

    if (fdInputFile.Length() > 0 && isTTbb) {

      addProc(procInfo_NRB, it->second, false, ttbb_norm_val);

    } else {

      addProc(procInfo_NRB, it->second, false);

    }

  }

  // Subtract NonQCD from B,C,D only. Do not subtract A.

  for (auto chData = dataProcIt->second.channels.begin();

       chData != dataProcIt->second.channels.end(); ++chData) {

    if (std::find(selCh.begin(), selCh.end(), chData->second.channel) == selCh.end()) continue;

    if (chData->first.find("veto_") == string::npos) continue;

    if (chData->first.find("_SR_")  == string::npos) continue;

    if (chData->first.find("_3b")   == string::npos) continue;

    if (chData->first.find("_A_") != string::npos) continue;

    auto chNRB = procInfo_NRB.channels.find(chData->first);

    if (chNRB == procInfo_NRB.channels.end()) continue;

    auto& shapesInfoDest = chData->second.shapes;

    auto& shapesInfoSrc  = chNRB->second.shapes;

    for (auto sh = shapesInfoSrc.begin(); sh != shapesInfoSrc.end(); ++sh) {

      if (shapesInfoDest.find(sh->first) == shapesInfoDest.end()) continue;

      for (auto uncS = sh->second.uncShape.begin();

           uncS != sh->second.uncShape.end(); ++uncS) {

        if (uncS->first != "") continue;

        if (!uncS->second) continue;

        if (shapesInfoDest[sh->first].uncShape.find("") == shapesInfoDest[sh->first].uncShape.end()) continue;

        if (!shapesInfoDest[sh->first].uncShape[""]) continue;

        shapesInfoDest[sh->first].uncShape[""]->Add(uncS->second, -1.0);

      }

    }

  }

  TString DDProcName = "ddqcd";

  for (auto p = sorted_procs.begin(); p != sorted_procs.end(); ) {

    if ((*p) == DDProcName.Data()) p = sorted_procs.erase(p);

    else ++p;

  }

  sorted_procs.push_back(DDProcName.Data());

  procs[DDProcName.Data()] = ProcessInfo_t();

  ProcessInfo_t& procInfo_DD = procs[DDProcName.Data()];

  procInfo_DD.shortName = "ddqcd";

  procInfo_DD.isData = true;

  procInfo_DD.isSign = false;

  procInfo_DD.isBckg = true;

  procInfo_DD.xsec = 0.0;

  procInfo_DD.br = 1.0;

  std::vector<string> toBeDelete;

  for (auto it = procs.begin(); it != procs.end(); ++it) {

    if (!it->second.isBckg || it->second.isData) continue;

    TString procName = it->first.c_str();

    if (!procName.Contains("QCD")) continue;

    addProc(procInfo_DD, it->second);

    for (auto p = sorted_procs.begin(); p != sorted_procs.end(); ) {

      if ((*p) == it->first) p = sorted_procs.erase(p);

      else ++p;

    }

    toBeDelete.push_back(it->first);

  }

  for (auto p = toBeDelete.begin(); p != toBeDelete.end(); ++p) {

    procs.erase(procs.find(*p));

  }

  // Predict ddqcd only in main A.

  for (auto chData = dataProcIt->second.channels.begin();

       chData != dataProcIt->second.channels.end(); ++chData) {

    if (std::find(selCh.begin(), selCh.end(), chData->second.channel) == selCh.end()) continue;

    if (chData->first.find("veto_") == string::npos) continue;

    if (chData->first.find("_SR_")  == string::npos) continue;

    if (chData->first.find("_3b")   == string::npos) continue;

    if (chData->first.find("_A_")   == string::npos) continue;

    auto chDD = procInfo_DD.channels.find(chData->first);

    if (chDD == procInfo_DD.channels.end()) {

      procInfo_DD.channels[chData->first] = ChannelInfo_t();

      chDD = procInfo_DD.channels.find(chData->first);

      chDD->second.bin = chData->second.bin;

      chDD->second.channel = chData->second.channel;

    }

    TString baseName = chData->second.channel.c_str();

    TString dName = baseName;

    TString bName = baseName;

    TString cName = baseName;

    dName.ReplaceAll("A_", "D_");

    bName.ReplaceAll("A_", "B_");

    cName.ReplaceAll("A_", "C_");

    TString keyD = dName + "_" + chData->second.bin.c_str() + year;

    TString keyB = bName + "_" + chData->second.bin.c_str() + year;

    TString keyC = cName + "_" + chData->second.bin.c_str() + year;

    TH1* hCtrl_SB = 0x0; // D

    TH1* hCtrl_SI = 0x0; // B

    TH1* hChan_SB = 0x0; // C

    auto itD = dataProcIt->second.channels.find(keyD.Data());

    auto itB = dataProcIt->second.channels.find(keyB.Data());

    auto itC = dataProcIt->second.channels.find(keyC.Data());

    if (itD != dataProcIt->second.channels.end() &&

        itD->second.shapes.find(mainHisto.Data()) != itD->second.shapes.end())

      hCtrl_SB = itD->second.shapes[mainHisto.Data()].histo();

    if (itB != dataProcIt->second.channels.end() &&

        itB->second.shapes.find(mainHisto.Data()) != itB->second.shapes.end())

      hCtrl_SI = itB->second.shapes[mainHisto.Data()].histo();

    if (itC != dataProcIt->second.channels.end() &&

        itC->second.shapes.find(mainHisto.Data()) != itC->second.shapes.end())

      hChan_SB = itC->second.shapes[mainHisto.Data()].histo();

    if (!hCtrl_SB || !hCtrl_SI || !hChan_SB) {

      printf("\n *** Missing DD-QCD input histograms for %s\n", chData->first.c_str());

      printf("     D key = %s, found = %d\n", keyD.Data(), hCtrl_SB != 0x0);

      printf("     B key = %s, found = %d\n", keyB.Data(), hCtrl_SI != 0x0);

      printf("     C key = %s, found = %d\n", keyC.Data(), hChan_SB != 0x0);

      continue;

    }

    resetNegativeBinsAndErrors(hChan_SB, 1.0);

    resetNegativeBinsAndErrors(hCtrl_SB, 1.0);

    resetNegativeBinsAndErrors(hCtrl_SI, 0.0);

    double B_err = 0.0, C_err = 0.0, D_err = 0.0;

    double B_val = hCtrl_SI->IntegralAndError(1, hCtrl_SI->GetNbinsX(), B_err);

    double C_val = hChan_SB->IntegralAndError(1, hChan_SB->GetNbinsX(), C_err);

    double D_val = hCtrl_SB->IntegralAndError(1, hCtrl_SB->GetNbinsX(), D_err);

    double alpha = 0.0;

    double alpha_err = 0.0;

    if (C_val > 0.0 && D_val > 0.0) {

      alpha = C_val / D_val;

      alpha_err = alpha * sqrt(pow(C_err / C_val, 2) + pow(D_err / D_val, 2));

    }

    double relTF = (alpha > 0.0) ? alpha_err / alpha : 0.0;

    printf("\n[DD DEBUG] Channel: %s\n", chData->first.c_str());

    printf("   B     = %.3f +/- %.3f\n", B_val, B_err);

    printf("   C     = %.3f +/- %.3f\n", C_val, C_err);

    printf("   D     = %.3f +/- %.3f\n", D_val, D_err);

    printf("   alpha = C/D = %.6f +/- %.6f, relTF = %.3f\n", alpha, alpha_err, relTF);

    TH1* hDDA_orig = chDD->second.shapes[mainHisto.Data()].histo();

    double alphaMC = 0.0;

    double alphaMC_err = 0.0;

    double valMC = 0.0;

    double valMC_err = 0.0;

    double valDD_MC = 0.0;

    double valDD_MC_err = 0.0;

    double ratioMC = 0.0;

    double ratioMC_err = 0.0;

    auto itMCB = procInfo_DD.channels.find(keyB.Data());

    auto itMCC = procInfo_DD.channels.find(keyC.Data());

    auto itMCD = procInfo_DD.channels.find(keyD.Data());

    TH1* hDD_B_MC = 0x0;

    TH1* hDD_C_MC = 0x0;

    TH1* hDD_D_MC = 0x0;

    if (itMCB != procInfo_DD.channels.end() &&

        itMCB->second.shapes.find(mainHisto.Data()) != itMCB->second.shapes.end())

      hDD_B_MC = itMCB->second.shapes[mainHisto.Data()].histo();

    if (itMCC != procInfo_DD.channels.end() &&

        itMCC->second.shapes.find(mainHisto.Data()) != itMCC->second.shapes.end())

      hDD_C_MC = itMCC->second.shapes[mainHisto.Data()].histo();

    if (itMCD != procInfo_DD.channels.end() &&

        itMCD->second.shapes.find(mainHisto.Data()) != itMCD->second.shapes.end())

      hDD_D_MC = itMCD->second.shapes[mainHisto.Data()].histo();

    if (hDDA_orig && hDD_B_MC && hDD_C_MC && hDD_D_MC) {

      double errMC_C = 0.0;

      double errMC_D = 0.0;

      double valMC_C = hDD_C_MC->IntegralAndError(1, hDD_C_MC->GetNbinsX(), errMC_C);

      double valMC_D = hDD_D_MC->IntegralAndError(1, hDD_D_MC->GetNbinsX(), errMC_D);

      if (valMC_C > 0.0 && valMC_D > 0.0) {

        alphaMC = valMC_C / valMC_D;

        alphaMC_err = alphaMC * sqrt(pow(errMC_C / valMC_C, 2) + pow(errMC_D / valMC_D, 2));

      }

      valMC = hDDA_orig->IntegralAndError(1, hDDA_orig->GetNbinsX(), valMC_err);

      TH1* hDD_MC_pred = (TH1*) hDD_B_MC->Clone("hDD_MC_closure");

      hDD_MC_pred->Scale(alphaMC);

      valDD_MC = hDD_MC_pred->IntegralAndError(1, hDD_MC_pred->GetNbinsX(), valDD_MC_err);

      if (valMC > 0.0 && valDD_MC > 0.0) {

        ratioMC = valDD_MC / valMC;

        ratioMC_err = ratioMC * sqrt(pow(valDD_MC_err / valDD_MC, 2) + pow(valMC_err / valMC, 2));

      }

      printf("   MC closure: alphaMC = %.6f +/- %.6f, predMC = %.3f +/- %.3f, obsMC = %.3f +/- %.3f, ratio = %.3f +/- %.3f\n",

             alphaMC, alphaMC_err, valDD_MC, valDD_MC_err, valMC, valMC_err, ratioMC, ratioMC_err);

      delete hDD_MC_pred;

    }

    TH1* hDD = chDD->second.shapes[mainHisto.Data()].histo();

    if (!hDD) hDD = (TH1*) hCtrl_SI->Clone();

    hDD->Reset();

    hDD->Add(hCtrl_SI, 1.0);

    hDD->Scale(alpha);

    hDD->SetTitle(DDProcName.Data());

    double valDD_err = 0.0;

double valDD = hDD->IntegralAndError(

    1,

    hDD->GetXaxis()->GetNbins() + 1,

    valDD_err

);


    if (valDD < 1e-6) {

        valDD = 0.0;

        valDD_err = 0.0;

    }

    TString bstatHistBase = TString("ddqcd_BtemplateStat_") + chData->first.c_str();

    TH1* hDD_BstatUp   = (TH1*) hCtrl_SI->Clone(bstatHistBase + "_Up");

    TH1* hDD_BstatDown = (TH1*) hCtrl_SI->Clone(bstatHistBase + "_Down");

    for (int ibin = 1; ibin <= hCtrl_SI->GetNbinsX(); ++ibin) {

      double b  = hCtrl_SI->GetBinContent(ibin);

      double eb = hCtrl_SI->GetBinError(ibin);

      hDD_BstatUp->SetBinContent(ibin, b + eb);

      hDD_BstatDown->SetBinContent(ibin, std::max(0.0, b - eb));

      hDD_BstatUp->SetBinError(ibin, 0.0);

      hDD_BstatDown->SetBinError(ibin, 0.0);

    }

    hDD_BstatUp->Scale(alpha);

    hDD_BstatDown->Scale(alpha);

    if (chDD->second.shapes[mainHisto.Data()].histo() == NULL) {

      hDD->SetFillColor(634);

      hDD->SetLineColor(1);

      hDD->SetMarkerColor(634);

      hDD->SetFillStyle(1001);

      hDD->SetLineWidth(1);

      hDD->SetMarkerStyle(20);

      hDD->SetLineStyle(1);

      chDD->second.shapes[mainHisto.Data()].uncShape[""] = hDD;

    }

    chDD->second.shapes[mainHisto.Data()].clearSyst();

    TString qcdSystName = chData->second.channel.c_str();

    std::string tfName =

      string("CMS_haa4b_sys_ddqcd_TF_") +

      qcdSystName.Data() + "_" +

      chData->second.bin.c_str() +

      year.Data() +

      systpostfix.Data();

    chDD->second.shapes[mainHisto.Data()].uncScale[tfName] = valDD * (relTF);

    //shapeInfo.uncScale["CMS_haa4b_sys_ddqcd_closure"] =

    //valDD * 0.1181;

    /*std::string closName =

      string("CMS_haa4b_sys_ddqcd_closure_") +

      qcdSystName.Data() + "_" +

      chData->second.bin.c_str() +

      year.Data() +

      systpostfix.Data();

    chDD->second.shapes[mainHisto.Data()].uncScale[closName] = valDD * (0.316);

    std::string NONQCDName =

      string("CMS_haa4b_sys_ddqcd_nonQCDsub_") +

      qcdSystName.Data() + "_" +

      chData->second.bin.c_str() +

      year.Data() +

      systpostfix.Data();

    */

    //chDD->second.shapes[mainHisto.Data()].uncScale[NONQCDName] = valDD;

    std::string bstatName =

      string("CMS_haa4b_sys_ddqcd_BtemplateStat_") +

      qcdSystName.Data() + "_" +

      chData->second.bin.c_str() +

      year.Data() +

      systpostfix.Data();

    chDD->second.shapes[mainHisto.Data()].uncShape[bstatName + "Up"]   = hDD_BstatUp;

    chDD->second.shapes[mainHisto.Data()].uncShape[bstatName + "Down"] = hDD_BstatDown;

    chDD->second.shapes[mainHisto.Data()].uncScale[bstatName] = -1.0;



    sprintf(Lcol, "%s%s", Lcol, "|c");

    sprintf(Lchan, "%s%25s", Lchan,

            (string(" & $ ") + chData->second.channel + string(" - ") +

             chData->second.bin + string(" $ ")).c_str());

    sprintf(Lalph1, "%s%25s", Lalph1,

            (string(" &") + utils::toLatexRounded(alpha, alpha_err)).c_str());

    sprintf(Lyield, "%s%25s", Lyield,

            (string(" &") + utils::toLatexRounded(valDD, valDD_err, valDD * relTF)).c_str());

    sprintf(LyieldMC, "%s%25s", LyieldMC,

            (string(" &") + utils::toLatexRounded(valMC, valMC_err)).c_str());

    sprintf(LalphMC, "%s%25s", LalphMC,

            (string(" &") + utils::toLatexRounded(alphaMC, alphaMC_err)).c_str());

    sprintf(LyieldPred, "%s%25s", LyieldPred,

            (string(" &") + utils::toLatexRounded(valDD_MC, valDD_MC_err)).c_str());

    sprintf(RatioMC, "%s%25s", RatioMC,

            (string(" &") + utils::toLatexRounded(ratioMC, ratioMC_err)).c_str());

    sprintf(LBstat, "%s%25s", LBstat,

          (string(" &") + utils::toLatexRounded(hDD_BstatUp->Integral() - hDD->Integral(), 0.0)).c_str());

    sprintf(LTF, "%s%25s", LTF,

            (string(" &") + utils::toLatexRounded(relTF, 0.0)).c_str());

       }

  for (auto itDD = procInfo_DD.channels.begin(); itDD != procInfo_DD.channels.end(); ) {

    const string& chname = itDD->first;

    bool keep =

      chname.find("veto_") != string::npos &&

      chname.find("_SR_")  != string::npos &&

      chname.find("_3b")   != string::npos &&

      chname.find("_A_")   != string::npos;

    if (!keep) itDD = procInfo_DD.channels.erase(itDD);

    else ++itDD;

  }

  procs["NonQCD"] = ProcessInfo_t();

  computeTotalBackground();

  if (pFile) {

    fprintf(pFile,

      "\\documentclass{article}\n"

      "\\usepackage[utf8]{inputenc}\n"

      "\\usepackage{rotating}\n"

      "\\begin{document}\n"

      "\\begin{sidewaystable}[htp]\n"

      "\\tiny\n"

      "\\begin{center}\n"

      "\\caption{Data-driven QCD background estimation.}\n"

      "\\label{tab:table}\n"

    );

    fprintf(pFile, "\\begin{tabular}{%s|}\\hline\n", Lcol);

    fprintf(pFile, "channel               %s\\\\\\hline\n", Lchan);

    fprintf(pFile, "$\\texttt{SF}_{qcd}$ measured    %s\\\\\n", Lalph1);

    fprintf(pFile, "QCD yield predicted data            %s\\\\\n", Lyield);

    fprintf(pFile, "\\hline\\hline\n");

    fprintf(pFile, "B-template stat shape yield shift %s\\\\\n", LBstat);

    fprintf(pFile, "C/D transfer-factor relative uncertainty %s\\\\\n", LTF);

    fprintf(pFile, "\\hline\\hline\n");

    fprintf(pFile, "$\\texttt{SF}_{qcd}$ MC            %s\\\\\n", LalphMC);

    fprintf(pFile, "QCD yield predicted MC              %s\\\\\n", LyieldPred);

    fprintf(pFile, "QCD yield observed MC               %s\\\\\n", LyieldMC);

    fprintf(pFile, "MC closure ratio pred/obs           %s\\\\\n", RatioMC);

    fprintf(pFile, "\\hline\n");

    fprintf(pFile, "\\end{tabular}\n\\end{center}\n\\end{sidewaystable}\n\\end{document}\n");

  }

  if (verbose) printf("\n\n --- verbose : AllInfo_t::doBackgroundSubtraction : end\n\n");

}

//

// Sum up all background processes and add this as a total process

//

void AllInfo_t::computeTotalBackground(){

  if ( verbose ) { printf("\n  --- verbose: computeTotalBackground : begin.\n" ) ; fflush(stdout) ; }

  for(std::vector<string>::iterator p=sorted_procs.begin(); p!=sorted_procs.end();p++){if((*p)=="total"){sorted_procs.erase(p);break;}}           

  sorted_procs.push_back("total");

  procs["total"] = ProcessInfo_t(); //reset

  ProcessInfo_t& procInfo_Bckgs = procs["total"];

  procInfo_Bckgs.shortName = "total";

  procInfo_Bckgs.isData = false;

  procInfo_Bckgs.isSign = false;

  procInfo_Bckgs.isBckg = true;

  procInfo_Bckgs.xsec   = 0.0;

  procInfo_Bckgs.br     = 1.0;

  //Compute total background nominal

  for(std::map<string, ProcessInfo_t>::iterator it=procs.begin(); it!=procs.end();it++){

    if(it->first=="total" || it->second.isBckg!=true)continue;

    if ( verbose ) { printf("    --- verbose: computeTotalBackground :  calling addProc( procInfo_Bckgs, it->second, false)  it->first = %s\n", (it->first).c_str() ) ; fflush(stdout) ; }

    addProc(procInfo_Bckgs, it->second, false);

  }

  //Compute total background systematics

  for(std::map<string, ProcessInfo_t>::iterator it=procs.begin(); it!=procs.end();it++){

    if(it->first=="total" || it->second.isBckg!=true)continue;

    if ( verbose ) { 

      printf("    --- verbose: computeTotalBackground :  calling addProc( procInfo_Bckgs, it->second, true)  it->first = %s\n", (it->first).c_str() ) ; fflush(stdout) ; }

    addProc(procInfo_Bckgs, it->second, true);

  }

  if ( verbose ) { printf(" ---  verbose: computeTotalBackground : end.\n\n" ) ; fflush(stdout) ; }

}


//

// Replace the Data process by TotalBackground

//

void AllInfo_t::blind() {

   if ( verbose ) { printf("\n  --- verbose : AllInfo_t::blind : begin.\n") ; fflush(stdout) ; }

  if(procs.find("total")==procs.end())computeTotalBackground();

  if(true){ //always replace data

    //if(procs.find("data")==procs.end()){ //true only if there is no "data" samples in the json file

    sorted_procs.push_back("data");           

    if ( verbose ) { printf("  verbose :  AllInfo_t::blind() :  before resetting data\n" ) ; procs["data"].printProcess() ; }

    procs["data"] = ProcessInfo_t(); //reset

    ProcessInfo_t& procInfo_Data = procs["data"];

    procInfo_Data.shortName = "data";

    procInfo_Data.isData = true;

    procInfo_Data.isSign = false;

    procInfo_Data.isBckg = false;

    procInfo_Data.xsec   = 0.0;

    procInfo_Data.br     = 1.0;

    for(std::map<string, ProcessInfo_t>::iterator it=procs.begin(); it!=procs.end();it++){

      if(it->first!="total")continue;

      /*

      for (std::map<string, ChannelInfo_t>::iterator ch=it->second.channels.begin(); ch!=it->second.channels.end();ch++){ 

        printf(" ---> Blind in channel %s :\n",ch->second.channel.c_str()); //find("CR")==string::npos));

      }

      */

      addProc(procInfo_Data, it->second);

    }

    if ( verbose ) { printf("  verbose :  AllInfo_t::blind() :  after resetting data\n" ) ; procs["data"].printProcess() ; }

  }

   if ( verbose ) { printf(" --- verbose : AllInfo_t::blind : end.\n\n") ; fflush(stdout) ; }

}



//---------------------------------------------------------------

void AllInfo_t::replaceHighSensitivityBinsWithBG() {

  if ( verbose ) { printf("\n  --- verbose : AllInfo_t::replaceHighSensitivityBinsWithBG : begin.\n") ; fflush(stdout) ; }

  if(procs.find("total")==procs.end())computeTotalBackground();

  std::map<string, ProcessInfo_t>::iterator itbg=procs.find("total");

  if ( itbg==procs.end() ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG : no total process???\n\n") ; return ; }

  std::map<string, ProcessInfo_t>::iterator idata=procs.find("data");

  if ( idata==procs.end() ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG : no data process???\n\n") ; return ; }

  ProcessInfo_t& total_proc = itbg -> second ;

  ProcessInfo_t& data_proc = idata -> second ;

  if ( verbose ) { printf(" AllInfo_t::replaceHighSensitivityBinsWithBG : before replacement.\n") ; total_proc.printProcess() ; data_proc.printProcess() ; }

  for ( std::map<string, ChannelInfo_t>::iterator ic = data_proc.channels.begin(); ic!= data_proc.channels.end(); ic++ ) {

     string chan_key = ic -> first ;

     ChannelInfo_t& data_chan = ic -> second ;

     if ( chan_key.find("SR")!=string::npos ) {

        std::map<string, ChannelInfo_t>::iterator itbgc = total_proc.channels.find( chan_key ) ;

        if ( itbgc == total_proc.channels.end() ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG : can't find channel %s in total BG!  bailing out.\n\n", chan_key.c_str() ) ; return ; }

        ChannelInfo_t& total_chan = itbgc -> second ;

        for ( std::map<string, ShapeData_t>::iterator is = data_chan.shapes.begin(); is!= data_chan.shapes.end(); is++ ) {

           string shape_key = is -> first ;

           ShapeData_t& data_shape = is -> second ;

           std::map<string, ShapeData_t>::iterator itbgs = total_chan.shapes.find( shape_key ) ;

           if ( itbgs == total_chan.shapes.end() ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG : can't find shape %s for channel %s in in total BG!  bailing out.\n\n", shape_key.c_str(), chan_key.c_str() ) ; return ; }

           ShapeData_t& total_shape = itbgs -> second ;

           TH1* data_hist = data_shape.histo() ;

           TH1* total_hist = total_shape.histo() ;

           if ( data_hist == 0x0 ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG : data hist is null pointer!!! bailing out.\n\n") ; return ; }

           if ( total_hist == 0x0 ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG : total BG hist is null pointer!!! bailing out.\n\n") ; return ; }

	   //HEREEEEE


      if ( data_hist -> GetNbinsX() != 5 ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG :  was expecting 5 bins.  found %d in data hist.  bailing out.\n\n", data_hist -> GetNbinsX() ) ; }

	   //           if ( total_hist -> GetNbinsX() != 5 ) { printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG :  was expecting 5 bins.  found %d in total BG hist.  bailing out.\n\n", total_hist -> GetNbinsX() ) ; }

	   if ( total_hist -> GetNbinsX() != 4 ) { 

	     printf("\n\n *** AllInfo_t::replaceHighSensitivityBinsWithBG :  was expecting 5 bins.  found %d in total BG hist.  bailing out.\n\n", total_hist -> GetNbinsX() ) ; 

	     int nbinlast = total_hist->GetNbinsX();

	     int nbinlastmone = total_hist->GetNbinsX() - 1;

	     // this case applies only to the Zh channel, 4B category:

	     data_hist -> SetBinContent( nbinlastmone, total_hist -> GetBinContent( nbinlastmone ) ) ;

	     data_hist -> SetBinContent( nbinlast, total_hist -> GetBinContent( nbinlast ) ) ;

	   } else {

	     // Standard cases:

	     data_hist -> SetBinContent( 3, total_hist -> GetBinContent( 3 ) ) ;

	     data_hist -> SetBinContent( 4, total_hist -> GetBinContent( 4 ) ) ;

	   }

	}

     }

  }

     // ic

  if ( verbose ) { printf(" AllInfo_t::replaceHighSensitivityBinsWithBG : after replacement.\n") ; data_proc.printProcess() ; }


  if ( verbose ) { printf("\n  --- verbose : AllInfo_t::replaceHighSensitivityBinsWithBG : end.\n") ; fflush(stdout) ; }

  }// AllInfo_t::replaceHighSensitivityBinsWithBG

//---------------------------------------------------------------



void AllInfo_t::getYieldsFromShape(FILE* pFile, std::vector<TString>& selCh, string histoName, FILE* pFileInc){

  if(!pFileInc)pFileInc=pFile;

  std::vector<string> VectorProc;

  std::map<string, bool> MapChannel;

  std::map<string, std::map<string, string> > MapProcChYields;         

  std::map<string, bool> MapChannelBin;

  std::map<string, std::map<string, string> > MapProcChYieldsBin;         

  std::map<string, std::map<string, std::vector<double> > > MapProcChBinYields;         

  std::map<string, std::map<string, std::vector<double> > > MapProcChBinErrors;         

  std::map<string, std::vector<int> > MapProcChBin;         

  std::vector<double> MapSignChBinYields;

  std::vector<double> MapBckgChBinYields;

  std::map<string, string> rows;

  std::map<string, string> rowsBin;

  string rows_header = "\\begin{tabular}{|c|";

  string rows_title  = "channel";

  //order the proc first

  sortProc();

  for(unsigned int p=0;p<sorted_procs.size();p++){

    string procName = sorted_procs[p];

    std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

    if(it==procs.end())continue;

    rows_header += "c|";

    rows_title  += "& " + it->second.shortName;

    std::map<string, double> bin_valerr;

    std::map<string, double> bin_val;

    std::map<string, double> bin_systUp;

    std::map<string, double> bin_systDown;

    VectorProc.push_back(it->first);

    for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

      if(std::find(selCh.begin(), selCh.end(), ch->second.channel)==selCh.end())continue;

      if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

      // Only get yields from shapes for regions A 

      if(modeDD && ((ch->first.find("_B_")!=string::npos) || (ch->first.find("_C_")!=string::npos) ||(ch->first.find("_D_")!=string::npos)))continue;   

      if((ch->first.find("_CR_")!=string::npos)) continue; // do not show CRs

      printf("Get yields from shapes:\n");

      printf("Process: %s , channel: %s \n",procName.c_str(),ch->first.c_str());

      fflush(stdout) ;

      TH1* h = ch->second.shapes[histoName].histo();

      double valerr = 0.;

      double val  = 0.;

      if (h!=NULL) {

        int start_bin = (docut ? h->FindBin(sstyCut) : 1);

        val = h->IntegralAndError(start_bin,h->GetXaxis()->GetNbins(),valerr);

      }

      double syst_scale = std::max(0.0, ch->second.shapes[histoName].getScaleUncertainty());

      double syst_shapeUp = std::max(0.0, ch->second.shapes[histoName].getIntegratedShapeUncertainty((it->first+ch->first).c_str(), "Up"));

      double syst_shapeDown = std::max(0.0, ch->second.shapes[histoName].getIntegratedShapeUncertainty((it->first+ch->first).c_str(), "Down"));

      double systUp = sqrt(pow(syst_scale,2)+pow(syst_shapeUp,2));

      double systDown = sqrt(pow(syst_scale,2)+pow(syst_shapeDown,2));

      systUp= (systUp >0)?systUp: -1; //Set to -1 if no syst, to be coherent with other convention in this file

      systDown= (systDown >0)?systDown: -1; //Set to -1 if no syst, to be coherent with other convention in this file

      if(val<1E-5 && valerr>=10*val && procName.find("ww")!=std::string::npos){val=0.0;}

      else if(val<1E-5 && valerr>=10*val){val=0.0; systUp=-1;systDown=-1;}

      else if(val<1E-6){val=0.0; valerr=0.0; systUp=-1;systDown=-1;}

      if(it->first=="data"){valerr=-1.0; systUp=-1;systDown=-1;}

      string YieldText = "";

      if(it->first=="data" || it->first=="total")YieldText += "\\boldmath ";

      if(it->first=="data"){char tmp[256];sprintf(tmp, "$%.0f$", val); YieldText += tmp;

      }else{                YieldText += utils::toLatexRounded(val,valerr, systUp, true, systDown);     }


      printf("%f %f %f %f --> %s\n", val, valerr, systUp, systDown, utils::toLatexRounded(val,valerr, systUp, true, systDown).c_str());

      if(rows.find(ch->first)==rows.end())rows[ch->first] = string("$ ")+ch->first+" $";

      rows[ch->first] += string("&") + YieldText;

      TString LabelText = TString("$") + ch->second.channel+ "\\ " +ch->second.bin + TString("$");

      LabelText.ReplaceAll("eq"," ="); LabelText.ReplaceAll("g =","\\geq"); LabelText.ReplaceAll("l =","\\leq"); LabelText.ReplaceAll("mu","\\mu"); LabelText.ReplaceAll("_","\\_");

      //      LabelText.ReplaceAll("_OS","OS "); LabelText.ReplaceAll("el","e"); LabelText.ReplaceAll("mu","\\mu");  LabelText.ReplaceAll("ha","\\tau_{had}");

      TString BinText = TString("$") + ch->second.bin + TString("$");

      BinText.ReplaceAll("eq"," ="); BinText.ReplaceAll("g =","\\geq"); BinText.ReplaceAll("l =","\\leq");

      //      BinText.ReplaceAll("_OS","OS "); BinText.ReplaceAll("el","e"); BinText.ReplaceAll("mu","\\mu");  BinText.ReplaceAll("ha","\\tau_{had}");


      bin_val   [BinText.Data()] = val;

      bin_valerr[BinText.Data()] = pow(valerr,2);

      bin_systUp  [BinText.Data()] = systUp>=0?pow(systUp,2):-1;

      bin_systDown  [BinText.Data()] = systDown>=0?pow(systDown,2):-1;

      if(systUp<0)bin_systUp  [BinText.Data()]=-1;

      if(systDown<0)bin_systDown  [BinText.Data()]=-1;

      bin_val   [" Inc."] += val;

      bin_valerr[" Inc."] += pow(valerr,2);

      bin_systUp  [" Inc."] += systUp>=0?pow(systUp,2):0; //We are doing a quadratic sum here, have to add 0 if we have negative value

      bin_systDown  [" Inc."] += systDown>=0?pow(systDown,2):0; //We are doing a quadratic sum here, have to add 0 if we have negative value

      MapChannel[LabelText.Data()] = true;

      MapProcChYields[it->first][LabelText.Data()] = YieldText;

    }

    if(bin_systUp  [" Inc."] <= 0) bin_systUp  ["Inc"]=-1; //If negative value, or 0, set it to -1

    if(bin_systDown  [" Inc."] <= 0) bin_systDown  ["Inc"]=-1; //If negative value, or 0, set it to -1

    for(std::map<string, double>::iterator bin=bin_val.begin(); bin!=bin_val.end(); bin++){

      string YieldText = "";                 

      if(it->first=="data" || it->first=="total" || bin->first==" Inc.")YieldText += "\\boldmath ";

      if(it->first=="data"){char tmp[256];sprintf(tmp, "%.0f", bin_val[bin->first]); YieldText += tmp;  //unblinded

                                //                 if(it->first=="data"){char tmp[256];sprintf(tmp, "-"); rowsBin[bin->first] += tmp;  //blinded

      }else{                YieldText += utils::toLatexRounded(bin_val[bin->first],sqrt(bin_valerr[bin->first]), bin_systUp[bin->first]<0?-1:sqrt(bin_systUp[bin->first]), true, bin_systDown[bin->first]<0?-1:sqrt(bin_systDown[bin->first]));   }

      if(rowsBin.find(bin->first)==rowsBin.end())rowsBin[bin->first] = string("$ ")+bin->first+" $";

      rowsBin[bin->first] += string("&") + YieldText;

      MapChannelBin[bin->first] = true;

      MapProcChYieldsBin[it->first][bin->first] = YieldText;                

      if(bin->first==" Inc."){

        MapChannel[bin->first] = true;

        MapProcChYields[it->first][bin->first] = YieldText;                

      }

    }

  }


    //All Channels

  fprintf(pFile,"\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{geometry}\n\\geometry{\n\tleft=10mm,\n\tright=10mm,\n\ttop=10mm,\n\tbottom=10mm\n}\n\\usepackage[utf8]{inputenc}\n\\usepackage{rotating}\n\\begin{document}\n\\begin{sidewaystable}[htp]\n\\begin{center}\n\\caption{Event yields expected for background and signal processes and observed in data.}\n\\label{tab:table}\n\\resizebox{\\textwidth}{!}{\n ");

  fprintf(pFile, "\\begin{tabular}{|c|"); for(auto ch = MapChannel.begin(); ch!=MapChannel.end();ch++){ fprintf(pFile, "c|"); } fprintf(pFile, "}\\hline\n");

  fprintf(pFile, "channel");   for(auto ch = MapChannel.begin(); ch!=MapChannel.end();ch++){ fprintf(pFile, " & %s", ch->first.c_str()); } fprintf(pFile, "\\\\\\hline\n");

  for(auto proc = VectorProc.begin();proc!=VectorProc.end(); proc++){

    if(*proc=="total")fprintf(pFile, "\\hline\n");

    auto ChannelYields = MapProcChYields.find(*proc);

    if(ChannelYields == MapProcChYields.end())continue;

    TString procName = (*proc).c_str(); if(procName.Contains("#")){procName = "$" + procName + "$";} procName.ReplaceAll("#","\\");

    fprintf(pFile, "%s ", procName.Data()); 

//    fprintf(pFile, "%s ", proc->c_str()); 

    for(auto ch = MapChannel.begin(); ch!=MapChannel.end();ch++){ 

      fprintf(pFile, " & ");

      if(ChannelYields->second.find(ch->first)!=ChannelYields->second.end()){

        fprintf(pFile, " %s", (ChannelYields->second)[ch->first].c_str());

      }

    }

    fprintf(pFile, "\\\\\n");

    if(*proc=="data")fprintf(pFile, "\\hline\n");             

  }

  fprintf(pFile,"\\hline\n");

  fprintf(pFile,"\\end{tabular}\n}\n\\end{center}\n\\end{sidewaystable}\n\\end{document}\n");

    //All Bins

  fprintf(pFileInc,"\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{rotating}\n\\begin{document}\n\\begin{sidewaystable}[htp]\n\\begin{center}\n\\caption{Event yields expected for background and signal processes and observed in data.}\n\\label{tab:table}\n");

  fprintf(pFileInc, "\\begin{tabular}{|c|"); for(auto ch = MapChannelBin.begin(); ch!=MapChannelBin.end();ch++){ fprintf(pFileInc, "c|"); } fprintf(pFileInc, "}\\hline\n");

  fprintf(pFileInc, "channel");   for(auto ch = MapChannelBin.begin(); ch!=MapChannelBin.end();ch++){ fprintf(pFileInc, " & %s", ch->first.c_str()); } fprintf(pFileInc, "\\\\\\hline\n");

  for(auto proc = VectorProc.begin();proc!=VectorProc.end(); proc++){

    if(*proc=="total")fprintf(pFileInc, "\\hline\n");

    auto ChannelYields = MapProcChYieldsBin.find(*proc);

    if(ChannelYields == MapProcChYieldsBin.end())continue;

    TString procName = (*proc).c_str(); procName.ReplaceAll("#","\\"); procName = "$" + procName + "$";

    fprintf(pFile, "%s ", procName.Data()); 

//    fprintf(pFileInc, "%s ", proc->c_str()); 

    for(auto ch = MapChannelBin.begin(); ch!=MapChannelBin.end();ch++){ 

      fprintf(pFileInc, " & ");

      if(ChannelYields->second.find(ch->first)!=ChannelYields->second.end()){

        fprintf(pFileInc, " %s", (ChannelYields->second)[ch->first].c_str());

      }

    }

    fprintf(pFileInc, "\\\\\n");

    if(*proc=="data")fprintf(pFileInc, "\\hline\n");             

  }

  fprintf(pFileInc,"\\hline\n");

  fprintf(pFileInc,"\\end{tabular}\n\\end{center}\n\\end{sidewaystable}\n\\end{document}\n");



}



  // Dump efficiencies



  void AllInfo_t::getEffFromShape(FILE* pFile, std::vector<TString>& selCh, string histoName)

  {

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end())continue;

      if(!it->second.isSign)continue;

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

        if(std::find(selCh.begin(), selCh.end(), ch->second.channel)==selCh.end())continue;

        if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

        TH1* h = ch->second.shapes[histoName].histo();

        double valerr = 0.;

        double val = 0.;

        if (h!=NULL) val = h->IntegralAndError(1,h->GetXaxis()->GetNbins(),valerr);

        fprintf(pFile,"%30s %30s %4.0f %6.2E %6.2E %6.2E %6.2E\n",ch->first.c_str(), it->first.c_str(), it->second.mass, it->second.xsec, it->second.br, val/(it->second.xsec*it->second.br), valerr/(it->second.xsec*it->second.br));

      }

    }

  }




  //

  // drop control channels

  //

  void AllInfo_t::dropCtrlChannels(std::vector<TString>& selCh)

  {

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end())continue;

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

        if(std::find(selCh.begin(), selCh.end(), ch->second.channel)==selCh.end()){

	  printf(" --- dropCtrlChannels::: process = %s, channel dropped : %s\n",procName.c_str(),ch->first.c_str());

	  it->second.channels.erase(ch); ch=it->second.channels.begin();}

      }

    }

  }



  //

  // drop background process that have a negligible yield

  //

  void AllInfo_t::dropSmallBckgProc(std::vector<TString>& selCh, string histoName, double threshold)

  {

    auto total = procs.find("total");

    if(total==procs.end()) {printf("dropSmallBckgProc: Error, cannot find process: total\n");return;}

    std::map<string, double> total_yields;

    std::map<string, std::map<string, double> > map_yields;

    for(std::map<string, ChannelInfo_t>::iterator ch = total->second.channels.begin(); ch!=total->second.channels.end(); ch++){

      if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

      TH1 *h=ch->second.shapes[histoName].histo();

      total_yields[ch->first] = 0.;

      if(h!=NULL) {total_yields[ch->first] = h->Integral();printf("dropsmallBckgProc, total in channel %s %f\n",ch->first.c_str(), total_yields[ch->first]);}

    }

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      if(procName.compare("total") == 0) continue;

      if ( procName.compare("ddqcd") == 0 ) {

         printf("  AllInfo_t::dropSmallBckgProc:  excluding ddqcd proc from consideration.  Always keep it.\n") ; fflush(stdout) ;

         continue ;

      }

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end())continue;

      if(!it->second.isBckg)continue;

//      printf("dropSmallBckgProc, Process: %s\n",  procName.c_str());

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end();ch++){

        if(std::find(selCh.begin(), selCh.end(), ch->second.channel)==selCh.end())continue;

        if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

        map_yields[it->first][ch->first] = 0.;  

        TH1 *h=ch->second.shapes[histoName].histo();

        if (h!=NULL) map_yields[it->first][ch->first] = h->Integral(); 

      }

    }

    for(std::map<string, std::map<string, double> >::iterator p = map_yields.begin();p!=map_yields.end();p++){

      for(std::map<string, double>::iterator ch = p->second.begin();ch!=p->second.end();ch++){

	if(p->first.find("t#bar{t} + light")<std::string::npos)continue;//never drop this background  

	if(p->first.find("t#bar{t} + b#bar{b}")<std::string::npos)continue;//never drop this background  

	if(p->first.find("t#bar{t} + c#bar{c}")<std::string::npos)continue;//never drop this background      

	if(p->first.find("Other Bkgs")<std::string::npos)continue;//never drop this background    

	if(p->first.find("Z#rightarrow  #nu #nu")<std::string::npos)continue;//never drop this background   

	//	if(p->first.find("Four-top (TTTT)")<std::string::npos)continue;//never drop this background  

	if(!runZh) { //Wh channel   

	  if(p->first.find("QCD")<std::string::npos)continue; //-Penny

	  if(p->first.find("W#rightarrow l#nu")<std::string::npos)continue;//never drop this background      

	}

        double tot = total_yields[ch->first];

        double yield = map_yields[p->first][ch->first];

        if(tot>0 && yield/tot<threshold){

          printf("Drop %s from the list of backgrounds in the channel %s because of negligible rate (%f of total bckq)\n", p->first.c_str(), ch->first.c_str(), yield/tot);

          procs.find(p->first)->second.channels.erase(procs.find(p->first)->second.channels.find(ch->first));

        }

      }

    }

  }



  //------------------------------------------------------------------------------------------------------

  //

  // Make a summary plot

  //

  void AllInfo_t::showShape(std::vector<TString>& selCh , TString histoName, TString SaveName)

  {

    if ( verbose ) {

       printf(" --- verbose : AllInfo_t::showShape :  begin \n") ;

       printf(" --- verbose : AllInfo_t::showShape :  channels : ") ;

       for ( int ci=0; ci<selCh.size(); ci++ ) { printf(" %s , ", selCh[ci].Data() ) ; }

       printf("\n") ;

       printf(" --- verbose : AllInfo_t::showShape :  histoName = %s , SaveName = %s\n", histoName.Data(), SaveName.Data() ) ;

       fflush(stdout) ;

    }

    int NLegEntry = 0;

    std::map<string, THStack*          > map_stack;

    std::map<string, TH1*              > map_mc;

    std::map<string, TGraphAsymmErrors*     > map_unc;

    std::map<string, TH1*              > map_uncH;

    std::map<string, TH1*              > map_data;

    std::map<string, TGraphAsymmErrors*> map_dataE;

    std::map<string, std::vector<TH1*> > map_signals;

    std::map<string, int               > map_legend;

    TLegend *legA = new TLegend(0.30,0.74,0.93,0.96, "NDC");

    legA->SetHeader("");

    legA->SetNColumns(3);   

    legA->SetBorderSize(0);

    legA->SetTextFont(42);   legA->SetTextSize(0.03);

    legA->SetLineColor(0);   legA->SetLineStyle(1);   legA->SetLineWidth(1);

    legA->SetFillColor(0); legA->SetFillStyle(0);//blind>-1E99?1001:0);

    std::vector<TLegendEntry*> legEntries;  //needed to have the entry in reverse order

    //order the proc first

    sortProc();

    //loop on sorted proc

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      TString process(procName.c_str());

      //      if( process.Contains("BOnly_B") || process.Contains("SandBandInterf_SBI") ) continue;

      if ( verbose ) { printf(" --- verbose : AllInfo_t::showShape :  proc = %s\n", process.Data() ) ; fflush(stdout) ; }

      if(it==procs.end())continue;

      //loop on channels for each process

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

        if(std::find(selCh.begin(), selCh.end(), ch->second.channel)==selCh.end())continue;

        if(ch->second.shapes.find(histoName.Data())==(ch->second.shapes).end())continue;

        if(modeDD && ch->first.find("_A_")==std::string::npos) continue; //  only consider region A

        TH1* h = ch->second.shapes[histoName.Data()].histo();

        if (!h) continue;

        if ( verbose ) { printf(" --- verbose : AllInfo_t::showShape :  proc = %s , chan = %s, hist = %s\n", process.Data(), ch->first.c_str(), h->GetName() ) ; fflush(stdout) ; }

        //if(process.Contains("SOnly_S") ) h->Scale(10);  

        if(it->first=="total"){

          double syst_scale = std::max(0.0, ch->second.shapes[histoName.Data()].getScaleUncertainty());

	  double Uncertainty_scale=syst_scale / h->Integral();

          double Maximum = 0;

          TGraphAsymmErrors* errors = new TGraphAsymmErrors(h->GetXaxis()->GetNbins());



          errors->SetFillStyle(3005);

          errors->SetFillColor(kGray+3);                    

          errors->SetLineStyle(1);

          errors->SetLineColor(1);

          int icutg=0;

          for(int ibin=1; ibin<=h->GetXaxis()->GetNbins(); ibin++){

            if(h->GetBinContent(ibin)>0)

              errors->SetPoint(icutg,h->GetXaxis()->GetBinCenter(ibin), h->GetBinContent(ibin));

            //This is the part where we define which errors will be shown on the shape plot

            double syst_shape_binUp = std::max(0.0, ch->second.shapes[histoName.Data()].getBinShapeUncertainty((it->first+ch->first).c_str(), ibin, "Up"));

            double syst_shape_binDown = std::max(0.0, ch->second.shapes[histoName.Data()].getBinShapeUncertainty((it->first+ch->first).c_str(), ibin, "Down"));

            double syst_binUp = sqrt(pow(Uncertainty_scale*h->GetBinContent(ibin), 2) + pow(syst_shape_binUp,2));

            double syst_binDown = sqrt(pow(Uncertainty_scale*h->GetBinContent(ibin), 2) + pow(syst_shape_binDown,2));

            double Uncertainty_binUp = syst_binUp / h->GetBinContent(ibin);

            double Uncertainty_binDown = syst_binDown / h->GetBinContent(ibin);

            //errors->SetPointError(icutg,h->GetXaxis()->GetBinWidth(ibin)/2.0, sqrt(pow(h->GetBinContent(ibin)*Uncertainty_bin,2) + pow(h->GetBinError(ibin),2) ) );

            errors->SetPointError(icutg,h->GetXaxis()->GetBinWidth(ibin)/2.0,h->GetXaxis()->GetBinWidth(ibin)/2.0, sqrt(pow(h->GetBinContent(ibin)*Uncertainty_binDown,2) + pow(h->GetBinError(ibin),2) ), sqrt(pow(h->GetBinContent(ibin)*Uncertainty_binUp,2) + pow(h->GetBinError(ibin),2) ) );

            //                        printf("Unc=%6.2f  X=%6.2f Y=%6.2f+-%6.2f+-%6.2f=%6.2f\n", Uncertainty, h->GetXaxis()->GetBinCenter(ibin), h->GetBinContent(ibin), h->GetBinContent(ibin)*Uncertainty, h->GetBinError(ibin), sqrt(pow(h->GetBinContent(ibin)*Uncertainty,2) + pow(h->GetBinError(ibin),2) ) );

            //                        errors->SetPointError(icutg,h->GetXaxis()->GetBinWidth(ibin)/2.0, 0 );

            Maximum =  std::max(Maximum , h->GetBinContent(ibin) + errors->GetErrorYhigh(icutg));

            icutg++;

          }errors->Set(icutg);

          errors->SetMaximum(Maximum);

          map_unc[ch->first] = errors;

          map_uncH[ch->first] = (TH1D*)h->Clone((ch->first+"histPlusSyst").c_str()); //utils::root::checkSumw2((ch->first+"histPlusSyst").c_str());

          // loop over hist to set the stat+syst errors on map_uncH

          icutg=0;

          for(int ibin=1; ibin<=map_uncH[ch->first]->GetXaxis()->GetNbins(); ibin++){

            double ierr=errors->GetErrorY(icutg);

            map_uncH[ch->first]->SetBinError(ibin,ierr);

            icutg++;

          }

	  //	  if ( verbose ) { printf(" --- verbose : AllInfo_t::showShape :  proc = %s\n", process.Data() ) ; fflush(stdout) ; }

          continue;//otherwise it will fill the legend

        }else if(it->second.isBckg){                 

          if(map_stack.find(ch->first)==map_stack.end()){

            map_stack[ch->first] = new THStack((ch->first+"stack").c_str(),(ch->first+"stack").c_str());

            map_mc   [ch->first] = (TH1D*)h->Clone((ch->first+"mc").c_str()); //utils::root::checkSumw2((ch->first+"mc").c_str());

          }else{

            map_mc [ch->first]->Add(h); 

          }

          map_stack   [ch->first]->Add(h,"HIST");

          //if(h!=NULL && h->Integral()>0){map_mc   [ch->first] = (TH1D*)h->Clone((ch->first+"mc").c_str());utils::root::checkSumw2((ch->first+"mc").c_str());}else{map_mc   [ch->first]->Add(h);}

        }else if(it->second.isSign){                    

          map_signals [ch->first].push_back(h);

        }else if(it->first=="data"){

          h->SetFillStyle(0);

          h->SetFillColor(0);

          h->SetMarkerSize(0.7);

          h->SetMarkerStyle(20);

          h->SetMarkerColor(1);

          h->SetBinErrorOption(TH1::kPoisson);

          map_data[ch->first] = h;

          //poisson error bars

          const double alpha = 1 - 0.6827;

          TGraphAsymmErrors * g = new TGraphAsymmErrors(h);

          g->SetMarkerSize(0.7);

          g->SetMarkerStyle (20);

          for (int i = 0; i < g->GetN(); ++i) {

            int N = g->GetY()[i];

            double L =  (N==0) ? 0  : (ROOT::Math::gamma_quantile(alpha/2,N,1.));

            double U =  ROOT::Math::gamma_quantile_c(alpha/2,N+1,1) ;

            g->SetPointEYlow(i, N-L);

            g->SetPointEYhigh(i, U-N);

          }

          map_dataE[ch->first] = g;

        }

        if(map_legend.find(it->first)==map_legend.end()){

          map_legend[it->first]=1;

          if(it->first=="data"){

            legA->AddEntry(h,it->first.c_str(),"PE0");

          }else if(it->second.isSign){

            legEntries.insert(legEntries.begin(), new TLegendEntry(h, it->first.c_str(), "L") );

          }else{

            legEntries.push_back(new TLegendEntry(h, it->first.c_str(), "F") );

          }

          NLegEntry++;

        }

      }

    }

    //fill the legend in reverse order

    while(!legEntries.empty()){

      legA->AddEntry(legEntries.back()->GetObject(), legEntries.back()->GetLabel(), legEntries.back()->GetOption());

      legEntries.pop_back();      

    }

    if(map_unc.begin()!=map_unc.end())legA->AddEntry(map_unc.begin()->second, "Syst. + Stat.", "F");


    TCanvas* c[50];

    int I=1;

    for(std::map<string, THStack*>::iterator p = map_stack.begin(); p!=map_stack.end(); p++){

      //init tab

      string ires;

      ostringstream convert;

      convert << I;   

      int NBins = map_data.size()/selCh.size();

      c[I] = new TCanvas("c_"+(char)I,"c_",800,800); //selCh.size());

      TPad* t1 = new TPad("t1","t1", 0.0, 0.2, 1.0, 1.0);

      t1->SetFillColor(0);

      t1->SetBorderMode(0);

      t1->SetBorderSize(2);

      t1->SetTickx(1);

      t1->SetTicky(1);

      t1->SetLeftMargin(0.10);

      t1->SetRightMargin(0.05);

      t1->SetTopMargin(0.05);

      t1->SetBottomMargin(0.10);

      t1->SetFrameFillStyle(0);

      t1->SetFrameBorderMode(0);

      t1->SetFrameFillStyle(0);

      t1->SetFrameBorderMode(0);

      t1->Draw();

      t1->cd();

      t1->SetLogy(useLogy); 

      //print histograms

      TH1* axis = (TH1*)map_data[p->first]->Clone("axis");

      axis->Reset();      

      if (histoName.Contains("bdt")) axis->GetXaxis()->SetRangeUser(0.0, 1.0); //-Penny

      axis->SetMaximum(1.5*std::max(map_unc[p->first]->GetMaximum(), map_data[p->first]->GetMaximum()));       

      //hard code range

      if(useLogy){

        if(procs["data"].channels[p->first].bin.find("vbf")!=string::npos){

          axis->SetMinimum(1E-1);

          axis->SetMaximum(std::max(axis->GetMaximum(), 5E1));

        }else{

          axis->SetMinimum(1E-2);

          axis->SetMaximum(std::max(axis->GetMaximum(), 1E5));

        }

      }

      axis->GetXaxis()->SetLabelOffset(0.007);

      axis->GetXaxis()->SetLabelSize(0.04);

      axis->GetXaxis()->SetTitleOffset(1.2);

      axis->GetXaxis()->SetTitleFont(42);

      axis->GetXaxis()->SetTitleSize(0.04);

      axis->GetYaxis()->SetLabelFont(42);

      axis->GetYaxis()->SetLabelOffset(0.007);

      axis->GetYaxis()->SetLabelSize(0.04);

      axis->GetYaxis()->SetTitleOffset(1.35);

      axis->GetYaxis()->SetTitleFont(42);

      axis->GetYaxis()->SetTitleSize(0.04);

      if ( startsWith(p->first,"veto_A_SR_3b")) {

        int bbin=axis->FindBin(0.9); //map_data[p->first]->FindBin(0.1); // Penny

        for(unsigned int i=bbin;i<axis->GetNbinsX()+1; i++){   

          axis->SetBinContent(i, 0); axis->SetBinError(i, 0);

        }

      }

      //if((I-1)%NBins!=0)

      axis->GetYaxis()->SetTitle("Events");

      axis->Draw();

      t1->Update();

      p->second->Draw("same"); // MC stack histogram

      map_unc [p->first]->Draw("2 same");

      for(unsigned int i=0;i<map_signals[p->first].size();i++){

        TH1* hs= map_signals[p->first][i];

        if (hs) {

          if ( startsWith(p->first,"veto_A_SR_3b"))// || startsWith(p->first,"mu_A_SR_4b") ||

            hs->Scale(signalScale);

          hs->Draw("HIST same");

        }

      }

      if (histoName.Contains("bdt")) axis->GetXaxis()->SetRangeUser(0.0, 1.0); //-Penny

      if(blindSR){

	if ( startsWith(p->first,"veto_A_SR_3b"))

	  {

         std::cout << "Channel name: " << p->first << std::endl;

         int bbin=axis->FindBin(0.54); // Penny

         int totNbins = map_dataE[p->first]->GetN();

         for (int i=bbin-1; i<totNbins; i++){

           map_dataE[p->first]->RemovePoint(bbin-1);

         }

         //TH1 *hist=(TH1*)p->second->GetHistogram(); \\ Penny

         TPave* blinding_box = new TPave(axis->GetBinLowEdge(axis->FindBin(0.9)), axis->GetMinimum(),axis->GetXaxis()->GetXmax(), axis->GetMaximum(), 0, "NB" );  

         blinding_box->SetFillColor(15); blinding_box->SetFillStyle(3013); blinding_box->Draw("same F");

        }

      }

      if(!blindData) map_dataE[p->first]->Draw("P0 same");

      bool printBinContent = false;

      if(printBinContent){

        TLatex* tex = new TLatex();

        tex->SetTextSize(0.04); tex->SetTextFont(42);

        tex->SetTextAngle(60);

        TH1* histdata = map_data[p->first];

        double Xrange = axis->GetXaxis()->GetXmax()-axis->GetXaxis()->GetXmin();

        for(int xi=1;xi<=histdata->GetNbinsX();++xi){

          double x=histdata->GetBinCenter(xi);

          double y=histdata->GetBinContent(xi);

          double yData=histdata->GetBinContent(xi);

          int graphBin=-1;  for(int k=0;k<map_unc[p->first]->GetN();k++){if(fabs(map_unc[p->first]->GetX()[k] - x)<histdata->GetBinWidth(xi)){graphBin=k;}} 

          if(graphBin<0){

            printf("MC bin not found for X=%f\n", x);

          }else{

            double yMC =  map_unc[p->first]->GetY()[graphBin];

            double yMCerr = map_unc[p->first]->GetErrorY(graphBin);

            y = std::max(y, yMC+yMCerr);

            if(yMC>=1){tex->DrawLatex(x-0.02*Xrange,y*1.15,Form("#color[4]{B=%.1f#pm%.1f}",yMC, yMCerr));

            }else{     tex->DrawLatex(x-0.02*Xrange,y*1.15,Form("#color[4]{B=%.2f#pm%.2f}",yMC, yMCerr));

            }

          }

          tex->DrawLatex(x+0.02*Xrange,y*1.15,Form("D=%.0f",yData));                 

        }

      }

      //print tab channel header

      TPaveText* Label = new TPaveText(0.2,0.81,0.84,0.89, "NDC");

      Label->SetFillColor(0);  Label->SetFillStyle(0);  Label->SetLineColor(0); Label->SetBorderSize(0);  Label->SetTextAlign(31);

      TString LabelText = procs["data"].channels[p->first].channel+"  "+procs["data"].channels[p->first].bin;

      LabelText.ReplaceAll("veto_","veto "); 

      gPad->RedrawAxis();

      legA->Draw("same");    legA->SetTextFont(42);

      //double iLumi=36.3;

      double iLumi=108960; // -Penny

      double iEcm=13.6;

      if(lumi > 0) iLumi = lumi;

      c[I]->cd();

      TPad *t2 = new TPad("t2", "t2",0.0,0.0, 1.0,0.2);

      t2->SetFillColor(0);

      t2->SetBorderMode(0);

      t2->SetBorderSize(2);

      t2->SetGridy();

      t2->SetTickx(1);

      t2->SetTicky(1);

      t2->SetLeftMargin(0.10);

      t2->SetRightMargin(0.05);

      t2->SetTopMargin(0.0);

      t2->SetBottomMargin(0.20);

      t2->SetFrameFillStyle(0);

      t2->SetFrameBorderMode(0);

      t2->SetFrameFillStyle(0);

      t2->SetFrameBorderMode(0);

      t2->Draw();

      t2->cd();

      t2->SetGridy(true);

      t2->SetPad(0,0.0,1.0,0.2);


      TH1D* denSystUncH = (TH1D*)map_uncH[p->first];

      utils::root::checkSumw2(denSystUncH);

      if(blindSR){

	if ( startsWith(p->first,"veto_A_SR_3b")){

	   int bbin=denSystUncH->FindBin(0.54); // Penny 

          for(unsigned int i=bbin;i<=denSystUncH->GetNbinsX()+1; i++){   

            denSystUncH->SetBinContent(i, 0); denSystUncH->SetBinError(i, 0);

          }

        }

      }

      int GPoint=0;

      TGraphErrors *denSystUnc=new TGraphErrors(denSystUncH->GetXaxis()->GetNbins()); 

      for(int xbin=1; xbin<=denSystUncH->GetXaxis()->GetNbins(); xbin++){

        denSystUnc->SetPoint(GPoint, denSystUncH->GetBinCenter(xbin), 1.0);

        denSystUnc->SetPointError(GPoint,  denSystUncH->GetBinWidth(xbin)/2, denSystUncH->GetBinContent(xbin)!=0?denSystUncH->GetBinError(xbin)/denSystUncH->GetBinContent(xbin):0);

        GPoint++;

        //if(denSystUncH->GetBinContent(xbin)==0) {std::cout << "!!!!!!! bin i: " << xbin << " is 0" << std::endl;continue;}

        if(denSystUncH->GetBinContent(xbin)==0) {continue;}

        Double_t err=denSystUncH->GetBinError(xbin)/denSystUncH->GetBinContent(xbin);

        denSystUncH->SetBinContent(xbin,1);

        denSystUncH->SetBinError(xbin,err);

      }denSystUnc->Set(GPoint);

      denSystUnc->SetLineColor(1);

      denSystUnc->SetFillStyle(3004);

      denSystUnc->SetFillColor(kGray+2);

      denSystUnc->SetMarkerColor(1);

      denSystUnc->SetMarkerStyle(1);

      denSystUncH->Reset("ICE");       

      denSystUncH->SetTitle("");

      denSystUncH->SetStats(kFALSE);

      denSystUncH->Draw();

      denSystUnc->Draw("2 0 SAME");

      float yscale = (1.0-0.2)/(0.2);       

      denSystUncH->GetYaxis()->SetTitle("Data/#Sigma Bkg.");

      denSystUncH->GetXaxis()->SetTitle(""); //drop the tile to gain space

      //denSystUncH->GetYaxis()->CenterTitle(true);

      denSystUncH->SetMinimum(0.4);

      denSystUncH->SetMaximum(1.6);

      denSystUncH->GetXaxis()->SetLabelFont(42);

      denSystUncH->GetXaxis()->SetLabelOffset(0.007);

      denSystUncH->GetXaxis()->SetLabelSize(0.04 * yscale);

      denSystUncH->GetXaxis()->SetTitleFont(42);

      denSystUncH->GetXaxis()->SetTitleSize(0.035 * yscale);

      denSystUncH->GetXaxis()->SetTitleOffset(0.8);

      denSystUncH->GetYaxis()->SetLabelFont(42);

      denSystUncH->GetYaxis()->SetLabelOffset(0.007);

      denSystUncH->GetYaxis()->SetLabelSize(0.03 * yscale);

      denSystUncH->GetYaxis()->SetTitleFont(42);

      denSystUncH->GetYaxis()->SetTitleSize(0.035 * yscale);

      denSystUncH->GetYaxis()->SetTitleOffset(0.3);



         //add comparisons

      TString name("CompHistogram"); //name+=icd;

      TH1 *dataToObsH = (TH1D*)map_data[p->first]->Clone(name);

      utils::root::checkSumw2(dataToObsH);

      TH1 *mc = (TH1D*)map_mc[p->first]->Clone("mc");

      utils::root::checkSumw2(mc);

      dataToObsH->Divide(mc);

      TGraphErrors* dataToObs = new TGraphErrors(dataToObsH);

      dataToObs->SetMarkerColor(1);

      dataToObs->SetMarkerStyle(20);

      dataToObs->SetMarkerSize(0.7);

      if(blindSR){

        if ( startsWith(p->first,"veto_A_SR_3b")){

         int bbin=axis->FindBin(0.54);

         int totNbins = dataToObs->GetN();

         for (int i=bbin-1; i<totNbins; i++){

           dataToObs->RemovePoint(bbin-1);

         }

        }

      }

      dataToObs->Draw("P 0 SAME");


      if(blindSR){

        if ( startsWith(p->first,"veto_A_SR_3b")){

          TPave* blinding_box = new TPave(axis->GetBinLowEdge(axis->FindBin(0.9)), 0.4,axis->GetXaxis()->GetXmax(), 1.6, 0, "NB" );  

          blinding_box->SetFillColor(15); blinding_box->SetFillStyle(3013); blinding_box->Draw("same F");

        }

      }

      TLegend *legR = new TLegend(0.56,0.78,0.93,0.96, "NDC");

      legR->SetHeader("");

      legR->SetNColumns(2);

      legR->SetBorderSize(1);

      legR->SetTextFont(42);   legR->SetTextSize(0.03 * yscale);

      legR->SetLineColor(1);   legR->SetLineStyle(1);   legR->SetLineWidth(1);

      legR->SetFillColor(0);   legR->SetFillStyle(1001);//blind>-1E99?1001:0);

      //legR->AddEntry(denRelUnc, "Stat. Unc.", "F");

      legR->AddEntry(denSystUnc, "Syst. + Stat.", "F");

      //legR->Draw("same");

      //  gPad->RedrawAxis();

      t1->cd();

      c[I]->cd();

      utils::root::DrawPreliminary(iLumi, iEcm, t1);

      c[I]->Modified();  

      c[I]->Update();

      //save canvas

      LabelText.ReplaceAll(" ","_"); 

      c[I]->SaveAs(LabelText+"_Shape"+year+".root");

      c[I]->SaveAs(LabelText+"_Shape"+year+".pdf");

      c[I]->SaveAs(LabelText+"_Shape"+year+".C");

      delete c[I];

      I++;

    }

    if ( verbose ) {

       printf(" --- verbose : AllInfo_t::showShape :  end \n") ;

       fflush(stdout) ;

    }

  } // showShape


  //------------------------------------------------------------------------------------------------------

  //

  // Make a summary plot

  //

  void AllInfo_t::showUncertainty(std::vector<TString>& selCh , TString histoName, TString SaveName)

  {

    string UncertaintyOnYield="";  char txtBuffer[4096];

    TFile *unc_f = TFile::Open("unc.root", "recreate");

    sprintf(txtBuffer,"\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{geometry}\n\\geometry{\n\tleft=10mm,\n\tright=10mm,\n\ttop=10mm,\n\tbottom=10mm\n}\n\\usepackage[utf8]{inputenc}\n\\usepackage{rotating}\n\\begin{document}\n"); UncertaintyOnYield += txtBuffer;    

    //loop on sorted proc

    for(unsigned int p=0;p<sorted_procs.size();p++){

      int NLegEntry = 0;

      std::map<string, int               > map_legend;

      std::vector<TH1*>                    toDelete;             

      TLegend* legA  = new TLegend(0.03,0.89,0.97,0.95, "");

      legA->SetTextSize(0.015);

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end())continue;

      //if(it->first=="total" || it->first=="data")continue;  //only do samples which have systematics

      if(it->first=="data")continue;  //only do samples which have systematics  

      std::map<string, bool> mapUncType;

      std::map<string, std::map< string, double> > mapYieldPerBin;

      std::map<string, std::pair< double, double> > mapYieldInc;


      int NBins = it->second.channels.size()/selCh.size() > 1 ? it->second.channels.size()/selCh.size() : 2;

//      if((it->second.channels.size()-NBins*selCh.size())>0) NBins++;

      TCanvas* c1 = new TCanvas("c1","c1",300*NBins,300*selCh.size());

      c1->SetTopMargin(0.00); c1->SetRightMargin(0.00); c1->SetBottomMargin(0.00);  c1->SetLeftMargin(0.00);

      TPad* t2 = new TPad("t2","t2", 0.03, 0.90, 1.00, 1.00, -1, 1);  t2->Draw();  c1->cd();

      t2->SetTopMargin(0.00); t2->SetRightMargin(0.00); t2->SetBottomMargin(0.00);  t2->SetLeftMargin(0.00);

      TPad* t1 = new TPad("t1","t1", 0.03, 0.03, 1.00, 0.90, 4, 1);  t1->Draw();  t1->cd();

      t1->SetTopMargin(0.00); t1->SetRightMargin(0.00); t1->SetBottomMargin(0.00);  t1->SetLeftMargin(0.00);

      t1->Divide(NBins, selCh.size(), 0, 0);

      int I=1;

      mapYieldInc[""].first = 0;  mapYieldInc[""].second = 0;

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++, I++){

        if(std::find(selCh.begin(), selCh.end(), ch->second.channel)==selCh.end())continue;

        if(ch->second.shapes.find(histoName.Data())==(ch->second.shapes).end())continue;


        //-- Look for the total BG process.  If found, find the corresponding total background histogram.

        //     Will pass it to makeStatUnc so that it can decide whether to include BinByBin stat uncertainty based on err_i / sqrt( N(total)_i ) for each bin i.

        TH1* h_total(0x0) ;

        std::map<string, ProcessInfo_t>::iterator tpi = procs.find("total") ;

        if ( tpi != procs.end() ) {

           ProcessInfo_t total_proc = tpi->second ;

           std::map<string, ChannelInfo_t>::iterator tci = total_proc.channels.find( ch->first ) ;

           if ( tci != total_proc.channels.end() ) {

              ChannelInfo_t total_chan = tci -> second ;

              std::map<string, ShapeData_t>::iterator tsi = total_chan.shapes.find( histoName.Data() ) ;

              if ( tsi != total_chan.shapes.end() ) {

                 ShapeData_t total_shape = tsi -> second ;

                 h_total = total_shape.histo() ;

              } // tsi

           } // tci

        } // tpi


        //add the stat uncertainty is there;

        //ch->second.shapes[histoName.Data()].makeStatUnc("_CMS_haa4b_", (TString("_")+ch->first+"_"+it->second.shortName).Data(),systpostfix.Data(), false );//add stat uncertainty to the uncertainty map;

        ch->second.shapes[histoName.Data()].makeStatUnc(

      "CMS_haa4b_",

      (TString("_") + ch->first + "_" + it->second.shortName).Data(),

      systpostfix.Data(),

      false,

      h_total

    );

        TVirtualPad* pad = t1->cd(I); 

        pad->SetTopMargin(0.06); pad->SetRightMargin(0.03); pad->SetBottomMargin(0.07);  pad->SetLeftMargin(0.06);

        //pad->SetLogy(true); 

        //TH1* h = (TH1*)(ch->second.shapes[histoName.Data()].histo()->Clone((it->first+ch->first+"Nominal").c_str())); 

        TH1* hh = ch->second.shapes[histoName.Data()].histo();

        if (hh==NULL) continue;

        TH1* h = (TH1*)(hh->Clone((it->first+ch->first+"Nominal").c_str())); 

        double yield = h->Integral();

        toDelete.push_back(h);

        mapYieldPerBin[""][ch->first] = yield;

        mapYieldInc[""].first  += yield;

        mapYieldInc[""].second = 1;

        //print histograms

        TH1* axis = (TH1*)h->Clone("axis");

        axis->Reset();


	if(histoName.Contains("bdt")) axis->GetXaxis()->SetRangeUser(0.0, 1.0); 


        axis->GetYaxis()->SetRangeUser(0.5, 1.5); 

        if((I-1)%NBins!=0)axis->GetYaxis()->SetTitle("");

        axis->Draw();

        toDelete.push_back(axis);


        //print tab channel header

        TPaveText* Label = new TPaveText(0.1,0.81,0.94,0.89, "NDC");

        Label->SetFillColor(0);  Label->SetFillStyle(0);  Label->SetLineColor(0); Label->SetBorderSize(0);  Label->SetTextAlign(31);

        TString LabelText = ch->second.channel+"  -  "+ch->second.bin;

        LabelText.ReplaceAll("eq","="); LabelText.ReplaceAll("l=","#leq");LabelText.ReplaceAll("g=","#geq"); 

	//        LabelText.ReplaceAll("_OS","OS "); LabelText.ReplaceAll("el","e"); LabelText.ReplaceAll("mu","#mu");  LabelText.ReplaceAll("ha","#tau_{had}");

        Label->AddText(LabelText);  Label->Draw();

        TLine* line = new TLine(axis->GetXaxis()->GetXmin(), 1.0, axis->GetXaxis()->GetXmax(), 1.0);

        toDelete.push_back((TH1*)line);

        line->SetLineWidth(2);  line->SetLineColor(1); line->Draw("same");

        if(I==1){legA->AddEntry(line,"Nominal","L");  NLegEntry++;}

        int ColorIndex=3;

        //draw scale uncertainties

        for(std::map<string, double>::iterator var = ch->second.shapes[histoName.Data()].uncScale.begin(); var!=ch->second.shapes[histoName.Data()].uncScale.end(); var++){

          if(h->Integral()<=0)continue;

          double ScaleChange   = var->second/h->Integral();

          double ScaleUp   = 1 + ScaleChange;

          double ScaleDn   = 1 - ScaleChange;

          TString systName = var->first.c_str();

          systName.ToLower();

          systName.ReplaceAll("cms","");

          systName.ReplaceAll("haa4b","");

          systName.ReplaceAll("sys","");

          systName.ReplaceAll("13p6tev","");

          systName.ReplaceAll("_","");

          systName.ReplaceAll("up","");

          systName.ReplaceAll("down","");

          TLine* lineUp = new TLine(axis->GetXaxis()->GetXmin(), ScaleUp, axis->GetXaxis()->GetXmax(), ScaleUp);

          TLine* lineDn = new TLine(axis->GetXaxis()->GetXmin(), ScaleDn, axis->GetXaxis()->GetXmax(), ScaleDn);

          toDelete.push_back((TH1*)lineUp);  toDelete.push_back((TH1*)lineDn);


          int color = ColorIndex;

          if(map_legend.find(systName.Data())==map_legend.end()){

            map_legend[systName.Data()]=color;

            legA->AddEntry(lineUp,systName.Data(),"L");

            NLegEntry++;

            ColorIndex++;

          }else{

            color = map_legend[systName.Data()];

          }

          if(mapYieldPerBin[systName.Data()].find(ch->first)==mapYieldPerBin[systName.Data()].end()){

            mapYieldPerBin[systName.Data()][ch->first] = fabs( ScaleChange );                       

            mapUncType[systName.Data()] = false;

          }else{

            mapYieldPerBin[systName.Data()][ch->first] = std::max( ScaleChange , mapYieldPerBin[systName.Data()][ch->first]);

          }

          if(mapYieldInc.find(systName.Data())==mapYieldInc.end()){ mapYieldInc[systName.Data()].first = 0; mapYieldInc[systName.Data()].second = 0;  }

          mapYieldInc[systName.Data()].first += ScaleChange*h->Integral();

          mapYieldInc[systName.Data()].second += h->Integral();

          lineUp->SetLineWidth(1);  lineUp->SetLineColor(color); lineUp->Draw("same");

          lineDn->SetLineWidth(1);  lineDn->SetLineColor(color); lineDn->Draw("same");

        }

        //draw shape uncertainties

        //double syst_shape = std::max(0.0, ch->second.shapes[histoName.Data()].getShapeUncertainty((it->first+ch->first).c_str()));

        for(std::map<string, TH1*>::iterator var = ch->second.shapes[histoName.Data()].uncShape.begin(); var!=ch->second.shapes[histoName.Data()].uncShape.end(); var++){

          if(var->first=="")continue;

          TH1* hvar = (TH1*)(var->second->Clone((it->first+ch->first+var->first).c_str())); 

          double varYield = hvar->Integral();

          hvar->Divide(h);

          toDelete.push_back(hvar);

          TString systName = var->first.c_str();

          systName.ToLower();

          systName.ReplaceAll("cms","");

          systName.ReplaceAll("haa4b","");

          systName.ReplaceAll("sys","");

          systName.ReplaceAll("13p6tev","");

          systName.ReplaceAll("_","");

          //systName.ReplaceAll("up","");

          //systName.ReplaceAll("down","");

          //if(systName.Index("jes")<0 && systName.Index("umet")<0 && systName.Index("resj")<0) continue;

	  if(!showOneUncertainty.IsNull()) {

	    if(!systName.Contains(showOneUncertainty.Data() )) continue; // georgia, Nov 25 

	  }

          int color = ColorIndex;

          if(systName.Contains("stat")){systName = "stat"; color=2;}

          if(map_legend.find(systName.Data())==map_legend.end()){

            map_legend[systName.Data()]=color;

            legA->AddEntry(hvar,systName.Data(),"L");

            NLegEntry++;

            ColorIndex++;

          }else{

            color = map_legend[systName.Data()];

          }

          if(yield>0){

            if(mapYieldPerBin[systName.Data()].find(ch->first)==mapYieldPerBin[systName.Data()].end()){

              mapYieldPerBin[systName.Data()][ch->first] = ( 1 - (varYield/yield)); // fabs( 1 - (varYield/yield));

              mapUncType[systName.Data()] = true;                        

            }else{

              mapYieldPerBin[systName.Data()][ch->first] = std::max(fabs( 1 - (varYield/yield) ), mapYieldPerBin[systName.Data()][ch->first]);

            }

            if(mapYieldInc.find(systName.Data())==mapYieldInc.end()){ mapYieldInc[systName.Data()].first = 0; mapYieldInc[systName.Data()].second = 0;  }

            mapYieldInc[systName.Data()].first +=  fabs( 1 - (varYield/yield))*yield;

            mapYieldInc[systName.Data()].second += yield;

          }

          hvar->SetFillColor(0);                  

          hvar->SetLineStyle(1);

          hvar->SetLineColor(color);

          hvar->SetLineWidth(1);

          hvar->Draw("HIST esame");                   

          hvar->Write(vh_tag + "_" + systName+"_Uncertainty_"+ch->second.channel+"_"+ch->second.bin+"_"+it->second.shortName);

        }

        //remove the stat uncertainty

        ch->second.shapes[histoName.Data()].removeStatUnc(); 

      }

      //print legend

      c1->cd(0);

      legA->SetFillColor(0); legA->SetFillStyle(0); legA->SetLineColor(0);  legA->SetBorderSize(0); legA->SetHeader("");

      legA->SetNColumns((NLegEntry/2) + 1);

      legA->Draw("same");    legA->SetTextFont(42);

      //print canvas header

      t2->cd(0);

      TPaveText* T = new TPaveText(0.1,0.7,0.9,1.0, "NDC");

      T->SetFillColor(0);  T->SetFillStyle(0);  T->SetLineColor(0); T->SetBorderSize(0);  T->SetTextAlign(22);

      if(systpostfix.Contains('3'))      { T->AddText((string("CMS preliminary, #sqrt{s}=13.0 TeV,   ")+it->first).c_str());

      }else if(systpostfix.Contains('8')){ T->AddText((string("CMS preliminary, #sqrt{s}=8.0 TeV,   ")+it->first).c_str());

      }else{                               T->AddText((string("CMS preliminary, #sqrt{s}=7.0 TeV,   ")+it->first).c_str());

      }T->Draw();

      //save canvas

      ////////////c1->SaveAs(SaveName+vh_tag+"_Uncertainty_"+it->second.shortName+".png");  //-- owen: temporarily disable this (file is huge)

      c1->SaveAs(SaveName+vh_tag+"_Uncertainty_"+it->second.shortName+".pdf");  //-- owen: temporarily disable this (file is huge)

      c1->SaveAs(SaveName+vh_tag+"_Uncertainty_"+it->second.shortName+".root");  //-- owen: temporarily disable this (file is huge)

      delete c1;             

      for(unsigned int i=0;i<toDelete.size();i++){delete toDelete[i];} //clear the objects


      //add inclusive uncertainty as a channel

      //

      for(auto systIt=mapYieldInc.begin(); systIt!=mapYieldInc.end(); systIt++){ mapYieldPerBin[systIt->first][" Inc"] = systIt->second.first/systIt->second.second;  }

      //print uncertainty on yield            

      sprintf(txtBuffer,"\\begin{table}[htp]\n\\tiny\n\\begin{center}\n\\caption{Uncertainty on the yield for the process: \\bf{%s}}\n\\label{tab:tablesys}\n\\resizebox{\\textwidth}{!}{\n ",it->first.c_str()); UncertaintyOnYield += txtBuffer; 

      sprintf(txtBuffer, "\\begin{tabular}{ccccccc} \n"); UncertaintyOnYield+= txtBuffer; 

      //      sprintf(txtBuffer, "\\begin{tabular}{ccccccc} \n {\\bf{%s}} & & & & & & \\\\ \\hline \n", it->first.c_str());  UncertaintyOnYield+= txtBuffer; 

      //      sprintf(txtBuffer, "\\multicolumn{%i}{'c'}{\\bf{%s}}\\\\ \n", I+1, it->first.c_str());  UncertaintyOnYield+= txtBuffer;

      sprintf(txtBuffer, "%10s & %25s", "Type", "Uncertainty");

      for(auto chIt=mapYieldPerBin[""].begin();chIt!=mapYieldPerBin[""].end();chIt++){ sprintf(txtBuffer, "%s & %12s ", txtBuffer, chIt->first.c_str()); } sprintf(txtBuffer, "%s\\\\ \\hline\n", txtBuffer);  UncertaintyOnYield += txtBuffer;

      sprintf(txtBuffer, "%10s & %25s", "", "Nominal yields ");

      for(auto chIt=mapYieldPerBin[""].begin();chIt!=mapYieldPerBin[""].end();chIt++){ sprintf(txtBuffer, "%s & %12.4E ", txtBuffer, chIt->second); } sprintf(txtBuffer, "%s\\\\ \n", txtBuffer);  UncertaintyOnYield += txtBuffer;

      for(auto varIt=mapYieldPerBin.begin();varIt!=mapYieldPerBin.end();varIt++){

        if(varIt->first == "")continue;

        sprintf(txtBuffer, "%10s & %25s", mapUncType[varIt->first.c_str()]?"shape":"scale", varIt->first.c_str());

        for(auto chIt=mapYieldPerBin[""].begin();chIt!=mapYieldPerBin[""].end();chIt++){

          if(varIt->second.find(chIt->first)==varIt->second.end()){ sprintf(txtBuffer, "%s & %10s   "    , txtBuffer, "-" );                        

          }else{                                                    sprintf(txtBuffer, "%s & %+10.1f\\%% ", txtBuffer,100.0 * varIt->second[chIt->first] ); 

          }

        }sprintf(txtBuffer, "%s\\\\ \n", txtBuffer);   UncertaintyOnYield += txtBuffer;

      }sprintf(txtBuffer, "\\hline \n"); UncertaintyOnYield += txtBuffer;

      sprintf(txtBuffer,"\\end{tabular}\n}\n\\end{center}\n\\end{table}\n");UncertaintyOnYield += txtBuffer; 

    }

    sprintf(txtBuffer,"\\end{document}\n");UncertaintyOnYield += txtBuffer;

    FILE* pFile = fopen(SaveName+vh_tag+"_Uncertainty.txt", "w");

    if(pFile){ fprintf(pFile, "%s\n", UncertaintyOnYield.c_str()); fclose(pFile);}

    unc_f->Close();

  }




  //

  // Turn to cut&count (rebin all histo to 1 bin only)

  //

  void AllInfo_t::turnToCC(string histoName){

    //order the proc first

    sortProc();

    //Loop on processes and channels

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end())continue;

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

        TString chbin = ch->first;

        if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

        //   ShapeData_t& shapeInfo = ch->second.shapes[histoName];      

        // TH1* h = shapeInfo.histo();

        ShapeData_t& shapeInfo = ch->second.shapes[histoName];

        TH1 * hh = shapeInfo.histo();

        double integral = 0.; if (hh!=NULL) integral = hh->Integral();

        TString proc = it->second.shortName.c_str();

        for(std::map<string, TH1*  >::iterator unc=shapeInfo.uncShape.begin();unc!=shapeInfo.uncShape.end();unc++){

          TString syst   = unc->first.c_str();

          TH1*    hshape = unc->second;

          hshape->SetDirectory(0);

          hshape = hshape->Rebin(hshape->GetXaxis()->GetNbins()); 

          //make sure to also count the underflow and overflow

          double bin  = hshape->GetBinContent(0) + hshape->GetBinContent(1) + hshape->GetBinContent(2);

          double bine = sqrt(hshape->GetBinError(0)*hshape->GetBinError(0) + hshape->GetBinError(1)*hshape->GetBinError(1) + hshape->GetBinError(2)*hshape->GetBinError(2));

          hshape->SetBinContent(0,0);              hshape->SetBinError  (0,0);

          hshape->SetBinContent(1,bin);            hshape->SetBinError  (1,bine);

          hshape->SetBinContent(2,0);              hshape->SetBinError  (2,0);

        }

      }

    }

  }

// Make a summary plot

//

void AllInfo_t::saveHistoForLimit(string histoName, TFile* fout){

  if (verbose) {

    printf(" --- verbose : AllInfo_t::saveHistoForLimit : begin with histoName = %s, file = %s\n",

           histoName.c_str(), fout->GetName());

    fflush(stdout);

  }

  sortProc();

  for (unsigned int p = 0; p < sorted_procs.size(); p++) {

    string procName = sorted_procs[p];

    std::map<string, ProcessInfo_t>::iterator it = procs.find(procName);

    if (it == procs.end()) continue;

    for (std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin();

         ch != it->second.channels.end(); ch++) {

      TString chbin = ch->first;

      fout->cd();

      if (!fout->GetDirectory(chbin)) {

        fout->mkdir(chbin);

      }

      fout->cd(chbin);

      if (ch->second.shapes.find(histoName) == ch->second.shapes.end()) continue;

      ShapeData_t& shapeInfo = ch->second.shapes[histoName];

      TH1* h = shapeInfo.histo();

      if (h == NULL) continue;

      //

      // Find total background histogram

      //

      TH1* h_total = 0x0;

      std::map<string, ProcessInfo_t>::iterator tpi = procs.find("total");

      if (tpi != procs.end()) {

        std::map<string, ChannelInfo_t>::iterator tci =

          tpi->second.channels.find(ch->first);

        if (tci != tpi->second.channels.end()) {

          std::map<string, ShapeData_t>::iterator tsi =

            tci->second.shapes.find(histoName);

          if (tsi != tci->second.shapes.end()) {

            h_total = tsi->second.histo();

          }

        }

      }

      TString proc = it->second.shortName.c_str();

      //

      // Create MC stat nuisances

      // BUT NOT for ddqcd

      //

      if (proc != "ddqcd"&& proc != "data" && proc != "total" ) {

        shapeInfo.makeStatUnc(

          "CMS_haa4b_",

          (TString("_") + ch->first + "_" + it->second.shortName).Data(),

          systpostfix.Data(),

          false,

          h_total

        );

      }

      fout->cd(chbin);

      //

      // Write nominal + systematics

      //

      for (std::map<string, TH1*>::iterator unc = shapeInfo.uncShape.begin();

           unc != shapeInfo.uncShape.end(); unc++) {

        TString syst = unc->first.c_str();

        TH1* hshape = unc->second;

        if (hshape == NULL) continue;

        hshape->SetDirectory(0);

        //

        // Nominal

        //

        if (syst == "") {

          hshape->SetName(proc);

          // DD-QCD template statistics are represented by the dedicated
          // B-template shape variation.  Clear nominal DD-QCD bin errors so
          // Combine autoMCStats does not add a second uncertainty for it.
          if (autoMCStats && proc == "ddqcd") {
            for (int ibin = 0; ibin <= hshape->GetNbinsX() + 1; ++ibin) {
              hshape->SetBinError(ibin, 0.0);
            }
          }

          if (it->first == "data") {

            hshape->Write("data_obs");

          } else {

            hshape->Write(proc + postfix);

          }

          shapeInfo.uncScale[syst.Data()] = hshape->Integral();

        }

        //

        // Existing Up/Down systematics

        //

        else if (

          proc != "data" &&

          (

            runSystematics ||

            (

              proc == "ddqcd" &&

              syst.Contains("CMS_haa4b_sys_ddqcd_BtemplateStat")

            )

          ) &&

          (

            syst.EndsWith("Up") ||

            syst.EndsWith("Down")

          )

        ) {

          if (!syst.Contains("stat") &&

              (hshape->Integral() < h->Integral()*0.01 ||

               isnan((float)hshape->Integral()))) {

            hshape->Reset();

            hshape->Add(h, 1);

          }

          if (hshape->Integral() <= 0) {

            for (int ibin = 1;

                 ibin <= hshape->GetXaxis()->GetNbins();

                 ibin++) {

              hshape->SetBinContent(ibin, 1E-10);

            }

          }

          //

          // JES naming fix

          //

          if (syst.Contains("CMS_haa4b_AbsoluteStat_jes") ||

              syst.Contains("CMS_haa4b_RelativeJEREC1_jes") ||

              syst.Contains("CMS_haa4b_RelativeJEREC2_jes") ||

              syst.Contains("CMS_haa4b_RelativePtEC1_jes") ||

              syst.Contains("CMS_haa4b_RelativePtEC2_jes") ||

              syst.Contains("CMS_haa4b_RelativeSample_jes") ||

              syst.Contains("CMS_haa4b_RelativeStatEC_jes") ||

              syst.Contains("CMS_haa4b_RelativeStatFSR_jes") ||

              syst.Contains("CMS_haa4b_RelativeStatHF_jes") ||

              syst.Contains("CMS_haa4b_TimePtEta_jes") ||

              syst.Contains("CMS_haa4b_umet") ||

              syst.Contains("CMS_res_j")) {

            if (syst.Contains("Up")) {

              syst = syst.ReplaceAll("Up", year + "Up");

            }

            if (syst.Contains("Down")) {

              syst = syst.ReplaceAll("Down", year + "Down");

            }

          }

	  /*

          TString writeName = proc;

          writeName += "_";

          writeName += syst;

          hshape->SetName(writeName);

          hshape->Write(writeName);*/

	  TString systForName = syst;

	  if (!systForName.BeginsWith("_")) {

	    systForName.Prepend("_");

	  }

	  TString writeName = proc + systForName;

	  hshape->SetName(writeName);

	  hshape->Write(writeName);

        }

        //

        // One-sided nuisances

        //

        else if (runSystematics && proc != "data") {

          TString writeNameUp = proc;

          writeNameUp += "_";

          writeNameUp += syst;

          writeNameUp += "Up";

          hshape->SetName(writeNameUp);

          hshape->Write(writeNameUp);

          TString writeNameDown = proc;

          writeNameDown += "_";

          writeNameDown += syst;

          writeNameDown += "Down";

          TH1* hmirrorshape = (TH1*) hshape->Clone(writeNameDown);

          for (int ibin = 1;

               ibin <= hmirrorshape->GetXaxis()->GetNbins();

               ibin++) {

            double bin =

              2*h->GetBinContent(ibin) -

              hmirrorshape->GetBinContent(ibin);

            if (bin < 0) bin = 0;

            hmirrorshape->SetBinContent(ibin, bin);

          }

          if (hmirrorshape->Integral() <= 0) {

            hmirrorshape->SetBinContent(1, 1E-10);

          }

          hmirrorshape->Write(writeNameDown);

        }

      }

      fout->cd();

    }

  }

}

  //

  // add hardcoded uncertainties

  //

  void AllInfo_t::addHardCodedUncertainties(string histoName){

    // set year                                                                                                                                                       

    TString iyear("");            

    if(inFileUrl.Contains("2016")){  iyear="2016"; }

    else  if(inFileUrl.Contains("2017")){  iyear="2017";} 

    else  if(inFileUrl.Contains("2018")){  iyear="2018";}

    else  if(inFileUrl.Contains("2024")){  iyear="2024";}

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end() || it->first=="total")continue;

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

        TString chbin = ch->first;

        if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

        ShapeData_t& shapeInfo = ch->second.shapes[histoName];      

        double integral = 0.;

        TH1* h=shapeInfo.histo(); if (h) integral=h->Integral();

	//lumi

	if(!it->second.isData && systpostfix.Contains('3')) {

	  if(inFileUrl.Contains("2016")) shapeInfo.uncScale["lumi_13TeV_2016"] = integral*0.010; 

	  if(inFileUrl.Contains("2017")) shapeInfo.uncScale["lumi_13TeV_2017"] = integral*0.020; 

	  if(inFileUrl.Contains("2018")) shapeInfo.uncScale["lumi_13TeV_2018"] = integral*0.015;

    if(inFileUrl.Contains("2024")) shapeInfo.uncScale["lumi_13p6TeV_2024"] = integral*0.016;

	  if(correlatedLumi) {

	    //https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiLUM

	    if(inFileUrl.Contains("2016")) shapeInfo.uncScale["lumi_13TeV_correlated"] = integral*0.006;

	    if(inFileUrl.Contains("2017")) {

	      shapeInfo.uncScale["lumi_13TeV_correlated"] = integral*0.009;

	      shapeInfo.uncScale["lumi_13TeV_1718"] = integral*0.006;

	    }

	    if(inFileUrl.Contains("2018")) {

	      shapeInfo.uncScale["lumi_13TeV_correlated"] = integral*0.020;  

	      shapeInfo.uncScale["lumi_13TeV_1718"] = integral*0.002;

	    }

	  }

	}

	//	} // end rateparam

	//Id+Trigger efficiencies combined

	if( (!it->second.isData) ) {// && (it->second.shortName.find("ddqcd")==string::npos) ){

	    if(chbin.Contains("veto" ))  shapeInfo.uncScale[string("CMS_eff_met_")+iyear.Data()] = integral*0.04; //0.072124;

	    if(chbin.Contains("lep1"))  shapeInfo.uncScale[string("CMS_eff_lep_")+iyear.Data()] = integral*0.02; //0.061788;

	
	}

        //Normalization uncertainties (THEORY)

        if(it->second.shortName.find("otherbkg")!=string::npos){shapeInfo.uncScale["norm_otherbkgs"] = integral*0.5;} //0.27;} 

	//	if(it->second.shortName.find("four-top")!=string::npos){shapeInfo.uncScale["norm_tttt"] = integral*0.46;}     

        if(runZh) {

          if(it->second.shortName.find("wjets")!=string::npos){shapeInfo.uncScale["norm_wjet"] = integral*0.02;}

	  if(it->second.shortName.find("ttbarcba")!=string::npos){shapeInfo.uncScale["norm_ttcc"] = integral*0.50;}

      if(it->second.shortName.find("ttbarlig")!=string::npos){shapeInfo.uncScale["norm_tt_light"] = integral*0.06;} 

       if(it->second.shortName.find("znunu")!=string::npos){shapeInfo.uncScale["norm_znunu"] = integral*0.02;}

        } else {

	  if(it->second.shortName.find("zll")!=string::npos){shapeInfo.uncScale["norm_zll"] = integral*0.06;}

	  if(it->second.shortName.find("ttbarcba")!=string::npos){shapeInfo.uncScale["norm_ch1_ttcc"] = integral*0.50;} 

	}


        // Signal production cross-section uncertainties.

        // Use the signal flag rather than depending on the input JSON label.

        if (mass > 0 && it->second.isSign) {

          if (runZh) {

            shapeInfo.uncScale["QCDscale_zh"] = integral * 0.038;

            shapeInfo.uncScale["PDFscale_zh"] = integral * 0.016;

          } else {

            shapeInfo.uncScale["QCDscale_wh"] = integral * 0.007;

            shapeInfo.uncScale["PDFscale_wh"] = integral * 0.019;

          }

        }

      }

    }


  }

 void AllInfo_t::buildDataCards(string histoName, TString url)

{

  if (verbose) {

    printf(" --- verbose : AllInfo_t::buildDataCards : histoName = %s, url = %s\n",

           histoName.c_str(), url.Data());

    fflush(stdout);

  }

  std::vector<string> clean_procs;

  std::vector<string> sign_procs;

  std::map<string, bool> allChannels;

  std::map<string, bool> allSysts;



  // Collect processes, channels, and nuisances.

  for (unsigned int p = 0; p < sorted_procs.size(); p++) {

    string procName = sorted_procs[p];

    auto it = procs.find(procName);

    if (it == procs.end()) continue;

    if (it->first == "total") continue;

    if (it->second.isSign) {

      sign_procs.push_back(procName);

      printf("pushing into sign_procs: %s\n", it->first.c_str());

    }

    if (it->second.isBckg) {

      clean_procs.push_back(procName);

    }

    for (auto ch = it->second.channels.begin();

         ch != it->second.channels.end(); ch++) {

      TString chbin = ch->first;

      if (ch->second.shapes.find(histoName) == ch->second.shapes.end()) continue;

      allChannels[ch->first] = true;

      ShapeData_t& shapeInfo = ch->second.shapes[histoName];

      for (auto unc = shapeInfo.uncScale.begin();

           unc != shapeInfo.uncScale.end(); unc++) {

        if (unc->first == "") continue;

        if (verbose) {

          printf(" --- verbose : buildDataCards : proc = %20s , chan = %20s , syst = %s\n",

                 procName.c_str(), chbin.Data(), unc->first.c_str());

        }

        allSysts[unc->first] = (unc->second == -1.0);

      }

      for (auto uncS = shapeInfo.uncShape.begin();

           uncS != shapeInfo.uncShape.end(); uncS++) {

        TString syst = uncS->first.c_str();

        if (syst == "") continue;

        if (!syst.EndsWith("Up") && !syst.EndsWith("Down")) continue;

        syst.ReplaceAll("Up", "");

        syst.ReplaceAll("Down", "");

        if (syst.First("_") == 0) syst.Remove(0, 1);

        if (syst == "") continue;

        allSysts[syst.Data()] = true;

      }

    }

  }

  clean_procs.insert(clean_procs.begin(), sign_procs.begin(), sign_procs.end());

  printf("\n\n Now building the datacards. We have %lu signal points available \n",

         sign_procs.size());

  TString combinedcard = "";

  TString srcard = "";

  TString crcard = "";

  for (auto C = allChannels.begin(); C != allChannels.end(); C++) {

    if (

      modeDD &&

      C->first.find("veto_") != string::npos &&

      (

        C->first.find("_B_")     != string::npos ||

        C->first.find("_C_")     != string::npos ||

        C->first.find("_D_")     != string::npos 

      )

    ) continue;

    TString dcName = url;

    dcName.ReplaceAll(".root", "_" + TString(C->first.c_str()) + ".dat");

    combinedcard += (C->first + "=").c_str() + dcName + " ";

    bool isSRCard =

      C->first.find("veto_") != string::npos &&

      C->first.find("_SR_")  != string::npos;

    bool isCR =

      C->first.find("lep1_") != string::npos &&

      C->first.find("_CR_")  != string::npos;

    if (isSRCard) srcard += (C->first + "=").c_str() + dcName + " ";

    if (simfit && isCR) crcard += (C->first + "=").c_str() + dcName + " ";

    std::vector<string> valid_procs;

    for (unsigned int j = 0; j < clean_procs.size(); j++) {

      if (procs[clean_procs[j]].channels.find(C->first) ==

          procs[clean_procs[j]].channels.end()) continue;

      if (procs[clean_procs[j]].channels[C->first].shapes.find(histoName) ==

          procs[clean_procs[j]].channels[C->first].shapes.end()) continue;

      if (procs[clean_procs[j]].channels[C->first].shapes[histoName].histo() != NULL) {

        valid_procs.push_back(clean_procs[j]);

      }

    }

    dcName.ReplaceAll("[", "");

    dcName.ReplaceAll("]", "");

    dcName.ReplaceAll("+", "");

    FILE* pFile = fopen(dcName.Data(), "w");

    fprintf(pFile, "imax 1\n");

    fprintf(pFile, "jmax *\n");

    fprintf(pFile, "kmax *\n");

    fprintf(pFile, "-------------------------------\n");

    if (shape) {

      fprintf(pFile,

              "shapes * * %s %s/$PROCESS %s/$PROCESS_$SYSTEMATIC\n",

              url.Data(), C->first.c_str(), C->first.c_str());

      fprintf(pFile, "-------------------------------\n");

    }

    double obs = 0.0;

    if (procs["data"].channels.find(C->first) != procs["data"].channels.end() &&

        procs["data"].channels[C->first].shapes.find(histoName) !=

        procs["data"].channels[C->first].shapes.end() &&

        procs["data"].channels[C->first].shapes[histoName].histo() != NULL) {

      obs = procs["data"].channels[C->first].shapes[histoName].histo()->Integral();

    }

    fprintf(pFile, "bin bin1\n");

    fprintf(pFile, "Observation %f\n", obs);

    fprintf(pFile, "-------------------------------\n");

    fprintf(pFile, "%55s  ", "bin");

    for (unsigned int j = 0; j < valid_procs.size(); j++) fprintf(pFile, "%8s ", "bin1");

    fprintf(pFile, "\n");

    fprintf(pFile, "%55s  ", "process");

    for (unsigned int j = 0; j < valid_procs.size(); j++) {

      fprintf(pFile, "%8s ", procs[valid_procs[j]].shortName.c_str());

    }

    fprintf(pFile, "\n");

    int nsign_this = 0;

    for (unsigned int j = 0; j < valid_procs.size(); j++) {

      if (procs[valid_procs[j]].isSign) nsign_this++;

    }

    int isig = 0;

    int ibkg = 1;

    fprintf(pFile, "%55s  ", "process");

    for (unsigned int j = 0; j < valid_procs.size(); j++) {

      int pid = 0;

      if (procs[valid_procs[j]].isSign) {

        pid = -nsign_this + 1 + isig;

        isig++;

      } else {

        pid = ibkg;

        ibkg++;

      }

      fprintf(pFile, "%8i ", pid);

    }

    fprintf(pFile, "\n");

    fprintf(pFile, "%55s  ", "rate");

    for (unsigned int j = 0; j < valid_procs.size(); j++) {

      double fval = 0.0;

      if (procs[valid_procs[j]].channels[C->first].shapes[histoName].histo() != NULL) {

        fval = procs[valid_procs[j]].channels[C->first].shapes[histoName].histo()->Integral();

      }

      fprintf(pFile, "%8.6f ", fval);  

    }

    fprintf(pFile, "\n");

    fprintf(pFile, "-------------------------------\n");

    std::set<std::string> writtenSysts;

    for (auto U = allSysts.begin(); U != allSysts.end(); U++) {

      if ((statUncMode == kStatNone || autoMCStats) &&

          U->first.find("CMS_haa4b_stat_") != string::npos) continue;

      if (U->first.find("CMS_haa4b_pdf") != string::npos) continue;

      char line[8192];

      sprintf(line, "%-45s %-10s ", U->first.c_str(), U->second ? "shape" : "lnN");

      bool isNonNull = false;

      for (unsigned int j = 0; j < valid_procs.size(); j++) {

        ShapeData_t& shapeInfo =

          procs[valid_procs[j]].channels[C->first].shapes[histoName];

        double integral = 0.0;

        if (shapeInfo.histo() != NULL) integral = shapeInfo.histo()->Integral();

        bool hasThisSyst = false;

        if (shapeInfo.uncScale.find(U->first) != shapeInfo.uncScale.end()) {

          hasThisSyst = true;

        }

        if (!hasThisSyst && U->second) {

          TString upName = U->first.c_str();

          TString dnName = U->first.c_str();

          upName += "Up";

          dnName += "Down";

	  /*

          if (shapeInfo.uncShape.find(upName.Data()) != shapeInfo.uncShape.end() ||

              shapeInfo.uncShape.find(dnName.Data()) != shapeInfo.uncShape.end()) {

            hasThisSyst = true;

	    }*/

	  TString upNameWithUnderscore = "_" + upName;

	  TString dnNameWithUnderscore = "_" + dnName;

	  if (

	      shapeInfo.uncShape.find(upName.Data()) !=

	      shapeInfo.uncShape.end() ||

	      shapeInfo.uncShape.find(dnName.Data()) !=

	      shapeInfo.uncShape.end() ||

	      shapeInfo.uncShape.find(upNameWithUnderscore.Data()) !=

	      shapeInfo.uncShape.end() ||

	      shapeInfo.uncShape.find(dnNameWithUnderscore.Data()) !=

	      shapeInfo.uncShape.end()

	      ) {

	    hasThisSyst = true;

	  }

        }

        if (hasThisSyst) {

          isNonNull = true;

          if (U->second) {

            sprintf(line, "%s%8s ", line, "1.0");

          } else {

            double unc = shapeInfo.uncScale[U->first];

            if (integral > 0.0) sprintf(line, "%s%8f ", line, 1.0 + unc / integral);

            else                sprintf(line, "%s%8f ", line, 1.0 + unc);

          }

        } else {

          sprintf(line, "%s%8s ", line, "-");

        }

      }

      if (isNonNull) {

        fprintf(pFile, "%s\n", line);

        writtenSysts.insert(U->first);

      }

    }


    bool hasTT = false;

    for (unsigned int j = 0; j < valid_procs.size(); j++) {

      TString sn = procs[valid_procs[j]].shortName.c_str();

      TString pn = valid_procs[j].c_str();

      if (sn.Contains("ttbarbba") ||

          pn.Contains("t#bar{t} + b#bar{b}")) {

        hasTT = true;

        break;

      }

    }

   if (hasTT && (isCR || isSRCard)) {

    fprintf(

        pFile,

        "ttbb_norm rateParam * ttbarbba 1 [0.1,4.0]\n"

    );

    writtenSysts.insert("ttbb_norm");

}

    fprintf(pFile, "-------------------------------\n");

    fprintf(pFile, "normSysts group = ");

    for (auto const& s : writtenSysts) {

      if (

        s.find("norm_") != string::npos ||

        s.find("lumi_") != string::npos ||

        s.find("eff_")  != string::npos ||

        s.find("PDFscale_") != string::npos ||

        s.find("QCDscale_") != string::npos ||

        s.find("thxsec_")   != string::npos ||

        s.find("CMS_haa4b_sys_ddqcd_TF") != string::npos ||
        //s.find("CMS_haa4b_sys_ddqcd_closure") != string::npos ||

        s.find("ttbb_norm") != string::npos

      ) {

        fprintf(pFile, "%s ", s.c_str());

      }

    }

    fprintf(pFile, "\n");

    fprintf(pFile, "shapeSysts group = ");

    for (auto const& s : writtenSysts) {

      if (

        s.find("CMS_haa4b_sys_ddqcd_BtemplateStat") != string::npos ||

        s.find("CMS_haa4b_stat_") != string::npos

      ) {

        fprintf(pFile, "%s ", s.c_str());

      }

    }

    fprintf(pFile, "\n");

    fprintf(pFile, "ddqcd group = ");

    for (auto const& s : writtenSysts) {

      if (s.find("CMS_haa4b_sys_ddqcd") != string::npos) {

        fprintf(pFile, "%s ", s.c_str());

      }

    }

    fprintf(pFile, "\n");

    fprintf(pFile, "-------------------------------\n");

    if (autoMCStats) {

      fprintf(pFile, "* autoMCStats 10\n");

      fprintf(pFile, "-------------------------------\n");

    }

    fclose(pFile);

  }

  FILE* pFile = fopen("combineCards" + vh_tag + ".sh", "w");

  fprintf(pFile, "%s;\n",

          (TString("combineCards.py ") + combinedcard +

           " > card_combined" + vh_tag + ".dat").Data());

  fprintf(pFile, "%s;\n",

          (TString("combineCards.py ") + srcard +

           " > card_sr" + vh_tag + ".dat").Data());

  fprintf(pFile, "%s;\n",

          (TString("combineCards.py ") + crcard +

           " > card_cr" + vh_tag + ".dat").Data());

  fclose(pFile);

  if (verbose) {

    printf(" --- verbose : AllInfo_t::buildDataCards : done.\n");

    fflush(stdout);

  }

}

  //

  // Load histograms from root file and json to memory

  //

void AllInfo_t::getShapeFromFile(TFile* inF, std::vector<string> channelsAndShapes, int cutBin, JSONWrapper::Object &Root,  double minCut, double maxCut, bool onlyData ){

    std::vector<TString> BackgroundsInSignal;

    // set year

    TString iyear("");

    if(inFileUrl.Contains("2016")){  iyear="2016"; }

    else  if(inFileUrl.Contains("2017")){  iyear="2017";}

    else  if(inFileUrl.Contains("2018")){  iyear="2018";}

    else  if(inFileUrl.Contains("2024")){  iyear="2024";}

    //iterate over the processes required

    std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();

    for(unsigned int i=0;i<Process.size();i++){

      string matchingKeyword="";

      if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue; //only consider samples passing key filtering

      TString procCtr(""); procCtr+=i;

      TString proc=Process[i].getString("tag", "noTagFound");

      //      printf("\n\nProcess (get shape from file) = %s\n",proc.Data());

      //      printf("1:channelsANdShapes size= %d\n",(int)channelsAndShapes.size());    

      string dirName = proc.Data(); 

            //std::<TString> keys = Process[i].getString("keys", "noKeysFound");

      if(Process[i].isTagFromKeyword(matchingKeyword, "mctruthmode") ) { char buf[255]; sprintf(buf,"_filt%d",(int)Process[i].getIntFromKeyword(matchingKeyword, "mctruthmode", 0)); dirName += buf; }

      string procSuffix = Process[i].getStringFromKeyword(matchingKeyword, "suffix", "");

      if(procSuffix!=""){dirName += "_" + procSuffix;}

      while(dirName.find("/")!=std::string::npos)dirName.replace(dirName.find("/"),1,"-");         


      TFile *inF17=NULL; TDirectory *pdir;

      // if 2018 era, replace W process from 2017 -- // But if all_plotter file and "4b": do NOT replace

      if(proc.Contains("W#rightarrow l#nu") && inFileUrl.Contains("2018") ) {

	inF17 = TFile::Open(inFileUrl17);

	if( !inF17 || inF17->IsZombie() ){ printf("Invalid file name for 2017 W sample replacement.\n");} 

	pdir = (TDirectory *)inF17->Get(dirName.c_str());  

      } else {

	pdir = (TDirectory *)inF->Get(dirName.c_str());         

      }

      if(!pdir){printf("Directory (%s) for proc=%s is not in the file!\n", dirName.c_str(), proc.Data()); continue;}


      bool isData = Process[i].getBool("isdata", false);

      if(onlyData && !isData) { continue; }//just here to speedup the NRB prediction     

      //      if(proc.Contains(")cp0"))continue; // skip those samples

      bool isSignal = Process[i].getBool("issignal",false);

      if( (proc.Contains("ggH") || proc.Contains("qqH") || proc.Contains("Wh") || proc.Contains("Zh") ))isSignal=true;

      //LQ bool isInSignal = Process[i].getBool("isinsignal", false);

      int color = Process[i].getInt("color", 1);

      int lcolor = Process[i].getInt("lcolor", 1);

      int mcolor = Process[i].getInt("mcolor", color);

      int lwidth = Process[i].getInt("lwidth", 1);

      int lstyle = Process[i].getInt("lstyle", 1);

      int fill   = Process[i].getInt("fill"  , 1001);

      int marker = Process[i].getInt("marker", 20);

      /*

      if(isSignal && signalTag!=""){

        if(!proc.Contains(signalTag.c_str()) )continue;

      }

      */

      double procMass = 0.;

      char procMassStr[128] = "";

      // Accept both native Zh labels and legacy Wh labels in the input file.

      const bool isWhSignal = proc.Contains("Wh (");

      const bool isZhSignal = proc.Contains("Zh (");

      const bool isVHSignal = isWhSignal || isZhSignal;

      if (isSignal && mass > 0 && isVHSignal) {

        const int pos = proc.First("(");

        if (pos == kNPOS ||

            sscanf(proc.Data() + pos + 1, "%lf", &procMass) != 1) {

          printf("ERROR: cannot extract signal mass from process '%s'\n",

                 proc.Data());

          continue;

        }

        printf("Signal candidate: %s --> mass %.0f; requested mass = %i\n",

               proc.Data(), procMass, mass);

        // For interpolation retain the two side masses; otherwise retain only

        // the signal sample matching --m.

        if (massL != -1 && massR != -1) {

          if (procMass != massL && procMass != massR) continue;

        } else {

          if (procMass != mass) continue;

        }

        sprintf(procMassStr, "%i", static_cast<int>(procMass));

      }

      TString procSave = proc;

      // Canonicalize the internal process name according to the analysis mode.

      // With --runZh, even a legacy input label such as "Wh (60)" becomes

      // "Zh60", and its datacard short name becomes "zh".

      if (isSignal && mass > 0 && isVHSignal) {

        proc = (runZh ? TString("Zh") : TString("Wh")) + procMassStr;

      }

      TString shortName = proc;

      shortName.ToLower();

      shortName.ReplaceAll(procMassStr,"");

      shortName.ReplaceAll("#bar{t}","tbar");

      shortName.ReplaceAll("Z#rightarrow ll","dy");

      shortName.ReplaceAll("Z#rightarrow #nu $nu","znunu");

      shortName.ReplaceAll("#rightarrow","");

      shortName.ReplaceAll("(",""); shortName.ReplaceAll(")","");    shortName.ReplaceAll("+","");    shortName.ReplaceAll(" ","");   shortName.ReplaceAll("/","");  shortName.ReplaceAll("#",""); 

      shortName.ReplaceAll("=",""); shortName.ReplaceAll(".","");    shortName.ReplaceAll("^","");    shortName.ReplaceAll("}","");   shortName.ReplaceAll("{","");  shortName.ReplaceAll(",","");

      //      shortName.ReplaceAll("ggh", "ggH");

      //      shortName.ReplaceAll("qqh", "qqH");

      if(shortName.Length()>8)shortName.Resize(8);

      if(procs.find(proc.Data())==procs.end()){sorted_procs.push_back(proc.Data());}

      ProcessInfo_t& procInfo = procs[proc.Data()];

      procInfo.jsonObj = Process[i]; 

      procInfo.isData = isData;

      procInfo.isSign = isSignal;

      procInfo.isBckg = !procInfo.isData && !procInfo.isSign;

      procInfo.mass   = procMass;

      procInfo.shortName = shortName.Data();

      if(procInfo.isSign){

        std::string xsec_str = procInfo.jsonObj["data"].daughters()[0].getString("xsec", "1.0");

        std::string delimiter = "*";

        size_t pos = 0;

        if((pos = xsec_str.find(delimiter)) != std::string::npos){

          double m1 = stod(xsec_str.substr(0,pos));

          double m2 = stod(xsec_str.erase(0,pos+delimiter.length()));

          procInfo.xsec = m1 * m2;

        }else{

          procInfo.xsec = stod(xsec_str);

        }

//        procInfo.xsec = procInfo.jsonObj["data"].daughters()[0].getDouble("xsec", 1);

//      std::cout << proc.Data() << ", xsec: " << procInfo.xsec << std::endl;

        if(procInfo.jsonObj["data"].daughters()[0].isTag("br")){

          std::vector<JSONWrapper::Object> BRs = procInfo.jsonObj["data"].daughters()[0]["br"].daughters();

          double totalBR=1.0; for(size_t ipbr=0; ipbr<BRs.size(); ipbr++){totalBR*=BRs[ipbr].toDouble();}   

          procInfo.br = totalBR;

        }

      }

      //Loop on all channels, bins and shape to load and store them in memory structure

      TH1* syst = (TH1*)pdir->Get("all_optim_systs");

      if(syst==NULL){

        printf("Please check all_optim_systs histo is missing!\n\n");

        syst=new TH1F("all_optim_systs","all_optim_systs",1,0,1);syst->GetXaxis()->SetBinLabel(1,"");

      }

      for(unsigned int c=0;c<channelsAndShapes.size();c++){

        TString chName    = (channelsAndShapes[c].substr(0,channelsAndShapes[c].find(";"))).c_str();

        TString binName   = (channelsAndShapes[c].substr(channelsAndShapes[c].find(";")+1, channelsAndShapes[c].rfind(";")-channelsAndShapes[c].find(";")-1)).c_str();

        TString shapeName = (channelsAndShapes[c].substr(channelsAndShapes[c].rfind(";")+1)).c_str();

        TString ch        = chName+TString("_")+binName;

        TString ch_postfix= (year == "") ? ch : ch + year;

                //printf("channel= %s, bin= %s, shape name= %s, ch name= %s\n",chName.Data(), binName.Data(), shapeName.Data(), ch.Data());

        ChannelInfo_t& channelInfo = procInfo.channels[ch_postfix.Data()];

        channelInfo.bin        = binName.Data();

        channelInfo.channel    = chName.Data();

        ShapeData_t& shapeInfo = channelInfo.shapes[shapeName.Data()];

	//	printf("%s SYST SIZE=%i\n", (ch+"_"+shapeName).Data(), syst->GetNbinsX());

        for(int ivar = 1; ivar<=syst->GetNbinsX();ivar++){                

          TH1D* hshape   = NULL;

	  TString varName   = syst->GetXaxis()->GetBinLabel(ivar);

	  TString histoName = ch+"_"+shapeName+(isSignal?signalSufix:"")+varName ; 

          //if(isSignal && ivar==1)printf("Syst %i = %s\n", ivar, varName.Data()); 

	  //	  if(ivar>syst->GetNbinsX()) 

	  //printf("Histo %s  for syst:%s has Integral:%f\n", histoName.Data(), varName.Data(),);          

          TH2* hshape2D = (TH2*)pdir->Get(histoName ); 

          //      else {hshape = (TH1D*)pdir->Get(histoName ); }

          if(!hshape2D){

	    //	    printf("Histo %s is NULL for syst:%s\n", histoName.Data(), varName.Data());   

	    if(shapeName==histo && histoVBF!="" && ch.Contains("vbf")){   hshape2D = (TH2*)pdir->Get(TString("all_")+histoVBF+(isSignal?signalSufix:"")+varName);

            }else{                                                        hshape2D = (TH2*)pdir->Get(TString("all_")+shapeName+varName);

            }

            if(hshape2D){

              hshape2D->Reset();

            }else{  //if still no histo, skip this proc...

	      //	      printf("Histo %s does not exist for syst:%s\n", histoName.Data(), varName.Data());

              continue;

            }

          }

          //special treatment for side mass points

          int cutBinUsed = cutBin;

          if(shapeName == histo && !ch.Contains("vbf") && procMass==massL)cutBinUsed = indexcutML[channelInfo.bin];

          if(shapeName == histo && !ch.Contains("vbf") && procMass==massR)cutBinUsed = indexcutMR[channelInfo.bin];

          histoName.ReplaceAll(ch,ch+"_proj"+procCtr);

          hshape   = hshape2D->ProjectionY(histoName,cutBinUsed,cutBinUsed);

	  //	  if(ivar>syst->GetNbinsX()) 

	  //	    printf("proc = %s . Histo %s  for syst:%s has Integral:%f\n", proc.Data(), histoName.Data(), varName.Data(),hshape->Integral()); 

          //  else {hshape = hshape2D; }

          filterBinContent(hshape);

          if(isnan((float)hshape->Integral())){hshape->Reset();}

          hshape->SetDirectory(0);

          hshape->SetTitle(proc);

          utils::root::fixExtremities(hshape,false,true);

          hshape->SetFillColor(color); hshape->SetLineColor(lcolor); hshape->SetMarkerColor(mcolor);

          hshape->SetFillStyle(fill);  hshape->SetLineWidth(lwidth); hshape->SetMarkerStyle(marker); hshape->SetLineStyle(lstyle);

          //if current shape is the one to cut on, then apply the cuts

          if(shapeName == histo){

            //if(ivar==1 && isSignal)printf("A %s %s Integral = %f\n", ch.Data(), shortName.Data(), hshape->Integral() );

            for(int x=0;x<=hshape->GetXaxis()->GetNbins()+1;x++){

              if(hshape->GetXaxis()->GetBinCenter(x)<=minCut || hshape->GetXaxis()->GetBinCenter(x)>=maxCut){ hshape->SetBinContent(x,0); hshape->SetBinError(x,0); }

            }

            if(rebinVal>1){ hshape->Rebin(rebinVal); }

            hshape->GetYaxis()->SetTitle("Entries");// (/25GeV)");

          }

          hshape->Scale(MCRescale);

	  // And rescale W sample for 2018 after replacement from 2017 era sample:

	  if(proc.Contains("W#rightarrow l#nu") && inFileUrl.Contains("2018")) {  hshape->Scale(wscale17); }

          if (postfit) {

            ///if(!(ch.Contains("_A_"))) {

              ///printf("W/Top NORMALIZATIONs: Process = %s and channel = %s\n\n",proc.Data(),ch.Data());

              if (

                  proc.Contains("t#bar{t} + b#bar{b}") 

                ) {

                  double scale_factor = rfr_tt_norm;

                  TString hname(hshape->GetName());

                  if (!hname.Contains("shapes_")) {

                    printf(

                      "     postfit: %15s %20s %10s : %-40s : tt_norm = %6.3f\n",

                      shortName.Data(), chName.Data(), year.Data(), hshape->GetName(), scale_factor

                    );

                  }

                  hshape->Scale(scale_factor);

                }

            ///} // _A_?

          } // postfit?

          if(isSignal)hshape->Scale(SignalRescale);


          //Do Renaming and cleaning

          varName.ReplaceAll("down","Down");

          varName.ReplaceAll("up","Up");

          if(varName==""){//does nothing

	    //	  }else if(varName.EndsWith("_jes")){

	  }else if(varName.EndsWith("_jes")){ // CHECK!!!    

	    //varName.ReplaceAll("_jes","_CMS_scale_j");

	    if(runZh) { varName.ReplaceAll("_jes",string("_CMS_ch2_jes_")+iyear.Data());  }  

	    else { varName.ReplaceAll("_jes",string("_CMS_ch1_jes_")+iyear.Data());  }  

	  } else if(varName.BeginsWith("_pu")){

	    if(runZh) { varName.ReplaceAll("_pu",string("_CMS_ch2_pu_")+iyear.Data());  }

	    else { varName.ReplaceAll("_pu",string("_CMS_ch1_pu_")+iyear.Data());  }     

	  } else if(varName.BeginsWith("_resRho_e")){

	    if(runZh) { varName.ReplaceAll("_resRho_e",string("_CMS_ch2_resRho_e_")+iyear.Data());  }     

	    else { varName.ReplaceAll("_resRho_e",string("_CMS_ch1_resRho_e_")+iyear.Data());  }

	  } else if(varName.BeginsWith("_sys_e")){

	    if(runZh) { varName.ReplaceAll("_sys_e",string("_CMS_ch2_sys_e_")+iyear.Data());  }  

	    else { varName.ReplaceAll("_sys_e",string("_CMS_ch1_sys_e_")+iyear.Data());  }   

	  }else if(varName.BeginsWith("_umet")) { //continue; //skip this one for now

	    if(runZh) { varName.ReplaceAll("_umet",string("_CMS_ch2_umet_")+iyear.Data());  }  

	    else { varName.ReplaceAll("_umet",string("_CMS_ch1_umet_")+iyear.Data());  }

	  }else if(varName.BeginsWith("_jer")){ 

	    //varName.ReplaceAll("_jer","_CMS_res_j"); //+iyear.Data());

	    if(runZh) { varName.ReplaceAll("_jer",string("_CMS_ch2_res_j_")+iyear.Data());  }     

	    else { varName.ReplaceAll("_jer",string("_CMS_ch1_res_j_")+iyear.Data());  }     

          }else if(varName.BeginsWith("_les")){

            continue; // skip this one for now

            //      if(ch.Contains("e"  ))varName.ReplaceAll("_les","_CMS_scale_e");

            //      if(ch.Contains("mu"))varName.ReplaceAll("_les","_CMS_scale_m");

          }else if(varName.BeginsWith("_btag"  )){

	    if(runZh) { varName.ReplaceAll("_btag",string("_CMS_ch2_eff_B_")+iyear.Data());  } 

	    else { varName.ReplaceAll("_btag",string("_CMS_ch1_eff_B_")+iyear.Data());  }        

          }else if(varName.BeginsWith("_ctag"  )){

	    if(runZh) { varName.ReplaceAll("_ctag",string("_CMS_ch2_eff_C_")+iyear.Data());  } 

	    else { varName.ReplaceAll("_ctag",string("_CMS_ch1_eff_C_")+iyear.Data());  }

          }else if(varName.BeginsWith("_ltag"  )){

	    if(runZh) { varName.ReplaceAll("_ltag",string("_CMS_ch2_eff_mistag_")+iyear.Data());  } 

	    else { varName.ReplaceAll("_ltag",string("_CMS_ch1_eff_mistag_")+iyear.Data());  }  

            //          }else if(varName.BeginsWith("_pu"    )){varName.ReplaceAll("_pu", "_CMS_haa4b_pu");

            //    }else if(varName.BeginsWith("_pdf" )){

            //      if (proc.Contains("wh")!=std::string::npos) {continue; }

          }else if(varName.BeginsWith("_bnorm"  )){continue; //skip this one

          }else{ varName="_CMS_haa4b"+varName;}

          hshape->SetTitle(proc+varName);

          if(shapeInfo.uncShape.find(varName.Data())==shapeInfo.uncShape.end()){

            shapeInfo.uncShape[varName.Data()] = hshape;

          }else{

            shapeInfo.uncShape[varName.Data()]->Add(hshape);

          }

        } // end ivar

      }

      if(proc.Contains("W#rightarrow l#nu") && inFileUrl.Contains("2018")) {  

	printf("Closing file used for W sample replacement in 2018.\n"); inF17->Close(); // close file used for W sample replacement in 2018

      }

    }

} // getShapeFromFile

  //

  // Rebin histograms to make sure that high BDT region have no empty bins

  //

  void AllInfo_t::rebinMainHisto(string histoName)

  {

    //Loop on processes and channels

    for(unsigned int p=0;p<sorted_procs.size();p++){

      string procName = sorted_procs[p];

      std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

      if(it==procs.end())continue;

      for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

        if(ch->second.shapes.find(histoName)==(ch->second.shapes).end())continue;

        ShapeData_t& shapeInfo = ch->second.shapes[histoName];      

        for(std::map<string, TH1*  >::iterator unc=shapeInfo.uncShape.begin();unc!=shapeInfo.uncShape.end();unc++){

          TH1* histo = unc->second;

          if(!histo)continue;

            TString jetBin = ch->second.bin.c_str();

            if(jetBin.Contains("3b")){

             //-----------------

	      if (runZh) {

		 double xbins[] = {

        0.0,

        0.04,  

	//boost

	//0.54,

	//0.74,

	//res

	0.86,

	0.96,                                   

        1.0

		 }; 

		int nbins=sizeof(xbins)/sizeof(double);

		unc->second = histo->Rebin(nbins-1, histo->GetName(), (double*)xbins); 

		utils::root::fixExtremities(unc->second, false, true); 

	      }

	    }

        }

      }

    }

  }

    //

    // merge histograms from different bins together... but keep the channel separated 

    //

    void AllInfo_t::mergeBins(std::vector<string>& binsToMerge, string NewName){

      printf("Merge the following bins of the same channel together: "); for(unsigned int i=0;i<binsToMerge.size();i++){printf("%s ", binsToMerge[i].c_str());}

      printf("The resulting bin will be called %s\n", NewName.c_str());

      for(unsigned int p=0;p<sorted_procs.size();p++){

        string procName = sorted_procs[p];

        std::map<string, ProcessInfo_t>::iterator it=procs.find(procName);

        if(it==procs.end())continue;

        for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

          if(find(binsToMerge.begin(), binsToMerge.end(), ch ->second.bin)==binsToMerge.end())continue;  //make sure this bin should be merged

          for(std::map<string, ChannelInfo_t>::iterator ch2 = ch; ch2!=it->second.channels.end(); ch2++){

            if(ch->second.channel != ch2->second.channel)continue; //make sure we merge bin in the same channel

            if(ch->second.bin     == ch2->second.bin    )continue; //make sure we do not merge with itself

            if(find(binsToMerge.begin(), binsToMerge.end(), ch2->second.bin)==binsToMerge.end())continue;  //make sure this bin should be merged

            addChannel(ch->second, ch2->second, false); // add nominal 

            addChannel(ch->second, ch2->second, true, false); // add systematics

            it->second.channels.erase(ch2);  

            ch2=ch;

          }

          ch->second.bin = NewName;

        }

        //also update the map keys

        std::map<string, ChannelInfo_t> newMap;

        for(std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin(); ch!=it->second.channels.end(); ch++){

          //newMap[ch->second.channel+ch->second.bin] = ch->second;

          newMap[ch->second.channel+"_"+ch->second.bin] = ch->second;

        }

        it->second.channels = newMap;

      }

    }


    void AllInfo_t::HandleEmptyBins(string histoName)

{

  for (unsigned int p = 0; p < sorted_procs.size(); p++) {

    string procName = sorted_procs[p];

    std::map<string, ProcessInfo_t>::iterator it = procs.find(procName);

    if (it == procs.end()) continue;

    if (it->second.isData) continue;

    if (it->first == "total") continue;

    for (std::map<string, ChannelInfo_t>::iterator ch = it->second.channels.begin();

         ch != it->second.channels.end(); ch++) {

      TString chbin = ch->first;

      if (ch->second.shapes.find(histoName) == ch->second.shapes.end()) continue;

      ShapeData_t& shapeInfo = ch->second.shapes[histoName];

      TH1* histo = (TH1*) shapeInfo.histo();

      if (!histo) {

        printf("Histo does not exist... skip it\n");

        fflush(stdout);

        continue;

      }

      double procWeight = 1E-6;

      double entries  = histo->GetEntries();

      double integral = histo->Integral();

      if (entries > 0.0 && integral > 0.0) {

        procWeight = fabs(integral / entries);

      }

      if (procWeight < 1E-6) {

        procWeight = 1E-6;

      }

      if (procWeight > std::max(1.0, 0.50 * fabs(integral))) {

        procWeight = std::max(1E-6, 0.10 * fabs(integral));

      }

      double StartIntegral = histo->Integral();

      for (int binx = 1; binx <= histo->GetNbinsX(); binx++) {

        if (histo->GetBinContent(binx) <= 1E-3) {

          histo->SetBinContent(binx, 1E-3);


            histo->SetBinError(binx, 1e-3);


          if (verbose) {

            printf("--- verbose : AllInfo_t::HandleEmptyBins : found empty bin");

            printf(" proc = %s", procName.c_str());

            printf(", year = %s", year.Data());

            printf(", channel = %s", chbin.Data());

            printf(", histoName = %s", histoName.c_str());

            printf(", bin = %i", binx);

            printf(", filled content = 1E-6");

            printf(", assigned error = %.6g", it->second.isSign ? 1E-6 : procWeight);

            printf("\n");

            fflush(stdout);

          }

        }

      }

      double EndIntegral = histo->Integral();

      shapeInfo.rescaleScaleUncertainties(StartIntegral, EndIntegral);

      for (std::map<string, TH1*>::iterator unc = shapeInfo.uncShape.begin();

           unc != shapeInfo.uncShape.end(); unc++) {

        if (!unc->second) continue;

        for (int binx = 1; binx <= unc->second->GetNbinsX(); binx++) {

          if (unc->second->GetBinContent(binx) <= 0.0) {

            unc->second->SetBinContent(binx, 1E-6);

          }

        }

      }

    }

  }

  computeTotalBackground();

}

  void AllInfo_t::printInventory() {

     printf("\n\n\n =============== AllInfo_t::printInventory :  begin\n\n") ;

     for ( std::map<string, ProcessInfo_t>::iterator ip = procs.begin(); ip!= procs.end(); ip++ ) {

        string proc_key = ip -> first ;

        ProcessInfo_t proc = ip -> second ;

        proc.printProcess() ;

     } // ip

     printf("\n\n\n =============== AllInfo_t::printInventory :  end\n\n") ;

  } // AllInfo_t::printInventory
