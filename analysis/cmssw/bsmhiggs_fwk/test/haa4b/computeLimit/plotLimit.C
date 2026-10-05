#include <string>
#include <vector>

#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TCutG.h"
#include "TGraph2D.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TMultiGraph.h"
#include "TPaveText.h"
#include "Math/Interpolator.h"
#include "TSpline.h"
#include "TBox.h"
#include "CMS_lumi.C"
#include "tdrstyle.C"
#include <TColor.h>

//#include "UserCode/bsmhiggs_fwk/interface/tdrstyle.h"
//#include "UserCode/bsmhiggs_fwk/src/tdrstyle.C"
#include "UserCode/bsmhiggs_fwk/interface/RootUtils.h"
// create custom colors (only once)
int yellowBand = TColor::GetColor("#ffcc00");
int blueBand   = TColor::GetColor("#228b22");
//tree variables
double Tmh, Tlimit, TlimitErr; float TquantExp;

TGraph* multiplyGraph(TGraph* A, TGraph* B){   double x, y;   for(int i=0;i<A->GetN();i++){A->GetPoint(i, x, y); A->SetPoint(i, x, y*B->Eval(x));}    return A;   }
TGraph* divideGraph (TGraph* A, TGraph* B){ double x, y; for(int i=0;i<A->GetN();i++){A->GetPoint(i, x, y); A->SetPoint(i, x, y/B->Eval(x));} return A; } 

TGraph* getLimitGraph(TTree* tree, float Quantil){
  TGraph* toReturn = new TGraph(999);
  int i=0; 
  for(int ientry=0;ientry<tree->GetEntriesFast();ientry++){
     tree->GetEntry(ientry);
     
     if(TquantExp==Quantil){
        //printf("Quantil = %f - mH=%f --> %f\n",TquantExp,Tmh,Tlimit);
        //
        if(Tmh<12) continue;
        toReturn->SetPoint(i, Tmh, Tlimit);
        i++;
     }
  }toReturn->Set(i);
  return toReturn;

}


TCutG* GetErrorBand(string name, TGraph* Low, TGraph* High, int fillColor)
{
   TCutG* cutg = new TCutG(name.c_str(),Low->GetN()+High->GetN()+2);
   cutg->SetFillColor(fillColor);
   cutg->SetLineStyle(0);
   cutg->SetLineColor(0);
   int I = 0;
   for(int i=0;i<Low->GetN();i++){  cutg->SetPoint(I,Low ->GetX()[i]               , Low ->GetY()[i]               );I++; }
                                    cutg->SetPoint(I,Low ->GetX()[Low ->GetN()-1]  , Low ->GetY()[Low ->GetN()-1]  );I++;
                                    cutg->SetPoint(I,High->GetX()[High->GetN()-1]  , High->GetY()[High->GetN()-1]  );I++;
   for(int i=0;i<High->GetN() ;i++){cutg->SetPoint(I,High->GetX()[High->GetN()-1-i], High->GetY()[High->GetN()-1-i]);I++;}
   return cutg;
}

void scaleGraph(TGraph* Limit, double scale){ 
   for(int i=0;i<Limit->GetN();i++){ 
      Limit->SetPoint(i, Limit->GetX()[i], Limit->GetY()[i]*scale);
   }   
}

void printLimits(FILE* pFile, TGraph* graph, double Mmin=15, double Mmax=60){ //-Penny
   double previous = graph->Eval(Mmin);
   fprintf(pFile, "Exclude ");
   bool opened = false;
   for(double M=Mmin;M<=Mmax;M+=1.0){
      double NEW = graph->Eval(M);
      //printf("%04f - %6.3f\n", M, NEW);
      if(previous>1 && NEW<=1){fprintf(pFile,"[%f,",M); opened=true;}
      if(previous<1 && NEW>=1){fprintf(pFile,"%f]",M-1); opened=false;}
      if(M==Mmax    && opened){fprintf(pFile,">%f]",Mmax); opened=false;}
      previous = NEW;
   }fprintf(pFile,"\n");
}



void plotLimit(string outputDir="./", TString inputs="", TString inputXSec="", bool strengthLimit=true, bool blind=false, double energy=13.6, double luminosity=109.82, TString legendName="Zh channels", bool logY=true)
{
   
   gStyle->SetPadTopMargin   (0.05);
   gStyle->SetPadBottomMargin(0.12);
   gStyle->SetPadRightMargin (0.16);
   gStyle->SetPadLeftMargin  (0.17);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.45);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);

   //   iPeriod = 4;
   //   iPos = 11;

  //get the limits from the tree
  TFile* file = TFile::Open(inputs);
  printf("Looping on %s\n",inputs.Data());
  if(!file) return;
  if(file->IsZombie()) return;

  TTree* tree = (TTree*)file->Get("limit");
  tree->GetBranch("mh"              )->SetAddress(&Tmh      );
  tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
  tree->GetBranch("limitErr"        )->SetAddress(&TlimitErr);
  tree->GetBranch("quantileExpected")->SetAddress(&TquantExp);
  TGraph* ObsLimit   = getLimitGraph(tree,-1   ); 
  TGraph* ExpLimitm2 = getLimitGraph(tree,0.025);
  TGraph* ExpLimitm1 = getLimitGraph(tree,0.160);
  TGraph* ExpLimit   = getLimitGraph(tree,0.500);
  TGraph* ExpLimitp1 = getLimitGraph(tree,0.840);
  TGraph* ExpLimitp2 = getLimitGraph(tree,0.975);
  file->Close(); 

  FILE* pFileSStrenght = fopen((outputDir+"SignalStrenght").c_str(),"w");
  std::cout << "Printing Signal Strenght" << std::endl;
  for(int i=0;i<ExpLimit->GetN();i++){
     double M = ExpLimit->GetX()[i];
     std::cout << "Mass: " << M << "; ExpLimit: " << ExpLimit->Eval(M) << std::endl; 
     printf("$%86.E$ & $%8.6E$ & $[%8.6E,%8.6E]$ & $[%8.6E,%8.6E]$ \\\\\\hline\n",M, ExpLimit->Eval(M), ExpLimitm1->Eval(M), ExpLimitp1->Eval(M), ExpLimitm2->Eval(M),  ExpLimitp2->Eval(M));
     fprintf(pFileSStrenght, "$%8.6E$ & $%8.6E$ & $[%8.6E,%8.6E]$ & $[%8.6E,%8.6E]$ & $%8.6E$ \\\\\\hline\n",M, ExpLimit->Eval(M), ExpLimitm1->Eval(M), ExpLimitp1->Eval(M), ExpLimitm2->Eval(M),  ExpLimitp2->Eval(M), ObsLimit->Eval(M));
    if(int(ExpLimit->GetX()[i])%50!=0)continue; //printf("%f ",ObsLimit->Eval(M));
  }printf("\n");
  fclose(pFileSStrenght); 
 

  //get the pValue
  inputs = inputs.ReplaceAll("/LimitTree.root", "/PValueTree.root");
  file = TFile::Open(inputs);
  
  printf("Looping on %s\n",inputs.Data());
  if(!file) return;
  if(file->IsZombie()) return;
  
  tree = (TTree*)file->Get("limit");
  tree->GetBranch("limit")->SetAddress(&Tlimit   );
  
  TGraph* pValue = getLimitGraph(tree,-1);
  
  file->Close();

  
  //make TH Cross-sections
   string suffix = outputDir;

   //Double_t mA[5]={12.,15.,20.,25.,30.}; //-Penny
   Double_t mA[10]={15.0, 20.0, 25.0, 30.0, 35., 40., 45., 50., 55., 60.};

  string prod = "pp";
  if(outputDir.find("GGF")!=std::string::npos)prod="gg";
  if(outputDir.find("VBF")!=std::string::npos)prod="qq";
 
  if(outputDir.find("all")!=std::string::npos)prod="all";    
  if(outputDir.find("Wh")!=std::string::npos)prod="wh";
  if(outputDir.find("Zh")!=std::string::npos)prod="zh";        
  //  if(outputDir.find("all")!=std::string::npos)prod="all";     

  strengthLimit = true; //-Penny
  if(prod=="pp")strengthLimit=true;
 
  TGraph* THXSec = new TGraph(999); int N=10; //-Penny
  for (int i=0; i<N; i++) {       
    if (prod=="wh") THXSec->SetPoint(i,mA[i],(1.94*1.076*0.3257));  // Penny: do i change this?
    if (prod=="zh") THXSec->SetPoint(i,mA[i],(0.94));  
    if (prod=="all") THXSec->SetPoint(i,mA[i],(1.37*1.076*0.3257+0.8839*1.0474*0.10099));   
  }


  if (!strengthLimit) {
    multiplyGraph(   ObsLimit, THXSec);
    multiplyGraph( ExpLimitm2, THXSec);
    multiplyGraph( ExpLimitm1, THXSec);
    multiplyGraph(   ExpLimit, THXSec);
    multiplyGraph( ExpLimitp1, THXSec);
    multiplyGraph( ExpLimitp2, THXSec);
  }
  /*
  //Scale exclusion XSec in fb
  scaleGraph(ObsLimit  , 0.001); //pb to fb
  scaleGraph(ExpLimitm2, 0.001); //pb to fb
  scaleGraph(ExpLimitm1, 0.001); //pb to fb
  scaleGraph(ExpLimit  , 0.001); //pb to fb
  scaleGraph(ExpLimitp1, 0.001); //pb to fb
  scaleGraph(ExpLimitp2, 0.001); //pb to fb

  //scal eTH cross-section and limits according to scale factor 
  //this only apply to NarrowResonnance case
  if(strengthLimit){
     divideGraph(ObsLimit   , THXSec);
     divideGraph(ExpLimitm2 , THXSec);
     divideGraph(ExpLimitm1 , THXSec);
     divideGraph(ExpLimit   , THXSec);
     divideGraph(ExpLimitp1 , THXSec);
     divideGraph(ExpLimitp2 , THXSec);
     divideGraph(THXSec     , THXSec);
  }
  */

  //limits in terms of signal strength
  TCanvas* c = new TCanvas("c", "c",1000,700);
  c->SetGridx(); 
  c->SetGridy();
  //TH1F* framework = new TH1F(inputs.Data(),"Graph",1,15,30); //3000);
  TH1F* framework = new TH1F(inputs.Data(),"Graph",1,15,60); //3000); // Penny
  framework->SetStats(false);
  framework->SetTitle("");
  framework->GetXaxis()->SetTitle("M_{a} [GeV]");
  framework->GetYaxis()->SetTitleOffset(1.70);
  if(strengthLimit){
    framework->GetYaxis()->SetTitle("#sigma_{ZH} B(H#rightarrowaa#rightarrowbbbb) / #sigma_{SM}");
    if(logY){
      framework->GetYaxis()->SetRangeUser(1E-3,1E2);
      c->SetLogy(true);
      outputDir += "log/";
    }else{
      framework->GetYaxis()->SetRangeUser(0,4.0);
      //framework->GetYaxis()->SetRangeUser(0,2.5);
      c->SetLogy(false);
      outputDir += "linear/";
    }
  }else{
    framework->GetYaxis()->SetTitle((string("#sigma_{95%} (") + prod +") x BR (pb)").c_str());
    if(logY){
      framework->GetYaxis()->SetRangeUser(1E-3,1E2);
      c->SetLogy(true);
      outputDir += "log/";
    }else{
      framework->GetYaxis()->SetRangeUser(0,4.0);
      //framework->GetYaxis()->SetRangeUser(0,2.5);
      c->SetLogy(false);
      outputDir += "linear/";
    }
  }
  system(string("mkdir -p " + outputDir).c_str());
  framework->GetXaxis()->SetLabelOffset(0.007);
  framework->GetXaxis()->SetLabelSize(0.045);
  framework->GetXaxis()->SetTitleOffset(1.1);
  framework->GetXaxis()->SetTitleFont(42);
  framework->GetXaxis()->SetTitleSize(0.05);
  framework->GetYaxis()->SetLabelFont(42);
  framework->GetYaxis()->SetLabelOffset(0.007);
  framework->GetYaxis()->SetLabelSize(0.045);
  framework->GetYaxis()->SetTitleOffset(1.1);
  framework->GetYaxis()->SetTitleFont(42);
  framework->GetYaxis()->SetTitleSize(0.05);
  framework->Draw();
  
  // Shade the region mu >= 1
  if (strengthLimit) {
    double xmin = framework->GetXaxis()->GetXmin();
    double xmax = framework->GetXaxis()->GetXmax();
    
    double yminShade = 1.0;
    double ymaxShade = logY ? 1E2 : 4.0;  // same as your y-axis max
    
    /*TBox* excludedBox = new TBox(xmin, yminShade, xmax, ymaxShade);
    excludedBox->SetFillColorAlpha(kGray, 1);  // very light
    excludedBox->SetFillStyle(3004); 
    excludedBox->SetLineColor(0);
    excludedBox->Draw("same");*/
  }
  TGraph* TGObsLimit   = ObsLimit;  TGObsLimit->SetLineWidth(2);
  TGraph* TGExpLimit   = ExpLimit;  TGExpLimit->SetLineWidth(2); TGExpLimit->SetLineStyle(2);
  TCutG* TGExpLimit1S  = GetErrorBand("1S", ExpLimitm1, ExpLimitp1,blueBand);  
  TCutG* TGExpLimit2S  = GetErrorBand("2S", ExpLimitm2, ExpLimitp2,yellowBand); 
  //  THXSec->SetLineWidth(2); THXSec->SetLineStyle(1); THXSec->SetLineColor(4);

  TGExpLimit->SetLineColor(1);  TGExpLimit->SetLineStyle(2);
  TGObsLimit->SetLineWidth(2);  TGObsLimit->SetMarkerStyle(20);
  TGExpLimit2S->Draw("fc same");
  TGExpLimit1S->Draw("fc same");

  TSpline3 *s = new TSpline3("grs",TGObsLimit); 
  if(!blind){
    s->SetLineWidth(2);
    s->Draw("same"); TGObsLimit->Draw("same P");
  }

  TGExpLimit->Draw("same c");

  
  if(strengthLimit){
     TLine* SMLine = new TLine(framework->GetXaxis()->GetXmin(),1.0,framework->GetXaxis()->GetXmax(),1.0);
     SMLine->SetLineWidth(2); SMLine->SetLineStyle(1); SMLine->SetLineColor(kBlack);      
     SMLine->Draw("same C");
  }
  else{
    THXSec->SetPoint(0,10.,0.94); //-Penny
    THXSec->SetLineWidth(2); THXSec->SetLineStyle(1); THXSec->SetLineColor(2); 
    THXSec->Draw("same C");
  }

  // HERE:
  //  utils::root::DrawPreliminary(luminosity, energy, c);
  //TLegend* LEG = new TLegend(0.55,0.7,0.85,0.95);   
  TLegend* LEG = new TLegend(0.55,0.75,0.85,0.95);
  LEG->SetHeader("");
  LEG->SetFillColor(0);
  LEG->SetFillStyle(0);
  LEG->SetTextFont(42);
  //  LEG->SetTextSize(0.1);
  LEG->SetBorderSize(0);
  //LEG->AddEntry(THXSec  , "Th prediction"  ,"L");
  LEG->AddEntry(TGExpLimit  , "median expected"  ,"L");
  LEG->AddEntry(TGExpLimit1S  , "expected #pm 1#sigma"  ,"F");
  LEG->AddEntry(TGExpLimit2S  , "expected #pm 2#sigma"  ,"F");
  //if(!blind) LEG->AddEntry(s , "observed"  ,"L");    
  if(!blind) LEG->AddEntry(TGObsLimit  , "observed"  ,"LP");
  LEG->Draw();

  double lumi_fb = luminosity;
  if (lumi_fb > 1000.) lumi_fb /= 1000.;  

  lumi_sqrtS = Form("%.2f fb^{-1} (%.1f TeV)", lumi_fb, energy);  
  // writing the lumi information and the CMS "logo"
  // CMS_lumi( c, 4, 11 ); //iPeriod, iPos );
  CMS_lumi( c, 0, 11 ); // Penny

  c->RedrawAxis();
  //c->SaveAs((outputDir+"Limit.png").c_str());
  c->SaveAs((outputDir+"Limit.C").c_str());
  c->SaveAs((outputDir+"Limit.pdf").c_str());
  c->SaveAs((outputDir+"Limit.root").c_str());

  
  //save a summary of the limits
  FILE* pFileSum = fopen((outputDir+"LimitSummary").c_str(),"w");
  for(int i=0;i<TGExpLimit->GetN();i++){
     double M = ExpLimit->GetX()[i];
     fprintf(pFileSum, "$%8.6E$ & $%8.6E$ & $[%8.6E,%8.6E]$ & $[%8.6E,%8.6E]$ & $%8.6E$ & Th=$%8.6E$ & pValue=$%8.6E$\\\\\\hline\n",M, ExpLimit->Eval(M), ExpLimitm1->Eval(M), ExpLimitp1->Eval(M), ExpLimitm2->Eval(M),  ExpLimitp2->Eval(M), ObsLimit->Eval(M), 1.37, pValue->Eval(M));
    if(int(ExpLimit->GetX()[i])%50!=0)continue; printf("%f ",ObsLimit->Eval(M));
  }printf("\n");
  fclose(pFileSum);

  pFileSum = fopen((outputDir+"LimitRange").c_str(),"w");
  fprintf(pFileSum, "EXPECTED LIMIT --> ");                   printLimits(pFileSum,TGExpLimit, TGExpLimit->GetX()[0], TGExpLimit->GetX()[TGExpLimit->GetN()-1]);
  if(!blind) fprintf(pFileSum, "OBSERVED LIMIT --> ");        printLimits(pFileSum,TGObsLimit, TGObsLimit->GetX()[0], TGObsLimit->GetX()[TGObsLimit->GetN()-1]);
     
  fprintf(pFileSum, "Exp Limits for Model are: "); for(int i=0;i<TGExpLimit->GetN();i++){
    //if(int(TGExpLimit->GetX()[i])%50==0) 
    fprintf(pFileSum, "%f+-%f ",TGExpLimit->GetY()[i], (ExpLimitp1->GetY()[i]-ExpLimitm1->GetY()[i])/2.0);
  }fprintf(pFileSum,"\n");
  if(!blind) { fprintf(pFileSum, "Obs Limits for Model are: "); for(int i=0;i<TGObsLimit->GetN();i++){
      //if(int(TGObsLimit->GetX()[i])%50==0) 
      fprintf(pFileSum, "%f ",TGObsLimit->GetY()[i]);}fprintf(pFileSum,"\n"); }
  fclose(pFileSum);
}




