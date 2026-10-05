// Run-3 0-lepton and 2-lepton plots with explicit cutflow/category labels.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <regex>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "TROOT.h"
#include "TAxis.h"
#include "TStyle.h"
#include "TPad.h"
#include "TPave.h"
#include "TList.h"
#include "TString.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TPaveText.h"
#include "THStack.h"

#include "UserCode/bsmhiggs_fwk/interface/tdrstyle.h"
#include "UserCode/bsmhiggs_fwk/interface/MacroUtils.h"
#include "UserCode/bsmhiggs_fwk/interface/RootUtils.h"
#include "UserCode/bsmhiggs_fwk/interface/JSONWrapper.h"
#include "UserCode/bsmhiggs_fwk/interface/th1fmorph.h"

#include <dirent.h>

using namespace std;

int cutIndex=-1;
string cutIndexStr="";
double iLumi = 2007;
double iEcm=13.0;
bool showChi2 = false;
bool showUnc=false;
double baseRelUnc=0.0;
bool noLog=false;
bool isSim=false;
bool doTree = true;
bool doInterpollation = false;
bool do2D  = true;
bool do1D  = true;
bool doTex = true;
bool doPowers = true;
bool StoreInFile = true;
bool doPlot = true;
bool splitCanvas = false;
bool onlyCutIndex = false;
bool showRatioBox=true;
bool fixUnderflow=true;
bool fixOverflow=true;
bool addExternalNorm=false;

int rebin=1;
double blind=-1E99;
double metxmax=600.;
double  mtxmax=900.;

double topNorm_e_3b=0.80;
double topNorm_mu_3b=0.99;

double wNorm_e_3b=1.24;
double wNorm_mu_3b=1.32;

double signalScale=1.0;
string inDir   = "OUTNew/";
string jsonFile = "../../data/beauty-samples.json";
string outDir  = "Img/";
std::vector<std::string> plotExt;
string outFile = "/tmp/plotter.root";
string fileOption = "RECREATE";
std::vector<string> keywords;

std::unordered_map<string, std::vector<string> > MissingFiles;
std::unordered_map<string, std::vector<string> > DSetFiles;

struct NameAndType{
  std::string name;
  int type;
  bool isIndexPlot;
  NameAndType(std::string name_,  int type_, bool isIndexPlot_){name = name_; type = type_; isIndexPlot = isIndexPlot_;}
  bool is1D() const  {return type==1;}
  bool is2D() const  {return type==2;}
  bool is3D() const  {return type==3;}
  bool isTree() const{return type==4;}

  std::string mergeKey() const {
    return name;
  }

  bool operator==(const NameAndType& a) { return mergeKey() == a.mergeKey(); }
  bool operator< (const NameAndType& a) { return mergeKey() < a.mergeKey();  }

};

TH1* CheckPositiveBins( TH1* h, string histo_name){
  TH1* h_new = (TH1*)h->Clone("h_local");
  h_new->SetNameTitle( histo_name.c_str(), histo_name.c_str());
  for( int bin=0; bin<h->GetNbinsX(); bin++){
    if( h->GetBinContent(bin)<0. ){ h_new->SetBinContent( bin, 0.); h_new->SetBinError( bin, h->GetBinError(bin)); }
    else if( h->GetBinContent(bin)>=0. ){ h_new->SetBinContent( bin, h->GetBinContent(bin)); h_new->SetBinError( bin, h->GetBinError(bin)); }
  }
  return h_new;
}

string getDirName(JSONWrapper::Object& Process, string matchingKeyword=""){
  string dirName = Process.getStringFromKeyword(matchingKeyword, "tag", "");
  if(Process.isTagFromKeyword(matchingKeyword, "mctruthmode") ) { char buf[255]; sprintf(buf,"_filt%d",(int)Process.getIntFromKeyword(matchingKeyword, "mctruthmode", 0)); dirName += buf; }
  string procSuffix = Process.getStringFromKeyword(matchingKeyword, "suffix", "");
  if(procSuffix!=""){dirName += "_" + procSuffix;}
  while(dirName.find("/")!=std::string::npos)dirName.replace(dirName.find("/"),1,"-");
  return dirName;
}

void GetListOfObject(JSONWrapper::Object& Root, std::string RootDir, std::list<NameAndType>& histlist, std::string parentPath="/",  TDirectory* dir=NULL){

  if(parentPath=="/"){
    int dataProcessed = 0;
    int signProcessed = 0;
    int bckgProcessed = 0;

    std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
    for(size_t ip=0; ip<Process.size(); ip++){
      if(Process[ip].isTag("interpollation") || Process[ip].isTag("mixing") || Process[ip].isTag("nosample"))continue;
      string matchingKeyword="";
      if(!utils::root::getMatchingKeyword(Process[ip], keywords, matchingKeyword))continue;
      string dirName = getDirName(Process[ip], matchingKeyword);

      if(dir){
	TObject* tmp = utils::root::GetObjectFromPath(dir,dirName,false);
	if(tmp && tmp->InheritsFrom("TDirectory")){
	  printf("Adding all objects from %25s to the list of considered objects:\n",  dirName.c_str());
	  GetListOfObject(Root,RootDir,histlist,"", (TDirectory*)tmp );
	  continue;
	}
      }

      bool isData (  Process[ip].getBoolFromKeyword(matchingKeyword, "isdata", false)  );
      bool isSign ( !isData &&  Process[ip].getBoolFromKeyword(matchingKeyword, "spimpose", false));
      bool isMC   = !isData && !isSign;
      string filtExt("");
      if(Process[ip].isTagFromKeyword(matchingKeyword, "mctruthmode") ) {
	char buf[255]; sprintf(buf,"_filt%d",(int)Process[ip].getIntFromKeyword(matchingKeyword, "mctruthmode")); filtExt += buf; }

      std::vector<JSONWrapper::Object> Samples = (Process[ip])["data"].daughters();
      for(size_t id=0; id<Samples.size(); id++){
	string dtag = Samples[id].getString("dtag", "");
	int fileProcessed=0;

	DIR           *dirp;
	struct dirent *directory;
	string path = RootDir + "DATA/";
	if(!isData) path = RootDir + "MC/";
	dirp = opendir(path.c_str());
	if(!dirp) {std::cout << "Cannot open the folder: " << path << std::endl; continue;}
	while((directory = readdir(dirp)) != NULL){

	  string FileName;

	  FileName = directory->d_name;

	  if (FileName.find(dtag) == std::string::npos || FileName.find(".root") == std::string::npos) continue;

	  if (FileName.find(".root") == std::string::npos) {
	    FileName = path + FileName + ".root";
	  } else {
	    FileName = path + FileName;
	  }

	  FILE* pFile = fopen(FileName.c_str(), "r");
	  if(!pFile){MissingFiles[dtag].push_back(FileName); continue;}else{fclose(pFile);}

	  TFile* File = new TFile(FileName.c_str());
	  if(!File || File->IsZombie() || !File->IsOpen() || File->TestBit(TFile::kRecovered) ){
	    MissingFiles[dtag].push_back(FileName);
	    continue;
	  }else{
	    DSetFiles[dtag+filtExt].push_back(FileName);
	  }

	  if(fileProcessed%5!=0){File->Close();fileProcessed++;continue;}

	  if(fileProcessed==0 && isData){if(dataProcessed>=40 ){ File->Close(); continue;}else{dataProcessed++;}}
	  if(fileProcessed==0 && isSign){if(signProcessed>=20 ){ File->Close(); continue;}else{signProcessed++;}}
	  if(fileProcessed==0 && isMC  ){if(bckgProcessed>=20 ){ File->Close(); continue;}else{bckgProcessed++;}}
	  fileProcessed++;

	  printf("Adding all objects from %25s to the list of considered objects:\n",  FileName.c_str());
	  GetListOfObject(Root,RootDir,histlist,"", (TDirectory*)File );
	  File->Close();
	}
	closedir(dirp);
      }
    }

    if(MissingFiles.size()>0){
      printf("The list of missing or corrupted files, that are ignored, can be found below:\n");
      for(std::unordered_map<string, std::vector<string> >::iterator it = MissingFiles.begin(); it!=MissingFiles.end(); it++){
	if(it->second.size()<=0)continue;
	printf("Missing file in dataset %s:\n", it->first.c_str());
	for(unsigned int f=0;f<it->second.size();f++){
	  printf("\t %s\n", it->second[f].c_str());
	}
      }
    }

    return;

  }else if(dir){
    TList* list = dir->GetListOfKeys();
    for(int i=0;i<list->GetSize();i++){
      TObject* tmp = utils::root::GetObjectFromPath(dir,list->At(i)->GetName(),false);

      if(tmp->InheritsFrom("TDirectory")){
	GetListOfObject(Root,RootDir,histlist,parentPath+ list->At(i)->GetName()+"/",(TDirectory*)tmp);
      }else if(tmp->InheritsFrom("TTree")){
	printf("found one object inheriting from a ttree\n");
	histlist.push_back(NameAndType(parentPath+list->At(i)->GetName(), 4, false ) );
      }else if(tmp->InheritsFrom("TH1")){
	int  type = 0;
	if(tmp->InheritsFrom("TH1")) type++;
	if(tmp->InheritsFrom("TH2")) type++;
	if(tmp->InheritsFrom("TH3")) type++;
	bool hasIndex = string(((TH1*)tmp)->GetXaxis()->GetTitle()).find("cut index")<string::npos;
	if(hasIndex){type=1;}
	histlist.push_back(NameAndType(parentPath+list->At(i)->GetName(), type, hasIndex ) );
      }else{
	printf("The file contain an unknown object named %s\n", list->At(i)->GetName() );
      }
      delete tmp;
    }
  }

}

void MixProcess(JSONWrapper::Object& Root, TFile* File, std::list<NameAndType>& histlist){
  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  for(unsigned int i=0;i<Process.size();i++){
    if(!Process[i].isTag("mixing"))continue;
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;

    string dirName = getDirName(Process[i], matchingKeyword);
    File->cd();

    TDirectory* subdir = File->GetDirectory(dirName.c_str());
    if(subdir && subdir!=File){
      printf("Skip process %s as it seems to be already processed\n", dirName.c_str());
      continue;
    }

    std::vector<JSONWrapper::Object> subProcess = Process[i]["mixing"].daughters();
    std::vector<std::pair<string, double> > subProcList;
    for(unsigned int sp=0;sp<subProcess.size();sp++){
      string subProcDir = subProcess[sp].getString("tag", "");
      if(!File->GetDirectory(subProcDir.c_str())){printf("subprocess directory not found: %s\n", subProcDir.c_str()); continue;}
      subProcList.push_back(std::make_pair(subProcDir, subProcess[sp].getDouble("scale", 1.0)) );
    }
    if(subProcList.size()<=0){printf("No subProcess defined to construct the mixed process %s\n", dirName.c_str()); continue;  };

    subdir = File->mkdir(dirName.c_str());
    subdir->cd();

    int ictr = 0;
    int TreeStep = std::max(1,(int)(histlist.size()/50));
    printf("Mixing %20s :", dirName.c_str());
    for(std::list<NameAndType>::iterator it= histlist.begin(); it!= histlist.end(); it++,ictr++){
      if(ictr%TreeStep==0){printf(".");fflush(stdout);}
      NameAndType& HistoProperties = *it;

      TH1* obj1_new = NULL;
      TH1* obj1 = (TH1*)utils::root::GetObjectFromPath(File,subProcList[0].first + "/" + HistoProperties.name);
      if(!obj1)continue;
      obj1 = (TH1*)obj1->Clone(HistoProperties.name.c_str());
      utils::root::checkSumw2(obj1);
      obj1->Scale(subProcList[0].second);

      for(unsigned int sp=1;sp<subProcList.size();sp++){
	TH1* obj2 = (TH1*)utils::root::GetObjectFromPath(File,subProcList[sp].first + "/" + HistoProperties.name);
	if(!obj2)continue;
	obj1->Add(obj2, subProcList[sp].second);
      }
      utils::root::setStyleFromKeyword(matchingKeyword,Process[i], obj1);

      obj1_new = CheckPositiveBins( obj1, HistoProperties.name.c_str());
      subdir->cd();
      obj1_new->Write(HistoProperties.name.c_str());
      gROOT->cd();
      delete obj1_new;
    }printf("\n");
  }
}

void NRBProcess(JSONWrapper::Object& Root, TFile* File, std::list<NameAndType>& histlist){
  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  for(unsigned int i=0;i<Process.size();i++){
    if(!Process[i].isTag("NRB"))continue;
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;

    string dirName = getDirName(Process[i], matchingKeyword);
    File->cd();

    TDirectory* subdir = File->GetDirectory(dirName.c_str());
    if(subdir && subdir!=File){
      printf("Skip process %s as it seems to be already processed\n", dirName.c_str());
      continue;
    }

    std::vector<JSONWrapper::Object> subProcess = Process[i]["NRB"].daughters();
    std::vector<std::pair<string, double> > subProcList;
    for(unsigned int sp=0;sp<subProcess.size();sp++){
      string subProcDir = subProcess[sp].getString("tag", "");
      if(!File->GetDirectory(subProcDir.c_str())){printf("subprocess directory not found: %s\n", subProcDir.c_str()); continue;}
      subProcList.push_back(std::make_pair(subProcDir, subProcess[sp].getDouble("scale", 1.0)) );
    }
    if(subProcList.size()<=0){printf("No subProcess defined to construct the mixed process %s\n", dirName.c_str()); continue;  };

    subdir = File->mkdir(dirName.c_str());
    subdir->cd();

    int ictr = 0;
    int TreeStep = std::max(1,(int)(histlist.size()/50));
    printf("NRB %20s :", dirName.c_str());
    for(std::list<NameAndType>::iterator it= histlist.begin(); it!= histlist.end(); it++,ictr++){
      if(ictr%TreeStep==0){printf(".");fflush(stdout);}
      NameAndType& HistoProperties = *it;

      TH1* obj1 = (TH1*)utils::root::GetObjectFromPath(File,subProcList[0].first + "/" + HistoProperties.name);
      if(!obj1)continue;
      obj1 = (TH1*)obj1->Clone(HistoProperties.name.c_str());
      utils::root::checkSumw2(obj1);
      obj1->Reset();

      std::vector<std::pair<string,double>> channels = {std::make_pair("ee",0.369), std::make_pair("mumu",0.683), std::make_pair("ll",0.369+0.683)};
      for(auto ch=channels.begin();ch!=channels.end();ch++){
	if(HistoProperties.name.find(ch->first)==0){
	  std::string emName = HistoProperties.name;  emName.replace(0,ch->first.size(),"emu");
	  for(unsigned int sp=0;sp<subProcList.size();sp++){
	    TH1* obj2 = (TH1*)utils::root::GetObjectFromPath(File,subProcList[sp].first + "/" + emName);
	    if(!obj2)continue;
	    obj1->Add(obj2, subProcList[sp].second*ch->second);
	  }
	}
      }
      utils::root::setStyleFromKeyword(matchingKeyword,Process[i], obj1);

      subdir->cd();
      obj1->Write(HistoProperties.name.c_str());
      gROOT->cd();
      delete obj1;
    }printf("\n");
  }
}

void SumBins(JSONWrapper::Object& Root, TFile* File, std::list<NameAndType>& histlist){
  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  for(unsigned int i=0;i<Process.size();i++){
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;

    string dirName = getDirName(Process[i], matchingKeyword);
    File->cd();

    TDirectory* subdir = File->GetDirectory(dirName.c_str());
    if(! (subdir && subdir!=File)){
      printf("skip missing process %s\n", dirName.c_str());
      continue;
    }
    subdir->cd();

    int ictr = 0;
    int TreeStep = std::max(1,(int)(histlist.size()/50));
    printf("Sum %20s :", dirName.c_str());
    for(std::list<NameAndType>::iterator it= histlist.begin(); it!= histlist.end(); it++,ictr++){
      if(ictr%TreeStep==0){printf(".");fflush(stdout);}
      NameAndType& HistoProperties = *it;

      if( HistoProperties.name.find("geq1jets_")!=std::string::npos){

	TString Incname = HistoProperties.name.c_str();   Incname.ReplaceAll("geq1jets_", "_");
	if(utils::root::GetObjectFromPath(File,dirName + "/" + Incname.Data())!=NULL)continue;

	TH1* obj1 = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + HistoProperties.name);
	if(!obj1)continue;
	obj1 = (TH1*)obj1->Clone(HistoProperties.name.c_str());
	utils::root::checkSumw2(obj1);

	if(true){
	  TString name = HistoProperties.name.c_str();   name.ReplaceAll("geq1jets_", "eq0jets_");
	  TH1* obj2 = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + name.Data());
	  if(!obj2)continue;
	  obj1->Add(obj2, 1);
	}

	if(true){
	  TString name = HistoProperties.name.c_str();   name.ReplaceAll("geq1jets_", "vbf_");
	  TH1* obj2 = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + name.Data());
	  if(!obj2)continue;
	  obj1->Add(obj2, 1);
	}

	utils::root::setStyleFromKeyword(matchingKeyword,Process[i], obj1);

	subdir->cd();

	obj1->Write(Incname.Data());
	gROOT->cd();
	delete obj1;
      }
    }printf("\n");
  }
}

void InterpollateProcess(JSONWrapper::Object& Root, TFile* File, std::list<NameAndType>& histlist){
  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  for(unsigned int i=0;i<Process.size();i++){
    if(!Process[i].isTag("interpollation"))continue;
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;

    string dirName = getDirName(Process[i], matchingKeyword);
    File->cd();

    TDirectory* subdir = File->GetDirectory(dirName.c_str());
    if(!subdir || subdir==File){ subdir = File->mkdir(dirName.c_str());
    }else{
      printf("Skip process %s as it seems to be already processed\n", dirName.c_str());
      continue;
    }
    subdir->cd();

    string signal   = Process[i].getStringFromKeyword(matchingKeyword, "tag");
    string signalL  = Process[i]["interpollation"][0]["tagLeft"].c_str();
    string signalR  = Process[i]["interpollation"][0]["tagRight"].c_str();
    double mass     = Process[i]["interpollation"][0]["mass"].toDouble();
    double massL    = Process[i]["interpollation"][0]["massLeft"].toDouble();
    double massR    = Process[i]["interpollation"][0]["massRight"].toDouble();
    double Ratio = ((double)mass - massL); Ratio/=(massR - massL);

    double xsecXbr  = utils::root::getXsecXbr(Process[i]);
    double xsecXbrL = xsecXbr;  for(unsigned int j=0;j<Process.size();j++){if(Process[j]["tag"].c_str()==signalL){xsecXbrL = utils::root::getXsecXbr(Process[j]); break;}}
    double xsecXbrR = xsecXbr;  for(unsigned int j=0;j<Process.size();j++){if(Process[j]["tag"].c_str()==signalR){xsecXbrR = utils::root::getXsecXbr(Process[j]); break;}}

    while(signalL.find("/")!=std::string::npos)signalL.replace(signalL.find("/"),1,"-");
    while(signalR.find("/")!=std::string::npos)signalR.replace(signalR.find("/"),1,"-");

    int ictr = 0;
    int TreeStep = std::max(1,(int)(histlist.size()/50));
    printf("Interpol %20s :", signal.c_str());

    for(std::list<NameAndType>::iterator it= histlist.begin(); it!= histlist.end(); it++,ictr++){
      if(ictr%TreeStep==0){printf(".");fflush(stdout);}
      NameAndType& HistoProperties = *it;

      TH1* objL = (TH1*)utils::root::GetObjectFromPath(File,signalL + "/" + HistoProperties.name);
      TH1* objR = (TH1*)utils::root::GetObjectFromPath(File,signalR + "/" + HistoProperties.name);
      if(!objL || !objR)continue;

      TH1* histoInterpolated = NULL;
      if(HistoProperties.isIndexPlot){
	TH2F* histo2DL = (TH2F*) objL;
	TH2F* histo2DR = (TH2F*) objR;
	if(!histo2DL or !histo2DR)continue;
	histo2DL->Scale(1.0/xsecXbrL);
	histo2DR->Scale(1.0/xsecXbrR);
	TH2F* histo2D  = (TH2F*) histo2DL->Clone(histo2DL->GetName());
	histo2D->Reset();

	for(unsigned int cutIndex=0;cutIndex<=(unsigned int)(histo2DL->GetNbinsX()+1);cutIndex++){
	  TH1D* histoL = histo2DL->ProjectionY("tempL", cutIndex, cutIndex);
	  TH1D* histoR = histo2DR->ProjectionY("tempR", cutIndex, cutIndex);
	  if(histoL->GetSum() >0 && histoR->GetSum()>0){

	    TH1D* histo  = th1fmorph("interpolTemp","interpolTemp", histoL, histoR, massL, massR, mass, (1-Ratio)*histoL->Integral() + Ratio*histoR->Integral(), 0);
	    for(unsigned int y=0;y<=(unsigned int)(histo2DL->GetNbinsY()+1);y++){
	      histo2D->SetBinContent(cutIndex, y, histo->GetBinContent(y));
	      histo2D->SetBinError(cutIndex, y, histo->GetBinError(y));
	    }
	    delete histo;
	  }
	  delete histoR;
	  delete histoL;
	}
	delete histo2DL;
	delete histo2DR;
	histo2D->Scale(xsecXbr);
	histoInterpolated = histo2D;
      }else if(HistoProperties.is1D()){
	TH1F* histoL = (TH1F*) objL;
	TH1F* histoR = (TH1F*) objR;
	if(!histoL or !histoR)continue;
	if(histoL->GetSum() <=0 || histoR->GetSum()<=0)continue;
	histoL->Scale(1.0/xsecXbrL);
	histoR->Scale(1.0/xsecXbrR);
	if(histoL->Integral()>0 && histoR->Integral()>0){
	  double Integral = (1-Ratio)*histoL->Integral() + Ratio*histoR->Integral();
	  TH1F* histo  = (TH1F*)histoL->Clone(HistoProperties.name.c_str());  histo->Reset();

	  histo->Add(th1fmorph("interpolTemp","interpolTemp", histoL, histoR, massL, massR, mass, Integral, 0), 1.0);

	  histo->Scale(xsecXbr);
	  histoInterpolated = histo;
	}
	delete histoR;
	delete histoL;
      }

      if(histoInterpolated){
	subdir->cd();
	histoInterpolated->Write(HistoProperties.name.c_str());
	gROOT->cd();
	delete histoInterpolated;
      }

    }printf("\n");
  }
}

static bool applyManualBinLabels(TH1*, const std::string&);

void SavingToFile(JSONWrapper::Object& Root, std::string RootDir, TFile* OutputFile, std::list<NameAndType>& histlist){
  std::vector<TObject*> ObjectToDelete;
  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();

  int IndexFiles = 0;
  int NFilesStep = 0;
  for(std::unordered_map<string, std::vector<string> >::iterator it = DSetFiles.begin(); it!=DSetFiles.end(); it++){NFilesStep+=it->second.size();}
  printf("Total number of files to be processed  : %d\n", NFilesStep);

  for(unsigned int i=0;i<Process.size();i++){

    if(Process[i].isTag("interpollation") || Process[i].isTag("mixing") || Process[i].isTag("nosample"))continue;
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;

    string filtExt("");
    if(Process[i].isTagFromKeyword(matchingKeyword, "mctruthmode") ) { char buf[255]; sprintf(buf,"_filt%d",(int)Process[i].getIntFromKeyword(matchingKeyword, "mctruthmode")); filtExt += buf; }

    string dirName = getDirName(Process[i], matchingKeyword);
    OutputFile->cd();
    TDirectory* subdir = OutputFile->GetDirectory(dirName.c_str());
    if(!subdir || subdir==OutputFile){
      subdir = OutputFile->mkdir(dirName.c_str());
    }else{
      printf("Skip process %s as it seems to be already processed\n", dirName.c_str());
      continue;
    }

    subdir->cd();

    float  Weight = 1.0;
    std::vector<JSONWrapper::Object> Samples = (Process[i])["data"].daughters();
    for(unsigned int j=0;j<Samples.size();j++){
      std::vector<string>& fileList = DSetFiles[(Samples[j])["dtag"].toString()+filtExt];
      if(!Process[i].getBoolFromKeyword(matchingKeyword, "isdata", false) && !Process[i].getBoolFromKeyword(matchingKeyword, "isdatadriven", false)){

	Weight=iLumi;

      } else {Weight=1.0;}

      for(int f=0;f<fileList.size();f++){

	IndexFiles++;printf("\r %d%%(%d/%d)",100*IndexFiles/NFilesStep,IndexFiles,NFilesStep);fflush(stdout);
	TFile* File = new TFile(fileList[f].c_str());

	for(std::list<NameAndType>::iterator it= histlist.begin(); it!= histlist.end(); it++){
	  NameAndType& HistoProperties = *it;

	  TObject* inobj  = utils::root::GetObjectFromPath(File,HistoProperties.name);  if(!inobj)continue;
	  TObject* outobj = utils::root::GetObjectFromPath(subdir,HistoProperties.name);

	  if(HistoProperties.isTree()){
	    TTree* outtree = (TTree*)outobj;
	    if(!outtree){
	      subdir->cd();
	      outtree =  ((TTree*)inobj)->CloneTree(-1, "fast");
	      outtree->SetDirectory(subdir);

	      TTree* weightTree = new TTree((HistoProperties.name+"_PWeight").c_str(),"plotterWeight");
	      weightTree->Branch("plotterWeight",&weightTree,"plotterWeight/F");
	      weightTree->SetDirectory(subdir);
	      for(unsigned int i=0;i<((TTree*)inobj)->GetEntries();i++){weightTree->Fill();}
	    }else{
	      outtree->CopyEntries(((TTree*)inobj), -1, "fast");
	      TTree* weightTree = (TTree*)utils::root::GetObjectFromPath(subdir,HistoProperties.name+"_PWeight");
	      for(unsigned int i=0;i<((TTree*)inobj)->GetEntries();i++){weightTree->Fill();}
	    }
	  }else{
	    if(HistoProperties.name.find("optim_")==std::string::npos) ((TH1*)inobj)->Scale(Weight);
	    TH1* outhist = (TH1*)outobj;
	    if(!outhist){
	      subdir->cd();
	      outhist = (TH1*)(inobj)->Clone(inobj->GetName());
	      utils::root::setStyleFromKeyword(matchingKeyword,Process[i], outhist);
	      utils::root::checkSumw2(outhist);
	    }else{
	      outhist->Add((TH1*)inobj);
	    }
	  }
	}
	delete File;
      }
    }

    for(const auto& properties : histlist) {
      if(!properties.isTree()) {
        TH1* histogram = dynamic_cast<TH1*>(
            utils::root::GetObjectFromPath(subdir, properties.name));
        if(histogram) applyManualBinLabels(histogram, properties.name);
      }
    }
    subdir->cd();
    subdir->Write();

  }printf("\n");
}

void Draw2DHistogramSplitCanvas(JSONWrapper::Object& Root, TFile* File, NameAndType& HistoProperties){
  if(HistoProperties.isIndexPlot && cutIndex<0)return;

  std::string SaveName = "";

  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  std::vector<TObject*> ObjectToDelete;
  for(unsigned int i=0;i<Process.size();i++){
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;
    if(Process[i].getBoolFromKeyword(matchingKeyword, "isinvisible", false))continue;

    TCanvas* c1 = new TCanvas("c1","c1",500,500);
    c1->SetLogz(true);

    string dirName = getDirName(Process[i], matchingKeyword);
    TH1* hist = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + HistoProperties.name);
    if(!hist)continue;
    utils::root::setStyleFromKeyword(matchingKeyword,Process[i], hist);

    SaveName = hist->GetName();
    ObjectToDelete.push_back(hist);
    hist->SetTitle("");
    hist->SetStats(kFALSE);

    hist->Draw("COLZ");

    TPaveText* leg = new TPaveText(0.20,0.95,0.40,0.80, "NDC");
    leg->SetFillColor(0);
    leg->SetFillStyle(0);  leg->SetLineColor(0);
    leg->SetTextAlign(12);
    leg->AddText(Process[i]["tag"].c_str());
    leg->Draw("same");
    ObjectToDelete.push_back(leg);

    utils::root::DrawPreliminary(iLumi, iEcm);

    string SavePath = utils::root::dropBadCharacters(SaveName + "_" + (Process[i])["tag"].toString());
    if(outDir.size()) SavePath = outDir +"/"+ SavePath;
    for(auto ext = plotExt.begin(); ext != plotExt.end(); ++ext)
      {
        system(string(("rm -f ") + SavePath + *ext).c_str());
        c1->SaveAs((SavePath + *ext).c_str());
      }
    delete c1;
  }

  for(unsigned int d=0;d<ObjectToDelete.size();d++){delete ObjectToDelete[d];}ObjectToDelete.clear();
}

void Draw2DHistogram(JSONWrapper::Object& Root, TFile* File, NameAndType& HistoProperties){
  if(HistoProperties.isIndexPlot && cutIndex<0)return;

  std::string SaveName = "";

  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  int NSampleToDraw = 0;
  for(unsigned int i=0;i<Process.size();i++){
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;
    if(Process[i].getBoolFromKeyword(matchingKeyword, "isinvisible", false))continue;
    NSampleToDraw++;
  }
  int CanvasX = 3;
  int CanvasY = ceil(NSampleToDraw/CanvasX);
  TCanvas* c1 = new TCanvas("c1","c1",CanvasX*350,CanvasY*350);
  c1->Divide(CanvasX,CanvasY,0,0);

  std::vector<TObject*> ObjectToDelete;
  for(unsigned int i=0;i<Process.size();i++){
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;
    if(Process[i].getBoolFromKeyword(matchingKeyword, "isinvisible", false))continue;

    TVirtualPad* pad = c1->cd(i+1);
    pad->SetLogz(true);
    pad->SetTopMargin(0.0); pad->SetBottomMargin(0.10);  pad->SetRightMargin(0.20);

    string dirName = getDirName(Process[i], matchingKeyword);
    TH1* hist = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + HistoProperties.name);
    if(!hist)continue;
    utils::root::setStyleFromKeyword(matchingKeyword,Process[i], hist);

    SaveName = hist->GetName();
    ObjectToDelete.push_back(hist);
    hist->SetTitle("");
    hist->SetStats(kFALSE);

    hist->Draw("COLZ");

    TPaveText* leg = new TPaveText(0.10,0.995,0.30,0.90, "NDC");
    leg->SetFillColor(0);
    leg->SetFillStyle(0);  leg->SetLineColor(0);
    leg->SetTextAlign(12);
    leg->AddText(Process[i]["tag"].c_str());
    leg->Draw("same");
    ObjectToDelete.push_back(leg);
  }
  c1->cd(0);
  utils::root::DrawPreliminary(iLumi, iEcm);

  string SavePath = utils::root::dropBadCharacters(SaveName);
  if(outDir.size()) SavePath = outDir +"/"+ SavePath;
  for(auto ext = plotExt.begin(); ext != plotExt.end(); ++ext){
    system(string(("rm -f ") + SavePath + *ext).c_str());
    c1->SaveAs((SavePath + *ext).c_str());
  }
  for(unsigned int d=0;d<ObjectToDelete.size();d++){delete ObjectToDelete[d];}ObjectToDelete.clear();
  delete c1;
}

void addShapeUnc(TFile* File, string& dirName, NameAndType& HistoProperties, TH1* systHist, bool categorical){
  TH1* syst = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + "all_optim_systs");
  if(!syst){printf("all_optim_systs histogram is not there\n"); return;}

  for(int ivar = 1; ivar<=syst->GetNbinsX();ivar++){
    TH1* hist = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + HistoProperties.name + syst->GetXaxis()->GetBinLabel(ivar) );

    if(!hist){continue;}
    if(!categorical && abs(rebin)>0){

      hist = hist->Rebin(abs(rebin)); hist->Scale(1.0/abs(rebin), rebin<0?"width":"");
    }
    for(int ibin=1; ibin<=systHist->GetXaxis()->GetNbins(); ibin++){
      systHist->SetBinError(ibin, sqrt(pow(systHist->GetBinError(ibin),2)+pow(systHist->GetBinContent(ibin) - hist->GetBinContent(ibin),2)));
    }
  }
}

static inline std::string tolower_copy(std::string s){
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return std::tolower(c); });
  return s;
}

static inline bool contains_token(const std::string& s, const std::string& token){
  std::string ls = tolower_copy(s), lt = tolower_copy(token);
  size_t pos = ls.find(lt);
  while(pos != std::string::npos){
    bool left_ok  = (pos==0) || (ls[pos-1]=='_' || ls[pos-1]=='/' );
    bool right_ok = (pos+lt.size()>=ls.size()) || !std::isalpha((unsigned char)ls[pos+lt.size()]);
    if(left_ok && right_ok) return true;
    pos = ls.find(lt, pos+1);
  }
  return false;
}

static inline std::string inferAxisLabel(const std::string& name)
{
  const std::string lower = tolower_copy(name);
  if(lower.find("categories") != std::string::npos) return "Event category";
  if(lower.find("eventflow") != std::string::npos) return "Selection cut";
  if(lower == "ttbar_flavour_counts") return "ttbar flavour";
  if(lower.find("nlighttruth") != std::string::npos) return "N_{light jets}^{truth}";
  if(lower.find("nctruth") != std::string::npos) return "N_{c jets}^{truth}";
  if(lower.find("nbtruth") != std::string::npos) return "N_{b jets}^{truth}";
  if(lower.find("_ge3j_nt_") != std::string::npos) return "N_{jets passing T}";
  if(lower.find("_ge3j_nm_") != std::string::npos) return "N_{jets passing M}";
  if(lower.find("_ge3j_nl_") != std::string::npos) return "N_{jets passing L}";
  if(lower.find("event_btagsf") != std::string::npos) return "Event b-tag SF";
  if(lower.find("btagsf_wpm") != std::string::npos) return "b-tag SF (M WP)";
  if(lower.find("btagsf_wpt") != std::string::npos) return "b-tag SF (T WP)";
  if(lower.find("btagscore") != std::string::npos) return "UParTAK4 b-tag discriminant";
  if(lower.find("btagupartak4probbb") != std::string::npos) return "UParTAK4 P(bb)";
  if(lower.find("upartak4probbb") != std::string::npos) return "UParTAK4 P(bb)";
  if(lower.find("btagupartak4b") != std::string::npos) return "UParTAK4B";
  if(lower.find("ndbjets") != std::string::npos || lower.find("n_dbjets") != std::string::npos)
    return "N_{double-b jets}";
  if(lower.find("nbjets") != std::string::npos || lower.find("n_bjets") != std::string::npos)
    return "N_{b jets}";
  if(lower.find("njets") != std::string::npos || lower.find("n_jets") != std::string::npos)
    return "N_{jets}";

  const std::vector<std::pair<std::string, std::string>> titles = {
    {"puppimet_phi", "#phi(E_{T}^{miss})"},
    {"puppimet_pt", "E_{T}^{miss} [GeV]"},
    {"dphi_hz", "|#Delta#phi(Z,H)|"},
    {"deta_hz", "|#Delta#eta(Z,H)|"},
    {"dr_hz", "#Delta R(Z,H)"},
    {"dphi_bb_met_min", "#Delta#phi_{min}(bb,E_{T}^{miss})"},
    {"dphi_b_met_min", "#Delta#phi_{min}(b,E_{T}^{miss})"},
    {"dphi_j_met_min", "#Delta#phi_{min}(j,E_{T}^{miss})"},
    {"dphi_h_met", "|#Delta#phi(H,E_{T}^{miss})|"},
    {"dr_bb_bb_ave", "#LT#Delta R(bb,bb)#GT"},
    {"dr_bb_ave", "#LT#Delta R(bb)#GT"},
    {"dr_ll", "#Delta R(ll)"},
    {"dm_bb_bb_min", "|#Delta m(bb,bb)|_{min} [GeV]"},
    {"mbbj", "m_{bbj} [GeV]"},
    {"z_pt", "p_{T}^{Z} [GeV]"},
    {"pt_ll", "p_{T}^{ll} [GeV]"},
    {"h_pt", "p_{T}^{H} [GeV]"},
    {"h_mass", "m_{H} [GeV]"},
    {"mass_ll", "m_{ll} [GeV]"},
    {"mass_z", "m_{ll} [GeV]"},
    {"ht", "H_{T} [GeV]"},
    {"bdt", "BDT score"},
    {"lep0_pt", "p_{T}^{l1} [GeV]"},
    {"lep1_pt", "p_{T}^{l2} [GeV]"}
  };
  for(const auto& title : titles)
    if(contains_token(lower, title.first)) return title.second;
  return "";
}

// Edit these labels when the processors change their stored bin order.
// ROOT plots and LaTeX tables use separate strings for the same bins.
struct BinLabel {
  std::string raw;
  std::string root;
  std::string tex;
  BinLabel(const std::string& r, const std::string& p, const std::string& t)
      : raw(r), root(p), tex(t) {}
};

struct HistogramLabels {
  std::vector<BinLabel> bins;
  bool sequential;
  HistogramLabels(const std::vector<BinLabel>& b = {}, bool s = false)
      : bins(b), sequential(s) {}
};

static std::string histogramBasename(const std::string& name)
{
  const size_t slash = name.find_last_of('/');
  return slash == std::string::npos ? name : name.substr(slash + 1);
}

static const HistogramLabels* getManualLabels(const std::string& name)
{
  static const std::map<std::string, HistogramLabels> labels = [] {
    std::map<std::string, HistogramLabels> result;
    const BinLabel veto("veto", "0-lepton veto", "0-lepton veto");
    const BinLabel oneLepton("1lep", "1 lepton", "1 lepton");
    const BinLabel ossf(">=2lep OSSF", "#geq2 leptons, OSSF", "$N_{\\ell}\\geq2$, OSSF");
    const BinLabel osdf(">=2lep OSDF", "#geq2 leptons, OSDF", "$N_{\\ell}\\geq2$, OSDF");
    const BinLabel trigger("trigger", "Trigger", "Trigger");
    const BinLabel met("met", "p_{T}^{miss}>150 GeV", "$p_{T}^{\\rm miss}>150$ GeV");
    const BinLabel mll("mll", "75<m_{ll}<105 GeV", "$75<m_{\\ell\\ell}<105$ GeV");
    const BinLabel ge3j(">=3jets", "#geq3j", "$N_j\\geq3$");
    const BinLabel ge3j2b(">=3jets & >=2bjets", "#geq3j, #geq2b", "$N_j\\geq3,\\ N_b\\geq2$");
    const BinLabel ge3j3b(">=3jets & >=3bjets", "#geq3j, #geq3b", "$N_j\\geq3,\\ N_b\\geq3$");
    const BinLabel resolved(">=3jets & >=3bjets & <2dbjets", "#geq3j, #geq3b, <2db",
                            "$N_j\\geq3,\\ N_b\\geq3,\\ N_{db}<2$");
    const BinLabel ge2j(">=2jets", "#geq2j", "$N_j\\geq2$");
    const BinLabel ge2j1db(">=2jets & >=1dbjet", "#geq2j, #geq1db", "$N_j\\geq2,\\ N_{db}\\geq1$");
    const BinLabel ge2j2db(">=2jets & >=2dbjets", "#geq2j, #geq2db", "$N_j\\geq2,\\ N_{db}\\geq2$");

    // 0-lepton: keep MET and both final A-region cuts as separate steps.
    result["eventflow_SR_resolved"] = HistogramLabels({
      veto, trigger, met, ge3j, ge3j2b, ge3j3b, resolved,
      BinLabel("A_mH", "75#leqm_{H}<225 GeV", "$75\\leq m_H<225$ GeV"),
      BinLabel("A_dphi", "A SR: #Delta#phi_{min}>1", "$A$ SR: $\\Delta\\phi_{\\min}(b,{\\rm MET})>1$")
    }, true);
    result["eventflow_SR_boosted"] = HistogramLabels({
      veto, trigger, met, ge2j, ge2j1db, ge2j2db,
      BinLabel("A_mH", "75#leqm_{H}<175 GeV", "$75\\leq m_H<175$ GeV"),
      BinLabel("A_dphi", "A SR: #Delta#phi_{min}>1", "$A$ SR: $\\Delta\\phi_{\\min}(bb,{\\rm MET})>1$")
    }, true);
    result["eventflow_TTCR_resolved"] = HistogramLabels({
      oneLepton, trigger, met, ge3j, ge3j2b, ge3j3b, resolved,
      BinLabel("mH", "m_{H} sideband", "$m_H$ sideband")
    }, true);
    result["eventflow_TTCR_boosted"] = HistogramLabels({
      oneLepton, trigger, met, ge2j, ge2j1db, ge2j2db,
      BinLabel("mH", "m_{H} sideband", "$m_H$ sideband")
    }, true);
    const HistogramLabels qcd({
      BinLabel("A", "A", "$A$"), BinLabel("B", "B", "$B$"),
      BinLabel("C", "C", "$C$"), BinLabel("D", "D", "$D$"),
      BinLabel("Astar", "A*", "$A^{*}$"), BinLabel("Bstar", "B*", "$B^{*}$")
    });
    result["eventflow_QCDCR_resolved"] = qcd;
    result["eventflow_QCDCR_boosted"] = qcd;

    // 2-lepton: ee and mumu include mll; the emu TTCR does not.
    for(const std::string channel : {"ee", "mumu"}) {
      result[channel + "_eventflow_SR_resolved"] = HistogramLabels({
        ossf, trigger, mll, ge3j, ge3j2b, ge3j3b, resolved
      }, true);
      result[channel + "_eventflow_SR1_boosted"] = HistogramLabels({
        ossf, trigger, mll, ge2j, ge2j1db, ge2j2db
      }, true);
    }
    result["emu_eventflow_TTCR_resolved"] = HistogramLabels({
      osdf, trigger, ge3j, ge3j2b, ge3j3b, resolved
    }, true);
    result["emu_eventflow_TTCR1_boosted"] = HistogramLabels({
      osdf, trigger, ge2j, ge2j1db, ge2j2db
    }, true);

    // These are diagnostic multiplicities, not separate 3b/4b analysis regions.
    const HistogramLabels fixedWP({
      BinLabel("eq0b", "0b", "$N_b=0$"), BinLabel("eq1b", "1b", "$N_b=1$"),
      BinLabel("eq2b", "2b", "$N_b=2$"), BinLabel("eq3b", "3b", "$N_b=3$"),
      BinLabel("eq4b", "4b", "$N_b=4$"), BinLabel("ge5b", "#geq5b", "$N_b\\geq5$")
    });
    for(const std::string suffix : {"nosf", "withSF"}) {
      result["veto_SR_ge3j_nb_categories_fixedMWP_resolved_" + suffix] = fixedWP;
      for(const std::string channel : {"ee", "mumu"})
        result[channel + "_SR_ge3j_nb_categories_fixedTWP_" + suffix] = fixedWP;
      result["emu_TTCR_ge3j_nb_categories_fixedTWP_" + suffix] = fixedWP;
    }

    const HistogramLabels boosted({
      BinLabel("eq2j_eq1db", "(2j,1db)", "$(2j,1db)$"),
      BinLabel("eq2j_eq2db", "(2j,2db)", "$(2j,2db)$"),
      BinLabel("eq3j_eq1db", "(3j,1db)", "$(3j,1db)$"),
      BinLabel("eq3j_eq2db", "(3j,2db)", "$(3j,2db)$"),
      BinLabel("eq3j_eq3db", "(3j,3db)", "$(3j,3db)$"),
      BinLabel("eq4j_eq1db", "(4j,1db)", "$(4j,1db)$"),
      BinLabel("eq4j_eq2db", "(4j,2db)", "$(4j,2db)$"),
      BinLabel("eq4j_eq3db", "(4j,3db)", "$(4j,3db)$"),
      BinLabel("eq4j_eq4db", "(4j,4db)", "$(4j,4db)$"),
      BinLabel("geq5j_eq1db", "(#geq5j,1db)", "$(\\geq5j,1db)$"),
      BinLabel("geq5j_eq2db", "(#geq5j,2db)", "$(\\geq5j,2db)$"),
      BinLabel("geq5j_eq3db", "(#geq5j,3db)", "$(\\geq5j,3db)$"),
      BinLabel("geq5j_eq4db", "(#geq5j,4db)", "$(\\geq5j,4db)$"),
      BinLabel("geq5j_eq5db", "(#geq5j,#geq5db)", "$(\\geq5j,\\geq5db)$")
    });
    result["ee_SR_jet_dbtag_categories_boosted"] = boosted;
    result["mumu_SR_jet_dbtag_categories_boosted"] = boosted;
    result["emu_TTCR_jet_dbtag_categories_boosted"] = boosted;
    result["ttbar_flavour_counts"] = HistogramLabels({
      BinLabel("parent_total", "Parent total", "Parent total"),
      BinLabel("ttBB", "tt+bb", "$t\\bar t+bb$"),
      BinLabel("ttCC", "tt+cc", "$t\\bar t+cc$"),
      BinLabel("ttLF", "tt+light", "$t\\bar t+{\\rm light}$")
    });
    return result;
  }();
  const auto found = labels.find(histogramBasename(name));
  return found == labels.end() ? nullptr : &found->second;
}

static bool hasCategoricalBins(const std::string& name)
{
  const std::string lower = tolower_copy(name);
  return getManualLabels(name) || lower.find("eventflow") != std::string::npos
      || lower.find("evtflow") != std::string::npos
      || lower.find("categories") != std::string::npos;
}

static bool checkLabelLayout(const TH1* histogram, const std::string& name,
                             const HistogramLabels& labels)
{
  if(!histogram || histogram->GetDimension() != 1) return false;
  if(histogram->GetNbinsX() != static_cast<int>(labels.bins.size())) {
    std::cerr << "[labels] " << name << ": expected " << labels.bins.size()
              << " bins, found " << histogram->GetNbinsX() << "; keeping original labels\n";
    return false;
  }
  for(int bin = 1; bin <= histogram->GetNbinsX(); ++bin) {
    const std::string current = histogram->GetXaxis()->GetBinLabel(bin);
    const BinLabel& expected = labels.bins[bin - 1];
    if(!current.empty() && current != expected.raw && current != expected.root) {
      std::cerr << "[labels] " << name << ": unexpected label '" << current
                << "' in bin " << bin << "; keeping original labels\n";
      return false;
    }
  }
  return true;
}

static bool applyManualBinLabels(TH1* histogram, const std::string& name)
{
  const HistogramLabels* labels = getManualLabels(name);
  if(!labels || !checkLabelLayout(histogram, name, *labels)) return false;
  for(size_t bin = 0; bin < labels->bins.size(); ++bin)
    histogram->GetXaxis()->SetBinLabel(bin + 1, labels->bins[bin].root.c_str());
  histogram->GetXaxis()->LabelsOption("v");
  return true;
}

static void formatCategoricalAxis(TAxis* axis, bool categorical, bool showLabels)
{
  if(!categorical) return;
  axis->SetLabelSize(showLabels ? 0.035 : 0.0);
  if(showLabels) axis->LabelsOption("v");
}

struct EventflowYield {
  double yield;
  double error;
  EventflowYield(double y = 0.0, double e = 0.0) : yield(y), error(e) {}
};

static void addEventflowYield(EventflowYield& target, const EventflowYield& source)
{
  target.yield += source.yield;
  target.error = std::hypot(target.error, source.error);
}

static double asimovSignificance(double signal, double background)
{
  if(signal <= 0.0 || background <= 0.0) return 0.0;
  const double term = 2.0 * ((signal + background) * std::log1p(signal / background) - signal);
  return term > 0.0 ? std::sqrt(term) : 0.0;
}

void Draw1DHistogram(JSONWrapper::Object& Root, TFile* File, NameAndType& HistoProperties){

  const std::string xLabel = inferAxisLabel(HistoProperties.name);
  const bool categorical = hasCategoricalBins(HistoProperties.name);
  if(HistoProperties.isIndexPlot && cutIndex<0)return;

  TCanvas* c1 = new TCanvas("c1","c1",800,800);

  TPad* t1 = new TPad("t1","t1", 0.0, 0.2, 1.0, 1.0);
  t1->SetFillColor(0);
  t1->SetBorderMode(0);
  t1->SetBorderSize(2);
  t1->SetTickx(1);
  t1->SetTicky(1);
  t1->SetLeftMargin(0.10);
  t1->SetRightMargin(0.05);
  t1->SetTopMargin(0.05);
  t1->SetBottomMargin(categorical ? 0.34 : 0.10);
  t1->SetFrameFillStyle(0);
  t1->SetFrameBorderMode(0);
  t1->SetFrameFillStyle(0);
  t1->SetFrameBorderMode(0);

  t1->Draw();
  t1->cd();
  if(!noLog) t1->SetLogy(true);

  TLegend *legA = new TLegend(0.30,0.74,0.93,0.96, "NDC");
  legA->SetHeader("");
  legA->SetNColumns(3);
  legA->SetBorderSize(0);
  legA->SetTextFont(42);   legA->SetTextSize(0.03);
  legA->SetLineColor(0);   legA->SetLineStyle(1);   legA->SetLineWidth(1);
  legA->SetFillColor(0);   legA->SetFillStyle(0);

  THStack* stack = new THStack("MC","MC");
  TH1*     data  = NULL;
  TH1 *     mc   = NULL;
  TH1 *     mcPlusSyst   = NULL;
  TH1 *     mcPlusRelUnc = NULL;
  std::vector<TH1*> spimpose;
  std::vector<TString> spimposeOpts;
  std::vector<TObject*> ObjectToDelete;
  std::string SaveName = HistoProperties.name;
  std::vector<JSONWrapper::Object> Process = Root["proc"].daughters();
  double Maximum = -1E100;
  double Minimum =  1E100;
  double SignalMin = 5E-2;
  ObjectToDelete.push_back(stack);

  std::vector<TLegendEntry*> legEntries;
  for(unsigned int i=0;i<Process.size();i++){
    string matchingKeyword="";
    if(!utils::root::getMatchingKeyword(Process[i], keywords, matchingKeyword))continue;
    if(Process[i].getBoolFromKeyword(matchingKeyword, "isinvisible", false))continue;
    string dirName = getDirName(Process[i], matchingKeyword);
    TH1* hist = (TH1*)utils::root::GetObjectFromPath(File,dirName + "/" + HistoProperties.name);
    if(!hist){

      if(Process[i].getBoolFromKeyword(matchingKeyword, "isdata", false)){
	TH1D* dummy = new TH1D("dummy", "dummy", 1, 0, 1);
	utils::root::setStyleFromKeyword(matchingKeyword,Process[i], dummy);
	legA->AddEntry(dummy, Process[i].getStringFromKeyword(matchingKeyword, "tag", "").c_str(), "P E");
	ObjectToDelete.push_back(dummy);
      }

      continue;
    }
    // Work on a detached copy so plots never rebin or relabel the stored counts.
    hist = (TH1*)hist->Clone();
    hist->SetDirectory(nullptr);
    if(!categorical && abs(rebin)>0) {
      hist = hist->Rebin(abs(rebin));
      hist->Scale(1.0/abs(rebin), rebin<0 ? "width" : "");
    }
    applyManualBinLabels(hist, HistoProperties.name);
    utils::root::setStyleFromKeyword(matchingKeyword,Process[i], hist);

    if(!categorical) utils::root::fixExtremities(hist,fixOverflow,fixUnderflow);
    if(Process[i].isTagFromKeyword(matchingKeyword, "normto")) hist->Scale( Process[i].getDoubleFromKeyword(matchingKeyword, "normto", 1.0)/hist->Integral() );

    if(Maximum<hist->GetMaximum())Maximum=hist->GetMaximum();
    if(Minimum>hist->GetMinimum())Minimum=hist->GetMinimum();
    ObjectToDelete.push_back(hist);

    if(Process[i].getBoolFromKeyword(matchingKeyword, "isdata", false)){
      if(!data){legA->AddEntry(hist, Process[i].getStringFromKeyword(matchingKeyword, "tag", "").c_str(), "P E");}
      if(!data){data = (TH1D*)hist->Clone("data");utils::root::checkSumw2(data);}
      else{data->Add(hist);}
    }else if(Process[i].getBoolFromKeyword(matchingKeyword, "spimpose", false)){
      legEntries.insert(legEntries.begin(), new TLegendEntry(hist, Process[i].getStringFromKeyword(matchingKeyword, "tag", "").c_str(), "L") );
      hist->Scale(signalScale);
      spimposeOpts.push_back( "hist" );
      spimpose.push_back(hist);
      if(SignalMin>hist->GetMaximum()*1E-2) SignalMin=hist->GetMaximum()*1E-2;
    }else{

      if (addExternalNorm) {

	if (dirName.find("t#bar{t}+jets")!=std::string::npos) {
	  if( HistoProperties.name.find("lep1_")!=std::string::npos){
	    if( HistoProperties.name.find("3b")!=std::string::npos) hist->Scale(topNorm_e_3b);
	  }
	  if( HistoProperties.name.find("MU_")!=std::string::npos){
	    if( HistoProperties.name.find("3b")!=std::string::npos) hist->Scale(topNorm_mu_3b);
	  }
	}
	if (dirName.find("W#rightarrow l#nu")!=std::string::npos) {
	  if( HistoProperties.name.find("E_")!=std::string::npos){
	    if( HistoProperties.name.find("3b")!=std::string::npos) hist->Scale(wNorm_e_3b);
	  }
	  if( HistoProperties.name.find("MU_")!=std::string::npos){
	    if( HistoProperties.name.find("3b")!=std::string::npos) hist->Scale(wNorm_mu_3b);
	  }
	}
      }

      stack->Add(hist, "HIST");
      legEntries.push_back(new TLegendEntry(hist, Process[i].getStringFromKeyword(matchingKeyword, "tag", "").c_str(), "F"));
      if(!mc){mc = (TH1D*)hist->Clone("mc");utils::root::checkSumw2(mc);
      }
      else{mc->Add(hist);}

      if(showUnc){
	TH1* histPlusSyst = (TH1D*)hist->Clone("histPlusSyst");

	double syst = 0.0;

	if(baseRelUnc>0){
	  syst+=pow(baseRelUnc,2);
	}

	if(Process[i].isTagFromKeyword(matchingKeyword, "syst")){
	  std::vector<JSONWrapper::Object> systs = Process[i]["syst"].daughters();
	  for(size_t isyst=0; isyst<systs.size(); isyst++){syst+=pow(systs[isyst].toDouble(),2);}
	}

	if(syst>0){
	  syst = sqrt(syst);
	  for(int ibin=1; ibin<=histPlusSyst->GetXaxis()->GetNbins(); ibin++){
	    histPlusSyst->SetBinError(ibin, sqrt(pow(histPlusSyst->GetBinError(ibin),2)+pow(histPlusSyst->GetBinContent(ibin)*syst,2)));
	  }
	}

	addShapeUnc(File, dirName, HistoProperties, histPlusSyst, categorical);

	if(!mcPlusSyst){ mcPlusSyst = (TH1D*)hist->Clone("mcPlusSyst");mcPlusSyst->Reset(); utils::root::checkSumw2(mcPlusSyst);}
	mcPlusSyst->Add(histPlusSyst);
      }
    }
  }

  while(!legEntries.empty()){
    legA->AddEntry(legEntries.back()->GetObject(), legEntries.back()->GetLabel(), legEntries.back()->GetOption());
    ObjectToDelete.push_back(legEntries.back());
    legEntries.pop_back();
  }

  if(mc   && Maximum<mc  ->GetMaximum()) Maximum=mc  ->GetMaximum()*1.1;
  if(data && Maximum<data->GetMaximum()) Maximum=data->GetMaximum()*1.1;
  Maximum = noLog?Maximum*1.5:Maximum*2E3;
  Minimum = noLog?0:std::min(SignalMin, std::max(5E-2, Minimum)) * 0.9;

  if(stack->GetNhists()<=0){
    TH1* dummy = NULL;
    if(!dummy && data)dummy=(TH1*)data->Clone("dummy");
    if(!dummy && spimpose.size()>0)dummy=(TH1*)spimpose[0]->Clone("dummy");

    if(!dummy) printf("Error the frame is still empty\n");

    stack->Add(dummy);
    ObjectToDelete.push_back(dummy);
  } else {

    stack->Draw("");

    stack->SetTitle("");

    if(!xLabel.empty())
      stack->GetXaxis()->SetTitle(xLabel.c_str());
    else
      stack->GetXaxis()->SetTitle(((TH1*)stack->GetStack()->At(0))->GetXaxis()->GetTitle());

    stack->GetYaxis()->SetTitle("Events");

    stack->GetXaxis()->SetLabelOffset(0.007);
    stack->GetXaxis()->SetLabelSize(0.04);
    formatCategoricalAxis(stack->GetXaxis(), categorical, true);
    stack->GetXaxis()->SetTitleOffset(1.2);
    stack->GetXaxis()->SetTitleFont(42);
    stack->GetXaxis()->SetTitleSize(0.04);
    stack->GetYaxis()->SetLabelFont(42);
    stack->GetYaxis()->SetLabelOffset(0.007);
    stack->GetYaxis()->SetLabelSize(0.04);
    stack->GetYaxis()->SetTitleOffset(1.35);
    stack->GetYaxis()->SetTitleFont(42);
    stack->GetYaxis()->SetTitleSize(0.04);
    stack->GetYaxis()->SetRangeUser(Minimum, Maximum);
    stack->SetMinimum(Minimum);
    stack->SetMaximum(Maximum);

    t1->Update();

    if(showUnc && mc){
      TGraphErrors* systBand = new TGraphErrors(mc->GetNbinsX()); int IPoint=0;
      for(int ibin=1; ibin<=mcPlusSyst->GetXaxis()->GetNbins(); ibin++){
	systBand->SetPoint     (IPoint, mcPlusSyst->GetBinCenter(ibin), mcPlusSyst->GetBinContent(ibin) );
	systBand->SetPointError(IPoint, mcPlusSyst->GetBinWidth(ibin)/2, mcPlusSyst->GetBinError(ibin) );
	IPoint++;
      }systBand->Set(IPoint);
      mcPlusSyst->SetFillStyle(3004);
      mcPlusSyst->SetFillColor(kGray+2);
      mcPlusSyst->SetMarkerStyle(1);

      systBand->SetFillStyle(3004);
      systBand->SetFillColor(kGray+2);
      systBand->SetMarkerStyle(1);
      systBand->Draw("same 2 0");

      TGraphErrors* errBand = new TGraphErrors(mc->GetNbinsX()); IPoint=0;
      mcPlusRelUnc = (TH1 *) mc->Clone("totalmcwithunc");utils::root::checkSumw2(mcPlusRelUnc); mcPlusRelUnc->SetDirectory(0);
      for(int ibin=1; ibin<=mcPlusRelUnc->GetXaxis()->GetNbins(); ibin++){
	errBand->SetPoint     (IPoint, mcPlusRelUnc->GetBinCenter(ibin), mcPlusRelUnc->GetBinContent(ibin) );
	errBand->SetPointError(IPoint, mcPlusRelUnc->GetBinWidth(ibin)/2, mcPlusRelUnc->GetBinError(ibin) );
	IPoint++;
      }errBand->Set(IPoint);
      mcPlusRelUnc->SetFillStyle(3005);
      mcPlusRelUnc->SetFillColor(kGray+3);
      mcPlusRelUnc->SetMarkerStyle(1);

      errBand->SetFillStyle(3005);
      errBand->SetFillColor(kGray+3);
      errBand->SetMarkerStyle(1);
      errBand->Draw("same 2 0");
      legA->AddEntry(errBand, "Stat. Unc.", "F");
      legA->AddEntry(systBand, "Syst + Stat.", "F");
    }

    if(data){
      if(blind>-1E99){
	if(data->GetBinLowEdge(data->FindBin(blind)) != blind){printf("Warning blinding changed from %f to %f for %s inorder to stay on bin edges\n", blind, data->GetBinLowEdge(data->FindBin(blind)),  HistoProperties.name.c_str() );}
	for(unsigned int i=data->FindBin(blind);i<=data->GetNbinsX()+1; i++){
	  data->SetBinContent(i, 0); data->SetBinError(i, 0);
	}
	TH1 *hist=(TH1*)stack->GetHistogram();

	TPave* blinding_box = new TPave(data->GetBinLowEdge(data->FindBin(blind)),  hist->GetMinimum(), data->GetXaxis()->GetXmax(), hist->GetMaximum(), 0, "NB" );
	blinding_box->SetFillColor(15);         blinding_box->SetFillStyle(3013);         blinding_box->Draw("same F");
	legA->AddEntry(blinding_box, "blinded area" , "F");
	ObjectToDelete.push_back(blinding_box);
      }
      data->Draw("E1 same");
    }

    for(size_t ip=0; ip<spimpose.size(); ip++){
      TString opt=spimposeOpts[ip];
      spimpose[ip]->Draw(opt + "same");
    }

    if(showChi2){
      TPaveText *pave = new TPaveText(0.6,0.85,0.8,0.9,"NDC");
      pave->SetBorderSize(0);
      pave->SetFillStyle(0);
      pave->SetTextAlign(32);
      pave->SetTextFont(42);
      char buf[100];
      if(data && mc && data->Integral()>0 && mc->Integral()>0){
	sprintf(buf,"#chi^{2}/ndof : %3.2f", data->Chi2Test(mc,"WWCHI2/NDF") );
	pave->AddText(buf);
	sprintf(buf,"K-S prob: %3.2f", data->KolmogorovTest(mc));
	pave->AddText(buf);

      }else if(mc && spimpose.size()>0 && mc->Integral()>0){
	for(size_t ip=0; ip<spimpose.size(); ip++){
	  if(spimpose[ip]->Integral()<=0) continue;
	  sprintf(buf,"#chi^{2}/ndof : %3.2f, K-S prob: %3.2f", spimpose[ip]->Chi2Test(mc,"WWCHI2/NDF"), spimpose[ip]->KolmogorovTest(mc) );
	  pave->AddText(buf);
	}
      }
      pave->Draw();
    }

    legA->Draw("same");

    std::vector<TH1 *> compDists;
    if(data)                   compDists.push_back(data);
    else if(spimpose.size()>0) compDists=spimpose;

    if( !(mc && compDists.size() && showRatioBox) ){
      t1->SetPad(0,0,1,1);
    }else{
      if(categorical) {
        t1->SetBottomMargin(0.02);
        formatCategoricalAxis(stack->GetXaxis(), true, false);
        t1->SetPad(0, 0.35, 1, 1);
      }
      c1->cd();
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
      t2->SetBottomMargin(categorical ? 0.62 : 0.20);
      t2->SetFrameFillStyle(0);
      t2->SetFrameBorderMode(0);
      t2->SetFrameFillStyle(0);
      t2->SetFrameBorderMode(0);
      t2->Draw();
      t2->cd();
      t2->SetGridy(true);
      t2->SetPad(0,0.0,1.0,categorical ? 0.35 : 0.2);

      TH1D *denSystUncH=0;
      if(mcPlusSyst)        denSystUncH=(TH1D *) mcPlusSyst  ->Clone("mcrelunc");
      else if (mcPlusRelUnc)denSystUncH=(TH1D *) mcPlusRelUnc->Clone("mcrelunc");
      else                  denSystUncH=(TH1D *) mc          ->Clone("mcrelunc");
      utils::root::checkSumw2(denSystUncH);

      int GPoint=0;
      TGraphErrors *denSystUnc=new TGraphErrors(denSystUncH->GetXaxis()->GetNbins());
      for(int xbin=1; xbin<=denSystUncH->GetXaxis()->GetNbins(); xbin++){
        // Use the stored weighted errors, not a Poisson estimate of MC yields.
        const double content = denSystUncH->GetBinContent(xbin);
        const double relErr = content != 0.0 ?
            denSystUncH->GetBinError(xbin)/std::fabs(content) : 0.0;
        denSystUnc->SetPoint(GPoint, denSystUncH->GetBinCenter(xbin), 1.0);
        denSystUnc->SetPointError(GPoint, denSystUncH->GetBinWidth(xbin)/2, relErr);
        GPoint++;
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
      denSystUnc->Draw("2 0 same");
      float yscale = categorical ? (1.0-0.35)/0.35 : (1.0-0.2)/0.2;
      denSystUncH->GetYaxis()->SetTitle("Data/#Sigma Bkg.");
      denSystUncH->GetXaxis()->SetTitle("");

      denSystUncH->SetMinimum(0.4);
      denSystUncH->SetMaximum(1.6);

      denSystUncH->GetXaxis()->SetLabelFont(42);
      denSystUncH->GetXaxis()->SetLabelOffset(0.007);
      denSystUncH->GetXaxis()->SetLabelSize(0.04 * yscale);
      formatCategoricalAxis(denSystUncH->GetXaxis(), categorical, true);
      if(categorical) denSystUncH->GetXaxis()->SetLabelSize(0.035 * yscale);
      denSystUncH->GetXaxis()->SetTitleFont(42);
      denSystUncH->GetXaxis()->SetTitleSize(0.035 * yscale);
      denSystUncH->GetXaxis()->SetTitleOffset(0.8);
      denSystUncH->GetYaxis()->SetLabelFont(42);
      denSystUncH->GetYaxis()->SetLabelOffset(0.007);
      denSystUncH->GetYaxis()->SetLabelSize(0.03 * yscale);
      denSystUncH->GetYaxis()->SetTitleFont(42);
      denSystUncH->GetYaxis()->SetTitleSize(0.035 * yscale);
      denSystUncH->GetYaxis()->SetTitleOffset(0.3);

      TH1D *denRelUncH=0;
      if(mcPlusRelUnc) denRelUncH=(TH1D *) mcPlusRelUnc->Clone("mcrelunc");
      else             denRelUncH=(TH1D *) mc->Clone("mcrelunc");
      utils::root::checkSumw2(denRelUncH);

      GPoint=0;
      TGraphErrors *denRelUnc=new TGraphErrors(denRelUncH->GetXaxis()->GetNbins());
      for(int xbin=1; xbin<=denRelUncH->GetXaxis()->GetNbins(); xbin++) {
        const double content = denRelUncH->GetBinContent(xbin);
        const double relErr = content != 0.0 ?
            denRelUncH->GetBinError(xbin)/std::fabs(content) : 0.0;
        denRelUnc->SetPoint(GPoint, denRelUncH->GetBinCenter(xbin), 1.0);
        denRelUnc->SetPointError(GPoint, denRelUncH->GetBinWidth(xbin)/2, relErr);
        GPoint++;
      }denRelUnc->Set(GPoint);
      denRelUnc->SetLineColor(1);
      denRelUnc->SetFillStyle(3005);
      denRelUnc->SetFillColor(kGray+3);
      denRelUnc->SetMarkerColor(1);
      denRelUnc->SetMarkerStyle(1);

      denRelUnc->Draw("2 0 same");

      for(size_t icd=0; icd<compDists.size(); icd++){
	TString name("CompHistogram"); name+=icd;

        TH1* dataToObsH = (TH1*)compDists[icd]->Clone(name);
        utils::root::checkSumw2(dataToObsH);
        if(categorical) {
          // The MC uncertainty is already shown by the denominator bands.
          for(int bin=1; bin<=dataToObsH->GetNbinsX(); ++bin) {
            const double numerator = compDists[icd]->GetBinContent(bin);
            const double denominator = mc->GetBinContent(bin);
            dataToObsH->SetBinContent(bin, denominator > 0.0 ? numerator/denominator : 0.0);
            dataToObsH->SetBinError(bin, denominator > 0.0 ?
                compDists[icd]->GetBinError(bin)/denominator : 0.0);
          }
        } else {
          dataToObsH->Divide(mc);
        }
        TGraphErrors* dataToObs = new TGraphErrors(dataToObsH);
	dataToObs->SetMarkerColor(dataToObsH->GetMarkerColor());
	dataToObs->SetMarkerStyle(dataToObsH->GetMarkerStyle());
	dataToObs->SetMarkerSize(dataToObsH->GetMarkerSize());
	dataToObs->Draw("P 0 same");
      }

      if(data && blind>-1E99){
	TPave* blinding_box = new TPave(data->GetBinLowEdge(data->FindBin(blind)), 0.4, data->GetXaxis()->GetXmax(), 1.6, 0, "NB" );
	blinding_box->SetFillColor(15);         blinding_box->SetFillStyle(3013);         blinding_box->Draw("same F");
	ObjectToDelete.push_back(blinding_box);
      }

      TLegend *legR = new TLegend(0.56,0.78,0.93,0.96, "NDC");
      legR->SetHeader("");
      legR->SetNColumns(2);
      legR->SetBorderSize(1);
      legR->SetTextFont(42);   legR->SetTextSize(0.03 * yscale);
      legR->SetLineColor(1);   legR->SetLineStyle(1);   legR->SetLineWidth(1);
      legR->SetFillColor(0);   legR->SetFillStyle(1001);
      legR->AddEntry(denRelUnc, "Stat. Unc.", "F");
      legR->AddEntry(denSystUnc, "Syst. + Stat.", "F");

      gPad->RedrawAxis();

    }
  }
  t1->cd();
  c1->cd();
  utils::root::DrawPreliminary(iLumi, iEcm, t1);
  c1->Modified();
  c1->Update();

  string SavePath = utils::root::dropBadCharacters(SaveName);
  if(outDir.size()) SavePath = outDir +"/"+ SavePath;
  for(auto i = plotExt.begin(); i != plotExt.end(); ++i){

    c1->SaveAs((SavePath + *i).c_str());
  }

  delete c1;

}

struct TableEntry {
  std::string name;
  std::vector<EventflowYield> stages;
};

struct YieldTable {
  std::vector<std::string> labels;
  std::vector<TableEntry> backgrounds;
  std::vector<TableEntry> signals;
  std::vector<EventflowYield> totalBackground;
  std::vector<EventflowYield> data;
};

static std::string escapeTexText(const std::string& text)
{
  std::string escaped;
  for(char ch : text) {
    switch(ch) {
      case '\\': escaped += "\\textbackslash{}"; break;
      case '&': case '%': case '$': case '#': case '_': case '{': case '}':
        escaped += '\\'; escaped += ch; break;
      case '~': escaped += "\\textasciitilde{}"; break;
      case '^': escaped += "\\textasciicircum{}"; break;
      default: escaped += ch;
    }
  }
  return escaped;
}

static std::string rootLabelToTex(std::string label)
{
  if(label.find('$') != std::string::npos) return label;
  if(label.find('#') == std::string::npos) return escapeTexText(label);
  std::replace(label.begin(), label.end(), '#', '\\');
  return "$" + label + "$";
}

static std::string texProcessName(JSONWrapper::Object& process,
                                  const std::string& matchingKeyword)
{
  return rootLabelToTex(process.getStringFromKeyword(matchingKeyword, "tag", ""));
}

static void accumulateStages(std::vector<EventflowYield>& target,
                              const std::vector<EventflowYield>& source)
{
  if(target.empty()) target.resize(source.size());
  for(size_t bin = 0; bin < source.size(); ++bin)
    addEventflowYield(target[bin], source[bin]);
}

static YieldTable collectYieldTable(JSONWrapper::Object& root, TFile* file,
                                    const std::string& name, int firstBin)
{
  YieldTable table;
  const HistogramLabels* manual = getManualLabels(name);
  std::vector<JSONWrapper::Object> processes = root["proc"].daughters();
  for(auto& process : processes) {
    std::string matchingKeyword;
    if(!utils::root::getMatchingKeyword(process, keywords, matchingKeyword)) continue;
    const std::string directory = getDirName(process, matchingKeyword);
    TH1* histogram = dynamic_cast<TH1*>(
        utils::root::GetObjectFromPath(file, directory + "/" + name));
    if(!histogram || histogram->GetDimension() != 1 || firstBin > histogram->GetNbinsX())
      continue;
    if(manual && !checkLabelLayout(histogram, name, *manual)) continue;

    const size_t count = histogram->GetNbinsX() - firstBin + 1;
    if(!table.labels.empty() && table.labels.size() != count) {
      std::cerr << "[table] Inconsistent bin count for " << directory << "/" << name << '\n';
      continue;
    }
    if(table.labels.empty()) {
      for(int bin = firstBin; bin <= histogram->GetNbinsX(); ++bin) {
        const std::string label = histogram->GetXaxis()->GetBinLabel(bin);
        table.labels.push_back(manual ? manual->bins[bin - 1].tex :
                               (label.empty() ? "Bin " + std::to_string(bin) : rootLabelToTex(label)));
      }
    }
    TableEntry entry;
    entry.name = texProcessName(process, matchingKeyword);
    for(int bin = firstBin; bin <= histogram->GetNbinsX(); ++bin)
      entry.stages.emplace_back(histogram->GetBinContent(bin), histogram->GetBinError(bin));

    const bool isData = process.getBoolFromKeyword(matchingKeyword, "isdata", false);
    const bool isSignal = !isData && (
        process.getBoolFromKeyword(matchingKeyword, "issignal", false) ||
        process.getBoolFromKeyword(matchingKeyword, "spimpose", false));
    if(isData) accumulateStages(table.data, entry.stages);
    else if(isSignal) table.signals.push_back(entry);
    else {
      table.backgrounds.push_back(entry);
      accumulateStages(table.totalBackground, entry.stages);
    }
  }
  return table;
}

static void printYieldRow(FILE* output, const std::string& name,
                          const std::vector<EventflowYield>& stages)
{
  fprintf(output, "%s", name.c_str());
  for(const auto& stage : stages)
    fprintf(output, " & %s", utils::toLatexRounded(
        stage.yield, stage.error, -1, doPowers).c_str());
  fprintf(output, " \\\\\n");
}

static void printEfficiencyRow(FILE* output, const std::string& name,
                               const std::vector<EventflowYield>& stages)
{
  fprintf(output, "%s efficiency", name.c_str());
  const double triggerYield = stages.empty() ? 0.0 : stages.front().yield;
  for(const auto& stage : stages) {
    if(triggerYield != 0.0) fprintf(output, " & %.5g", stage.yield / triggerYield);
    else fprintf(output, " & --");
  }
  fprintf(output, " \\\\\n");
}

static void printSensitivityRow(FILE* output, const std::string& signalName,
                                const std::vector<EventflowYield>& signal,
                                const std::vector<EventflowYield>& background, int mode)
{
  const std::string quantity = mode == 0 ? "$S/B$" :
                               (mode == 1 ? "$S/\\sqrt{B}$" : "$Z_A$");
  fprintf(output, "%s [%s]", quantity.c_str(), signalName.c_str());
  for(size_t bin = 0; bin < signal.size(); ++bin) {
    const double s = signal[bin].yield;
    const double b = background[bin].yield;
    if(b <= 0.0 || s < 0.0) { fprintf(output, " & --"); continue; }
    const double value = mode == 0 ? s / b :
                         (mode == 1 ? s / std::sqrt(b) : asimovSignificance(s, b));
    fprintf(output, " & %.5g", value);
  }
  fprintf(output, " \\\\\n");
}

static void writeYieldTable(const YieldTable& table, const std::string& name,
                            bool efficiencies)
{
  if(table.labels.empty()) return;
  const std::string stem = name + (efficiencies ? "_efficiency" : "");
  std::string path = utils::root::dropBadCharacters(stem + ".tex");
  if(!outDir.empty()) path = outDir + "/" + path;
  FILE* output = fopen(path.c_str(), "w");
  if(!output) { std::cerr << "[table] Cannot write " << path << '\n'; return; }

  std::string columns = "|l" + std::string(table.labels.size(), 'c') + "|";
  fprintf(output, "%%\\usepackage{rotating,graphicx}\n\\begin{sidewaystable}[htp]\n\\centering\n");
  fprintf(output, "\\caption{Event yields%s for \\texttt{%s}.}\n",
          efficiencies ? " and efficiencies relative to the trigger step" : "",
          escapeTexText(name).c_str());
  fprintf(output, "\\label{tab:%s}\n\\resizebox{\\linewidth}{!}{%%\n\\begin{tabular}{%s} \\hline\n",
          utils::root::dropBadCharacters(stem).c_str(), columns.c_str());
  fprintf(output, "%s", efficiencies ? "Process / quantity" : "Process");
  for(const auto& label : table.labels) fprintf(output, " & %s", label.c_str());
  fprintf(output, " \\\\ \\hline\\hline\n");

  for(const auto& entry : table.backgrounds) {
    printYieldRow(output, entry.name + (efficiencies ? " events" : ""), entry.stages);
    if(efficiencies) printEfficiencyRow(output, entry.name, entry.stages);
  }
  if(!table.totalBackground.empty()) {
    fprintf(output, "\\hline\n");
    printYieldRow(output, "Total background", table.totalBackground);
    if(efficiencies) printEfficiencyRow(output, "Total background", table.totalBackground);
  }
  if(!table.data.empty()) {
    fprintf(output, "\\hline\n");
    printYieldRow(output, "Data", table.data);
    if(efficiencies) printEfficiencyRow(output, "Data", table.data);
  }
  for(const auto& entry : table.signals) {
    fprintf(output, "\\hline\n");
    printYieldRow(output, entry.name + (efficiencies ? " events" : ""), entry.stages);
    if(efficiencies) {
      printEfficiencyRow(output, entry.name, entry.stages);
      if(!table.totalBackground.empty())
        for(int mode = 0; mode < 3; ++mode)
          printSensitivityRow(output, entry.name, entry.stages, table.totalBackground, mode);
    }
  }
  fprintf(output, "\\hline\n\\end{tabular}}\n\\end{sidewaystable}\n");
  fclose(output);
  std::cout << "[table] Wrote " << path << '\n';
}

void ConvertToTex(JSONWrapper::Object& root, TFile* file, NameAndType& properties)
{
  if(properties.isIndexPlot && cutIndex < 0) return;
  writeYieldTable(collectYieldTable(root, file, properties.name, 1), properties.name, false);
  const HistogramLabels* labels = getManualLabels(properties.name);
  // QCD ABCD regions and multiplicity diagnostics are not sequential cutflows.
  if(labels && labels->sequential)
    writeYieldTable(collectYieldTable(root, file, properties.name, 2), properties.name, true);
}

int main(int argc, char* argv[]){
  gROOT->LoadMacro("../../src/tdrstyle.C");
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

  std::vector<string> histoNameMask;

  for(int i=1;i<argc;i++){
    string arg(argv[i]);

    if(arg.find("--help")!=string::npos){
      printf("--help   --> print this helping text\n");
      printf("--key     --> only samples including this keyword are considered\n");
      printf("--iLumi   --> integrated luminosity to be used for the MC rescale\n");
      printf("--iEcm    --> center of mass energy in TeV\n");
      printf("--isSim   --> print CMS Simulation instead of the standard title\n");
      printf("--inDir   --> path to the directory containing the .root files to process\n");
      printf("--outDir  --> path of the directory that will contains the output plots and tables\n");
      printf("--outFile --> path of the output summary .root file\n");
      printf("--json    --> containing list of process (and associated style) to process to process\n");
      printf("--only    --> processing only the objects matching this regex expression\n");
      printf("--index   --> will do the projection on that index for histos of type cutIndex\n");
      printf("--chi2    --> show the data/MC chi^2\n");
      printf("--showUnc --> show stat uncertainty (if number is given use it as relative bin by bin uncertainty (e.g. lumi)\n");
      printf("--noLog   --> use linear scale\n");
      printf("--noInterpollation --> do not motph histograms for missing samples\n");
      printf("--doInterpollation --> do motph histograms for missing samples\n");
      printf("--no1D   --> Skip processing of 1D objects\n");
      printf("--no2D   --> Skip processing of 2D objects\n");
      printf("--noTree --> Skip processing of Tree objects\n");
      printf("--noTex  --> Do not create latex table (when possible)\n");
      printf("--noPowers --> Do not use powers of 10 for numbers in tables\n");
      printf("--noRoot --> Do not make a summary .root file\n");
      printf("--noPlot --> Do not creates plot files (useful to speedup processing)\n");
      printf("--plotExt --> extension to save, you can specify multiple extensions by repeating this option\n");
      printf("--splitCanvas --> (only for 2D plots) save all the samples in separated pltos\n");
      printf("--removeRatioPlot --> if you want to remove ratio plots between Data ad Mc\n");
      printf("--removeUnderFlow --> Remove the Underflow bin in the final plots\n");
      printf("--removeOverFlow --> Remove the Overflow bin in the final plots\n");

      printf("command line example: runPlotter --json ../data/beauty-samples.json --iLumi 2007 --inDir OUT/ --outDir OUT/plots/ --outFile plotter.root --noRoot --noPlot\n");
      return 0;
    }

    if(arg.find("--iLumi"  )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%lf",&iLumi); i++; printf("Lumi = %f\n", iLumi); }
    if(arg.find("--iEcm"   )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%lf",&iEcm); i++; printf("Ecm = %f TeV\n", iEcm); }
    if(arg.find("--signalScale")!=string::npos && i+1<argc){ sscanf(argv[i+1],"%lf",&signalScale); i++; printf("scale signal by %f\n", signalScale); }
    if(arg.find("--blind"  )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%lf",&blind); i++; printf("Blind above = %f\n", blind); }
    if(arg.find("--metxmax"  )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%lf",&metxmax); i++; printf("xMax for MET = %f\n", metxmax); }
    if(arg.find("--mtxmax"  )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%lf",&mtxmax); i++; printf("xMax for MT = %f\n", mtxmax); }

    if(arg.find("--rebin"  )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%i",&rebin); i++; printf("Rebin by %i\n",rebin); }
    if(arg.find("--inDir"  )!=string::npos && i+1<argc){ inDir    = argv[i+1];  i++;  printf("inDir = %s\n", inDir.c_str());  }
    if(arg.find("--outDir" )!=string::npos && i+1<argc){ outDir   = argv[i+1];  i++;  printf("outDir = %s\n", outDir.c_str());  }
    if(arg.find("--outFile")!=string::npos && i+1<argc){ outFile  = argv[i+1];  i++; printf("output file = %s\n", outFile.c_str()); }
    if(arg.find("--json"   )!=string::npos && i+1<argc){ jsonFile = argv[i+1];  i++;  }
    if(arg.find("--key"    )!=string::npos && i+1<argc){ keywords.push_back(argv[i+1]); printf("Only samples matching this (regex) expression '%s' are processed\n", argv[i+1]); i++;  }
    if(arg.find("--only"   )!=string::npos && i+1<argc){ histoNameMask.push_back(argv[i+1]); printf("Only histograms matching (regex) expression '%s' are processed\n", argv[i+1]); i++;  }
    if(arg.find("--index"  )!=string::npos && i+1<argc){ sscanf(argv[i+1],"%d",&cutIndex); i++; onlyCutIndex=(cutIndex>=0); printf("index = %i\n", cutIndex);  }
    if(arg.find("--chi2"  )!=string::npos)             { showChi2 = true;  }
    if(arg.find("--showUnc") != string::npos) {
      showUnc=true;
      if(i+1<argc) {
	string nextArg(argv[i+1]);
	if(nextArg.find("--")==string::npos)
	  {
	    sscanf(argv[i+1],"%lf",&baseRelUnc);
	    i++;
	  }
      }
      printf("Uncertainty band will be included for MC with base relative uncertainty of: %3.2f\n",baseRelUnc);
    }
    if(arg.find("--isSim")!=string::npos){ isSim = true;    }
    if(arg.find("--noLog")!=string::npos){ noLog = true;    }
    if(arg.find("--noTree"  )!=string::npos){ doTree = false;    }
    if(arg.find("--noInterpollation"  )!=string::npos){ doInterpollation = false; }
    if(arg.find("--doInterpollation"  )!=string::npos){ doInterpollation = true; }
    if(arg.find("--no2D"  )!=string::npos){ do2D = false;    }
    if(arg.find("--no1D"  )!=string::npos){ do1D = false;    }
    if(arg.find("--noTex" )!=string::npos){ doTex= false;    }
    if(arg.find("--noPowers" )!=string::npos){ doPowers= false;    }
    if(arg.find("--noRoot")!=string::npos){ StoreInFile = false;    }
    if(arg.find("--noPlot")!=string::npos){ doPlot = false;    }
    if(arg.find("--addExternalNorm")!=string::npos){ addExternalNorm = true;    }
    if(arg.find("--removeRatioPlot")!=string::npos){ showRatioBox = false; printf("No ratio plot between Data and Mc \n"); }
    if(arg.find("--removeUnderFlow")!=string::npos){ fixUnderflow = false; printf("No UnderFlowBin \n"); }
    if(arg.find("--removeOverFlow")!=string::npos){ fixOverflow = false; printf("No OverFlowBin \n"); }
    if(arg.find("--plotExt" )!=string::npos && i+1<argc){ plotExt.push_back(argv[i+1]);  i++;  printf("saving plots as = %s\n", argv[i]);  }
    if(arg.find("--splitCanvas")!=string::npos){ splitCanvas = true;    }
    if(arg.find("--fileOption" )!=string::npos && i+1<argc){ fileOption = argv[i+1];  i++;  printf("FileOption = %s\n", fileOption.c_str());  }
  }
  if(doPlot)system( (string("mkdir -p ") + outDir).c_str());
  if(plotExt.size() == 0)
    plotExt.push_back(".png");

  char buf[255];
  sprintf(buf, "_Index%d", cutIndex);
  cutIndexStr = buf;

  TFile* OutputFile = new TFile(outFile.c_str(),fileOption.c_str());

  JSONWrapper::Object Root(jsonFile, true);
  std::list<NameAndType> histlist;
  GetListOfObject(Root,inDir,histlist,"/",OutputFile);
  histlist.sort();
  histlist.unique();

  printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");

  std::list<NameAndType>::iterator it= histlist.begin();
  while(it!= histlist.end()){
    bool passMasking = (histoNameMask.size()==0);
    for(const auto& mask : histoNameMask) {
      if(std::regex_match(it->name, std::regex(mask))) { passMasking = true; break; }
    }
    if(!passMasking){it=histlist.erase(it); continue; }
    if(!do2D   &&(it->is2D() || it->is3D())){it=histlist.erase(it); continue;}
    if(!do1D   && it->is1D()){it=histlist.erase(it); continue;}
    if(!doTree && it->isTree()){it=histlist.erase(it); continue;}
    it++;
  }

  if(fileOption!="READ"){
    SavingToFile(Root,inDir,OutputFile, histlist);
    if(doInterpollation)InterpollateProcess(Root, OutputFile, histlist);
    SumBins(Root, OutputFile, histlist);
    MixProcess(Root, OutputFile, histlist);
    NRBProcess(Root, OutputFile, histlist);
  }

  int ictr =0;
  int TreeStep = std::max(1,(int)(histlist.size()/50));
  printf("Plotting                     :");
  for(std::list<NameAndType>::iterator it= histlist.begin(); it!= histlist.end(); it++,ictr++){
    if(ictr%TreeStep==0){printf(".");fflush(stdout);}

    if(doPlot && doTex && it->is1D() && hasCategoricalBins(it->name)
        && it->name.find("optim_eventflow")==std::string::npos)
      ConvertToTex(Root,OutputFile,*it);
    if(doPlot && do2D  && it->is2D()){                      if(!splitCanvas){Draw2DHistogram(Root,OutputFile,*it); }else{Draw2DHistogramSplitCanvas(Root,OutputFile,*it);}}
    if(doPlot && do1D  && it->is1D()){ Draw1DHistogram(Root,OutputFile,*it); }
  }printf("\n");
  OutputFile->Close();
}
