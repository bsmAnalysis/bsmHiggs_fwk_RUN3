#ifndef macroutils_h
#define macroutils_h


#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/Common/interface/MergeableCounter.h"


//Load here all the dataformat that we will need
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"



#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"




#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/PatCandidates/interface/GenericParticle.h"


//------------------ needed headers for slew rate corrections -------------------//

#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <vector>
#include "TVector3.h"
#include "TMath.h"
#include "TGraph.h"
#include <Math/VectorUtil.h>


typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > LorentzVector;


namespace utils
{
  namespace cmssw
  {
   
      }

    //
    template<class T>
    float getArcCos(T &a,T &b)
      {
	TVector3 mom1(a.px(),a.py(),a.pz());
	TVector3 mom2(b.px(),b.py(),b.pz());
	float cosine = mom1.Dot(mom2)/(mom1.Mag()*mom2.Mag());
	float arcCosine = acos(cosine);
	return arcCosine;
      }

    template<class T>
    float getMT(T &visible, T &invisible, bool setSameMass=false)
    {
      float mt(-1);
      if(setSameMass){
	T sum=visible+invisible;
	mt= TMath::Power(TMath::Sqrt(TMath::Power(visible.pt(),2)+pow(visible.mass(),2))+TMath::Sqrt(TMath::Power(invisible.pt(),2)+pow(visible.mass(),2)),2);
	mt-=TMath::Power(sum.pt(),2);
	mt=TMath::Sqrt(mt);
      }else{
	double dphi=fabs((double)deltaPhi((double)invisible.phi(),(double)visible.phi()));
	mt=TMath::Sqrt(2*invisible.pt()*visible.pt()*(1-TMath::Cos(dphi)));
      }
      return mt;
    }

    template<class T1, class T2>
    float getMT(T1 &visible, T2 &invisible, bool setSameMass=false)
    {
      float mt(-1);
      if(setSameMass){
	LorentzVector sum= LorentzVector(visible)+LorentzVector(invisible);
	mt= TMath::Power(TMath::Sqrt(TMath::Power(visible.pt(),2)+pow(visible.mass(),2))+TMath::Sqrt(TMath::Power(invisible.pt(),2)+pow(visible.mass(),2)),2);
	mt-=TMath::Power(sum.pt(),2);
	mt=TMath::Sqrt(mt);
      }else{
	double dphi=fabs(deltaPhi((double)invisible.phi(),(double)visible.phi()));
	mt=TMath::Sqrt(2*invisible.pt()*visible.pt()*(1-TMath::Cos(dphi)));
      }
      return mt;
    }
 
  //round up and show in TeX
  std::string toLatexRounded(double value, double error, double systError=-1,bool doPowers=true, double systErrorDown = -1);

  //clean up ROOT version of TeX
  void TLatexToTex(TString &expr);

  inline float deltaPhi(float phi1, float phi2) { 
     float result = phi1 - phi2;
     while (result > float(M_PI)) result -= float(2*M_PI);
     while (result <= -float(M_PI)) result += float(2*M_PI);
     return result;
  }

// FWLITE CODE

  // loop on all the lumi blocks for an EDM file in order to count the number of events that are in a sample
  // this is useful to determine how to normalize the events (compute weight)
  double getTotalNumberOfEvents(std::vector<std::string>& urls, bool fast=false, bool weightSum=false);

  unsigned long getMergeableCounterValue(const std::vector<std::string>& urls, std::string counter);
}
class DuplicatesChecker{
 private :
  typedef std::unordered_map<std::string, bool > RunEventHashMap;
  RunEventHashMap map;
 public :
  DuplicatesChecker(){}
  ~DuplicatesChecker(){}
  void Clear(){map.clear();}
  bool isDuplicate(unsigned int Run, unsigned int Lumi, unsigned int Event){
    char tmp[255];sprintf(tmp,"%i_%i_%i",Run,Lumi,Event);
    RunEventHashMap::iterator it = map.find(tmp);
    if(it==map.end()){
      map[tmp] = true;
      return false;
    }
    return true;
  }
  bool isDuplicate(unsigned int Run, unsigned int Lumi, unsigned int Event,unsigned int cat){
    char tmp[255];sprintf(tmp,"%i_%i_%i_%i",Run,Lumi,Event,cat);
    RunEventHashMap::iterator it = map.find(tmp);
    if(it==map.end()){
      map[tmp] = true;
      return false;
    }
    return true;
  }
};


#endif
