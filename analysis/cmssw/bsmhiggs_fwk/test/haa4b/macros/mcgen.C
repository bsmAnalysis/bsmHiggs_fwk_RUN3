#include "TLorentzVector.h"
#include "TVector3.h"
#include "TRandom.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"

#include <vector>
#include <algorithm>

using namespace std;

Double_t HiggsMass = 125.;

Double_t Alpha1Mass = 50.;
Double_t Alpha2Mass = 50.; // a2 -> b1b2

Double_t b2Mass = 5;
Double_t b1Mass = 5.;
Double_t b3Mass = 5;
Double_t b4Mass = 5.;

Double_t r = b2Mass/Alpha2Mass;


namespace RandomNumberGenerator {

    TRandom Number1_th(34523);
    TRandom Number2_th(23534);
    TRandom Number3_th(64364);

    TRandom Number1_phi(46346);
    TRandom Number2_phi(34636);
    TRandom Number3_phi(34534);

};

namespace partInfo {

  TLorentzVector Alpha1LVector;
  TLorentzVector b1LVector;
  //TLorentzVector Lepton2LVector;
  // a2->b1b2
  TLorentzVector Alpha2Vector;
  TLorentzVector b2Vector;
  // a1->b3b4
  TLorentzVector b3Vector;
  TLorentzVector b4Vector;

  TLorentzVector Alpha2Vector_rest;
  TLorentzVector b1Vector_rest;
  
};

int SavePart(int thePar, TLorentzVector &MyLVector) {
 
    using namespace partInfo;

    // particles in lab frame   
  if (thePar == 0) {
    Alpha1LVector = MyLVector;
  }
  if (thePar == 1) {
    b1LVector = MyLVector;
  }
  if (thePar == 2) {
    b3Vector = MyLVector;
  }
  if (thePar == 3) {
    Alpha2Vector = MyLVector;
  }
  if (thePar == 4) {
    b2Vector = MyLVector;
  }
  if (thePar == 5) {
    b4Vector = MyLVector;
  }
  // particles in rest frame of parent
  if (thePar == 6) {
    Alpha2Vector_rest = MyLVector;
  }
  if (thePar == 7) {
    b1Vector_rest = MyLVector;
  }
  
  return 1;
};

bool ptsort(const TLorentzVector & x, const TLorentzVector & y) 
{ 
  return  (x.Pt() > y.Pt() ) ; 
};

double deltaPhi(double phi1, double phi2) {

  if (phi1<0) phi1+=2.*TMath::Pi();
  if (phi2<0) phi2+=2.*TMath::Pi();

  double result = phi1 - phi2;
  if(fabs(result) > 9999) return result;
  while (result > TMath::Pi()) result -= 2*TMath::Pi();
  while (result <= -TMath::Pi()) result += 2*TMath::Pi();
  return result;
}

double deltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = deltaPhi(phi1, phi2);
  return sqrt(deta*deta + dphi*dphi);
}


void Rotatez(TLorentzVector &pvect, TLorentzVector &pvect1){
// Make a rotation to the parent coordinate system of pvect
  double pvsave[3];
  double gamma, betgam;
  double pxy, pmod, axx, axy, axz, ayx, ayy, ayz, azx, azy, azz;

// Perform a rotation assuming that the original z-axis is given by pvect

  double mass = pvect.M();
  double mass1 = pvect1.M();

  pxy = pvect.Px()*pvect.Px() + pvect.Py()*pvect.Py();
  if (pxy > 0.) {
   pmod = pxy + pvect.Pz()*pvect.Pz();
   pxy = sqrt(pxy);
   pmod = sqrt(pmod);
   axx = - pvect.Py() /pxy;
   axy = - pvect.Pz() / pmod * pvect.Px() / pxy;
   axz =   pvect.Px() / pmod;
   ayx =   pvect.Px() / pxy;
   ayy = - pvect.Pz() / pmod * pvect.Py() / pxy;
   ayz =   pvect.Py() / pmod;
   azx =   0.;
   azy =   pxy / pmod;
   azz =   pvect.Pz() / pmod;
   pvsave[0] = pvect1.Px();
   pvsave[1] = pvect1.Py();
   pvsave[2] = pvect1.Pz();

   pvect1.SetPx(axx*pvsave[0] + axy*pvsave[1] + axz*pvsave[2]);
   pvect1.SetPy(ayx*pvsave[0] + ayy*pvsave[1] + ayz*pvsave[2]);
   pvect1.SetPz(azy*pvsave[1] + azz*pvsave[2]);

   pvect1.SetE( sqrt(pvect1.Px()*pvect1.Px() + pvect1.Py()*pvect1.Py()
                 + pvect1.Pz()*pvect1.Pz() + mass1*mass1) );
  }
}


void DecayTwoBody(Double_t &SMMass, Double_t &SusyMass, TLorentzVector &ParentLabLVector, TLorentzVector &ParentParLVector, int RunNumber)
{
    using namespace RandomNumberGenerator;
   
//    TRandom Number;
    
    TLorentzVector SusyLVector;
    TLorentzVector SMLVector;

    Double_t ParentMass = ParentLabLVector.M();

// Compute the energies and momentum in parent rest frame

    Double_t ErestSM = ( (ParentMass*ParentMass) + (SMMass*SMMass) - (SusyMass*SusyMass))/(2.*ParentMass);
    Double_t ErestSusy = ( (ParentMass*ParentMass) - (SMMass*SMMass) + (SusyMass*SusyMass))/(2.*ParentMass);
    Double_t Prest = sqrt( (ErestSusy*ErestSusy) - (SusyMass*SusyMass) );

    // Double_t Prest = (ParentMass*ParentMass - SusyMass*SusyMass) / (2.*ParentMass);

    Double_t costh, sinth;
    Double_t phi, cosphi, sinphi;

    if (RunNumber == 1) {
	costh = 2.*(Number1_th.Uniform(0,1) - 0.5);
	sinth = sqrt(1 - costh*costh);

	phi = 2.*TMath::Pi()*Number1_phi.Uniform(0,1);
	cosphi = TMath::Cos(phi);
	sinphi = sqrt(1-cosphi*cosphi);
    } else if (RunNumber == 2) {
	costh = 2.*(Number2_th.Uniform(0,1) - 0.5);
	sinth = sqrt(1 - costh*costh);

	phi = 2.*TMath::Pi()*Number2_phi.Uniform(0,1);
	cosphi = TMath::Cos(phi);
	sinphi = sqrt(1-cosphi*cosphi);
    } else if (RunNumber == 3) {
	costh = 2.*(Number3_th.Uniform(0,1) - 0.5);
	sinth = sqrt(1 - costh*costh);

	phi = 2.*TMath::Pi()*Number3_phi.Uniform(0,1);
	cosphi = TMath::Cos(phi);
	sinphi = sqrt(1-cosphi*cosphi);
    }


// Compute the momentum 4-vectors in parent rest frame

    Double_t SMArray[4], SusyArray[4];

    SMArray[0] = -Prest*sinth*cosphi;
    SMArray[1] = -Prest*sinth*sinphi;
    SMArray[2] = -Prest*costh;

    SMArray[3] = sqrt( SMArray[0]*SMArray[0] + SMArray[1]*SMArray[1] + SMArray[2]*SMArray[2] + SMMass*SMMass);

    SusyArray[0] = Prest*sinth*cosphi;
    SusyArray[1] = Prest*sinth*sinphi;
    SusyArray[2] = Prest*costh;

    SusyArray[3] = sqrt( SusyArray[0]*SusyArray[0] + SusyArray[1]*SusyArray[1] + SusyArray[2]*SusyArray[2] + SusyMass*SusyMass);

    SMLVector.SetPxPyPzE(SMArray[0], SMArray[1], SMArray[2], SMArray[3]);
    SusyLVector.SetPxPyPzE(SusyArray[0], SusyArray[1], SusyArray[2], SusyArray[3]);

// Lorentz transform to the lab frame:
//  first rotate to the same coordinate frame as the parent
//  then Lorentz transform

    if (RunNumber == 1) {

      TVector3 b1 = ParentLabLVector.BoostVector();
      //	b1 = -b1;
      
      Rotatez(ParentParLVector, SusyLVector);
      Rotatez(ParentParLVector, SMLVector);


      // Save costheta* opening angle in the parent rest frame
      SavePart(6,SusyLVector);
      
      TLorentzVector oldSusyLVector;
      oldSusyLVector = SusyLVector;	

      TLorentzVector oldSMLVector;
      oldSMLVector = SMLVector;
      
      // Boost decay products to lab frame...
      SMLVector.Boost(b1);
      SusyLVector.Boost(b1);
      
// We are here in the cms of the Higgs
      // Save Alpha1 4-Vector
      SavePart(0,SMLVector);
      SavePart(3,SusyLVector);
      
      DecayTwoBody(b1Mass, b2Mass, SusyLVector, oldSusyLVector, 2); // a2, a2_beforeboost
      DecayTwoBody(b3Mass, b4Mass, SMLVector, oldSMLVector, 3); // with a1, a1_beforeboost
      
    } else if (RunNumber == 2) {

      // We are here in the cms of the Alpha2
      // Going back to Higgs frame, we use the Alpha2 boost Vector

      /*
SusyLVector = slepton vector (in Alpha2 frame)
ParentLabVector = Alpha2 vector (in Higgs rest frame)
BoostLVector = Alpha2 vector (in Higgs rest frame)
*/

// Boost fermion1 to Higgs frame
      
      TVector3 b2 = ParentLabLVector.BoostVector();
      //	b2 = -b2;
      
      Rotatez(ParentParLVector, SusyLVector);
      Rotatez(ParentParLVector, SMLVector);

      SavePart(7,SusyLVector);
      
      TLorentzVector oldSusyLVector;
      oldSusyLVector = SusyLVector;	
      
      SMLVector.Boost(b2);
      SusyLVector.Boost(b2);
	
      // Boost alpha2 to its rest frame
      
      TLorentzVector Alpha2Vector(0.,0.,0.,Alpha2Mass);
      Alpha2Vector.Boost(b2);	
  
      // Save b1 4-Vector
      SavePart(1,SMLVector);
      SavePart(4,SusyLVector);
      
      // DecayTwoBody(plots, Lepton2Mass, LSPMass, SusyLVector, oldSusyLVector, 3);
    } else if (RunNumber == 3) {

      // We are here in the cms of the Alpha2
      // Going back to Higgs frame, we use the Alpha2 boost Vector

      /*
SusyLVector = slepton vector (in Alpha2 frame)
ParentLabVector = Alpha2 vector (in Higgs rest frame)
BoostLVector = Alpha2 vector (in Higgs rest frame)
*/

// Boost fermion1 to Higgs frame
      
      TVector3 b3 = ParentLabLVector.BoostVector();
      //	b2 = -b2;
      
      Rotatez(ParentParLVector, SusyLVector);
      Rotatez(ParentParLVector, SMLVector);
      
      TLorentzVector oldSusyLVector;
      oldSusyLVector = SusyLVector;	
      
      SMLVector.Boost(b3);
      SusyLVector.Boost(b3);
	
      // Boost alpha2 to its rest frame
      
      TLorentzVector Alpha1Vector(0.,0.,0.,Alpha1Mass);
      Alpha1Vector.Boost(b3);	
  
      // Save b1 4-Vector
      SavePart(2,SMLVector);
      SavePart(5,SusyLVector);
      
      // DecayTwoBody(plots, Lepton2Mass, LSPMass, SusyLVector, oldSusyLVector, 3);
    }
}

    
void mcgen() {

    using namespace partInfo;

// Open ntuple_data.root to get the 4-vector of the Higgs, for a given production mode, in the lab frame...

    TFile *f = new TFile("MC13TeV_Wh_amass50_0.root");

    TH1F *PT_D = (TH1F*)f->Get("raw_higgsPt;1");
    TH1F *ETA_D = (TH1F*)f->Get("raw_higgsEta;1");

// Generate events
    const Int_t NEVT = 500000;

    vector<TLorentzVector> b_partons;
    TLorentzVector HiggsLabVector, HiggsLVector;

    // Create a ROOT tree to fill variables for Unbinned DATA ...
    
    TFile *f1 = new TFile("analyze_kins.root","RECREATE");
    TTree *t = new TTree("t","A tree with the unbinned data ");

    Double_t hMass, hPt, hEta, hPhi;

    Double_t a1Mass, a1Pt, a1Eta, a1Phi;
    Double_t a2Mass, a2Pt, a2Eta, a2Phi;

    Double_t b1M, b1Pt, b1Eta, b1Phi;
    Double_t b2M, b2Pt, b2Eta, b2Phi;

    Double_t b3M, b3Pt, b3Eta, b3Phi;
    Double_t b4M, b4Pt, b4Eta, b4Phi;

    Double_t j1Pt, j2Pt, j3Pt, j4Pt;
    Double_t j1Eta, j2Eta, j3Eta, j4Eta;
    
    Double_t minbPt, maxbPt;
    Double_t mindRbb, maxdRbb;
    
    Double_t mbb1, mbb2, m4b;
    Double_t mbbmax, mbbmin;

    Double_t dRaa, dRbb1, dRbb2;

    Double_t thetastar_a2, thetastar_b1;
    Double_t theta_a2, theta_b1;
    
    t->Branch("hMass",&hMass,"hMass/D");
    t->Branch("hPt",&hPt,"hPt/D");
    t->Branch("hEta",&hEta,"hEta/D");
    t->Branch("hPhi",&hPhi,"hPhi/D");

    t->Branch("a1Mass",&a1Mass,"a1Mass/D");
    t->Branch("a1Pt",&a1Pt,"a1Pt/D");
    t->Branch("a1Eta",&a1Eta,"a1Eta/D");
    t->Branch("a1Phi",&a1Phi,"a1Phi/D");

    t->Branch("a2Mass",&a2Mass,"a2Mass/D");
    t->Branch("a2Pt",&a2Pt,"a2Pt/D");
    t->Branch("a2Eta",&a2Eta,"a2Eta/D");
    t->Branch("a2Phi",&a2Phi,"a2Phi/D");

    t->Branch("b1M",&b1M,"b1M/D");
    t->Branch("b1Pt",&b1Pt,"b1Pt/D");
    t->Branch("b1Eta",&b1Eta,"b1Eta/D");
    t->Branch("b1Phi",&b1Phi,"b1Phi/D");

    t->Branch("b2M",&b2M,"b2M/D");
    t->Branch("b2Pt",&b2Pt,"b2Pt/D");
    t->Branch("b2Eta",&b2Eta,"b2Eta/D");
    t->Branch("b2Phi",&b2Phi,"b2Phi/D");

    t->Branch("b3M",&b3M,"b3M/D");
    t->Branch("b3Pt",&b3Pt,"b3Pt/D");
    t->Branch("b3Eta",&b3Eta,"b3Eta/D");
    t->Branch("b3Phi",&b3Phi,"b3Phi/D");

    t->Branch("b4M",&b4M,"b4M/D");
    t->Branch("b4Pt",&b4Pt,"b4Pt/D");
    t->Branch("b4Eta",&b4Eta,"b4Eta/D");
    t->Branch("b4Phi",&b4Phi,"b4Phi/D");

    // Here start analysis level variables
    t->Branch("j1Pt",&j1Pt,"j1Pt/D");
    t->Branch("j2Pt",&j2Pt,"j2Pt/D");
    t->Branch("j3Pt",&j3Pt,"j3Pt/D");
    t->Branch("j4Pt",&j4Pt,"j4Pt/D");

    t->Branch("j1Eta",&j1Eta,"j1Eta/D");
    t->Branch("j2Eta",&j2Eta,"j2Eta/D");
    t->Branch("j3Eta",&j3Eta,"j3Eta/D");
    t->Branch("j4Eta",&j4Eta,"j4Eta/D");
    
    t->Branch("minbPt",&minbPt,"minbPt/D");
    t->Branch("maxbPt",&maxbPt,"maxbPt/D");
    t->Branch("mindRbb",&mindRbb,"mindRbb/D");
    t->Branch("maxdRbb",&maxdRbb,"maxdRbb/D");
    
    t->Branch("mbb1",&mbb1,"mbb1/D");
    t->Branch("mbb2",&mbb2,"mbb2/D");
    t->Branch("m4b",&m4b,"m4b/D");
    t->Branch("mbbmax",&mbbmax,"mbbmax/D");
    t->Branch("mbbmin",&mbbmin,"mbbmin/D");

    t->Branch("dRbb1",&dRbb1,"dRbb1/D");
    t->Branch("dRbb2",&dRbb2,"dRbb2/D");
    t->Branch("dRaa",&dRaa,"dRaa/D");

    t->Branch("theta_a2",&theta_a2,"theta_a2/D");
    t->Branch("thetastar_a2",&thetastar_a2,"thetastar_a2/D");
    t->Branch("theta_b1",&theta_b1,"theta_b1/D");
    t->Branch("thetastar_b1",&thetastar_b1,"thetastar_b1/D");
    
    TRandom Number;
    for (Int_t i=0; i<NEVT; i++) {

       // Higgs 4-momentum in the lab frame
      Double_t ptDerived = PT_D->GetRandom();
      Double_t etaDerived = ETA_D->GetRandom();
      Double_t phiDerived = 2.*TMath::Pi()*Number.Uniform(0,1);
      
      HiggsLabVector.SetPtEtaPhiM(ptDerived, etaDerived, phiDerived, HiggsMass);
      HiggsLVector.SetPxPyPzE(0.,0.,0.,HiggsMass);
      
      // Generate decays
      DecayTwoBody(Alpha1Mass, Alpha2Mass, HiggsLabVector, HiggsLVector, 1);
    
      // Fill the tree with unbinned data
      hMass = HiggsMass;
      hPt = HiggsLabVector.Pt();
      hEta = HiggsLabVector.Eta();
      hPhi = HiggsLabVector.Phi();

      a1Mass = Alpha1Mass;
      a1Pt = Alpha1LVector.Pt();
      a1Eta = Alpha1LVector.Eta();
      a1Phi = Alpha1LVector.Phi();

      a2Mass = Alpha2Mass;
      a2Pt = Alpha2Vector.Pt();
      a2Eta = Alpha2Vector.Eta();
      a2Phi = Alpha2Vector.Phi();

      b1M = b1Mass;
      b1Pt = b1LVector.Pt();
      b1Eta = b1LVector.Eta();
      b1Phi = b1LVector.Phi();

      b2M = b2Mass;
      b2Pt = b2Vector.Pt();
      b2Eta = b2Vector.Eta();
      b2Phi = b2Vector.Phi();

      b3M = b3Mass;
      b3Pt = b3Vector.Pt();
      b3Eta = b3Vector.Eta();
      b3Phi = b3Vector.Phi();
      
      b4M = b4Mass;
      b4Pt = b4Vector.Pt();
      b4Eta = b4Vector.Eta();
      b4Phi = b4Vector.Phi();

      minbPt = std::min(b1Pt, std::min(b2Pt, std::min(b3Pt,b4Pt)));
      maxbPt = std::max(b1Pt, std::max(b2Pt, std::max(b3Pt,b4Pt)));

      // Sorting a vector of TLorentzVector
      b_partons.clear();
      b_partons.push_back(b1LVector);
      b_partons.push_back(b2Vector);
      b_partons.push_back(b3Vector);
      b_partons.push_back(b4Vector);

       //sort gen b's in pt
      std::sort(b_partons.begin(), b_partons.end(), ptsort);

      //      std::sort(b_partons.begin(), b_partons.end(), reorder);
      j1Pt=b_partons[0].Pt(); j1Eta=b_partons[0].Eta();
      j2Pt=b_partons[1].Pt(); j2Eta=b_partons[1].Eta();
      j3Pt=b_partons[2].Pt(); j3Eta=b_partons[2].Eta();
      j4Pt=b_partons[3].Pt(); j4Eta=b_partons[3].Eta();
      
      mbb1 = (b1LVector+b2Vector).M();
      mbb2 = (b3Vector+b4Vector).M();
      m4b = (b1LVector+b2Vector+b3Vector+b4Vector).M();

      dRaa = deltaR(Alpha1LVector.Eta(),Alpha1LVector.Phi(),Alpha2Vector.Eta(),Alpha2Vector.Phi());
      dRbb1 = deltaR(b1LVector.Eta(),b1LVector.Phi(),b2Vector.Eta(),b2Vector.Phi());
      dRbb2 = deltaR(b3Vector.Eta(),b3Vector.Phi(),b4Vector.Eta(),b4Vector.Phi());

      mbbmax = TMath::Max(mbb1,mbb2);
      mbbmin = TMath::Min(mbb1,mbb2);

      thetastar_a2 = Alpha2Vector_rest.Theta();
      thetastar_b1 = b1Vector_rest.Theta();

      theta_a2 = Alpha2Vector.Theta();
      theta_b1 = b1LVector.Theta();
      
      t->Fill();
    
    } // end NEVT loop (Event loop)
 
      // Save the tree with unbinned data
    t->AutoSave();
    
    f1->Write();
    f1->Close();

}
