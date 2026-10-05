
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "TSystem.h"
#include "TLorentzVector.h"
#include "TFile.h"

char pname[100] ;

const char* mcname( int pdgid ) ;

bool isAncestor( const reco::Candidate* ancestor, const reco::Candidate* daughter ) ;

double drMatchPGP( double eta, double phi, pat::PackedGenParticleCollection& packed_gen, int& match_ipgp ) ;

//---------------------------

   int main(int argc, char ** argv){

      gSystem->Load( "libFWCoreFWLite" );
      FWLiteEnabler::enable();



      optutl::CommandLineParser parser ("Analyze FWLite Histograms");

      parser.integerValue ("maxEvents"  ) = 10;

      parser.parseArguments (argc, argv);

      std::vector<std::string> inputFiles_ = parser.stringVector("inputFiles");

      int maxEvents_ = parser.integerValue("maxEvents");

      int ievt(0) ;


      for(unsigned int iFile=0; iFile<inputFiles_.size(); ++iFile){

         printf("  Opening file %2d : %s\n", iFile, inputFiles_[iFile].c_str() ) ;
         TFile* inFile = TFile::Open(inputFiles_[iFile].c_str());

         if( inFile ){

            fwlite::Event event(inFile);

            for(event.toBegin(); !event.atEnd(); ++event){

               printf("\n\n ==== Run = %u , lumi %u , event %llu\n", event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(), event.eventAuxiliary().event() ) ;

               reco::VertexCollection vtx;
               fwlite::Handle< reco::VertexCollection > vtxHandle;
               vtxHandle.getByLabel(event, "offlineSlimmedPrimaryVertices");
               if(vtxHandle.isValid()){ vtx = *vtxHandle;}
               if ( vtx.empty() ) { printf("\n\n *** No reco Primary Vertex.\n\n") ; continue ; }

               const reco::Vertex &PV = vtx.front();
               printf("\n  Primary Vertex : x,y,z = %9.5f, %9.5f, %9.5f\n\n", PV.x(), PV.y(), PV.z() ) ;
       


               pat::JetCollection jets;
               fwlite::Handle< pat::JetCollection > jetsHandle;
               jetsHandle.getByLabel(event, "slimmedJets");
               if(jetsHandle.isValid()){ jets = *jetsHandle;} else { printf("\n\n *** bad slimmedJets handle.\n\n") ; gSystem->Exit(-1) ; }

       

               reco::GenParticleCollection gen;
               fwlite::Handle< reco::GenParticleCollection > genHandle;
               genHandle.getByLabel(event, "prunedGenParticles");
               if(genHandle.isValid()){ gen = *genHandle;} else { printf("\n\n *** bad handle for reco::GenParticleCollection\n\n") ; gSystem->Exit(-1) ; }

               std::vector<reco::GenParticle*> b_hadrons ;

           // vx, vy, vz seem to give the production vertex, not decay vertex
           // dxy and dz are not members of GenParticle class.  Only PackedGenParticle

               printf(" +++ prunedGenParticles\n" ) ;
               for (unsigned int igen=0; igen<gen.size(); igen++) {
                  char name[100] ;
                  sprintf( name, "%s", mcname( gen[igen].pdgId() ) ) ;
                  printf(" %3d %p : ID=%6d, %10s : m=%6.2f : pt=%6.1f, eta=%7.3f, phi=%7.3f : status = %2d , Nmom = %2lu , Ndau = %2lu : Vxyz = %9.5f, %9.5f, %9.5f : %s %s ",
                      igen,
                      &(gen[igen]),
                      gen[igen].pdgId(),
                      name,
                      gen[igen].mass(),
                      gen[igen].pt(),
                      ( gen[igen].pt() == 0 ? 99. : gen[igen].eta() ),
                      gen[igen].phi(),
                      gen[igen].status(),
                      gen[igen].numberOfMothers(),
                      gen[igen].numberOfDaughters(),
                      gen[igen].vx(),
                      gen[igen].vy(),
                      gen[igen].vz(),
                      (gen[igen].isLastCopy() ? "final" : "intermediate" ),
                      (gen[igen].statusFlags().isDirectHadronDecayProduct() ? "direct-had-dec-prod" : " " )
                   ) ;

                   int pdgId = gen[igen].pdgId() ;
                   if (  abs( pdgId ) == 511
                      || abs( pdgId ) == 521
                      || abs( pdgId ) == 531
                      || abs( pdgId ) == 541
                      || abs( pdgId ) == 5122
                      || abs( pdgId ) == 5112
                      || abs( pdgId ) == 5222
                      || abs( pdgId ) == 5132
                      || abs( pdgId ) == 5232
                      || abs( pdgId ) == 5332 ) {
                      printf( " ground state b hadron. " ) ;
                      b_hadrons.emplace_back( &(gen[igen]) ) ;
                   }

                   printf("\n" ) ;

               } // igen





               printf("\n\n +++ Ground state b hadrons\n" ) ;
               for ( int bi=0; bi<b_hadrons.size(); bi++ ) {
                  char name[100] ;
                  sprintf( name, "%s", mcname( b_hadrons[bi]->pdgId() ) ) ;
                  printf( " %2d %p : ID=%6d, %10s : m=%6.2f : pt=%6.1f, eta=%7.3f, phi=%7.3f\n",
                    bi,
                    b_hadrons[bi],
                    b_hadrons[bi]->pdgId(),
                    name,
                    b_hadrons[bi]->mass(),
                    b_hadrons[bi]->pt(),
                    b_hadrons[bi]->eta(),
                    b_hadrons[bi]->phi()
                    ) ;
               } // bi







               pat::PackedGenParticleCollection packed_gen;
               fwlite::Handle< pat::PackedGenParticleCollection > packed_genHandle;
               packed_genHandle.getByLabel(event, "packedGenParticles");
               if(packed_genHandle.isValid()){ packed_gen = *packed_genHandle; } else { printf("\n\n *** bad handle for pat::PackedGenParticleCollection\n\n") ; gSystem->Exit(-1) ; }

               printf("\n\n +++ packedGenParticles\n" ) ;

        // seems like dxy and dz always return zero.
        //      also vx, vy, vz always zero.
        //      If mother vertex is the production vertex of the mother, how is it possible
        //      to get the decay vertex position of the mother, which is where these pions and
        //      kaons come from???

               for ( unsigned int ipgp=0; ipgp<packed_gen.size(); ipgp++ ) {
                  char name[100] ;
                  sprintf( name, "%s", mcname( packed_gen[ipgp].pdgId() ) ) ;
                  printf(" %3d : ID=%6d, %10s : m=%6.2f : pt=%6.1f, eta=%7.3f, phi=%7.3f ",
                      ipgp,
                      packed_gen[ipgp].pdgId(),
                      name,
                      packed_gen[ipgp].mass(),
                      packed_gen[ipgp].pt(),
                      ( packed_gen[ipgp].pt() == 0 ? 99. : packed_gen[ipgp].eta() ),
                      packed_gen[ipgp].phi()
                   ) ;
            ////  printf(" : dxy=%9.5f, dz=%9.5f, Vxyz = %9.5f, %9.5f, %9.5f ",
            ////      packed_gen[ipgp].dxy(),
            ////      packed_gen[ipgp].dz(),
            ////      packed_gen[ipgp].vx(),
            ////      packed_gen[ipgp].vy(),
            ////      packed_gen[ipgp].vz()
            ////   ) ;
                  const reco::Candidate* mom = packed_gen[ipgp].mother(0) ;
                  if ( mom != 0x0 ) {
                     char momname[100] ;
                     sprintf( momname, "%s", mcname( mom->pdgId() ) ) ;
                     printf(" : mother ID=%6d, %10s : pt=%6.1f, eta=%7.3f, phi=%7.3f  ptr=%p",
                        mom->pdgId(),
                        momname,
                        mom->pt(),
                        ( mom->pt() == 0 ? 99 : mom->eta()),
                        mom->phi(),
                        mom) ;
                  } else {
                     printf(" : no mother" ) ;
                  }

                  for ( int bi=0; bi<b_hadrons.size(); bi++ ) {
                     const reco::Candidate* bhad = b_hadrons[bi] ;
                     const reco::Candidate* dau  = &(packed_gen[ipgp]) ;
                     if ( isAncestor( bhad, dau ) ) {
                        printf( " : from b hadron %d", bi ) ;
                     }
                  } // bi

                  printf("\n") ;

               } // ipgp






               printf("\n\n +++ Charged daughters from ground state b hadrons.\n" ) ;
               for ( int bi=0; bi<b_hadrons.size(); bi++ ) {
                  char name[100] ;
                  sprintf( name, "%s", mcname( b_hadrons[bi]->pdgId() ) ) ;
                  printf( " %2d %p : ID=%6d, %10s : m=%6.2f : pt=%6.1f, eta=%7.3f, phi=%7.3f\n",
                    bi,
                    b_hadrons[bi],
                    b_hadrons[bi]->pdgId(),
                    name,
                    b_hadrons[bi]->mass(),
                    b_hadrons[bi]->pt(),
                    b_hadrons[bi]->eta(),
                    b_hadrons[bi]->phi()
                    ) ;
                  const reco::Candidate* bhad = b_hadrons[bi] ;
                  for ( unsigned int ipgp=0; ipgp<packed_gen.size(); ipgp++ ) {
                     if ( packed_gen[ipgp].charge() == 0 ) continue ;
                     const reco::Candidate* dau  = &(packed_gen[ipgp]) ;
                     if ( isAncestor( bhad, dau ) ) {
                        char name[100] ;
                        sprintf( name, "%s", mcname( packed_gen[ipgp].pdgId() ) ) ;
                        char momname[100] ;
                        sprintf( momname, " " ) ;
                        const reco::Candidate* mom = packed_gen[ipgp].mother(0) ;
                        if ( mom != 0x0 ) {
                           sprintf( momname, "%s", mcname( mom->pdgId() ) ) ;
                        }
                        printf(" %3d : ID=%6d, %10s : m=%6.2f : pt=%6.1f, eta=%7.3f, phi=%7.3f : mother %6s",
                            ipgp,
                            packed_gen[ipgp].pdgId(),
                            name,
                            packed_gen[ipgp].mass(),
                            packed_gen[ipgp].pt(),
                            ( packed_gen[ipgp].pt() == 0 ? 99. : packed_gen[ipgp].eta() ),
                            packed_gen[ipgp].phi(),
                            momname
                         ) ;
                        printf("\n") ;
                     }
                  } // ipgp
                  printf("\n\n" ) ;
               } // bi









               reco::VertexCompositePtrCandidateCollection sec_vert ;
               fwlite::Handle< reco::VertexCompositePtrCandidateCollection > svHandle ;
               svHandle.getByLabel( event, "slimmedSecondaryVertices" ) ;
               if ( svHandle.isValid() ) { sec_vert = *svHandle ; } else { printf("\n\n *** bad handle for reco::VertexCompositePtrCandidateCollection\n\n") ; gSystem -> Exit(-1) ; }

               printf("\n\n +++ slimmedSecondaryVertices\n" ) ;
               for ( unsigned int isv=0; isv<sec_vert.size(); isv++ ) {
                  printf(" %3d :   x,y,z = %9.5f, %9.5f, %9.5f :  Ntrk = %2lu : chi2 = %7.3f, Ndof = %5.2f\n",
                     isv,
                     sec_vert[isv].position().x(),
                     sec_vert[isv].position().y(),
                     sec_vert[isv].position().z(),
                     sec_vert[isv].numberOfDaughters(),
                     sec_vert[isv].vertexChi2(),
                     sec_vert[isv].vertexNdof()
                     ) ;
                  double sv_dxy = sqrt( pow((sec_vert[isv].position().x() - PV.x()),2) + pow((sec_vert[isv].position().y() - PV.y()),2) ) ;
                  double sv_dxyz = sqrt( pow((sec_vert[isv].position().x() - PV.x()),2) + pow((sec_vert[isv].position().y() - PV.y()),2) + pow((sec_vert[isv].position().z() - PV.z()),2) ) ;
                  printf("      dx,dy,dz = %9.5f, %9.5f, %9.5f :  dxy = %9.5f , dxyz = %9.5f\n",
                       sec_vert[isv].position().x() - PV.x(),
                       sec_vert[isv].position().y() - PV.y(),
                       sec_vert[isv].position().z() - PV.z(),
                       sv_dxy,
                       sv_dxyz
                     ) ;
                  GlobalVector sv_p3 ;
                  TLorentzVector sv_p4 ;
                  for ( unsigned int id=0; id<sec_vert[isv].numberOfDaughters(); id++ ) {
                     reco::CandidatePtr dau = sec_vert[isv].daughterPtr(id) ;
                     TLorentzVector svd_p4( dau->px(), dau->py(), dau->pz(), dau->energy() ) ;
                     GlobalVector svd_p3( dau->px(), dau->py(), dau->pz() ) ;
                     sv_p3 += svd_p3 ;
                     sv_p4 += svd_p4 ;
                     printf("      trk %2d :  pt=%6.1f, eta=%7.3f, phi = %7.3f",
                           id,
                           dau->pt(),
                           dau->eta(),
                           dau->phi()
                     ) ;
                     int match_ipgp(-1) ;
                     double drm = drMatchPGP( dau->eta(), dau->phi(), packed_gen, match_ipgp ) ;
                     printf(" : dr=%5.3f ipgp=%3d",
                           drm,
                           match_ipgp
                     ) ;
                     if ( match_ipgp >= 0 ) {
                        for ( int bi=0; bi<b_hadrons.size(); bi++ ) {
                           const reco::Candidate* bhad = b_hadrons[bi] ;
                           const reco::Candidate* dau  = &(packed_gen[match_ipgp]) ;
                           if ( isAncestor( bhad, dau ) ) {
                              printf(" daughter of B had %d (pt=%5.1f, eta=%7.3f, phi=%7.3f)",
                                bi,
                                bhad->pt(),
                                bhad->eta(),
                                bhad->phi()
                                ) ;
                           }
                        } // bi
                     }
                     printf("\n") ;
                  } // id

                  GlobalVector dxyz( sec_vert[isv].position().x() - PV.x(), sec_vert[isv].position().y() - PV.y(), sec_vert[isv].position().z() - PV.z() ) ;
                  double cos_pv_sv = -2. ;
                  if ( sv_p3.mag() * dxyz.mag() > 0 ) {
                     cos_pv_sv = sv_p3.dot( dxyz ) / ( sv_p3.mag() * dxyz.mag() ) ;
                  }
                  double sv_pt = sqrt( pow( sv_p3.x(), 2. ) + pow( sv_p3.y(), 2. ) ) ;
                  printf("  secondary vertex pt = %6.1f, mass = %6.2f\n", sv_pt, sv_p4.M() ) ;
                  printf("  cos(pv,sv) = %6.3f\n", cos_pv_sv ) ;

                  const reco::Vertex sv( sec_vert[isv].position(), sec_vert[isv].error() ) ;
                  Measurement1D projected_flight_length = reco::SecondaryVertex::computeDist3d( PV, sv, sv_p3, true ) ;
                  printf("  projected flight length: val = %9.5f , err = %9.5f , signif = %9.5f\n",
                      projected_flight_length.value(), projected_flight_length.error(), projected_flight_length.significance() ) ;

                  double minDrJet15(9999.) ;
                  double minDrJet20(9999.) ;
                  double minDrJet25(9999.) ;
                  double minDrJet30(9999.) ;
                  int closest_jet15_idx(-1) ;
                  int closest_jet20_idx(-1) ;
                  int closest_jet25_idx(-1) ;
                  int closest_jet30_idx(-1) ;

                  //TLorentzVector sv_p4( sv_p3.x(), sv_p3.y(), sv_p3.z(), sv_p3.mag() ) ;

                  for ( unsigned int ij=0; ij<jets.size(); ij++ ) {

                     if ( jets[ij].pt() < 15 ) continue ;

                     double deta = fabs( sv_p4.Eta() - jets[ij].eta() ) ;
                     double dphi = fabs( sv_p4.Phi() - jets[ij].phi() ) ;
                     if ( dphi > 3.14159265 ) dphi -= 2*3.14159265 ;
                     if ( dphi <-3.14159265 ) dphi += 2*3.14159265 ;
                     double dr = sqrt( dphi*dphi + deta*deta ) ;

                     if ( jets[ij].pt() > 15 && dr < minDrJet15 ) {
                        minDrJet15 = dr ;
                        closest_jet15_idx = ij ;
                     }
                     if ( jets[ij].pt() > 20 && dr < minDrJet20 ) {
                        minDrJet20 = dr ;
                        closest_jet20_idx = ij ;
                     }
                     if ( jets[ij].pt() > 25 && dr < minDrJet25 ) {
                        minDrJet25 = dr ;
                        closest_jet25_idx = ij ;
                     }
                     if ( jets[ij].pt() > 30 && dr < minDrJet30 ) {
                        minDrJet30 = dr ;
                        closest_jet30_idx = ij ;
                     }

                  } // ij

                  if ( closest_jet15_idx >= 0 ) {
                     printf("  closest jet, pt>15 :  dr = %6.3f,  pt= %6.1f, eta= %7.3f, phi = %7.3f,  bCSV=%7.3f\n",
                        minDrJet15, jets[closest_jet15_idx].pt(), jets[closest_jet15_idx].eta(), jets[closest_jet15_idx].phi(),
                        jets[closest_jet15_idx].bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") ) ;
                  }
                  if ( closest_jet20_idx >= 0 ) {
                     printf("  closest jet, pt>20 :  dr = %6.3f,  pt= %6.1f, eta= %7.3f, phi = %7.3f,  bCSV=%7.3f\n",
                        minDrJet20, jets[closest_jet20_idx].pt(), jets[closest_jet20_idx].eta(), jets[closest_jet20_idx].phi(),
                        jets[closest_jet20_idx].bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") ) ;
                  }
                  if ( closest_jet25_idx >= 0 ) {
                     printf("  closest jet, pt>25 :  dr = %6.3f,  pt= %6.1f, eta= %7.3f, phi = %7.3f,  bCSV=%7.3f\n",
                        minDrJet25, jets[closest_jet25_idx].pt(), jets[closest_jet25_idx].eta(), jets[closest_jet25_idx].phi(),
                        jets[closest_jet25_idx].bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") ) ;
                  }
                  if ( closest_jet30_idx >= 0 ) {
                     printf("  closest jet, pt>30 :  dr = %6.3f,  pt= %6.1f, eta= %7.3f, phi = %7.3f,  bCSV=%7.3f\n",
                        minDrJet30, jets[closest_jet30_idx].pt(), jets[closest_jet30_idx].eta(), jets[closest_jet30_idx].phi(),
                        jets[closest_jet30_idx].bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") ) ;
                  }

                  bool sv_good( true ) ;
                  if ( sv_dxy > 3. ) {  // are the units of vertex position cm?
                     sv_good = false ;
                     printf("    Transverse fligth length too long (>3 cm) : %9.5f\n", sv_dxy ) ;
                  }
                  if ( projected_flight_length.significance() < 4. ) {
                     sv_good = false ;
                     printf("    Projected flight length significance too low (<4) : %5.2f\n", projected_flight_length.significance() ) ;
                  }
                  if ( cos_pv_sv < 0.98 ) {
                     sv_good = false ;
                     printf("    cos of angle between flight length and momentum too low (<0.98) : %6.3f\n", cos_pv_sv ) ;
                  }
                  if ( sec_vert[isv].numberOfDaughters() < 3 ) {
                     sv_good = false ;
                     printf("    too few daughters (<3) : %lu\n", sec_vert[isv].numberOfDaughters() ) ;
                  }
                  if ( sv_pt < 10. ) {
                     sv_good = false ;
                     printf("    pt too low (<10): %7.1f\n", sv_pt ) ;
                  }
                  if ( minDrJet20 < 0.4 ) {
                     sv_good = false ;
                     printf("    Too close to a jet with pt>20 :   dr = %6.3f,  pt= %6.1f, eta= %7.3f, phi = %7.3f\n",
                        minDrJet20, jets[closest_jet20_idx].pt(), jets[closest_jet20_idx].eta(), jets[closest_jet20_idx].phi() ) ;
                  }
                  if ( sv_good ) {
                     printf("    *** SV is good.\n" ) ;
                  }


                  printf("\n") ;

               } // isv










               ievt ++ ;

               if ( ievt >= maxEvents_ ) break ;

            } // event
         } // file pointer ok?

         delete inFile ;

         if ( ievt >= maxEvents_ ) break ;

      } // iFile

      printf("\n\n All done.\n\n") ;

   } // main

//=========================================================

const char* mcname( int pdgid ) {

   sprintf( pname, " " ) ;

   if ( pdgid == 1 ) sprintf( pname, "d" ) ;
   if ( pdgid == 2 ) sprintf( pname, "u" ) ;
   if ( pdgid == 3 ) sprintf( pname, "s" ) ;
   if ( pdgid == 4 ) sprintf( pname, "c" ) ;
   if ( pdgid == 5 ) sprintf( pname, "b" ) ;
   if ( pdgid == 6 ) sprintf( pname, "t" ) ;

   if ( pdgid == -1 ) sprintf( pname, "d-bar" ) ;
   if ( pdgid == -2 ) sprintf( pname, "u-bar" ) ;
   if ( pdgid == -3 ) sprintf( pname, "s-bar" ) ;
   if ( pdgid == -4 ) sprintf( pname, "c-bar" ) ;
   if ( pdgid == -5 ) sprintf( pname, "b-bar" ) ;
   if ( pdgid == -6 ) sprintf( pname, "t-bar" ) ;

   if ( pdgid == 11 ) sprintf( pname, "e-" ) ;
   if ( pdgid == 12 ) sprintf( pname, "nu_e" ) ;
   if ( pdgid == 13 ) sprintf( pname, "mu-" ) ;
   if ( pdgid == 14 ) sprintf( pname, "nu_mu" ) ;
   if ( pdgid == 15 ) sprintf( pname, "tau-" ) ;
   if ( pdgid == 16 ) sprintf( pname, "nu_tau" ) ;

   if ( pdgid == -11 ) sprintf( pname, "e+" ) ;
   if ( pdgid == -12 ) sprintf( pname, "nu_e-bar" ) ;
   if ( pdgid == -13 ) sprintf( pname, "mu+" ) ;
   if ( pdgid == -14 ) sprintf( pname, "nu_mu-bar" ) ;
   if ( pdgid == -15 ) sprintf( pname, "tau+" ) ;
   if ( pdgid == -16 ) sprintf( pname, "nu_tau-bar" ) ;

   if ( pdgid == 21 ) sprintf( pname, "gluon" ) ;
   if ( pdgid == 22 ) sprintf( pname, "photon" ) ;
   if ( pdgid == 23 ) sprintf( pname, "Z0" ) ;
   if ( pdgid == 24 ) sprintf( pname, "W+" ) ;
   if ( pdgid ==-24 ) sprintf( pname, "W-" ) ;
   if ( pdgid == 25 ) sprintf( pname, "h" ) ;
   if ( pdgid == 35 ) sprintf( pname, "H" ) ;
   if ( pdgid == 36 ) sprintf( pname, "a" ) ;

   if ( pdgid == 511 ) sprintf ( pname, "B0" ) ;
   if ( pdgid == 521 ) sprintf ( pname, "B+" ) ;
   if ( pdgid == 531 ) sprintf ( pname, "Bs0" ) ;
   if ( pdgid == 541 ) sprintf ( pname, "Bc+" ) ;
   if ( pdgid == 513 ) sprintf ( pname, "B*0" ) ;
   if ( pdgid == 523 ) sprintf ( pname, "B*+" ) ;
   if ( pdgid == 533 ) sprintf ( pname, "Bs*0" ) ;

   if ( pdgid == -511 ) sprintf ( pname, "B0-bar" ) ;
   if ( pdgid == -521 ) sprintf ( pname, "B-" ) ;
   if ( pdgid == -531 ) sprintf ( pname, "Bs0-bar" ) ;
   if ( pdgid == -541 ) sprintf ( pname, "Bc-" ) ;
   if ( pdgid == -513 ) sprintf ( pname, "B*0-bar" ) ;
   if ( pdgid == -523 ) sprintf ( pname, "B*-" ) ;
   if ( pdgid == -533 ) sprintf ( pname, "Bs*0-bar" ) ;

   if ( pdgid == 411 ) sprintf ( pname, "D+" ) ;
   if ( pdgid == 413 ) sprintf ( pname, "D*+" ) ;
   if ( pdgid == 421 ) sprintf ( pname, "D0" ) ;
   if ( pdgid == 423 ) sprintf ( pname, "D*0" ) ;
   if ( pdgid == 10411 ) sprintf ( pname, "D*+" ) ;
   if ( pdgid == 10421 ) sprintf ( pname, "D*0" ) ;
   if ( pdgid == 431 ) sprintf ( pname, "Ds+" ) ;
   if ( pdgid == 10431 ) sprintf ( pname, "Ds0*+" ) ;
   if ( pdgid == 433 ) sprintf ( pname, "Ds*+" ) ;

   if ( pdgid == -411 ) sprintf ( pname, "D-" ) ;
   if ( pdgid == -413 ) sprintf ( pname, "D*-" ) ;
   if ( pdgid == -421 ) sprintf ( pname, "D0-bar" ) ;
   if ( pdgid == -423 ) sprintf ( pname, "D*0-bar" ) ;
   if ( pdgid == -10411 ) sprintf ( pname, "D*-" ) ;
   if ( pdgid == -10421 ) sprintf ( pname, "D*0-bar" ) ;
   if ( pdgid == -431 ) sprintf ( pname, "Ds-" ) ;
   if ( pdgid == -10431 ) sprintf ( pname, "Ds0*-" ) ;
   if ( pdgid == -433 ) sprintf ( pname, "Ds*-" ) ;

   if ( pdgid == 2212 ) sprintf ( pname, "proton" ) ;
   if ( pdgid == 2112 ) sprintf ( pname, "neutron" ) ;

   if ( pdgid == 211 ) sprintf ( pname, "pi+" ) ;
   if ( pdgid == -211 ) sprintf ( pname, "pi-" ) ;
   if ( pdgid == 111 ) sprintf ( pname, "pi0" ) ;

   if ( pdgid == 130 ) sprintf ( pname, "Klong" ) ;
   if ( pdgid == 310 ) sprintf ( pname, "Kshort" ) ;
   if ( pdgid == 321 ) sprintf ( pname, "K+" ) ;
   if ( pdgid == -321 ) sprintf ( pname, "K-" ) ;

   return pname ;


} // mcname

//========================================================================

bool isAncestor( const reco::Candidate* ancestor, const reco::Candidate* daughter ) {

   //printf(" in isAncestor:  ancestor ID = %d, %p     ,     daughter ID = %d, %p\n", ancestor->pdgId(), ancestor, daughter->pdgId(), daughter ) ;

 //-- doesn't work
 //if ( ancestor == daughter ) return true ;

 //-- doesn't work
 //OverlapChecker overlaps ;
 //if ( overlaps( *(ancestor), *(daughter) ) ) {
 //   printf("  overlap checker returned true.\n" ) ;
 //   return true ;
 //}

  //-- totally stupid way that works...
   if ( ancestor->pdgId() == daughter->pdgId() ) {
      if ( fabs( ancestor->pt() - daughter->pt() ) < 0.1
         && fabs( ancestor->phi() - daughter->phi() ) < 0.01
         && fabs( ancestor->eta() - daughter->eta() ) < 0.01 )
         //printf(" these two are the same.\n" ) ;
         return true ;
   }

   for (size_t i=0; i< daughter->numberOfMothers(); i++) {
      if ( isAncestor( ancestor, daughter->mother(i) ) ) {
         return true ;
      }
   } // i
   return false ;

} // isAncestor

//========================================================================

double drMatchPGP( double eta, double phi, pat::PackedGenParticleCollection& packed_gen, int& match_ipgp ) {

   double minDr = 9999. ;
   match_ipgp = -1 ;

   for ( unsigned int ipgp=0; ipgp<packed_gen.size(); ipgp++ ) {

      if ( packed_gen[ipgp].charge() == 0 ) continue ;

      double gp_eta = packed_gen[ipgp].eta() ;
      double gp_phi = packed_gen[ipgp].phi() ;

      double deta = fabs( eta - gp_eta ) ;
      double dphi = fabs( phi - gp_phi ) ;
      if ( dphi > 3.14159265 ) dphi -= 2*3.14159265 ;
      if ( dphi <-3.14159265 ) dphi += 2*3.14159265 ;
      double dr = sqrt( dphi*dphi + deta*deta ) ;
      if ( dr < minDr ) {
         minDr = dr ;
         match_ipgp = ipgp ;
      }

   } // ipgp

   return minDr ;

} // drMatchPGP

//========================================================================

















