#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TFile.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TLine.h"
#include "TLatex.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"
#include "TStyle.h"
#include "CMS_lumi.C"
#include "tdrstyle.C"

namespace {
struct Limits {
   double obs = NAN, m2 = NAN, m1 = NAN, med = NAN, p1 = NAN, p2 = NAN;
   bool complete() const {
      return std::isfinite(m2) && std::isfinite(m1) &&
             std::isfinite(med) && std::isfinite(p1) && std::isfinite(p2);
   }
};
using LimitMap = std::map<double, Limits>;

TString treePath(TString path) {
   if (!path.EndsWith(".root")) {
      if (!path.EndsWith("/")) path += "/";
      path += "LimitTree.root";
   }
   return path;
}

bool readLimits(TString input, LimitMap& result) {
   TString path = treePath(input);
   TFile* file = TFile::Open(path, "READ");
   if (!file || file->IsZombie()) {
      Error("plotLimitCombined", "Cannot open %s", path.Data());
      if (file) { file->Close(); delete file; }
      return false;
   }
   TTree* tree = nullptr;
   file->GetObject("limit", tree);
   if (!tree || !tree->GetBranch("mh") || !tree->GetBranch("limit") ||
       !tree->GetBranch("quantileExpected")) {
      Error("plotLimitCombined", "Missing limit tree or branches in %s", path.Data());
      file->Close(); delete file;
      return false;
   }
   double mh = 0., limit = 0.;
   float quantile = 0.f;
   tree->SetBranchAddress("mh", &mh);
   tree->SetBranchAddress("limit", &limit);
   tree->SetBranchAddress("quantileExpected", &quantile);
   for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      if (!std::isfinite(mh) || !std::isfinite(limit)) continue;
      Limits& v = result[mh];
      if (quantile < 0.f) v.obs = limit;
      else if (std::abs(quantile - 0.025f) < 0.005f) v.m2 = limit;
      else if (std::abs(quantile - 0.160f) < 0.005f) v.m1 = limit;
      else if (std::abs(quantile - 0.500f) < 0.005f) v.med = limit;
      else if (std::abs(quantile - 0.840f) < 0.005f) v.p1 = limit;
      else if (std::abs(quantile - 0.975f) < 0.005f) v.p2 = limit;
   }
   file->Close(); delete file;
   return true;
}

struct Curves {
   TGraph* observed = new TGraph();
   TGraph* expected = new TGraph();
   TGraphAsymmErrors* oneSigma = new TGraphAsymmErrors();
   TGraphAsymmErrors* twoSigma = new TGraphAsymmErrors();
};

void append(Curves& c, double mass, const Limits& v, bool blind) {
   const int i = c.expected->GetN();
   c.expected->SetPoint(i, mass, v.med);
   c.oneSigma->SetPoint(i, mass, v.med);
   c.oneSigma->SetPointError(i, 0., 0., v.med-v.m1, v.p1-v.med);
   c.twoSigma->SetPoint(i, mass, v.med);
   c.twoSigma->SetPointError(i, 0., 0., v.med-v.m2, v.p2-v.med);
   if (!blind && std::isfinite(v.obs))
      c.observed->SetPoint(c.observed->GetN(), mass, v.obs);
}

void style(Curves& c) {
   c.twoSigma->SetFillColor(TColor::GetColor("#ffcc00"));
   c.twoSigma->SetLineColor(c.twoSigma->GetFillColor());
   c.oneSigma->SetFillColor(TColor::GetColor("#228b22"));
   c.oneSigma->SetLineColor(c.oneSigma->GetFillColor());
   c.expected->SetLineColor(kBlack);
   c.expected->SetLineStyle(2);
   c.expected->SetLineWidth(3);
   c.observed->SetLineColor(kBlack);
   c.observed->SetLineWidth(2);
   c.observed->SetMarkerStyle(20);
   c.observed->SetMarkerSize(0.8);
}

void draw(Curves& c, bool blind) {
   // Draw each regime independently; bands and curves must not bridge the boundary.
   c.twoSigma->Draw("3 SAME");
   c.oneSigma->Draw("3 SAME");
   c.expected->Draw("L SAME");
   if (!blind && c.observed->GetN()) c.observed->Draw("LP SAME");
}
} // namespace

// Each input can be its LimitTree.root path or its containing directory.
// The current plot includes the 25 GeV point in both regimes.
void plotLimitCombined(std::string outputDir,
                       TString boostedInput,
                       TString resolvedInput,
                       bool blind = true,
                       double energy = 13.6,
                       double luminosity = 109.82,
                       bool logY = true,
                       double yMin = 1e-3,
                       double yMax = 1e2)
{
   if (boostedInput.IsNull() || resolvedInput.IsNull() ||
       yMin <= 0. || yMax <= yMin) {
      Error("plotLimitCombined", "Provide both input paths and valid y-axis limits");
      return;
   }
   LimitMap boosted, resolved;
   if (!readLimits(boostedInput, boosted) || !readLimits(resolvedInput, resolved)) return;

   Curves b, r;
   std::vector<std::pair<double, Limits>> used;
   for (const auto& item : boosted) {
      double mass = item.first;
      if (mass < 12. || mass > 25.) continue;
      if (!item.second.complete()) {
         Warning("plotLimitCombined", "Skipping incomplete boosted mass %.1f", mass);
         continue;
      }
      append(b, mass, item.second, blind);
      used.push_back(item);
   }
   for (const auto& item : resolved) {
      double mass = item.first;
      if (mass < 25. || mass > 60.) continue;
      if (!item.second.complete()) {
         Warning("plotLimitCombined", "Skipping incomplete resolved mass %.1f", mass);
         continue;
      }
      append(r, mass, item.second, blind);
      used.push_back(item);
   }
   if (!b.expected->GetN() || !r.expected->GetN()) {
      Error("plotLimitCombined", "Need complete boosted points below 25 and resolved points from 25 to 60 GeV");
      return;
   }
   if (resolved.find(25.) == resolved.end())
      Warning("plotLimitCombined", "No resolved result at exactly 25 GeV; the resolved curve begins at its first available mass");

   style(b); style(r);
   setTDRStyle();
   gStyle->SetPadTopMargin(0.05);
   gStyle->SetPadBottomMargin(0.12);
   gStyle->SetPadRightMargin(0.16);
   gStyle->SetPadLeftMargin(0.17);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.45);
   TCanvas* c = new TCanvas("combinedLimit", "Combined boosted and resolved limits", 1000, 700);
   //c->SetGridx(); c->SetGridy();
   c->SetLogy(logY);
   TH1F* frame = new TH1F("combinedLimitFrame", "", 48, 12., 60.);
   frame->SetStats(false);
   frame->GetXaxis()->SetTitle("M_{a} [GeV]");
   frame->GetYaxis()->SetTitle("#sigma_{ZH} #it{B}(H#rightarrowaa#rightarrow4b) / #sigma_{SM}");
   frame->GetXaxis()->SetTitleOffset(1.1);
   frame->GetYaxis()->SetTitleOffset(1.1);
   frame->GetXaxis()->SetTitleFont(42);
   frame->GetYaxis()->SetTitleFont(42);
   frame->GetXaxis()->SetTitleSize(0.05);
   frame->GetYaxis()->SetTitleSize(0.05);
   frame->GetYaxis()->SetRangeUser(yMin, yMax);
   frame->Draw();
   draw(b, blind);
   draw(r, blind);

   // The separator stops at the higher +2 sigma band edge at 25 GeV.
   const auto b25 = boosted.find(25.);
   const auto r25 = resolved.find(25.);
   if (b25 == boosted.end() || r25 == resolved.end() ||
       !std::isfinite(b25->second.p2) || !std::isfinite(r25->second.p2)) {
      Error("plotLimitCombined", "Both inputs need a +2 sigma limit at 25 GeV");
      return;
   }
   const double splitTop = std::max(b25->second.p2, r25->second.p2);
   TLine* split = new TLine(25., yMin, 25., yMax);
   split->SetLineColor(kGray+2);
   split->SetLineStyle(2);
   split->SetLineWidth(3);
   split->Draw("SAME");
   //TLine* unit = new TLine(12., 1., 60., 1.);
   //unit->SetLineStyle(1);
   //unit->SetLineColor(kBlack);
   //unit->SetLineWidth(2);
   //unit->Draw("SAME");

   // TLatex region labels are disabled in the current plot.
   TLegend* leg = new TLegend(0.55, 0.75, 0.85, 0.95);
   leg->SetHeader("");
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   leg->SetTextFont(42);
   leg->SetBorderSize(0);
   leg->AddEntry(r.expected, "median expected", "L");
   leg->AddEntry(r.oneSigma, "expected #pm 1#sigma", "F");
   leg->AddEntry(r.twoSigma, "expected #pm 2#sigma", "F");
   if (!blind && (b.observed->GetN() || r.observed->GetN()))
      leg->AddEntry(r.observed->GetN() ? r.observed : b.observed, "observed", "LP");
   leg->Draw();

   double lumiFb = luminosity > 1000. ? luminosity / 1000. : luminosity;
   lumi_sqrtS = Form("%.2f fb^{-1} (%.1f TeV)", lumiFb, energy);
   CMS_lumi(c, 0, 11);
   c->RedrawAxis();

   if (!outputDir.empty() && outputDir.back() != '/') outputDir += '/';
   if (gSystem->mkdir(outputDir.c_str(), true) != 0 && gSystem->AccessPathName(outputDir.c_str())) {
      Error("plotLimitCombined", "Cannot create output directory %s", outputDir.c_str());
      return;
   }
   TString stem = TString(outputDir) + (logY ? "LimitCombined_log" : "LimitCombined_linear");
   c->SaveAs(stem + ".pdf");
   //c->SaveAs(stem + ".png");
   c->SaveAs(stem + ".root");

   FILE* summary = fopen((outputDir + "LimitCombined.csv").c_str(), "w");
   if (summary) {
      fprintf(summary, "regime,ma,observed,expected_minus_2sigma,expected_minus_1sigma,expected,expected_plus_1sigma,expected_plus_2sigma\n");
      for (const auto& item : used) {
         const double m = item.first;
         const Limits& v = item.second;
         fprintf(summary, "%s,%.6g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                 m < 25. ? "boosted" : "resolved", m, v.obs, v.m2, v.m1, v.med, v.p1, v.p2);
      }
      fclose(summary);
   }
}
