#include "UserCode/bsmhiggs_fwk/interface/MacroUtils.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

#include "TFile.h"
#include "TH1F.h"
#include "TSystem.h"

// FWLite / EDM
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/Common/interface/MergeableCounter.h"

namespace utils
{

// ----------------------------------------------------------------------
// Pretty-print a value ± stat (± syst) in LaTeX, optionally using powers
// This is called as utils::toLatexRounded(...) from runPlotter
// ----------------------------------------------------------------------
std::string toLatexRounded(double value,
                           double error,
                           double systError,
                           bool   doPowers,
                           double systErrorDown /* = -1 by caller when symmetric */)
{
  using std::max;
  using std::sqrt;

  // if systErrorDown not specified, treat as symmetric (encode with <0)
  if(systError == systErrorDown) systErrorDown = -1;

  bool valueWasNull = false;
  if(value == 0.0 && error == 0.0) return std::string("");
  if(value == 0.0) { value = error; valueWasNull = true; }

  if(!doPowers){
    char buf[255];
    if(systError < 0)
      std::snprintf(buf, sizeof(buf), "$%.2f\\pm%.2f$", value, error);
    else if(systErrorDown < 0)
      std::snprintf(buf, sizeof(buf), "$%.2f\\pm%.2f\\pm%.2f$", value, error, systError);
    else
      std::snprintf(buf, sizeof(buf), "$%.2f\\pm%.2f^{+%.2f}_{-%.2f}$", value, error, systError, systErrorDown);
    return std::string(buf);
  }

  // choose an exponent so that numbers are O(1–100)
  double power = std::floor(std::log10(std::fabs(value) > 0 ? std::fabs(value) : (error>0?error:1.)));
  if(power <= -2) { /* keep */ }
  else if(power >= 2) power -= 2;
  else power = 0;

  const double scale = std::pow(10.0, power);
  value /= scale;
  error /= scale;
  if(systError     >= 0) systError     /= scale;
  if(systErrorDown >= 0) systErrorDown /= scale;

  int valueDigits = 0;
  if(error > 0){
    if(systError < 0){
      valueDigits = 1 + (int)std::max(-std::log10(error), 0.0);
    }else if(systErrorDown < 0){
      valueDigits = 1 + (int)std::max(-std::log10(systError), std::max(-std::log10(error), 0.0));
    }else{
      valueDigits = 1 + (int)std::max({-std::log10(systErrorDown),
                                       -std::log10(systError),
                                       -std::log10(error), 0.0});
    }
  }else{
    if(systErrorDown < 0){
      valueDigits = 1 + (int)std::max(-std::log10(std::max(systError, 1e-99)), 0.0);
    }else{
      valueDigits = 1 + (int)std::max(-std::log10(std::max(systErrorDown, 1e-99)),
                                      -std::log10(std::max(systError,     1e-99)));
    }
  }
  int errorDigits = valueDigits;

  if(valueWasNull) value = 0.0;

  char buf[255];

  if(std::fabs(value) <= 1E-4){
    // show as an upper limit with quadrature error
    double err2 = 0.0;
    if(error       > 0) err2 += error*error;
    if(systError   > 0) err2 += systError*systError;
    if(systErrorDown > 0) err2 += systErrorDown*systErrorDown;
    std::snprintf(buf, sizeof(buf), "$<%.*f$", errorDigits, std::sqrt(err2));
    return std::string(buf);
  }

  if(power != 0){
    if(systError < 0){
      std::snprintf(buf, sizeof(buf),
                    "$(%%.%df\\pm%%.%df)\\times 10^{%%g}$",
                    valueDigits, errorDigits);
      std::snprintf(buf, sizeof(buf),
                    "$( %.*f\\pm%.*f)\\times 10^{%g}$",
                    valueDigits, value, errorDigits, error, power);
    }else if(systErrorDown < 0){
      std::snprintf(buf, sizeof(buf),
                    "$( %.*f\\pm%.*f\\pm%.*f)\\times 10^{%g}$",
                    valueDigits, value, errorDigits, error, errorDigits, systError, power);
    }else{
      std::snprintf(buf, sizeof(buf),
                    "$( %.*f\\pm%.*f^{+%.*f}_{-%.*f})\\times 10^{%g}$",
                    valueDigits, value, errorDigits, error,
                    errorDigits, systError, errorDigits, systErrorDown, power);
    }
  }else{
    if(systError < 0){
      std::snprintf(buf, sizeof(buf),
                    "$%.*f\\pm%.*f$",
                    valueDigits, value, errorDigits, error);
    }else if(systErrorDown < 0){
      std::snprintf(buf, sizeof(buf),
                    "$%.*f\\pm%.*f\\pm%.*f$",
                    valueDigits, value, errorDigits, error, errorDigits, systError);
    }else{
      std::snprintf(buf, sizeof(buf),
                    "$%.*f\\pm%.*f^{+%.*f}_{-%.*f}$",
                    valueDigits, value, errorDigits, error,
                    errorDigits, systError, errorDigits, systErrorDown);
    }
  }
  return std::string(buf);
}

// ----------------------------------------------------------------------
void TLatexToTex(TString &expr)
{
  expr = "$" + expr + "$";
  expr.ReplaceAll("mu","\\mu");
  expr.ReplaceAll("_"," ");
  expr.ReplaceAll("#","\\");
}

// ======================================================================
// FWLite helpers live in utils::cmssw
// ======================================================================
namespace cmssw
{

// loop over all lumi blocks to read a MergeableCounter
unsigned long getMergeableCounterValue(const std::vector<std::string>& urls,
                                       std::string counter)
{
  unsigned long total = 0;
  for(size_t f=0; f<urls.size(); ++f){
    TFile *file = TFile::Open(urls[f].c_str());
    if(!file || file->IsZombie()){ if(file) delete file; continue; }

    fwlite::LuminosityBlock ls(file);
    for(ls.toBegin(); !ls.atEnd(); ++ls){
      fwlite::Handle<edm::MergeableCounter> h;
      h.getByLabel(ls, counter.c_str());
      if(!h.isValid()) continue;
      total += h->value;
    }
    delete file;
  }
  return total;
}

double getTotalNumberOfEvents(std::vector<std::string>& urls,
                              bool fast,
                              bool weightSum)
{
  double total = 0.0;
  for(size_t f=0; f<urls.size(); ++f){
    TFile* file = TFile::Open(urls[f].c_str());
    if(!file || file->IsZombie()){ if(file) delete file; continue; }

    fwlite::Event ev(file);
    if(!fast){
      for(ev.toBegin(); !ev.atEnd(); ++ev){
        fwlite::Handle<GenEventInfoProduct> gen;
        gen.getByLabel(ev, "generator");
        if(!gen.isValid()){ fast = true; break; }
        if(weightSum) total += gen->weight();
        else          total += (gen->weight() < 0 ? -1.0 : 1.0);
      }
    }
    if(fast){
      total += ev.size();
    }
    delete file;
  }
  return total;
}

} // namespace cmssw
} // namespace utils
