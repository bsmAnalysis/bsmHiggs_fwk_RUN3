import ROOT as rt

import CMS_lumi, tdrstyle

from array import array

import os

import sys

import ctypes



import argparse





#########from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double # owen

from ROOT import gROOT, gBenchmark, gRandom, gSystem, gStyle #, Double



import ROOT



ROOT.gROOT.SetBatch(True)



parser = argparse.ArgumentParser()



parser.add_argument( "--do_liny", dest='do_liny', help="Use linear scale for y axis", action='store_true' )

parser.add_argument( "--do_linzoom", dest='do_linzoom', help="Zoom in on the highest bins with a linear vertical scale", action='store_true' )

parser.add_argument( "prodmode", type=str, help="Production mode (wh or zh)")

parser.add_argument( "limit_dir", type=str, help="Input directory for the mass point.  Example = all-batch-output-10x-july24a--new-bdt-binning2--autoMCStats/cards_SB13TeV_SM_Wh_2016_noSoftb/0040/")



args = parser.parse_args()



limit_dir = args.limit_dir



# Choose if you need to blind SRs

iblind=-1

#iblind=3



# Picks up the pre-fit BDT:

dir1="shapes_prefit"



#Picks up the b post-fit BDT:

#dir1="shapes_fit_b"



#Picks up the s+b post-fit BDT:

#dir1="shapes_fit_s"



#wz = "zh"

#wz = "wh"



wz = args.prodmode

if ( not (wz == "wh" or wz == "zh" )):

   print("\n\n *** set prodmode to wh or zh.  ", wz, " is not allowed.\n\n")

   quit()





outdir = ""

if ( wz == "zh" ) :

   outdir = "prefit-plots-zh"

   if ( "2016" in limit_dir ): outdir = "prefit-plots-zh-2016"

   if ( "2017" in limit_dir ): outdir = "prefit-plots-zh-2017"

   if ( "2018" in limit_dir ): outdir = "prefit-plots-zh-2018"

   if ( "2024" in limit_dir ): outdir = "prefit-plots-zh-2024"

if ( wz == "wh" ) :

   outdir = "prefit-plots-wh"

   if ( "2016" in limit_dir ): outdir = "prefit-plots-wh-2016"

   if ( "2017" in limit_dir ): outdir = "prefit-plots-wh-2017"

   if ( "2018" in limit_dir ): outdir = "prefit-plots-wh-2018"

   if ( "2018" in limit_dir ): outdir = "prefit-plots-wh-2018"

try:

        os.mkdir( outdir )

except:

        print("\n\n problem making %s" % outdir ) ;





if wz=="wh":

  #channels = ["mu_A_CR5j_3b", "mu_A_CR5j_4b", "mu_A_CR_3b", "mu_A_CR_4b", "mu_A_SR_3b", "mu_A_SR_4b", "e_A_CR5j_3b", "e_A_CR5j_4b", "e_A_CR_3b", "e_A_CR_4b", "e_A_SR_3b", "e_A_SR_4b"]

  #channels = ["mu_A_CR_3b", "mu_A_CR_4b", "mu_A_SR_3b", "mu_A_SR_4b", "e_A_CR_3b", "e_A_CR_4b", "e_A_SR_3b", "e_A_SR_4b"]

   channels = ["emu_A_CR_3b"] #-Penny 

   e_mu = [""]

elif wz=="zh":

#  channels = ["mumu_A_CR_3b", "mumu_A_CR_4b", "mumu_A_SR_3b", "mumu_A_SR_4b", "emu_A_CR_3b", "emu_A_CR_4b", "emu_A_SR_3b", "emu_A_SR_4b", "ee_A_CR_3b", "ee_A_CR_4b", "ee_A_SR_3b", "ee_A_SR_4b"]

   channels=["lep1_A_CR_3b"]

   #channels = ["ee_A_SR_3b", "mumu_A_SR_3b"]#, "ee_A_SR_3b", "mumu_A_SR_3b", "ee_A_CR_3b", "mumu_A_CR_3b"]

   e_mu = ["Test"]

   #CRs = ["mumu_A_CR_3b"]

verbose = True



do_liny = False

if args.do_liny:

   do_liny = True

   print("Will use linear axis\n")



do_linzoom = False

if args.do_linzoom:

   do_linzoom = True

   do_liny = True

   print("Will zoom in on senitive bins with linear vertical scale.\n")







#set the tdr style

tdrstyle.setTDRStyle()



#change the CMS_lumi variables (see CMS_lumi.py)

CMS_lumi.lumi_7TeV = "4.8 fb^{-1}"

CMS_lumi.lumi_8TeV = "18.3 fb^{-1}"

CMS_lumi.writeExtraText = 1

CMS_lumi.extraText = "Preliminary"

CMS_lumi.lumi_sqrtS = "" #"13 TeV" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)



if ( "2016" in limit_dir ): iPeriod = 6

if ( "2017" in limit_dir ): iPeriod = 7

if ( "2018" in limit_dir ): iPeriod = 8

if "2024" in limit_dir:

    iPeriod = 0  # free-form label mode                                         

    CMS_lumi.lumi_sqrtS = "13.6 TeV, 109.82 fb^{-1}"

iPos = 11

if( iPos==0 ): CMS_lumi.relPosX = 0.12



H_ref = 700; 

W_ref = 700; 

W = W_ref

H  = H_ref





gStyle.SetErrorX(0.5)  # owen: need to add this to get horizontal bars on data hist





#------------------------------------------------------------



def printEvtYields(hist, name):

  if not hist: return

#  printout = [name+":"]

  printout = ["{0: <19}:".format(name)]

  for ibin in range(1, hist.GetXaxis().GetNbins()+1):

#    content = hist.GetBinContent(ibin)

#    error = hist.GetBinError(ibin)

    #error = Double() # owen

    error = ctypes.c_double()

    content = hist.IntegralAndError(ibin, ibin, error)

    ##########yields = "{0:10.2f}".format(content) +" +- " + "{0:<10.2f}".format(error)

    yields = "{0:10.2f}".format(content) +" +- " + "{0:<10.2f}".format(error.value)

    #yields = " fixme "

    printout.append(yields)

#  error = Double()

#  content = hist.IntegralAndError(1, hist.GetXaxis().GetNbins(), error)

#  yields = "{:.2f}".format(content) +"+-" + "{:.2f}".format(error)

#  printout.append(yields)

  print("  ".join(printout))





#------------------------------------------------------------



def convertXRange(hist, name, edges):

  out_h = rt.TH1F(name, name, len(edges)-1, array('d', edges))

  for i in range(1, hist.GetNbinsX()+1):

    out_h.SetBinContent(i, hist.GetBinContent(i))

    out_h.SetBinError(i, hist.GetBinError(i))

  return out_h





#------------------------------------------------------------

# 

# Simple example of macro: plot with CMS name and lumi text

#  (this script does not pretend to work in all configurations)

# iPeriod = 1*(0/1 7 TeV) + 2*(0/1 8 TeV)  + 4*(0/1 13 TeV) 

# For instance: 

#               iPeriod = 3 means: 7 TeV + 8 TeV

#               iPeriod = 7 means: 7 TeV + 8 TeV + 13 TeV 

#               iPeriod = 0 means: free form (uses lumi_sqrtS)

# Initiated by: Gautier Hamel de Monchenault (Saclay)

# Translated in Python by: Joshua Hardenbrook (Princeton)

# Updated by:   Dinko Ferencek (Rutgers)

#



#iPeriod = 0



# references for T, B, L, R

T = 0.08*H_ref

B = 0.12*H_ref 

L = 0.12*W_ref

R = 0.04*W_ref

for ch in e_mu:



  if ( verbose ) : print("\n\n verbose:  ch = ", ch , "\n")



  if wz=="zh":

    file = rt.TFile(limit_dir+"fitDiagnostics{}.root".format(ch),"READ")

#    file = rt.TFile(limit_dir+"fitDiagnosticsTest.root","READ")      

  elif wz=="wh":

#    file = rt.TFile(limit_dir+"fitDiagnostics.root","READ")

    file = rt.TFile(limit_dir+"fitDiagnosticsTest.root","READ")



  if ( verbose ) : print("\n\n verbose: file = ", file.GetName(),"\n" )



  inf = rt.TFile(limit_dir+"haa4b_60_13p6TeV_{}.root".format(wz),"READ")

  #inf = rt.TFile(limit_dir+"haa4b_40_13TeV_{}.root".format(wz),"READ")

  #inf = rt.TFile(limit_dir+"haa4b_30_13TeV_{}.root".format(wz),"READ")

  #inf = rt.TFile(limit_dir+"haa4b_25_13TeV_{}.root".format(wz),"READ")

  if ( verbose ) : print("\n\n verbose inf = ", inf.GetName(), "\n" )



  for dir2 in channels:



    if ( verbose ) : print("  verbose:  dir2 = ", dir2)



    blind=iblind

    #if dir2 in CRs: blind=-1



    print("blind: " +str(blind))



    dir=dir1+"/"+dir2+"/"



    inh = inf.Get(dir2+"/data_obs")



    if ( verbose ) : print("  verbose:  Get arg : ", (dir2+"/data_obs") , " inh = ", inh.GetName(), "  ,  ", inh.GetTitle() )



    if not inh:

      print("!!!Warning: cannot find the data_obs in channel {}, skip the channel!".format(dir2))

      continue

    edges = []

    np = inh.GetNbinsX()

    for i in range(1, np+1):

      edges.append(inh.GetXaxis().GetBinLowEdge(i))

    edges.append(inh.GetXaxis().GetBinUpEdge(i))

    print(edges)



    canvas = rt.TCanvas("c2_" + dir2, "c2_" + dir2, 50, 50, W, H)

    canvas.SetFillColor(0)

    canvas.SetBorderMode(0)

    canvas.SetFrameFillStyle(0)

    canvas.SetFrameBorderMode(0)

    canvas.SetLeftMargin( L/W )

    canvas.SetRightMargin( R/W )

    canvas.SetTopMargin( T/H )

    canvas.SetBottomMargin( B/H )

    canvas.SetTickx(0)

    canvas.SetTicky(0)



    h = rt.TH1F("h_" + dir2, "h; BDT; Events", 4, 0, 4)



    xAxis = h.GetXaxis()

    xAxis.SetNdivisions(5,4,0)



    yAxis = h.GetYaxis()

    yAxis.SetNdivisions(5,4,0)

    yAxis.SetTitleOffset(1)





    bkgd_list = []

    lgname_list = []



    data = file.Get(dir+"data")





    if not data: continue



    if ( verbose ) : print("  verbose:  Get arg: ",(dir+"data")," data = ", data.GetName(), "  ,  ", data.GetTitle() )



    data.SetMarkerStyle(20)

    data.SetMarkerColor(1)



#    total = file.Get(dir+"total_background")

    tmp_hist = file.Get(dir+"total_background")

    if ( verbose ) : print("  verbose:  Get arg: ", (dir+"total_background"), " tmp_hist = ", tmp_hist.GetName(), "  ,  ", tmp_hist.GetTitle(), "\n" )





    total = convertXRange(file.Get(dir+"total_background"), "total_background", edges)



    sig = file.Get(dir+"zh")

    if sig:

      sig = convertXRange(sig, "zh", edges)

      sig.SetLineColor(rt.kRed+4)

      sig.SetLineWidth(2)

      sig.SetLineStyle(2)

#      if((dir1 == "shapes_fit_s") and (dir2 not in CRs)): sig.Scale(10)

    #sig.Scale(50)

    # sig.Scale(10)



    otherbkg = file.Get(dir+"otherbkg")

    if otherbkg:

      otherbkg = convertXRange(otherbkg, "otherbkg", edges)

      otherbkg.SetFillColor(852)

      otherbkg.SetLineColor(1)

      bkgd_list.append(otherbkg)

      lgname_list.append("Other bkgs")



    # -Penny mc qcd (qcd|QCD)/dd qcd (ddqcd|QCD (dd))

    ddqcd = file.Get(dir+"ddqcd")

    if ddqcd:

      ddqcd = convertXRange(ddqcd, "ddqcd", edges)

      ddqcd.SetFillColor(634)

      ddqcd.SetLineColor(1)

      bkgd_list.append(ddqcd)

      lgname_list.append("ddQCD")



    znunu = file.Get(dir+"znunu")

    if znunu:

      znunu = convertXRange(znunu, "znunu", edges)

      znunu.SetFillColor(624)

      znunu.SetLineColor(1)

      bkgd_list.append(znunu)

      lgname_list.append("Z#rightarrow #nu #nu")



    wjets = file.Get(dir+"wjets")

    if wjets:

       wjets = convertXRange(wjets, "wjets", edges)

       wjets.SetFillColor(622)

       wjets.SetLineColor(1)

       bkgd_list.append(wjets)

       lgname_list.append("W#rightarrow l#nu")

    '''

    zll = file.Get(dir+"zll")                                                                                                                                       

    if zll:                                                                                                                                                           

      zll = convertXRange(zll, "zll", edges)                                                                                                                      

      zll.SetFillColor(622)                                                                                                                                           

      zll.SetLineColor(1)                                                                                                                                             

      bkgd_list.append(zll)                                                                                                                                           

      lgname_list.append("Z#rightarrow ll")

    '''

    ttbarbba = file.Get(dir+"ttbarbba")

    if ttbarbba:

      ttbarbba = convertXRange(ttbarbba, "ttbarbba", edges)

      ttbarbba.SetFillColor(833)

      ttbarbba.SetLineColor(1)

      bkgd_list.append(ttbarbba)

      lgname_list.append("t#bar{t} + b#bar{b}")



    ttbarcba = file.Get(dir+"ttbarcba")

    if ttbarcba:

      ttbarcba = convertXRange(ttbarcba, "ttbarcba", edges)

      ttbarcba.SetFillColor(408)

      ttbarcba.SetLineColor(1)

      bkgd_list.append(ttbarcba)

      lgname_list.append("t#bar{t} + c#bar{c}")



    ttbarlig = file.Get(dir+"ttbarlig")

    if ttbarlig:

      ttbarlig = convertXRange(ttbarlig, "ttbarlig", edges)

      ttbarlig.SetFillColor(406)

      ttbarlig.SetLineColor(1)

      bkgd_list.append(ttbarlig)

      lgname_list.append("t#bar{t} + light")



    # Avoid THStack here.  With ROOT 6.30 the PyROOT/THStack painting path can
    # retain a stale C++ histogram pointer and crash in BuildAndPaint().
    # Cumulative TH1 layers reproduce the same stacked appearance.
    mc_stack_layers = []
    running_mc = None

    for index, bkgd in enumerate(bkgd_list):
      if not bkgd or not bkgd.InheritsFrom("TH1"):
        raise RuntimeError(
            "Invalid background histogram while building stack for " + dir2
        )
      if bkgd.GetNbinsX() != total.GetNbinsX():
        raise RuntimeError(
            "Bin mismatch for {} in {}: {} versus {}".format(
                bkgd.GetName(),
                dir2,
                bkgd.GetNbinsX(),
                total.GetNbinsX(),
            )
        )

      bkgd.SetDirectory(0)
      ROOT.SetOwnership(bkgd, False)

      if running_mc is None:
        running_mc = bkgd.Clone(
            "mc_running_{}_{}".format(dir2, index)
        )
      else:
        running_mc.Add(bkgd)

      running_mc.SetDirectory(0)
      ROOT.SetOwnership(running_mc, False)

      layer = running_mc.Clone(
          "MCstack_layer_{}_{}".format(dir2, index)
      )
      layer.SetDirectory(0)
      ROOT.SetOwnership(layer, False)
      layer.SetFillColor(bkgd.GetFillColor())
      layer.SetFillStyle(bkgd.GetFillStyle())
      layer.SetLineColor(bkgd.GetLineColor())
      layer.SetLineStyle(bkgd.GetLineStyle())
      layer.SetLineWidth(bkgd.GetLineWidth())
      mc_stack_layers.append(layer)

    if not mc_stack_layers:
      raise RuntimeError("No background histograms available for " + dir2)





    t1 = rt.TPad("t1_" + dir2, "t1_" + dir2, 0.0, 0.2, 1.0, 1.0)

    t1.SetFillColor(0)

    t1.SetBorderMode(0)

    t1.SetBorderSize(2)

    t1.SetTickx(1)

    t1.SetTicky(1)

    t1.SetLeftMargin(0.10)

    t1.SetRightMargin(0.05)

    t1.SetTopMargin(0.05)

    t1.SetBottomMargin(0.10)

    t1.SetFrameFillStyle(0)

    t1.SetFrameBorderMode(0)

    t1.SetFrameFillStyle(0)

    t1.SetFrameBorderMode(0)



    t1.Draw()

    t1.cd()



    #MC.Draw("hist")

    hdata = rt.TH1F(
        "hdata_" + dir2, "data bdt", np, array('d', edges)
    )
    hdata.SetDirectory(0)
    ROOT.SetOwnership(hdata, False)



    #xmin=-0.3

    #for i in range(0,hdata.GetXaxis().GetNbins()):

    #    label=str(xmin+i*hdata.GetXaxis().GetBinWidth(i))

    #    hdata.GetXaxis().SetBinLabel(i,label)



    #hdata.Rebin(rbin)



    htotal = total.Clone("htotal_" + dir2)
    htotal.SetDirectory(0)
    ROOT.SetOwnership(htotal, False)



    #px, py = Double(), Double() # owen

    #pyerr = Double() # owen

    px = array('d', [0.0])

    py = array('d', [0.0])

    #px = ctypes.c_double()

    #py = ctypes.c_double()

    #pyerr = ctypes.c_double()

    #pyerr = data.GetErrorY(i)

    alpha = 1 - 0.6827;



    nPoints=data.GetN()

    for i in range(0,nPoints):

        data.GetPoint(i,px,py)



        N=px[0]  # Update for asymmetric error bar in 0 observed events

        if N==0: 

           L=0

           U= rt.Math.gamma_quantile_c(alpha/2,N+1,1)     

           data.SetMarkerSize(0.5);

           data.SetMarkerStyle (20);

           data.SetPointEYlow(i,0) 

           data.SetPointEYhigh(i,U-N)





        pyerr = data.GetErrorY(i)  # this returns a float fine



        hdata.SetBinContent(i+1, py[0])

        hdata.SetBinError(i+1, pyerr)

 #   t1.Update();



    # Determine the range before removing the blinded data points.
    hmax = max(hdata.GetMaximum(), total.GetMaximum())
    upper_ymin = 0.1 if not do_liny else 0.0

    if not do_liny:
       if hmax > 50000:
          upper_ymax = 1000*hmax
       elif hmax > 10000:
          upper_ymax = 500*hmax
       elif hmax > 1000:
          upper_ymax = 100*hmax
       else:
          upper_ymax = 50*hmax
    else:
       if do_linzoom:
          zoombin = hdata.GetBinContent(4)
          upper_ymax = 3*zoombin
       else:
          upper_ymax = 1.4*hmax

    if upper_ymax <= upper_ymin:
       upper_ymax = 10.0 if not do_liny else 1.0

    systBand = rt.TGraphErrors(total.GetNbinsX()); IPoint=0

    for ibin in range(1,htotal.GetXaxis().GetNbins()+1):

       systBand.SetPoint(IPoint, htotal.GetBinCenter(ibin), htotal.GetBinContent(ibin) )

       systBand.SetPointError(IPoint, htotal.GetBinWidth(ibin)/2, htotal.GetBinError(ibin) )

       IPoint += 1



    systBand.Set(IPoint);

#    htotal.SetFillStyle(3004);

#    htotal.SetFillColor(kGray+2);

#    htotal.SetMarkerStyle(1);



    systBand.SetFillStyle(3004);

    systBand.SetFillColor(rt.kGray+2);

    systBand.SetMarkerStyle(1);

    # Draw an empty frame first.  The blinded area spans the complete visible
    # y range and is painted before MC, signal, data, and the legend.
    upper_frame = total.Clone("upper_frame_" + dir2)
    upper_frame.SetDirectory(0)
    ROOT.SetOwnership(upper_frame, False)
    upper_frame.Reset("ICE")
    upper_frame.SetStats(0)
    upper_frame.SetTitle("")
    upper_frame.SetMinimum(upper_ymin)
    upper_frame.SetMaximum(upper_ymax)
    upper_frame.GetXaxis().SetLabelOffset(0.007)
    upper_frame.GetXaxis().SetLabelSize(0.04)
    upper_frame.GetXaxis().SetTitleOffset(1.2)
    upper_frame.GetXaxis().SetTitleFont(42)
    upper_frame.GetXaxis().SetTitleSize(0.04)
    upper_frame.GetXaxis().SetTitle("BDT")
    upper_frame.GetYaxis().SetLabelFont(42)
    upper_frame.GetYaxis().SetLabelOffset(0.007)
    upper_frame.GetYaxis().SetLabelSize(0.04)
    upper_frame.GetYaxis().SetTitleOffset(1.35)
    upper_frame.GetYaxis().SetTitleFont(42)
    upper_frame.GetYaxis().SetTitleSize(0.04)
    upper_frame.GetYaxis().SetTitle("Events")

    t1.SetLogy(not do_liny)
    upper_frame.Draw("AXIS")

    if blind > 0:
        for i in range(blind, hdata.GetNbinsX()+1):
            hdata.SetBinContent(i, 0)
            hdata.SetBinError(i, 0)

        blinding_box = rt.TPave(
            hdata.GetXaxis().GetBinLowEdge(blind),
            upper_ymin,
            hdata.GetXaxis().GetXmax(),
            upper_ymax,
            0,
            "NB",
        )
        blinding_box.SetFillColor(15)
        blinding_box.SetFillStyle(3013)
        blinding_box.Draw("same F")

    # Draw all physics objects after the blinded-area background.
    for layer in reversed(mc_stack_layers):
        layer.Draw("hist same")
    systBand.Draw("same 2 0")
    if sig:
        sig.Draw("hist same")
    hdata.Draw("e0psame0")
    hdata.Draw("e1same")
    t1.RedrawAxis()



    #set the colors and size for the legend

    histLineColor = rt.kOrange+7

    histFillColor = rt.kOrange-2

    markerSize  = 1.0



    latex = rt.TLatex()

    n_ = 2



    x1_l = 0.95 #0.92

    y1_l = 0.90 #0.60



    dx_l = 0.30

    dy_l = 0.18 #0.18

    x0_l = x1_l-dx_l

    y0_l = y1_l-dy_l



    #######legend =  rt.TLegend(0.40,0.74,0.93,0.96, "NDC")

    #legend =  rt.TLegend(0.67,0.96,0.83,0.45, "NDC")

    legend =  rt.TLegend(0.30,0.74,0.93,0.96, "NDC") # -Penny

    legend.SetNColumns(3) # -Penny

    #legend = rt.TLegend(0.40,0.74,0.98,1.02, "NDC")

    #legend.SetFillColor( rt.kGray )

    legend.SetHeader("") #(dir)

    #legend.SetNColumns(2)  

    legend.SetBorderSize(0)

    legend.SetTextFont(42)

    legend.SetTextSize(0.035)

    legend.SetLineColor(0)

    legend.SetLineStyle(1)

    legend.SetLineWidth(1)

    legend.SetFillColor(0)

    legend.SetFillStyle(0)



    #legend.Draw("same")

    #legend.cd()



    ar_l = dy_l/dx_l

    #gap_ = 0.09/ar_l

    gap_ = 1./(n_+1)

    bwx_ = 0.12

    bwy_ = gap_/1.5



    x_l = [1.2*bwx_]

    #y_l = [1-(1-0.10)/ar_l]

    y_l = [1-gap_]

    ex_l = [0]

    ey_l = [0.04/ar_l]



    ## latex.DrawLatex(xx_+1.*bwx_,yy_,"Data")

    legend.AddEntry(hdata,"Data","LP")

#    legend.AddEntry(errBand, "Stat. Unc.", "F");

    legend.AddEntry(systBand, "Syst + Stat.", "F");

    #if sig: legend.AddEntry(sig,"Zh (40) x 10","L")

    print("-"*100)

    print(dir2) 

    for i in range(len(bkgd_list)):

      legend.AddEntry(bkgd_list[i], lgname_list[i], "F")

      printEvtYields(bkgd_list[i], lgname_list[i])

    printEvtYields(total, "total")

    print("\n")

    printEvtYields(hdata, "data")

    if sig: 

       if ( wz == "zh" ): legend.AddEntry(sig,"Signal ZH (60GeV)","L") 

       if ( wz == "wh" ): legend.AddEntry(sig,"Signal WH (60GeV)","L") 

       printEvtYields(sig, "signal")



    #legend.AddEntry(wlnu,"W#rightarrow l#nu","LF")

    #legend.AddEntry(zll,"Z#rightarrow ll","LF")

    #legend.AddEntry(ttbarbba,"t#bar{t} + b#bar{b}","LF")

    #legend.AddEntry(ttbarcba,"t#bar{t} + c#bar{c}","LF")

    #legend.AddEntry(ttbarlig,"t#bar{t} + light","LF")

    #legend.AddEntry(otherbkg,"Other bkgs","LF")

    #legend.AddEntry(qcd,"DD qcd","LF")



    if(blind>0):

        legend.AddEntry(blinding_box,"Blinded area","F")



    legend.Draw("same")

    if ( not do_liny ) : t1.SetLogy(True)



    #draw the lumi text on the canvas

    CMS_lumi.CMS_lumi(t1, iPeriod, iPos)



    t1.Update()



    canvas.cd()

    canvas.Update()



    ## ratio plot

    t2 = rt.TPad("t2_" + dir2, "t2_" + dir2, 0.0, 0.0, 1.0, 0.2)

    t2.SetFillColor(0)

    t2.SetBorderMode(0)

    t2.SetBorderSize(2)

    t2.SetGridy()

    t2.SetTickx(1)

    t2.SetTicky(1)

    t2.SetLeftMargin(0.10)

    t2.SetRightMargin(0.05)

    t2.SetTopMargin(0.0)

    t2.SetBottomMargin(0.20)

    t2.SetFrameFillStyle(0)

    t2.SetFrameBorderMode(0)

    t2.SetFrameFillStyle(0)

    t2.SetFrameBorderMode(0)

    t2.Draw()

    t2.cd()

    t2.SetGridy(1)

    t2.SetPad(0,0.0,1.0,0.2)





    hMCtotal = htotal.Clone("mcrelunc_" + dir2)
    hMCtotal.SetDirectory(0)
    ROOT.SetOwnership(hMCtotal, False)

#    hMCtotal.Sumw2()

 #   utils::root::checkSumw2(hMCtotal)



    # Make ratio with TGraphErrors

    gtotal = rt.TGraphErrors(hMCtotal.GetNbinsX())



    GPoint = 0

    for i in range(1,hMCtotal.GetXaxis().GetNbins()+1):

       gtotal.SetPoint(GPoint, hMCtotal.GetBinCenter(i), 1.0)

       if (hMCtotal.GetBinContent(i) != 0):

          gtotal.SetPointError(GPoint, hMCtotal.GetBinWidth(i)/2, hMCtotal.GetBinError(i)/hMCtotal.GetBinContent(i))

          GPoint += 1

       else:

          gtotal.SetPointError(GPoint, hMCtotal.GetBinWidth(i)/2, 0)

          GPoint += 1 ; continue



       err = hMCtotal.GetBinError(i)/hMCtotal.GetBinContent(i)

       hMCtotal.SetBinContent(i,1)

       hMCtotal.SetBinError(i,err)



    gtotal.Set(GPoint)

    gtotal.SetLineColor(1)

    gtotal.SetFillStyle(3001) #3004

    gtotal.SetFillColor(16) #rt.kGray+3)

    gtotal.SetMarkerColor(1)

    gtotal.SetMarkerStyle(20) #1)



    hMCtotal.Reset("ICE")

    hMCtotal.SetTitle("")

    hMCtotal.SetStats(0)

    yscale = (1.0-0.2)/(0.2)

    hMCtotal.GetYaxis().SetTitle("Data/#Sigma Bkg.")

    hMCtotal.GetXaxis().SetTitle() # drop the title to gain space

    hMCtotal.SetMinimum(0.2)

    hMCtotal.SetMaximum(1.8)



    #if "CR" in dir2:

    #    hMCtotal.SetMinimum(0.4)     

    #    hMCtotal.SetMaximum(1.6)



    hMCtotal.GetXaxis().SetLabelFont(42)

    hMCtotal.GetXaxis().SetLabelOffset(0.007)

    hMCtotal.GetXaxis().SetLabelSize(0.04 * yscale)

    hMCtotal.GetXaxis().SetTitleFont(42)

    hMCtotal.GetXaxis().SetTitleSize(0.035 * yscale)

    hMCtotal.GetXaxis().SetTitleOffset(0.8)

    hMCtotal.GetYaxis().SetLabelFont(42)

    hMCtotal.GetYaxis().SetLabelOffset(0.007)

    hMCtotal.GetYaxis().SetLabelSize(0.03 * yscale)

    hMCtotal.GetYaxis().SetTitleFont(42)

    hMCtotal.GetYaxis().SetTitleSize(0.035 * yscale)

    hMCtotal.GetYaxis().SetTitleOffset(0.3)

    # Draw the ratio frame and the full-height blinded area before the
    # uncertainty band and data/MC points.
    hMCtotal.Draw("AXIS")

    if blind > 0:
        blinding_box2 = rt.TPave(
            hMCtotal.GetXaxis().GetBinLowEdge(blind),
            hMCtotal.GetMinimum(),
            hMCtotal.GetXaxis().GetXmax(),
            hMCtotal.GetMaximum(),
            0,
            "NB",
        )
        blinding_box2.SetFillColor(15)
        blinding_box2.SetFillStyle(3013)
        blinding_box2.Draw("same F")

    gtotal.Draw("2 0 same")



#    hratio.GetXaxis().SetLabelSize(0.13)

#    hratio.GetXaxis().SetTitleSize(0.13)

#    hratio.GetXaxis().SetTitleOffset(1.2)

#    hratio.SetMarkerSize(0.5)

    #hratio.GetYaxis().SetTitle("ratio")

#    hratio.SetLineWidth(2)



#add comparisons

#    dataToObsH=hdata.Clone("myratio") 

#    dataToObsH.Sumw2()

#    utils::root::checkSumw2(dataToObsH);

#    hmc=htotal.Clone("mc"); hmc.Sumw2()

#    dataToObsH.Divide(htotal) 



    dataToObs = rt.TGraphErrors(hdata.GetNbinsX())



    RPoint = 0  

    for i in range(1,hdata.GetXaxis().GetNbins()+1): 

       dataToObs.SetPoint(RPoint, hdata.GetBinCenter(i), hdata.GetBinContent(i)/htotal.GetBinContent(i)) 

       if (hdata.GetBinContent(i) != 0): 

          dataToObs.SetPointError(RPoint, hdata.GetBinWidth(i)/2, hdata.GetBinError(i)/hdata.GetBinContent(i)) 

          RPoint += 1 

       else:

          dataToObs.SetPointError(RPoint, hdata.GetBinWidth(i)/2, 0)

          RPoint += 1 ; continue

#    dataToObs = rt.TGraphErrors(dataToObsH)

#    dataToObs.SetMarkerColor(dataToObsH.GetMarkerColor())

#    dataToObs.SetMarkerStyle(dataToObsH.GetMarkerStyle())

#    dataToObs.SetMarkerSize(dataToObsH.GetMarkerSize())

    dataToObs.SetLineWidth(2)

    dataToObs.Draw("P 0 same")

#    dataToObsH.Draw("same e0") 

#    hratio.Draw("same e2")







    line = rt.TLine(hdata.GetXaxis().GetXmin(),1,hdata.GetXaxis().GetXmax(),1);

    line.SetLineColor(rt.kBlack)

    line.Draw("same")

    t2.RedrawAxis()



    t2.Update()



    canvas.cd()

#    canvas.SetLogy(True)

    canvas.Update()

    canvas.RedrawAxis()

    #frame = canvas.GetFrame()

    #frame.Draw()



    pdf_file = "%s/%s_%s" % (outdir, dir1, dir2)

    root_file = "%s/%s_%s" % (outdir, dir1, dir2)

    c_file = "%s/%s_%s" % (outdir, dir1, dir2)

    if ( do_linzoom ):

      pdf_file = pdf_file + "-linzoom"

    elif ( do_liny ):

       pdf_file = pdf_file + "-liny"

    else:

      pdf_file = pdf_file + "-logy"

      root_file = pdf_file + "-logy"

      c_file = pdf_file + "-logy"



    pdf_file = pdf_file + ".pdf"

    root_file = root_file + ".root"

    c_file = c_file + ".C"

    canvas.SaveAs( pdf_file )

    canvas.SaveAs( root_file )

    canvas.SaveAs( c_file )

    canvas.Update()














