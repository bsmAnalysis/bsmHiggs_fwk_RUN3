#!/usr/bin/env python
import os,sys
import glob
import json
import ROOT as r
import getopt
import commands
import subprocess
from array import array
import math
from ROOT import SetOwnership

import CMS_lumi
CMS_lumi.writeExtraText = 1
CMS_lumi.extraText = "Preliminary"
iPos = 11
iPeriod = 0

bkgds = ["otherbkg", "ttbarbba", "ttbarcba", "ttbarlig", "zll", "wlnu", "ddqcd", "qcd"]
#startbin = 9 

low=-1
high=-1

def drawHist(hist, name):

    c = r.TCanvas(name,name,800,800)
    r.gPad.SetLogy(1)
    hist.SetStats(0)
    hist.SetLineColor(r.kBlack)
    hist.SetMarkerStyle(20)
    hist.SetLineWidth(2)
    hist.Draw("ehist")
    r.gPad.SetLogy(0)

    return c

def applyWeights(hist, name):
    thred_low = 0
    thred_high = 500
#    if vh_tag=="zh":
#        thred_low = int(low)
#	thred_high = int(high)
    for i in range(1, hist.GetNbinsX()+1):
        sf=1.0
	ht = hist.GetBinCenter(i)
	if ht>=thred_low and ht<=thred_high:
	    if '3b' in name:
		sf = math.exp(0.06117-0.00134*ht) # Wh 2016
		
	    elif '4b' in name:
		sf = math.exp(0.05567-0.00182*ht) # Wh 2016
		
	hist.SetBinContent(i, hist.GetBinContent(i)*sf)

        

def weightedAverage(ratio, hNLO, threshold, end=800):
    start = ratio.GetXaxis().FindBin(threshold)
    stop = ratio.GetXaxis().FindBin(end)
    mean_num = 0
    mean_den = 0
    error = 0
    for i in range(start, stop+1):
        weight = hNLO.GetBinContent(i)
	value = ratio.GetBinContent(i)
	err = ratio.GetBinError(i)
	mean_num += weight*value
	mean_den += weight
	error += weight**2*err**2
    if mean_den == 0: # if sum of weights is 0, then set zero to the error and mean
        mean_den = stop+1 - start
	error = 0
	mean_num = 0
    error = math.sqrt(error)/mean_den
    mean = mean_num/mean_den
    for i in range(start,ratio.GetXaxis().GetNbins()+1):
        ratio.SetBinContent(i,mean)
	ratio.SetBinError(i,error)
    return ratio

def ratioFit(ratio_h, name, outdir):
    c = r.TCanvas(name,name,800,800)
    ratio = ratio_h.Clone(name+"_95%CL")
    ratio.SetMaximum(min(2, max(1.3, 1.3*ratio.GetMaximum())))
    ratio.SetMinimum(max(0, 0.8*ratio.GetMinimum()))
#    ratio.GetXaxis().SetRangeUser(0,300)
    ratio.SetTitle(" ")
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(1.1)
    ratio.GetYaxis().SetTitle("Correction")
    ratio.GetYaxis().SetTitleSize(35)
    ratio.GetYaxis().SetTitleFont(43)
    ratio.GetYaxis().SetTitleOffset(1.1)
    ratio.GetYaxis().SetLabelFont(43)
    ratio.GetYaxis().SetLabelSize(30)
    ratio.GetXaxis().SetTitleOffset(1.5)
    ratio.GetXaxis().SetTitleSize(25)
    ratio.GetXaxis().SetTitleFont(43)
    ratio.GetXaxis().SetLabelFont(43)
    ratio.GetXaxis().SetLabelSize(30)

    thred = int(low)
    fitf = r.TF1(name+"_f_95%CL", "expo", thred, int(high))
#    fitf = r.TF1(name+"_f_95%CL", "expo", 0, 800)
    fitf.SetLineColor(r.kBlack)
    fitf.SetLineWidth(2)
    res = ratio.Fit(fitf, "R S")
    fitf.SetRange(0, 800)

    values = res.GetConfidenceIntervals(0.95, False)
    print("values:")
    print(len(values))
    interval = r.TGraphErrors(len(values))
    for i in range(len(values)):
	if i == 0:
	    xPos = thred + ratio.GetXaxis().GetBinLowEdge(i+1)
#	    interval.SetPoint(i, ratio.GetXaxis().GetBinLowEdge(i+1), fitf.Eval( ratio.GetXaxis().GetBinLowEdge(i+1) ))
	elif i == (len(values)-1):
	    xPos = thred + ratio.GetXaxis().GetBinLowEdge(i+2)
#	    interval.SetPoint(i, ratio.GetXaxis().GetBinLowEdge(i+2), fitf.Eval( ratio.GetXaxis().GetBinLowEdge(i+2) ))
	    #interval.SetPoint(i, 300, fitf.Eval( 300 ))
        else: 
	    xPos = thred + ratio.GetXaxis().GetBinCenter(i+1)
#	    interval.SetPoint(i, ratio.GetXaxis().GetBinCenter(i+1), fitf.Eval( ratio.GetXaxis().GetBinCenter(i+1) ))
	interval.SetPoint(i, xPos, fitf.Eval(xPos))
	interval.SetPointError(i, 0, values[i] )
    #interval.SetFillColor(r.kRed-9)
    interval.SetFillColor(r.kOrange)
    #interval.SetFillColor(r.kViolet-9)
    interval.Draw("3")
    ratio.Draw("same")
    fitf.SetRange(0, 800)
    fitf.Draw("same")

    leg = r.TLegend(0.6,0.74,0.89,0.89)
    leg.AddEntry(ratio, "Measured")
    text = "#splitline{Exponential Fit(95% CL)}{#chi^{2} = " + "{:.2f}".format(fitf.GetChisquare())  + " }"
    #leg.AddEntry(interval, "#splitline{Exponential Fit(95% CL)}{#chi^{2} = }", "p")
    leg.AddEntry(interval, text)
    leg.SetBorderSize(0)
    leg.Draw("same")

    CMS_lumi.CMS_lumi(c, iPeriod, iPos)
    c.cd()
    c.Update()
    
    c.SaveAs(os.path.join(outdir,"topPtSF_"+name[-2:]+"_Fit.pdf"))
#    raw_input("Press [ENTER] to exit...")
    return c

def ratioPlot(hLO,hNLO,ratio_h,name):
    global CC

    # Define the Canvas
    c = r.TCanvas(name,name,800,800)
    
    # Upper plot will be in pad1
    hLO.SetLineColor(r.kBlack)
    hLO.SetMarkerColor(r.kBlack)
    hLO.SetFillStyle(4000)
    hLO.SetLineWidth(2)
    hLO.SetTitle(name)
    hLO.SetMarkerStyle(21)
    hLO.SetMarkerSize(1)
    hNLO.SetLineColor(r.kBlue)
    hNLO.SetMarkerColor(r.kBlue)
    hNLO.SetFillStyle(4000)
    hNLO.SetLineWidth(2)
    hNLO.SetTitle(name)
    hNLO.SetMarkerStyle(21)
    hNLO.SetMarkerSize(1)
    
    ymin = min(hLO.GetMinimum(), hNLO.GetMinimum())
    ymax = 1.1*max(hLO.GetMaximum(), hNLO.GetMaximum())
    if ymin<0: ymin = 1.1*ymin
    else: ymin = 0.9*ymin
    hLO.GetYaxis().SetRangeUser(ymin,ymax)
#    hLO.GetXaxis().SetRangeUser(0,300)
    hLO.GetYaxis().SetTitleOffset(1.6)
    hLO.GetYaxis().SetTitleSize(25)
    hLO.GetYaxis().SetTitleFont(43)
    hLO.GetYaxis().SetLabelFont(43)
    hLO.GetYaxis().SetLabelSize(20)
    hLO.GetXaxis().SetTitleSize(25)
    hLO.GetXaxis().SetTitleFont(43)
    hLO.GetXaxis().SetLabelFont(43)
    hLO.GetXaxis().SetLabelSize(20)
    pad1 = r.TPad("pad1","pad1", 0., 0.3, 1, 1)
    pad1.SetBottomMargin(0) # Upper and lower plot are joined
#    pad1.SetGridx()         # Vertical grid
#    pad1.SetGridy()         # Vertical grid
    pad1.Draw()
    pad1.cd()		    # pad1 becomes the current pad
    hLO.SetStats(0)	    # No statistics on upper plot
    hLO.Draw("ehist")
    hNLO.Draw("ehistsame")
    leg = r.TLegend(0.65,0.75,0.9,0.85)
    leg.AddEntry(hLO,"Data - nonTop","lp")
    leg.AddEntry(hNLO,"Top","lp")
    leg.Draw("same")
    SetOwnership( leg, 0 ) # 0 = release (not keep), 1 = keep

    c.cd()		    # Go back to the main canvas before defining pad2
    pad2 = r.TPad("pad2", "pad2", 0., 0, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.2)
#    pad2.SetGridx()
    pad2.SetGridy()
    pad2.Draw()
    pad2.cd()

    # Define the ratio plot
    ratio = ratio_h.DrawCopy("ehist_"+name)
#    ratio.GetXaxis().SetRangeUser(0,300)
#    ratio.GetYaxis().SetRangeUser(0,2)
    ratio.GetYaxis().SetRangeUser(0.4,1.6)
#    ratio.GetYaxis().SetRangeUser(0.9*ratio.GetMinimum(),2)
#    ratio = hNLO.Clone(name+"_clone")
    ratio.SetLineColor(r.kBlack)
    ratio.SetLineWidth(2)
#    ratio.SetMinimum(0.8)
#    ratio.SetMaximum(1.35)
#    ratio.Sumw2()
    ratio.SetStats(0);      # No statistics on lower plot
#    ratio.Divide(hLO)
#    ratio.SetMarkerStyle(21)

    ratio.SetTitle("")      # Remove the ratio title
    ratio.GetYaxis().SetTitle("(Data-nonTop)/Top")
    ratio.GetYaxis().SetTitleSize(20)
    ratio.GetYaxis().SetTitleFont(43)
    ratio.GetYaxis().SetTitleOffset(1.5)
    ratio.GetYaxis().SetLabelFont(43)
    ratio.GetYaxis().SetLabelSize(20)
    ratio.GetXaxis().SetTitle("Pt(W)")
    ratio.GetXaxis().SetTitleSize(25)
    ratio.GetXaxis().SetTitleFont(43)
    ratio.GetXaxis().SetTitleOffset(2.5)
    ratio.GetXaxis().SetLabelFont(43)
    ratio.GetXaxis().SetLabelSize(20)

    pad2.cd()
    fitf = r.TF1(name+"_f", "exp([0]+[1]*x)", int(low), int(high))
    if CC=="False":
      print("name of function: {}".format(name+"_f"))
      ratio.Fit(name+"_f", "R")
      if showfull=="True":
	func = ratio.GetFunction(name+"_f")
        func.SetRange(0, 500)
    
    ratio.Draw("E0")
#    r.gPad.Update()
#    line = r.TLine(r.gPad.GetUxmin(), 1, r.gPad.GetUxmax(), 1)
#    line.Draw("same")
#    SetOwnership( line, 0 )
   
#    return c,func
    return c

def getHists(channels, inf):
    for ch in channels:
	data = inf.Get(ch+"/data_obs")
	if not data:
	    print("Error, cannot find data in the channel: {}".format(ch))
	    exit(-1)
	dataSub = data.Clone("dataSub_h_"+ch[-2:])
	for bkgd in bkgds:
	    if("ttbar" in bkgd): # add in ttbarMC
	    #if("ttbarlig" in bkgd): # add in ttbarMC
		h_mc = inf.Get(ch+"/"+bkgd)
		if h_mc:
		    if("ttbarMC" in vars() or "ttbarMC" in globals()): ttbarMC.Add(h_mc)
		    else: ttbarMC = h_mc.Clone("ttbarMC_h_"+ch[-2:])
	    else:   # add in nonttbarMC
		h_mc = inf.Get(ch+"/"+bkgd)
		if h_mc:
		    if("nonttbarMC" in vars() or "nonttbarMC" in globals()): nonttbarMC.Add(h_mc)
		    else: nonttbarMC = h_mc.Clone("nonttbarMC_h_"+ch[-2:])
	if not ("ttbarMC" in vars() or "ttbarMC" in globals()):
	    print("Error, cannot find ttbar process in the channel: {}".format(ch))
	    exit(-1)
	if("nonttbarMC" in vars() or "nonttbarMC" in globals()): dataSub.Add(nonttbarMC,-1)
	if("dataSub_ret" in vars() or "dataSub_ret" in globals()): dataSub_ret.Add(dataSub)
	else: dataSub_ret = dataSub.Clone("dataSub_ret_h_"+ch[-2:])
	if("ttbarMC_ret" in vars() or "ttbarMC_ret" in globals()): ttbarMC_ret.Add(ttbarMC)
	else: ttbarMC_ret = ttbarMC.Clone("ttbarMC_ret_h_"+ch[-2:])
	del ttbarMC,dataSub
	if("nonttbarMC" in vars() or "nonttbarMC" in globals()): del nonttbarMC

    return dataSub_ret, ttbarMC_ret

def computeTopPtSF(channels, inf, outdir, outf):
    global CC
    postfix=""
    if CC=="True": postfix="_CC"
    dataSub_h, ttbarMC_h = getHists(channels, inf)
    #if CC=="True":
    #    applyWeights(ttbarMC_h, "weights"+channels[0][-2:])
    weights = dataSub_h.Clone("weights"+channels[0][-2:]+postfix)
    weights.Divide(ttbarMC_h)
#    c = ratioPlot(dataSub_h, ttbarMC_h, weights, "topPtSF_"+channels[0][-2:]+postfix)
#    c.Write()
    
    c = ratioFit(weights, "topPtSF_"+channels[0][-2:], outdir)
#    c.SaveAs(os.path.join(outdir,"topPtSF_"+channels[0][-2:]+"_Fit.pdf"))
    c.Write()
    #c.SaveAs(os.path.join(outdir,"topPtSF_"+channels[0][-2:]+".pdf")+postfix)
    #outf.cd()
    #c.Write()


try:
     # retrive command line options
    shortopts  = "d:o:l:a:f:u:t:c:?" #RJ
    opts, args = getopt.getopt( sys.argv[1:], shortopts )
except getopt.GetoptError:
    # print help information and exit:
    print "ERROR: unknown options in argument %s" % sys.argv[1:]
#    usage()
    sys.exit(1)

CMSSW_BASE=os.environ.get('CMSSW_BASE')
if(len(CMSSW_BASE)==0):
    usage()
    sys.exit(1)

inputdir=''
outdir=''
vh_tag=''
CC="False" # do the closure check or not
showfull="True"

for o,a in opts:
    if o in('-d'): inputdir = a
    elif o in('-o'): outdir = a
    elif o in('-c'): vh_tag = a
    elif o in('-l'): low = a
    elif o in('-u'): high = a
    elif o in('-a'): CC = a
    elif o in('-f'): showfull = a

print("vh_tag = {}".format(vh_tag))
print("CC = {}".format(CC))

if vh_tag == "wh":
    hists_3b = ['e_A_CR_3b','mu_A_CR_3b']
    hists_4b = ['e_A_CR_4b','mu_A_CR_4b']
elif vh_tag == "zh":
    hists_3b = ['emu_A_CR_3b']
    hists_4b = ['emu_A_CR_4b']
else:
    print("Error, cannot recognize the tag: {}".format(vh_tag))
    print(vh_tag)
    exit(-1)

fpath = os.path.join(inputdir, "haa4b_60_13TeV_{}.root".format(vh_tag))
inf = r.TFile(fpath, "READ")
if not inf.IsOpen():
    print("The file cannot be opened, please check: " + fpath)
    exit(-1)


outname = "topSF.root"
if CC=="True": outname = "topSF_CC.root"
outf = r.TFile(os.path.join(outdir,outname), 'RECREATE')
######## 3b bin ##########
computeTopPtSF(hists_3b, inf, outdir, outf)

######## 4b bin ##########
computeTopPtSF(hists_4b, inf, outdir, outf)

inf.Close()
outf.Close()

#raw_input("Press [ENTER] to exit...")
