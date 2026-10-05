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

iLumi=35866.932
#iLumi=41529.152
#iLumi=59740.565

hists = ['ee_A_CR_3b', 'mumu_A_CR_3b', 'ee_A_SR_3b', 'mumu_A_SR_3b', 'ee_A_CR_4b', 'mumu_A_CR_4b', 'ee_A_SR_4b', 'mumu_A_SR_4b']

startbin = 1 

"""
Gets the value of a given item
(if not available a default value is returned)
"""
def getByLabel(desc,key,defaultVal=None) :
    try :
        return desc[key]
    except KeyError:
        return defaultVal

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

def weightedAverage(ratio, hNLO, threshold, end=5):
    start = ratio.GetXaxis().FindBin(threshold)
    stop = ratio.GetXaxis().FindBin(end)
    mean_num = 0
    mean_den = 0
    error = 0
    for i in range(start, stop+1):
        #weight = hNLO.GetBinContent(i)
        weight = 1
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

def scaleinFile(in_f, weight, hists):
    f = r.TFile(in_f, "UPDATE")
    for hist in hists:
	h = f.Get(hist+"_ptw")
	if h:
	    h.Scale(weight)
	    h.Write()
    f.Close()

def ratioOnly(ratio_h, thred, name):
    c = r.TCanvas(name,name,800,800)
    ratio = ratio_h.Clone(name)
#    ratio = ratio_h.DrawCopy(name)
    r.gPad.SetGridx()
    r.gPad.SetGridy()

    ratio.SetTitle("")
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(1.1)
    ratio.GetYaxis().SetTitle("Correction")
    ratio.GetYaxis().SetTitleSize(35)
    ratio.GetYaxis().SetTitleFont(43)
    ratio.GetYaxis().SetTitleOffset(1.1)
    ratio.GetYaxis().SetLabelFont(43)
    ratio.GetYaxis().SetLabelSize(30)
    ratio.GetXaxis().SetTitle("#Delta R (bb)^{ave}")
    ratio.GetXaxis().SetTitleOffset(1.0)
    ratio.GetXaxis().SetTitleSize(35)
    ratio.GetXaxis().SetTitleFont(43)
    ratio.GetXaxis().SetLabelFont(43)
    ratio.GetXaxis().SetLabelSize(30)
    
    fitf = r.TF1(name+"_f", "[0]", thred, 5)
#    fitf.SetLineColor(r.kBlack)
    fitf.SetLineWidth(2)
#    res = ratio.Fit(fitf, "R S")
    ratio.GetYaxis().SetRangeUser(0, 2)
    ratio.Draw("E0")

    leg = r.TLegend(0.6,0.79,0.89,0.89)
    leg.AddEntry(ratio, "Measured")
    leg.AddEntry(fitf, "Flattened")
#    text = "#splitline{Flatterning(95% CL)}{#chi^{2} = " + "{:.2f}".format(fitf.GetChisquare())  + " }"
#    leg.AddEntry(interval, text)
    leg.SetBorderSize(0)
    leg.Draw("same")
    
    line = r.TLine(0, 1, 5, 1)
    line.SetLineWidth(2)
    line.Draw("same")
    SetOwnership( line, 0 )

    CMS_lumi.CMS_lumi(c, iPeriod, iPos)
    c.cd()
    c.Update()
    c.SaveAs(outdir + "/"+name+"_Fit.pdf")
    return c


def ratioPlot(hLO,hNLO,ratio_h,name):

    thred = 3.8
#    ratio_h = weightedAverage(ratio_h,hNLO,thred)
    
    # Define the Canvas
    c = r.TCanvas(name,name,800,800)
    
    # Upper plot will be in pad1
    hLO.SetLineColor(r.kBlue)
    hLO.SetLineWidth(2)
    hLO.SetMarkerColor(r.kBlue)
    hNLO.SetLineColor(r.kBlack)
    hNLO.SetLineWidth(2)
    hNLO.SetMarkerColor(1)
    
    ymin = min(hLO.GetMinimum(), hNLO.GetMinimum())
    ymax = 1.1*max(hLO.GetMaximum(), hNLO.GetMaximum())
    if ymin<0: ymin = 1.1*ymin
    else: ymin = 0.9*ymin
    hLO.GetYaxis().SetRangeUser(ymin,ymax)
#    hLO.GetXaxis().SetRangeUser(0,400)
    hLO.GetYaxis().SetTitleOffset(1.55)
    hLO.GetYaxis().SetTitleSize(25)
    hLO.GetYaxis().SetTitleFont(43)
    hLO.GetYaxis().SetLabelFont(43)
    hLO.GetYaxis().SetLabelSize(25)
    hLO.GetXaxis().SetTitleSize(30)
    hLO.GetXaxis().SetTitleFont(43)
    hLO.GetXaxis().SetLabelFont(43)
    hLO.GetXaxis().SetLabelSize(25)
    pad1 = r.TPad("pad1","pad1", 0, 0.3, 1, 1)
    pad1.SetBottomMargin(0) # Upper and lower plot are joined
#    pad1.SetGridx()         # Vertical grid
#    pad1.SetGridy()         # Vertical grid
    pad1.Draw()
    pad1.cd()		    # pad1 becomes the current pad
    hLO.SetStats(0)	    # No statistics on upper plot
    hLO.Draw("ehist")
    hLO.SetFillColor(0)
    hNLO.SetFillColor(0)
    hNLO.Draw("ehistsame")
    leg = r.TLegend(0.65,0.7,0.9,0.85)
    leg.AddEntry(hLO,"2016 DY MC","l") 
    leg.AddEntry(hNLO,"2017/18 DY MC","l")
    leg.Draw("same")
    SetOwnership( leg, 0 ) # 0 = release (not keep), 1 = keep

    c.cd()		    # Go back to the main canvas before defining pad2
    pad2 = r.TPad("pad2", "pad2", 0, 0, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.25)
#    pad2.SetGridx()
    pad2.SetGridy()
    pad2.Draw()
    pad2.cd()

    # Define the ratio plot
    ratio = ratio_h.DrawCopy("ehist")
#    ratio.GetYaxis().SetRangeUser(ratio.GetMinimum()*0.8,1.2*ratio.GetMaximum())
    ratio.SetLineColor(r.kBlack)
    ratio.SetLineWidth(2)
#    ratio.SetMinimum(0.8)
#    ratio.SetMaximum(1.35)
    ratio.Sumw2()
    ratio.SetStats(0);      # No statistics on lower plot
    ratio.SetMarkerStyle(21)
    ratio.SetMarkerColor(1)

    ratio.SetTitle("")      # Remove the ratio title
    ratio.GetYaxis().SetTitle("201718/2016 ")
    ratio.GetYaxis().SetTitleSize(25)
    ratio.GetYaxis().SetTitleFont(43)
    ratio.GetYaxis().SetTitleOffset(1.55)
    ratio.GetYaxis().SetLabelFont(43)
    ratio.GetYaxis().SetLabelSize(25)
    ratio.GetXaxis().SetTitle("#Delta R (bb)^{ave}")
    ratio.GetXaxis().SetTitleSize(25)
    ratio.GetXaxis().SetTitleFont(43)
    ratio.GetXaxis().SetTitleOffset(3)
    ratio.GetXaxis().SetLabelFont(43)
    ratio.GetXaxis().SetLabelSize(25)
    c1 = ratioOnly(ratio,thred, name+"_ratio")
    pad2.cd()
    if fit == "True":
        fitf = r.TF1(name+"_f", "[0]", thred, 5)
        ratio.Fit(name+"_f", "R0")
        res_fit = fitf.GetParameter(0)
        err_fit = fitf.GetParError(0)
        print "For " + name + " , par= " + str(res_fit) + " and err= " + str(err_fit) 
        start = ratio.GetXaxis().FindBin(thred)
        for i in range(start,ratio.GetXaxis().GetNbins()+1):
            ratio.SetBinContent(i, res_fit)
	    ratio.SetBinError(i,err_fit)

#    ratio.GetXaxis().SetRangeUser(0,5)
    ratio.GetYaxis().SetRangeUser(0.4,1.6)
    ratio.Draw("E0")
    r.gPad.Update()
    line = r.TLine(r.gPad.GetUxmin(), 1, r.gPad.GetUxmax(), 1)
    line.Draw("same")
    SetOwnership( line, 0 )
#    func = ratio.GetFunction(name+"_f")
    
#    return c,func
    return c, ratio, c1


def produceDRSFs(input16, input17, output_name):
    print(output_name+":")
    inFile16 = r.TFile(input16, 'READ')
    inFile17 = r.TFile(input17, 'READ')
    output = r.TFile(output_name, 'RECREATE')
#    isDY = (input16.find("DY") >= 0)

    for hist in hists:
	for ztype in ['']:
    	    hist16 = inFile16.Get("Z#rightarrow ll/"+hist+ztype+"_dRave")
    	    hist17 = inFile17.Get("Z#rightarrow ll/"+hist+ztype+"_dRave")
    	    if not (hist16 and hist17): 
                print("Could not find DRave histos.. exiting ")
                exit
            hist16.Rebin(2)
            hist17.Rebin(2)
    	    hist16.Scale(1./abs(hist16.Integral(startbin, hist16.GetNbinsX())))
    	    hist17.Scale(1./abs(hist17.Integral(startbin, hist17.GetNbinsX())))
            hist16.SetDirectory(0)   
            hist17.SetDirectory(0)     
    	    ratios_out = hist17.Clone(hist+ztype+'_sf')
	    ratios_out.Sumw2()
	    ratios_out.Divide(hist16)
    	    print("  "+hist+ztype+'_sf')
	    c,ratio,c1 = ratioPlot(hist16,hist17, ratios_out, hist+ztype)
	    ratio.SetName(hist+ztype+'_sf')
#	    c = ratioPlot(hist16,hist17, ratios_out, hist+ztype)
	    hist16.Write()
	    hist17.Write()
    	    c.Write()
    	    c1.Write()
#	    f.Write()
	    c.SaveAs(os.path.split(output_name)[0]+'/'+hist+ztype+ ".pdf")
	    c1.SaveAs(os.path.split(output_name)[0]+'/'+hist+ztype+ "_ratio.pdf")
    	    ratio.Write()
    inFile16.Close()
    inFile17.Close()
    output.Close()

try:
     # retrive command line options
    shortopts  = "s:e:j:d:f:o:c:l:p:t:g:r:?" #RJ
    opts, args = getopt.getopt( sys.argv[1:], shortopts )
except getopt.GetoptError:
    # print help information and exit:
    print "ERROR: unknown options in argument %s" % sys.argv[1:]
    usage()
    sys.exit(1)

CMSSW_BASE=os.environ.get('CMSSW_BASE')
if(len(CMSSW_BASE)==0):
    usage()
    sys.exit(1)

inputdir=CMSSW_BASE+'/src/UserCode/bsmhiggs_fwk/test/haa4b'
outdir=CMSSW_BASE+'/src/UserCode/bsmhiggs_fwk/test/haa4b/Stephanes_test'

commands.getstatusoutput('mkdir '+outdir)

fit = "True"

who = commands.getstatusoutput('whoami')[1]

#for o,a in opts:
#    elif o in('-d'): inputdir = a
#    elif o in('-o'): outdir = a
#    elif o in('-f'): fit = a
print("Inputdir = "+inputdir)

if os.path.isfile(inputdir +'/plotter_ZH_2016_2020_06_19_forLimits_v1.root') and os.path.isfile(inputdir +'/plotter_ZH_201718_2020_02_05_forLimits.root'): 
    produceDRSFs(inputdir+'/plotter_ZH_2016_2020_06_19_forLimits_v1.root',inputdir+'/plotter_ZH_201718_2020_02_05_forLimits.root',outdir+'/drSF.root')


raw_input("Press [ENTER] to exit...")
