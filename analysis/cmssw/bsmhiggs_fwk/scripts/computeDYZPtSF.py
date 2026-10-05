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

#hists_dy = ['alljets','3jets','4jets','5+jets']
hists = ['alljets','3jets','4jets','5+jets']   

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

def weightedAverage(ratio, hNLO, threshold, end=300):
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
    ratio.SetTitle("")
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(1.1)
    ratio.GetYaxis().SetTitle("Correction")
    ratio.GetYaxis().SetTitleSize(35)
    ratio.GetYaxis().SetTitleFont(43)
    ratio.GetYaxis().SetTitleOffset(1.1)
    ratio.GetYaxis().SetLabelFont(43)
    ratio.GetYaxis().SetLabelSize(30)
    ratio.GetXaxis().SetTitle("Pt(V)")
    ratio.GetXaxis().SetTitleOffset(1.0)
    ratio.GetXaxis().SetTitleSize(35)
    ratio.GetXaxis().SetTitleFont(43)
    ratio.GetXaxis().SetLabelFont(43)
    ratio.GetXaxis().SetLabelSize(30)
    
    fitf = r.TF1(name+"_f", "[0]", thred, 500)
#    fitf.SetLineColor(r.kBlack)
    fitf.SetLineWidth(2)
    res = ratio.Fit(fitf, "R S")
    ratio.GetYaxis().SetRangeUser(0, 2)

#    values = res.GetConfidenceIntervals(0.95, False)
#    print("values:")
#    print(len(values))
#    interval = r.TGraphErrors(len(values))
#    for i in range(len(values)):
#	if i == 0:
#	    xPos = ratio.GetXaxis().GetBinLowEdge(i+1) + thred
#	elif i == (len(values)-1):
#	    xPos = ratio.GetXaxis().GetBinLowEdge(i+2) + thred
#	else:
#	    xPos = ratio.GetXaxis().GetBinCenter(i+1) + thred
#	interval.SetPoint(i, xPos, fitf.Eval( xPos ))
#	interval.SetPointError(i, 0, values[i] )
#	print("{} : {}".format(i, values[i]))
#    interval.SetFillColor(r.kOrange)
#    interval.Draw("3")
    ratio.Draw("E0")

    leg = r.TLegend(0.6,0.79,0.89,0.89)
    leg.AddEntry(ratio, "Measured")
    leg.AddEntry(fitf, "Flatterning")
#    text = "#splitline{Flatterning(95% CL)}{#chi^{2} = " + "{:.2f}".format(fitf.GetChisquare())  + " }"
#    leg.AddEntry(interval, text)
    leg.SetBorderSize(0)
    leg.Draw("same")
    
    line = r.TLine(0, 1, 500, 1)
    line.SetLineWidth(2)
    line.Draw("same")
    SetOwnership( line, 0 )

    CMS_lumi.CMS_lumi(c, iPeriod, iPos)
    c.cd()
    c.Update()
    c.SaveAs(outdir + "/"+name+"_Fit.pdf")
    return c


def ratioPlot(hLO,hNLO,ratio_h,name):

    thred = 500
    if '2jets' in name: thred = 150
    elif '3jets' in name: thred = 200
    elif '4jets' in name: thred = 200
    elif '5+jets' in name: thred = 250
#    ratio_h = weightedAverage(ratio_h,hNLO,thred)
    
    # Define the Canvas
    c = r.TCanvas(name,name,800,800)
    
    # Upper plot will be in pad1
    hLO.SetLineColor(r.kBlue)
    hLO.SetLineWidth(2)
    hNLO.SetLineColor(r.kBlack)
    hNLO.SetLineWidth(2)
    
    ymin = min(hLO.GetMinimum(), hNLO.GetMinimum())
    ymax = 1.1*max(hLO.GetMaximum(), hNLO.GetMaximum())
    if ymin<0: ymin = 1.1*ymin
    else: ymin = 0.9*ymin
    hLO.GetYaxis().SetRangeUser(ymin,ymax)
    hLO.GetXaxis().SetRangeUser(0,400)
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
    hNLO.Draw("ehistsame")
    leg = r.TLegend(0.65,0.7,0.9,0.85)
    leg.AddEntry(hLO,"LO V+Jets","lp") 
    leg.AddEntry(hNLO,"NLO V+Jets","lp")
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

    ratio.SetTitle("")      # Remove the ratio title
    ratio.GetYaxis().SetTitle("NLO/LO ")
    ratio.GetYaxis().SetTitleSize(25)
    ratio.GetYaxis().SetTitleFont(43)
    ratio.GetYaxis().SetTitleOffset(1.55)
    ratio.GetYaxis().SetLabelFont(43)
    ratio.GetYaxis().SetLabelSize(25)
    ratio.GetXaxis().SetTitle("Pt(Z)")
    ratio.GetXaxis().SetTitleSize(25)
    ratio.GetXaxis().SetTitleFont(43)
    ratio.GetXaxis().SetTitleOffset(3)
    ratio.GetXaxis().SetLabelFont(43)
    ratio.GetXaxis().SetLabelSize(25)
    c1 = ratioOnly(ratio,thred, name+"_ratio")
    pad2.cd()
    if fit == "True":
        fitf = r.TF1(name+"_f", "[0]", thred, 500)
        ratio.Fit(name+"_f", "R0")
        res_fit = fitf.GetParameter(0)
        err_fit = fitf.GetParError(0)
        start = ratio.GetXaxis().FindBin(thred)
        for i in range(start,ratio.GetXaxis().GetNbins()+1):
            ratio.SetBinContent(i, res_fit)
	    ratio.SetBinError(i,err_fit)
   
    ratio.GetXaxis().SetRangeUser(0,400)
    ratio.GetYaxis().SetRangeUser(0.4,1.6)
    ratio.Draw("E0")
    r.gPad.Update()
    line = r.TLine(r.gPad.GetUxmin(), 1, r.gPad.GetUxmax(), 1)
    line.Draw("same")
    SetOwnership( line, 0 )
#    func = ratio.GetFunction(name+"_f")
    
#    return c,func
    return c, ratio, c1


def produceZptSFs(inputLO, inputNLO, output_name):
    print(output_name+":")
    inFileLO = r.TFile(inputLO, 'READ')
    inFileNLO = r.TFile(inputNLO, 'READ')
    output = r.TFile(output_name, 'RECREATE')
    isDY = (inputLO.find("DY") >= 0)

    for hist in ['alljets', 'alljets_w']:
	histLO = inFileLO.Get(hist+"_jetsMulti")
	histNLO = inFileNLO.Get(hist+"_jetsMulti")
	if histLO:
	    canvas = drawHist(histLO,hist+'Multiplicity_LO')
	    canvas.Write()
	    canvas.SaveAs(os.path.split(output_name)[0]+'/'+hist+'Multiplicity_LO.pdf')
	if histNLO:
	    canvas = drawHist(histNLO,hist+'Multiplicity_NLO')
	    canvas.Write()
	    canvas.SaveAs(os.path.split(output_name)[0]+'/'+hist+'Multiplicity_NLO.pdf')
    
#    for hist in ['alljets','3jets','4jets','5+jets','2b_3j_jets','2b_4j_jets','2b_geq5j_jets','3b_3j_jets','3b_4j_jets','3b_geq5j_jets','4b_4j_jets','4b_geq5j_jets','5b_geq5j_jets']:
#    if isDY: hists = hists_dy
#    else: hists = hists_wj
    for hist in hists:
	for ztype in ['']:
    	    histLO = inFileLO.Get(hist+ztype+"_ptw")
    	    histNLO = inFileNLO.Get(hist+ztype+"_ptw")
    	    if not (histLO and histNLO): continue
    	    histLO.SetDirectory(0)
    	    histNLO.SetDirectory(0)
	    if '4' in hist or '5' in hist:
		histLO.Rebin(2)
		histNLO.Rebin(2)
    	    histLO.Scale(1./abs(histLO.Integral(startbin, histLO.GetNbinsX())))
    	    histNLO.Scale(1./abs(histNLO.Integral(startbin, histLO.GetNbinsX())))
    	    ratios_out = histNLO.Clone(hist+ztype+'_sf')
	    ratios_out.Sumw2()
	    ratios_out.Divide(histLO)
    	    print("  "+hist+ztype+'_sf')
	    c,ratio,c1 = ratioPlot(histLO,histNLO, ratios_out, hist+ztype)
	    ratio.SetName(hist+ztype+'_sf')
#	    c = ratioPlot(histLO,histNLO, ratios_out, hist+ztype)
	    histLO.Write()
	    histNLO.Write()
    	    c.Write()
    	    c1.Write()
#	    f.Write()
	    c.SaveAs(os.path.split(output_name)[0]+'/'+hist+ztype+ ".pdf")
	    c1.SaveAs(os.path.split(output_name)[0]+'/'+hist+ztype+ "_ratio.pdf")
    	    ratio.Write()
    inFileLO.Close()
    inFileNLO.Close()
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

samplesDB=''
inputdir=''
outdir=''
onlytag='all'
fit = "False"

DtagsList = []
who = commands.getstatusoutput('whoami')[1]

for o,a in opts:
    if o in('-j'): samplesDB = a
    elif o in('-d'): inputdir = a
    elif o in('-o'): outdir = a
    elif o in('-t'): onlytag = a
    elif o in('-f'): fit = a

jsonFile = open(samplesDB,'r')
procList=json.load(jsonFile,encoding='utf-8').items()
print("fit = {}".format(fit))

inputdir += '/MC'

DYs = []
WJs = []
print "Only files with dtags matching " + onlytag + " are processed."
#run over sample to merge LO and NLO DY files
for proc in procList :
    #run over processes
    for desc in proc[1] :

        mctruthmode=getByLabel(desc,'mctruthmode',0)
        tag = getByLabel(desc,'tag','')
	# extract Z pt reweights only from LO and NLO DY samples
	if not (tag == "Z#rightarrow ll" or tag == "W#rightarrow l#nu"): continue 
#	if not (tag == "Z#rightarrow ll"): continue 
	
        data = desc['data']
        for d in data :
            origdtag = getByLabel(d,'dtag','')
	    if "10to50" in origdtag: continue
            dtag = origdtag
	    if(onlytag!='all') :
		if(dtag.find(onlytag)<0) : continue
            if(mctruthmode!=0) : dtag+='_filt'+str(mctruthmode)

            outfile = outdir +'/'+ dtag + '_' + 'ZPt.root'
            status, output = commands.getstatusoutput('ls '+inputdir+'/'+dtag+'_*.root')
            if status and not os.path.isfile(outfile) > 0 :
                print "!!!!! Warning: No root files for the dtag: " + origdtag
                continue
	    if tag == "Z#rightarrow ll": 
	        DYs.append(dtag)
#		hists = hists_dy
	    elif tag == "W#rightarrow l#nu": 
	        WJs.append(dtag)
#		hists = hists_wj
	    if os.path.isfile(outfile): 
		continue
            segment = 0
            for file in glob.glob(inputdir+'/'+dtag+'_*.root'):
                out_temp = outdir +'/'+ dtag + '_' + str(segment) + '_zpt.root'
                commands.getstatusoutput('rootcp --recreate '+file+':*jets* '+out_temp)
                segment += 1
            commands.getstatusoutput('hadd -f '+outfile+' '+outdir +'/'+ dtag + '_*' + '_zpt.root')
            #commands.getstatusoutput('hadd -f '+outfile+' '+inputdir+'/'+dtag+'_*.root')
            commands.getstatusoutput('rm -rf '+outdir +'/'+ dtag + '_*' + '_zpt.root')
	    
#	    if "amcNLO" in dtag: 
#		status, output = commands.getstatusoutput('find {} -name "{}*.root" | wc -l'.format(inputdir, dtag))
#		print(dtag+": "+str(iLumi/int(output)))
#		scaleinFile(outfile, iLumi/int(output), hists)
#	    else: 
            print(dtag+": "+str(iLumi))
            scaleinFile(outfile, iLumi, hists)

DY_LO_lowpt = ''
DY_LO_highpt = ''
DY_NLO_lowpt = ''
DY_NLO_highpt = ''
WJ_LO = ''
WJ_NLO = ''

for dtag in DYs:
    root_file = outdir +'/'+ dtag + '_' + 'ZPt.root'
    if 'amcNLO' in dtag:
        if '10to50' in dtag: DY_NLO_lowpt += (' ' + root_file)
        else: DY_NLO_highpt += (' ' + root_file)
    else:
        if '10to50' in dtag: DY_LO_lowpt += (' ' + root_file)
        else: DY_LO_highpt += (' ' + root_file)

for dtag in WJs:
    root_file = outdir +'/'+ dtag + '_' + 'ZPt.root'
    if 'amcNLO' in dtag: WJ_NLO += (' ' + root_file)
    else: WJ_LO += (' ' + root_file)

if len(DY_LO_lowpt) > 0: commands.getstatusoutput('hadd -f '+outdir +'/LODY_lowPt.root'+' '+DY_LO_lowpt)
if len(DY_LO_highpt) > 0: commands.getstatusoutput('hadd -f '+outdir +'/LODY_highPt.root'+' '+DY_LO_highpt)
if len(DY_NLO_lowpt) > 0: commands.getstatusoutput('hadd -f '+outdir +'/NLODY_lowPt.root'+' '+DY_NLO_lowpt)
if len(DY_NLO_highpt) > 0: commands.getstatusoutput('hadd -f '+outdir +'/NLODY_highPt.root'+' '+DY_NLO_highpt)

if len(WJ_LO) > 0:commands.getstatusoutput('hadd -f '+outdir +'/LOWJet.root'+' '+WJ_LO) 
if len(WJ_NLO) > 0:commands.getstatusoutput('hadd -f '+outdir +'/NLOWJet.root'+' '+WJ_NLO)  

if os.path.isfile(outdir +'/LOWJet.root') and os.path.isfile(outdir +'/NLOWJet.root'): produceZptSFs(outdir+'/LOWJet.root',outdir+'/NLOWJet.root',outdir+'/WSF.root')

if os.path.isfile(outdir +'/LODY_lowPt.root') and os.path.isfile(outdir +'/NLODY_lowPt.root'): produceZptSFs(outdir+'/LODY_lowPt.root',outdir+'/NLODY_lowPt.root',outdir+'/DYSF_lowPt.root')
if os.path.isfile(outdir +'/LODY_highPt.root') and os.path.isfile(outdir +'/NLODY_highPt.root'): produceZptSFs(outdir+'/LODY_highPt.root',outdir+'/NLODY_highPt.root',outdir+'/DYSF_highPt.root')
raw_input("Press [ENTER] to exit...")
