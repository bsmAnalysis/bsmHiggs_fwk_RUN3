import ROOT as rt
import CMS_lumi, tdrstyle
import array
import os

from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double

#set the tdr style
tdrstyle.setTDRStyle()

#change the CMS_lumi variables (see CMS_lumi.py)
CMS_lumi.lumi_7TeV = "4.8 fb^{-1}"
CMS_lumi.lumi_8TeV = "18.3 fb^{-1}"
CMS_lumi.writeExtraText = 1
CMS_lumi.extraText = "Preliminary"
CMS_lumi.lumi_sqrtS = "" #"13 TeV" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)

iPos = 11
if( iPos==0 ): CMS_lumi.relPosX = 0.12

H_ref = 700; 
W_ref = 700; 
W = W_ref
H  = H_ref

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

iPeriod = 0

# references for T, B, L, R
T = 0.08*H_ref
B = 0.12*H_ref 
L = 0.12*W_ref
R = 0.04*W_ref

canvas = rt.TCanvas("c2","c2",800,800) #50,50,W,H)
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

h =  rt.TH1F("h","h; BDT; Events",5, 0, 5)

xAxis = h.GetXaxis()
xAxis.SetNdivisions(6,5,0)

yAxis = h.GetYaxis()
yAxis.SetNdivisions(6,5,0)
yAxis.SetTitleOffset(1)



#blind=-1
blind=3

limit_dir="plots_2018_09_19/limit_final/"

file = rt.TFile(limit_dir+"cards_SB13TeV_SM_Wh/0060/fitDiagnostics.root","READ")

dir1="shapes_prefit"
#dir1="shapes_fit_b"
dir2="e_A_SR_3b"
dir=dir1+"/"+dir2+"/"

data = file.Get(dir+"data")
data.SetMarkerStyle(20)
data.SetMarkerColor(1)

total = file.Get(dir+"total_background")

sig = file.Get(dir+"wh")
sig.SetLineColor(rt.kRed+4)
sig.SetLineWidth(2)
sig.SetLineStyle(2)
sig.Scale(50)

qcd = file.Get(dir+"ddqcd")
qcd.SetFillColor(634)
qcd.SetLineColor(1)
singleto = file.Get(dir+"singleto")
singleto.SetFillColor(424)
singleto.SetLineColor(1)

ttbargam = file.Get(dir+"ttbargam")
ttbargam.SetFillColor(424)
ttbargam.SetLineColor(1)

#vv = file.Get(dir+"vv")
#vv.SetFillColor(595)
#vv.SetLineColor(1)
zvv = file.Get(dir+"zvv")
zvv.SetFillColor(17)
zvv.SetLineColor(1)

zll = file.Get(dir+"zll")
zll.SetFillColor(624)
zll.SetLineColor(1)
wlnu = file.Get(dir+"wlnu")
wlnu.SetFillColor(622)
wlnu.SetLineColor(1)
#qcd = file.Get(dir+"qcd")

ttbarbba = file.Get(dir+"ttbarbba")
ttbarbba.SetFillColor(833)
ttbarbba.SetLineColor(1)

ttbarcba = file.Get(dir+"ttbarcba")
ttbarcba.SetFillColor(408)
ttbarcba.SetLineColor(1)

ttbarlig = file.Get(dir+"ttbarlig")
ttbarlig.SetFillColor(406)
ttbarlig.SetLineColor(1)


MC = rt.THStack()

MC.Add(ttbargam)

#MC.Add(vv)
MC.Add(zvv)
MC.Add(zll)
MC.Add(wlnu)

MC.Add(qcd)
MC.Add(singleto)

MC.Add(ttbarbba)
MC.Add(ttbarcba)
MC.Add(ttbarlig)
#MC.Add(qcd)

t1 = rt.TPad("t1","t1", 0.0, 0.2, 1.0, 1.0)
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

hdata = rt.TH1F("hdata","data bdt",5, 0, 5)

hdata.SetNdivisions(5)

#xmin=-0.3
#for i in range(0,hdata.GetXaxis().GetNbins()):
#    label=str(xmin+i*hdata.GetXaxis().GetBinWidth(i))
#    hdata.GetXaxis().SetBinLabel(i,label)
    
#hdata.Rebin(rbin)
htotal=total.Clone()

px, py = Double(), Double()
pyerr = Double()

nPoints=data.GetN()
for i in range(0,nPoints):
    data.GetPoint(i,px,py)
    pyerr=data.GetErrorY(i)
    hdata.Fill(px,py)
    hdata.SetBinError(i,pyerr)



hdata.GetXaxis().SetLabelOffset(0.007);
hdata.GetXaxis().SetLabelSize(0.04);
hdata.GetXaxis().SetTitleOffset(1.2);
hdata.GetXaxis().SetTitleFont(42);
hdata.GetXaxis().SetTitleSize(0.04);
hdata.GetYaxis().SetLabelFont(42);
hdata.GetYaxis().SetLabelOffset(0.007);
hdata.GetYaxis().SetLabelSize(0.04);
hdata.GetYaxis().SetTitleOffset(1.35);
hdata.GetYaxis().SetTitleFont(42);
hdata.GetYaxis().SetTitleSize(0.04);

hdata.GetYaxis().SetTitle("Events");
hdata.GetXaxis().SetTitle("BDT");
hdata.Draw();

t1.Update();

    
MC.Draw("histsame")
hdata.Draw("epsame0")

sig.Draw("histsame")

hdata.SetMaximum(1.5*MC.GetMaximum())


if (blind>0):
    for i in range(hdata.FindBin(blind),hdata.GetNbinsX()+1):
        hdata.SetBinContent(i,0)
        hdata.SetBinError(i,0)
    blinding_box = rt.TPave(hdata.GetBinLowEdge(hdata.FindBin(blind)),  hdata.GetMinimum(), hdata.GetXaxis().GetXmax(), hdata.GetMaximum(), 0, "NB" )  
    blinding_box.SetFillColor(15)
    blinding_box.SetFillStyle(3013)
    blinding_box.Draw("same F");

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

legend =  rt.TLegend(0.40,0.74,0.93,0.96, "NDC")
#legend.SetFillColor( rt.kGray )
#legend.SetHeader(dir)
legend.SetNColumns(3)  
legend.SetBorderSize(0)
legend.SetTextFont(42)
legend.SetTextSize(0.03)
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
legend.AddEntry(zvv,"(V)VV", "LF")
legend.AddEntry(wlnu,"W#rightarrow l#nu","LF")
legend.AddEntry(zll,"Z#rightarrow ll","LF")
legend.AddEntry(ttbarbba,"t#bar{t} + b#bar{b}","LF")
legend.AddEntry(ttbarcba,"t#bar{t} + c#bar{c}","LF")
legend.AddEntry(ttbarlig,"t#bar{t} + light","LF")
legend.AddEntry(singleto,"Single Top","LF")
legend.AddEntry(qcd,"DD qcd","LF")
legend.AddEntry(sig,"Wh (60)","L")

if(blind>0):
    legend.AddEntry(blinding_box,"Blinded area","F")

legend.Draw("same")

#draw the lumi text on the canvas
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)



## ratio plot
t2 = rt.TPad("t2", "t2",0.0,0.0, 1.0,0.2)
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



hratio=hdata.Clone("myratio")
hratio.Divide(htotal)

hratio.Draw("e2")
#hratio.DrawCopy("histsame"); 
hratio.SetFillColor(rt.kBlue);
hratio.SetFillStyle(3018);
hratio.Draw("same 2 0");
 


#hratio.SetFillColor(rt.kMagenta)

#hratio.Draw("e2")
#hratio.Draw("hist L same"); # you missed the option L
#hratio.Draw("same 2 0")


hratio.SetMinimum(0.4);
hratio.SetMaximum(1.6);

hratio.SetLineColor(1)
hratio.SetFillStyle(3005)
hratio.SetFillColor(rt.kGray+3)
hratio.SetMarkerColor(1)
hratio.SetMarkerStyle(20)
hratio.GetYaxis().SetLabelSize(0.13)
hratio.GetYaxis().SetTitleSize(0.13)
hratio.GetYaxis().SetTitleOffset(1.)
hratio.GetXaxis().SetLabelSize(0.13)
hratio.GetXaxis().SetTitleSize(0.13)
hratio.GetXaxis().SetTitleOffset(1.2)
hratio.SetMarkerSize(0.5)
#hratio.GetYaxis().SetTitle("ratio")
hratio.SetLineWidth(2)

if(blind>0):
    blinding_box2 = rt.TPave(hdata.GetBinLowEdge(hdata.FindBin(blind)), 0.4, hdata.GetXaxis().GetXmax(), 1.6, 0, "NB" )  
    blinding_box2.SetFillColor(15)
    blinding_box2.SetFillStyle(3013)
    blinding_box2.Draw("same F");


line = rt.TLine(0.,1,hdata.GetXaxis().GetXmax(),1);
line.SetLineColor(rt.kBlack)
line.Draw("same")

t2.Update()

canvas.cd()
canvas.Update()
canvas.RedrawAxis()
#frame = canvas.GetFrame()
#frame.Draw()

canvas.SaveAs(limit_dir+dir1+"_"+dir2+".pdf")

canvas.Update()

raw_input("Press Enter to end")
