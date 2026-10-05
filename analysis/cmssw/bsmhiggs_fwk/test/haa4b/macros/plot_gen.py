import ROOT as r
import sys

r.gROOT.SetBatch()
r.gStyle.SetOptStat(0)

infile1, infile2, infile3, outstub = sys.argv[1:]

fun1 = r.TF1("fun1", "x", 0, 400)
fun1.SetLineColor(r.kMagenta)

r.gStyle.SetTitleFontSize(.06)
r.gStyle.SetTitleXSize(.05)
r.gStyle.SetTitleYSize(.05)
r.gStyle.SetPadBottomMargin(.13)
r.gStyle.SetPadLeftMargin(.12)
r.gStyle.SetPadRightMargin(.09)
r.gStyle.SetPadTopMargin(.1)
r.gStyle.SetPalette(1)

accpt = "( b1Pt>15 && b2Pt>15 && b3Pt>15 && b4Pt>15 )"
acceta = "( abs(b1Eta)<2.5 && abs(b2Eta)<2.5 && abs(b3Eta)<2.5 && abs(b4Eta)<2.5 )"


for m in (infile1, infile2, infile3):
    outfile = "{}{}.pdf".format(outstub, m)

    f = r.TFile(m)
    tps=f.Get("t")
    
    c = r.TCanvas()
    c.SetLogz(True)
    c.SetRightMargin(c.GetRightMargin() * 1.5)
    c.SaveAs(outfile + '[')
    
    r.TH1.SetDefaultSumw2()
    
    r.gPad.SetGridx();
    r.gPad.SetGridy()

    # pT(H) vs a
    tps.Draw("a2Pt:hPt>>h_1(100,0,400,100,0,400)","","COLZ")
    h1=r.gDirectory.Get("h_1")
    h1.SetTitle("p_{T}(H)  vs p_{T}(a);p_{T}(H) (GeV); p_{T}(a)")
    fun1.Draw("SAME")
    c.SaveAs(outfile)
    
    tps.Draw("theta_a2:hPt>>h_2(100,0,400,100,0,3.1415","","COLZ")
    h2=r.gDirectory.Get("h_2")
    h2.SetTitle("p_{T}(H)  vs opening angle(a);p_{T}(H) (GeV); #theta(a)")
    c.SaveAs(outfile)
    
    tps.Draw("dRaa:hPt>>h_3(100,0,400,100,0,6)","","COLZ")
    h3=r.gDirectory.Get("h_3")
    h3.SetTitle("p_{T}(H)  vs #DeltaR(aa);p_{T}(H) (GeV); #DeltaR(aa)")
    c.SaveAs(outfile)
    
    #pT(a) vs b
    tps.Draw("b1Pt:a2Pt>>hh_1(100,0,400,100,0,400)","","COLZ")
    h1=r.gDirectory.Get("hh_1")
    h1.SetTitle("p_{T}(a)  vs p_{T}(b);p_{T}(a) (GeV); p_{T}(b)")
    fun1.Draw("SAME")
    c.SaveAs(outfile)
    
    tps.Draw("theta_b1:a2Pt>>hh_2(100,0,400,100,0,3.1415","","COLZ")
    h2=r.gDirectory.Get("hh_2")
    h2.SetTitle("p_{T}(a)  vs opening angle(b);p_{T}(H) (GeV); #theta(b)")
    c.SaveAs(outfile)
    
    tps.Draw("dRbb1:a2Pt>>hh_3(100,0,400,100,0,6)","","COLZ")
    h3=r.gDirectory.Get("hh_3")
    h3.SetTitle("p_{T}(a)  vs #DeltaR(bb);p_{T}(a) (GeV); #DeltaR(bb)")
    c.SaveAs(outfile)

    ## Add pt(b) min and max for each mass superimposed
    f1 = r.TFile(infile1)
    tps1 = f1.Get("t")

    f2 = r.TFile(infile2)
    tps2 = f2.Get("t")

    f3 = r.TFile(infile3)
    tps3 = f3.Get("t")

    r.gPad.SetLogy()
    r.gPad.SetGridx(0)
    r.gPad.SetGridy(0)

    # pT(b) max
    tps1.Draw("maxbPt>>h_0(100,0,400)","","HIST")
    h0=r.gDirectory.Get("h_0")
    h0.SetStats(1)
    h0.GetXaxis().SetTitle("p_{T}(b)^{max}")
    tps2.Draw("maxbPt>>h_1(100,0,400)","","HISTSAME")
    h1=r.gDirectory.Get("h_1")
    h1.SetLineColor(2)
    tps3.Draw("maxbPt>>h_2(100,0,400)","","HISTSAME")
    h2=r.gDirectory.Get("h_2")
    h2.SetLineColor(1)

    leg=r.TLegend()
    #leg=r.gDirectory.Get("leg")
    leg.AddEntry(h0,"m(a)=15 GeV","L")
    leg.AddEntry(h1,"m(a)=30 GeV","L")
    leg.AddEntry(h2,"m(a)=50 GeV","L")
    leg.Draw("SAME")

    c.SaveAs(outfile)
    
    # pT(b) min
    tps1.Draw("minbPt>>h_0(100,0,400)","","HIST")
    h0=r.gDirectory.Get("h_0")
    h0.SetStats(1)
    h0.GetXaxis().SetTitle("p_{T}(b)^{min}")
    tps2.Draw("minbPt>>h_1(100,0,400)","","HISTSAME")
    h1=r.gDirectory.Get("h_1")
    h1.SetLineColor(2)
    tps3.Draw("minbPt>>h_2(100,0,400)","","HISTSAME")
    h2=r.gDirectory.Get("h_2")
    h2.SetLineColor(1)

    leg=r.TLegend()
    #leg=r.gDirectory.Get("leg")
    leg.AddEntry(h0,"m(a)=15 GeV","L")
    leg.AddEntry(h1,"m(a)=30 GeV","L")
    leg.AddEntry(h2,"m(a)=50 GeV","L")
    leg.Draw("SAME")

    c.SaveAs(outfile)
    
    #DR(aa)
    r.gPad.SetLogy(0)
    tps1.Draw("dRaa>>h01(100,0,6)","","HIST")
    h01=r.gDirectory.Get("h01")
    h01.GetXaxis().SetTitle("#Delta R(aa)")
    #tps.Draw("dRaa>>h02(100,0,6)",accpt+" * "+acceta,"HISTSAME")
    tps2.Draw("dRaa>>h02(100,0,6)","","HISTSAME")
    h02=r.gDirectory.Get("h02")
    h02.SetLineColor(2)
    tps3.Draw("dRaa>>h03(100,0,6)","","HISTSAME")
    h03=r.gDirectory.Get("h03")
    h03.SetLineColor(1)

    leg=r.TLegend()
    #leg=r.gDirectory.Get("leg")
    leg.AddEntry(h01,"m(a)=15 GeV","L")
    leg.AddEntry(h02,"m(a)=30 GeV","L")
    leg.AddEntry(h03,"m(a)=50 GeV","L")
    leg.Draw("SAME")
    c.SaveAs(outfile)
    
    
    #DR(bb1)
    r.gPad.SetLogy(0)
    tps1.Draw("dRbb1>>h04(100,0,6)","","HIST")
    h04=r.gDirectory.Get("h04")
    h04.GetXaxis().SetTitle("#Delta R(bb)")
    tps2.Draw("dRbb1>>h05(100,0,6)","","HISTSAME")
    h05=r.gDirectory.Get("h05")
    h05.SetLineColor(2)
    tps3.Draw("dRbb1>>h06(100,0,6)","","HISTSAME")
    h06=r.gDirectory.Get("h06")
    h06.SetLineColor(1)

    leg=r.TLegend()
    #leg=r.gDirectory.Get("leg")
    leg.AddEntry(h04,"m(a)=15 GeV","L")
    leg.AddEntry(h05,"m(a)=30 GeV","L")
    leg.AddEntry(h06,"m(a)=50 GeV","L")
    leg.Draw("SAME")
    c.SaveAs(outfile)

     #DR(bb2)
    r.gPad.SetLogy(0)
    tps1.Draw("dRbb2>>h04(100,0,6)","","HIST")
    h04=r.gDirectory.Get("h04")
    h04.GetXaxis().SetTitle("#Delta R(bb)")
    tps2.Draw("dRbb2>>h05(100,0,6)","","HISTSAME")
    h05=r.gDirectory.Get("h05")
    h05.SetLineColor(2)
    tps3.Draw("dRbb2>>h06(100,0,6)","","HISTSAME")
    h06=r.gDirectory.Get("h06")
    h06.SetLineColor(1)

    leg=r.TLegend()
    #leg=r.gDirectory.Get("leg")
    leg.AddEntry(h04,"m(a)=15 GeV","L")
    leg.AddEntry(h05,"m(a)=30 GeV","L")
    leg.AddEntry(h06,"m(a)=50 GeV","L")
    leg.Draw("SAME")
    c.SaveAs(outfile)

    
    c.SaveAs(outfile + ']')
    
