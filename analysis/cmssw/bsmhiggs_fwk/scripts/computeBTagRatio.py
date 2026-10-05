#!/usr/bin/env python
import os,sys
import glob
import json
import ROOT as r
import getopt
import commands
import subprocess
from array import array
"""
Gets the value of a given item
(if not available a default value is returned)
"""
def getByLabel(desc,key,defaultVal=None) :
    try :
        return desc[key]
    except KeyError:
        return defaultVal
def produceEfficiencyMaps(binxX, binsY, inputPath, outputPath, useDeepCSV):
    inputFile = r.TFile(inputPath, 'READ')
    outputFile = r.TFile(outputPath, 'RECREATE')
    #csvTag = "DeepCSV"
    #if(not useDeepCSV): csvTag = "CSV"
    for partonFlavor in ['b', 'c', 'udsg']:
        #denominatorHisto = csvTag + '_BTaggingEff_Denom_' + partonFlavor
        denominatorHisto = 'BTaggingEff_Denom_' + partonFlavor
        denominatorIn = inputFile.Get(denominatorHisto)
        if not denominatorIn: continue
        print(denominatorHisto)
        xShift = denominatorIn.GetXaxis().GetBinWidth(1)/2.
        yShift = denominatorIn.GetYaxis().GetBinWidth(1)/2.
        #denominatorOut = r.TH2D(csvTag+'_denominator_' + partonFlavor, '', (len(binsX)-1), binsX, (len(binsY)-1), binsY)
        denominatorOut = r.TH2D('denominator_' + partonFlavor, '', (len(binsX)-1), binsX, (len(binsY)-1), binsY)
	#for method in ['CSV','DeepCSV']:
	for method in ['','DeepCSV']:
    	    for WP in ['Loose', 'Medium', 'Tight']:
                #numeratorHisto = csvTag + '_' + WP + 'BTaggingEff_Num_' + partonFlavor
                numeratorHisto = method + WP + 'BTaggingEff_Num_' + partonFlavor
                numeratorIn = inputFile.Get(numeratorHisto)
                if not numeratorIn: continue
                print(numeratorHisto)
                #numeratorOut   = r.TH2D(csvTag+'_'+WP+'_numerator_' + partonFlavor, '', (len(binsX)-1), binsX, (len(binsY)-1), binsY)
                numeratorOut   = r.TH2D(WP+'_numerator_' + partonFlavor, '', (len(binsX)-1), binsX, (len(binsY)-1), binsY)
                #efficiencyOut  = r.TH2D(csvTag+'_'+WP+'_efficiency_' + partonFlavor, '', (len(binsX)-1), binsX, (len(binsY)-1), binsY)
                efficiencyOut  = r.TH2D(WP+'_efficiency_' + partonFlavor, '', (len(binsX)-1), binsX, (len(binsY)-1), binsY)
                for i in range(1,denominatorOut.GetXaxis().GetNbins()+1):
                    for j in range(1,denominatorOut.GetYaxis().GetNbins()+1):
                        binXMin = denominatorIn.GetXaxis().FindBin(denominatorOut.GetXaxis().GetBinLowEdge(i)+xShift)
                        binXMax = denominatorIn.GetXaxis().FindBin(denominatorOut.GetXaxis().GetBinUpEdge(i)-xShift)
                        binYMinPos = denominatorIn.GetYaxis().FindBin(denominatorOut.GetYaxis().GetBinLowEdge(j)+yShift)
                        binYMaxPos = denominatorIn.GetYaxis().FindBin(denominatorOut.GetYaxis().GetBinUpEdge(j)-yShift)
                        binYMinNeg = denominatorIn.GetYaxis().FindBin(-denominatorOut.GetYaxis().GetBinUpEdge(j)+yShift)
                        binYMaxNeg = denominatorIn.GetYaxis().FindBin(-denominatorOut.GetYaxis().GetBinLowEdge(j)-yShift)
    
                        denominator = denominatorIn.Integral(binXMin,binXMax,binYMinPos,binYMaxPos)
                        denominator = denominator + denominatorIn.Integral(binXMin,binXMax,binYMinNeg,binYMaxNeg)
                        numerator = numeratorIn.Integral(binXMin,binXMax,binYMinPos,binYMaxPos)
                        numerator = numerator + numeratorIn.Integral(binXMin,binXMax,binYMinNeg,binYMaxNeg)
    
                        if(i==denominatorOut.GetXaxis().GetNbins()): # also add overflow to the last bin in jet pT
                            denominator = denominator + denominatorIn.Integral(binXMax+1,denominatorIn.GetXaxis().GetNbins()+1,binYMinPos,binYMaxPos)
                            denominator = denominator + denominatorIn.Integral(binXMax+1,denominatorIn.GetXaxis().GetNbins()+1,binYMinNeg,binYMaxNeg)
                            numerator = numerator + numeratorIn.Integral(binXMax+1,numeratorIn.GetXaxis().GetNbins()+1,binYMinPos,binYMaxPos)
                            numerator = numerator + numeratorIn.Integral(binXMax+1,numeratorIn.GetXaxis().GetNbins()+1,binYMinNeg,binYMaxNeg)
    
                        denominatorOut.SetBinContent(i,j,denominator)
                        numeratorOut.SetBinContent(i,j,numerator)
                        if(denominator>0.): efficiencyOut.SetBinContent(i,j,numerator/denominator)
                # check if there are any bins with 0 or 100% efficiency
                for i in range(1,denominatorOut.GetXaxis().GetNbins()+1):
                    for j in range(1,denominatorOut.GetYaxis().GetNbins()+1):
                        efficiency = efficiencyOut.GetBinContent(i,j)
                        if(efficiency==0. or efficiency==1.):
                            print 'Warning! Bin(%i,%i) for %s jets has a b-tagging efficiency of %.3f'%(i,j,partonFlavor,efficiency)
                for i in range(1,denominatorOut.GetXaxis().GetNbins()+1):
                    efficiencyOut.SetBinContent(i, denominatorOut.GetYaxis().GetNbins()+1, efficiencyOut.GetBinContent(i, denominatorOut.GetYaxis().GetNbins()))
                for j in range(1,denominatorOut.GetYaxis().GetNbins()+2):
                    efficiencyOut.SetBinContent(denominatorOut.GetXaxis().GetNbins()+1, j, efficiencyOut.GetBinContent(denominatorOut.GetXaxis().GetNbins(), j))
                
                #outputFile.cd()
                #numeratorOut.Write()
                efficiencyOut.Write()
            #denominatorOut.Write()
    inputFile.Close()
    outputFile.Close()


try:
     # retrive command line options
    shortopts  = "s:e:j:d:o:c:l:p:t:g:r:?" #RJ
    opts, args = getopt.getopt( sys.argv[1:], shortopts )
except getopt.GetoptError:
    # print help information and exit:
    print "ERROR: unknown options in argument %s" % sys.argv[1:]
    usage()
    sys.exit(1)

samplesDB=''
inputdir=''
outdir=''
onlytag='all'
useDeepCSV=True

DtagsList = []
who = commands.getstatusoutput('whoami')[1]

for o,a in opts:
    if o in('-j'): samplesDB = a
    elif o in('-d'): inputdir = a
    elif o in('-o'): outdir = a
    elif o in('-t'): onlytag = a
    elif o in('-u'): useDeepCSV = a

jsonFile = open(samplesDB,'r')
procList=json.load(jsonFile,encoding='utf-8').items()

print "Only files with dtags matching " + onlytag + " are processed."
#run over sample
for proc in procList :
    #run over processes
    for desc in proc[1] :

        #run over items in process
        isdata=getByLabel(desc,'isdata',False)
        if(isdata): continue
        mctruthmode=getByLabel(desc,'mctruthmode',0)
        tag = getByLabel(desc,'tag','')
        print tag

        data = desc['data']
        for d in data :
            origdtag = getByLabel(d,'dtag','')
            dtag = origdtag
            suffix = str(getByLabel(d,'suffix' ,""))
            if(onlytag!='all') :
                if(dtag.find(onlytag)<0) : continue
            if(mctruthmode!=0) : dtag+='_filt'+str(mctruthmode)

            outfile = outdir +'/'+ dtag + '_' + 'BTaggNums.root'
            destfile = outdir +'/'+ dtag + '_' + 'BTaggEff.root'
            ntplpath = '/eos/cms/store/user/' + who + '/'+inputdir + '/*/crab_' + origdtag + '*/*/*/'
            status, output = commands.getstatusoutput('ls '+ntplpath+'analysis_*.root')
            if status > 0 :
                print "Warning: No NTuples for tag: " + dtag
                continue
            segment = 0
            commands.getstatusoutput('rm -rf '+outdir +'/'+ dtag + suffix + '_*' + '_BTagTemp.root')
            for file in glob.glob(ntplpath+'analysis_*.root'):
                out_temp = outdir +'/'+ dtag + suffix + '_' + str(segment) + '_BTagTemp.root'
                commands.getstatusoutput('rootcp --recreate '+file+':mainNtuplizer/*BTagging* '+out_temp)
                segment += 1
            commands.getstatusoutput('hadd -f '+outfile+' '+outdir +'/'+ dtag + suffix + '_*' + '_BTagTemp.root')
            commands.getstatusoutput('rm -rf '+outdir +'/'+ dtag + suffix + '_*' + '_BTagTemp.root')
            
            binsX = array('d', [0., 40., 60., 80., 100., 150., 200., 300., 400., 1000.])
            binsY = array('d', [0., 0.6, 1.2, 2.5])
            produceEfficiencyMaps(binsX, binsY, outfile, destfile, useDeepCSV)
            print "Output file is: " + destfile

