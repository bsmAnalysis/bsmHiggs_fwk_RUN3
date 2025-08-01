#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:32:45 2025

@author: konpas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ROOT

###############################################################################
#Function to read ROOT file and convert it to a DataFrame
###############################################################################

def Read_Root_File(file_or_dir, rebin = 1):
    
    if file_or_dir in [None, "", []]:
        # Return empty dummy data
        return pd.DataFrame(), pd.DataFrame()
    
    # Only open if it's a file path (string)
    if isinstance(file_or_dir, str):
        root_obj = ROOT.TFile.Open(file_or_dir)
    else:
        root_obj = file_or_dir  # already a ROOT object (TFile or TDirectory)   
    keys = root_obj.GetListOfKeys()
    print(f"[INFO] Found {len(keys)} keys")
    for key in keys:
        print(f"  Key: {key.GetName()} - Class: {key.GetClassName()}")    
    hist_data = {}
    df_data = pd.DataFrame()
    hist_statistics = {}
    df_statistics = pd.DataFrame()
    
    for key in keys:
        try:
            obj = key.ReadObj()
            if not isinstance(obj, ROOT.TH1D):
                continue

            hist_name = key.GetName()
            hist = root_obj.Get(hist_name)
            if not hist:
                print(f"[WARN] Could not get histogram: {hist_name}")
                continue

            if hist.GetNbinsX() > rebin:
                hist = hist.Rebin(rebin)

            bin_values = [hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)]
            bin_centers = [hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)]
            bin_errors = [hist.GetBinError(i)   for i in range(1, hist.GetNbinsX() + 1)]

            hist_data = {
                hist_name + "_values": bin_values,
                hist_name + "_bins": bin_centers,
                hist_name + "_errors": bin_errors
            }
            df_data = pd.concat([df_data, pd.DataFrame.from_dict(hist_data)], axis=1)

            hist_statistics = {
                "mean": hist.GetMean(),
                "std": hist.GetStdDev(),
                "entries": hist.GetEntries(),
                "title": hist.GetTitle(),
                "xlabel": hist.GetXaxis().GetTitle(),
                "ylabel": hist.GetYaxis().GetTitle(),
                "key": hist_name
            }
            df_statistics = pd.concat([df_statistics, pd.DataFrame([hist_statistics])], ignore_index=True)
        
        except Exception as e:
            print(f"[ERROR] Failed to process {key.GetName()}: {e}")
    
    df_statistics = df_statistics.set_index("key")

    return df_data, df_statistics

###############################################################################
#Function to extract bin edges and bin values for given DataFrame and a given
#histogram name 
###############################################################################

def Bin_Edges_Values_Errors(dataframe, name):
    bin_centers = dataframe[name + "_bins"]\
        [~np.isnan(dataframe[name + "_bins"])].values
    bin_values = dataframe[name + "_values"]\
        [~np.isnan(dataframe[name + "_values"])].values
    bin_errors = dataframe[name + "_errors"]\
        [~np.isnan(dataframe[name + "_errors"])].values

    #equal bin widths
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = bin_centers - bin_width / 2
    bin_edges = np.append(bin_edges, bin_centers[-1] 
                                 + bin_width / 2)
    return bin_edges, bin_values, bin_errors


###############################################################################
#Read Signal and Background paths
###############################################################################
Rebin_Factor = 1;

file_path_12 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-12_2024.root"
f12 = ROOT.TFile.Open(file_path_12)
gen_dir_12      = f12.Get("gen")
boosted_dir_12  = f12.Get("boosted")
merged_dir_12   = f12.Get("merged")
resolved_dir_12 = f12.Get("resolved")

file_path_15 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-15_2024.root"
f15 = ROOT.TFile.Open(file_path_15)
gen_dir_15      = f15.Get("gen")
boosted_dir_15  = f15.Get("boosted")
merged_dir_15   = f15.Get("merged")
resolved_dir_15 = f15.Get("resolved")

file_path_20 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-20_2024.root"
f20 = ROOT.TFile.Open(file_path_20)
gen_dir_20      = f20.Get("gen")
boosted_dir_20  = f20.Get("boosted")
merged_dir_20   = f20.Get("merged")
resolved_dir_20 = f20.Get("resolved")

file_path_25 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-25_2024.root"
f25 = ROOT.TFile.Open(file_path_25)
gen_dir_25      = f25.Get("gen")
boosted_dir_25  = f25.Get("boosted")
merged_dir_25   = f25.Get("merged")
resolved_dir_25 = f25.Get("resolved")

file_path_30 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-30_2024.root"
f30 = ROOT.TFile.Open(file_path_30)
gen_dir_30      = f30.Get("gen")
boosted_dir_30  = f30.Get("boosted")
merged_dir_30   = f30.Get("merged")
resolved_dir_30 = f30.Get("resolved")

file_path_35 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-35_2024.root"
f35 = ROOT.TFile.Open(file_path_35)
gen_dir_35      = f35.Get("gen")
boosted_dir_35  = f35.Get("boosted")
merged_dir_35   = f35.Get("merged")
resolved_dir_35 = f35.Get("resolved")

file_path_40 = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-40_2024.root"
f40 = ROOT.TFile.Open(file_path_40)
gen_dir_40      = f40.Get("gen")
boosted_dir_40  = f40.Get("boosted")
merged_dir_40   = f40.Get("merged")
resolved_dir_40 = f40.Get("resolved")

file_path_45 = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-45_2024.root"
f45 = ROOT.TFile.Open(file_path_45)
gen_dir_45      = f45.Get("gen")
boosted_dir_45  = f45.Get("boosted")
merged_dir_45   = f45.Get("merged")
resolved_dir_45 = f45.Get("resolved")

file_path_50 = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-50_2024.root"
f50 = ROOT.TFile.Open(file_path_50)
gen_dir_50      = f50.Get("gen")
boosted_dir_50  = f50.Get("boosted")
merged_dir_50   = f50.Get("merged")
resolved_dir_50 = f50.Get("resolved")

file_path_55 = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-55_2024.root"
f55 = ROOT.TFile.Open(file_path_55)
gen_dir_55      = f55.Get("gen")
boosted_dir_55  = f55.Get("boosted")
merged_dir_55   = f55.Get("merged")
resolved_dir_55 = f55.Get("resolved")

file_path_60 = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-60_2024.root"
f60 = ROOT.TFile.Open(file_path_60)
gen_dir_60      = f60.Get("gen")
boosted_dir_60  = f60.Get("boosted")
merged_dir_60   = f60.Get("merged")
resolved_dir_60 = f60.Get("resolved")

file_path_60 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-60_2024.root"
f60 = ROOT.TFile.Open(file_path_60)
gen_dir_60      = f60.Get("gen")
boosted_dir_60  = f20.Get("boosted")
merged_dir_60   = f20.Get("merged")
resolved_dir_60 = f60.Get("resolved")

file_path_qcd = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/QCD_Bin-PT-MuEnr_2024.root"
f_qcd = ROOT.TFile.Open(file_path_qcd)
boosted_dir_qcd  = f_qcd.Get("boosted")
merged_dir_qcd   = f_qcd.Get("merged")
resolved_dir_qcd = f_qcd.Get("resolved")

file_path_wlnu = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WtoLNu-4Jets_Bin-4J_2024.root"
f_wlnu = ROOT.TFile.Open(file_path_wlnu)
boosted_dir_wlnu  = f_wlnu.Get("boosted")
merged_dir_wlnu   = f_wlnu.Get("merged")
resolved_dir_wlnu = f_wlnu.Get("resolved")

file_path_other = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/OtherBkg.root"
f_other = ROOT.TFile.Open(file_path_other)
boosted_dir_other  = f_other.Get("boosted")
merged_dir_other   = f_other.Get("merged")
resolved_dir_other = f_other.Get("resolved")

file_path_ttLF = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/TTtoLNu_ttLF.root"
f_ttLF = ROOT.TFile.Open(file_path_ttLF)
boosted_dir_ttLF  = f_ttLF.Get("boosted")
merged_dir_ttLF   = f_ttLF.Get("merged")
resolved_dir_ttLF = f_ttLF.Get("resolved")

file_path_ttCC = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/TTtoLNu_ttCC.root"
f_ttCC = ROOT.TFile.Open(file_path_ttCC)
boosted_dir_ttCC  = f_ttCC.Get("boosted")
merged_dir_ttCC   = f_ttCC.Get("merged")
resolved_dir_ttCC = f_ttCC.Get("resolved")

file_path_ttBB = "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/TTtoLNu_ttBB.root"
f_ttBB = ROOT.TFile.Open(file_path_ttBB)
boosted_dir_ttBB  = f_ttBB.Get("boosted")
merged_dir_ttBB   = f_ttBB.Get("merged")
resolved_dir_ttBB = f_ttBB.Get("resolved")

#plt.style.use(hep.style.CMS)
#extract bins and values


###############################################################################
# Detector Plots
###############################################################################

def Plot_Hists(name, description, offset, xmin = None, xmax = None, 
                    log = False, normalize = False, scale=20, 
                    int_lumi = 109080, Rebin=1, ylabel = "", xlabel = ""):
    
    hist_20, stat_20 = Read_Root_File(resolved_dir_20, Rebin)
    hist_60, stat_60 = Read_Root_File(resolved_dir_60, Rebin)

    hist_qcd, stat_qcd     = Read_Root_File(resolved_dir_qcd,   Rebin)
    hist_wlnu, stat_wlnu   = Read_Root_File(resolved_dir_wlnu,  Rebin)
    hist_other, stat_other = Read_Root_File(resolved_dir_other, Rebin)
    hist_ttLF, stat_ttLF   = Read_Root_File(resolved_dir_ttLF,  Rebin)
    hist_ttCC, stat_ttCC   = Read_Root_File(resolved_dir_ttCC,  Rebin)
    hist_ttBB, stat_ttBB   = Read_Root_File(resolved_dir_ttBB,  Rebin)
    
    #extract bins and values
    bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
    bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)
    
    bin_edges_qcd, bin_values_qcd, error_qcd       = Bin_Edges_Values_Errors(hist_qcd,   name)
    bin_edges_wlnu, bin_values_wlnu, error_wlnu    = Bin_Edges_Values_Errors(hist_wlnu,  name)
    bin_edges_other, bin_values_other, error_other = Bin_Edges_Values_Errors(hist_other, name)
    bin_edges_ttLF, bin_values_ttLF, error_ttLF    = Bin_Edges_Values_Errors(hist_ttLF,  name)
    bin_edges_ttBB, bin_values_ttBB, error_ttBB    = Bin_Edges_Values_Errors(hist_ttBB,  name)
    bin_edges_ttCC, bin_values_ttCC, error_ttCC    = Bin_Edges_Values_Errors(hist_ttCC,  name)

    #normalize signal with respect to the background area
    if normalize:    
        bin_values_20 = bin_values_20 * scale
        bin_values_60 = bin_values_60 * scale
        
    #normalize all MC to 2024 luminosity
    bin_values_qcd, error_qcd     = int_lumi * bin_values_qcd,   int_lumi * error_qcd  
    bin_values_wlnu, error_wlnu   = int_lumi * bin_values_wlnu,  int_lumi * error_wlnu  
    bin_values_other, error_other = int_lumi * bin_values_other, int_lumi * error_other 
    bin_values_ttLF, error_ttLF   = int_lumi * bin_values_ttLF,  int_lumi * error_ttLF 
    bin_values_ttCC, error_ttCC   = int_lumi * bin_values_ttCC,  int_lumi * error_ttCC
    bin_values_ttBB, error_ttBB   = int_lumi * bin_values_ttBB,  int_lumi * error_ttBB 
    
    bin_values_20 = int_lumi * bin_values_20
    bin_values_60 = int_lumi * bin_values_60
    
    # #define bin centers and bin width
    # bin_width = (bin_edges_wlnu[1:]-bin_edges_wlnu[:-1])/2
    # bin_centers = (bin_edges_wlnu[1:]+bin_edges_wlnu[:-1])/2
    
    #Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # --- STACKING --- #
    h_other = ax.stairs(bin_values_other, bin_edges_wlnu,
                        color='royalblue', fill=True, edgecolor='black', linewidth=1.2)
    
    h_ttLF = ax.stairs(bin_values_other + bin_values_ttLF, bin_edges_wlnu,
                       baseline=bin_values_other,
                       color='darkseagreen', fill=True, edgecolor='black', linewidth=1.2)
    
    h_ttCC = ax.stairs(bin_values_other + bin_values_ttLF + bin_values_ttCC, bin_edges_wlnu,
                       baseline=bin_values_other + bin_values_ttLF,
                       color='mediumseagreen', fill=True, edgecolor='black', linewidth=1.2)
    
    h_ttBB = ax.stairs(bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB, bin_edges_wlnu,
                       baseline=bin_values_other + bin_values_ttLF + bin_values_ttCC,       
                       color='seagreen', fill=True, edgecolor='black', linewidth=1.2)
    
    # to be added 
    # h_dy = ax.stairs(bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB + bin_values_dy, bin_edges_wlnu,
    #                  baseline=bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB,  
    #                  color='palevioletred', fill=True, edgecolor='black', linewidth=1.2)
    
    h_wlnu = ax.stairs(bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB + bin_values_wlnu, bin_edges_wlnu,
                       baseline=bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB,
                       color='pink', fill=True, edgecolor='black', linewidth=1.2)
    
    h_qcd = ax.stairs(bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB + bin_values_wlnu + bin_values_qcd, bin_edges_wlnu,
                      baseline=bin_values_other + bin_values_ttLF + bin_values_ttCC + bin_values_ttBB + bin_values_wlnu,
                      color='darkred', fill=True, edgecolor='black', linewidth=1.2)
    

    h_sig20 = ax.stairs(bin_values_20, bin_edges_20, color='red', linestyle='--', linewidth=1.5)
    h_sig60 = ax.stairs(bin_values_60, bin_edges_60, color='blue', linewidth=1.5)

           
    if log:    
        ax.set_yscale('log')
                      
    handles = [h_sig20, h_sig60, h_qcd, h_wlnu, h_ttBB, h_ttCC, h_ttLF, h_other]

    labels = [
        r"$WH \rightarrow 4b,~m_{a}=20~(\times{"+str(scale)+r"})$",
        r"$WH \rightarrow 4b,~m_{a}=60~(\times{"+str(scale)+r"})$",
        "QCD",
        r"$W \rightarrow \ell\nu$",
        r"$t\bar{t}+b\bar{b}$",
        r"$t\bar{t}+c\bar{c}$",
        r"$t\bar{t}+$ light",
        "Other bkgds"
    ]

    # Draw the legend
    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.40, 0.98),  
        ncol=4,
        columnspacing=1.5,
        handletextpad=0.5,
        frameon=False,
        fontsize=15
    )
    # labels = [
    #     r"$WH \rightarrow 4b,~m_{a}=20~(\times" + str(scale) + ")$",
    #     r"$WH \rightarrow 4b,~m_{a}=60~(\times" + str(scale) + ")$",
    #     "QCD",
    #     r"$W \rightarrow \ell\nu$",
    #     r"$t\bar{t}+b\bar{b}$",
    #     r"$t\bar{t}+c\bar{c}$",
    #     r"$t\bar{t}+$ light",
    #     "Other bkgds"      
    #     ]    
    
    # leg = ax.legend(
    # handles=handles,
    # labels=labels,
    # fontsize=15,
    # frameon=False,
    # loc="upper right", 
    # bbox_to_anchor=(1, 0.9)
    # )
    
    # Set uniform legend line width
    for line in leg.get_lines():
        line.set_linewidth(3)
    
    #create x and y axis labels    
    ax.set_ylabel(f"{ylabel}", fontsize=22, loc="top")        
    ax.set_xlabel(f"{xlabel}", fontsize=22, loc="right")
    
    #create minor ticks and modify their appearance
    ax.minorticks_on()
    ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                   right=True, top=True, labelsize=12)
    ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                   right=True, top=True, labelsize=12)

    #title = stat_20.loc[name]["title"]
    ax.set_title(r'$\bf{CMS}$' + " " +  r'$\it{Preliminary}$', fontsize=18, 
                 loc="left")
    ax.set_title("109.08 " + r"$fb^{-1}$" + ", (2024, 13.6 TeV)", loc="right", 
                 fontsize=15)
    
    ax.text(
    offset, 0.85, description, 
    transform=plt.gca().transAxes,
    fontsize=17,
    verticalalignment='top',
    horizontalalignment='left',
    )
    
    #define x and y axis limits
    if not log:
        ax.set_ylim([min(min(bin_values_20),     min(bin_values_60), 
                         min(bin_values_qcd),    min(bin_values_wlnu), 
                         min(bin_values_other),  min(bin_values_ttLF), 
                         min(bin_values_ttCC),   min(bin_values_ttBB)), 
                     max(max(bin_values_qcd + bin_values_wlnu +  bin_values_other +
                             bin_values_ttBB + bin_values_ttCC + bin_values_ttLF)*1.2, 
                         max(bin_values_20), 
                         max(bin_values_60))])
    else:
        ax.set_ylim([min(min(bin_values_20),     min(bin_values_60), 
                         min(bin_values_qcd),    min(bin_values_wlnu), 
                         min(bin_values_other),  min(bin_values_ttLF), 
                         min(bin_values_ttCC),   min(bin_values_ttBB)), 
                     max(max(bin_values_qcd  + bin_values_wlnu + bin_values_other +
                             bin_values_ttBB + bin_values_ttCC + bin_values_ttLF)*100, 
                         max(bin_values_20)*100,
                         max(bin_values_60)*100)])
    
    if xmax == None or xmin == None:
        ax.set_xlim([min(min(bin_edges_20),    min(bin_edges_60), 
                         min(bin_edges_qcd),   min(bin_edges_wlnu), 
                         min(bin_edges_other), min(bin_values_ttLF), 
                         min(bin_values_ttCC), min(bin_values_ttBB)), 
                     max(max(bin_edges_20),    max(bin_edges_60), 
                         max(bin_edges_qcd),   max(bin_edges_wlnu), 
                         max(bin_edges_other), max(bin_values_ttLF),
                         max(bin_values_ttCC), max(bin_values_ttBB))])

    else:
        ax.set_xlim([xmin, xmax])

    out_path = \
        "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/Plots/"
        
    plt.tight_layout()  # Helps ensure your plot doesn't overlap layout

    # if log:
    #     plt.savefig(out_path + name + "_log.pdf")
    # else:
    #     plt.savefig(out_path + name + ".pdf")


# Example
    Plot_Hists(
        name="mass_H_resolved", 
        description="Higgs mass (resolved regime)", 
        offset=0.65,
        xmin=0, 
        xmax=1000, 
        log=False, 
        normalize=True,
        #ylabel=r"$\frac{1}{N} \frac{dN}{dx}$",
        ylabel = r"$N_{Events}$",
        xlabel=r"$m_h~[GeV]$",
        Rebin = 1
    )
    

###############################################################################
# Generator Plots
###############################################################################


###############################################################################
# 1st Plot: pt b1 
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)

name = "pt_gen:b1"
mean20 = round(stat_20.loc[name]["mean"],2)
mean60 = round(stat_60.loc[name]["mean"],2)

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$" + "\n" + "mean: " + f"${mean20}$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$" + "\n" + "mean: " + f"${mean60}$", linewidth=1.2)


ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T,{b_1}}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 500])
ax.set_ylim([0, 1.2*max(bin_values_60)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)

ax.text(
    0.7, 0.98, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + r"$b_{1}:\max ~ p_{T}$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 2nd Plot: pt b4
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)

name = "pt_gen:b4"
mean20 = round(stat_20.loc[name]["mean"],2)
mean60 = round(stat_60.loc[name]["mean"],2)

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$" + "\n" + "mean: " + f"${mean20}$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$" + "\n" + "mean: " + f"${mean60}$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T,{b_4}}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 300])
ax.set_ylim([0, 1.2*max(bin_values_20)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.7, 0.98, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + r"$b_{4}:\min ~ p_{T}$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 3rd Plot: deltaR bb1
###############################################################################
hist_12, stat_12 = Read_Root_File(gen_dir_12, Rebin_Factor)
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "dr_gen:bb1"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_12, bin_values_12, _ = Bin_Edges_Values_Errors(hist_12, name)
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_12 /= sum(bin_values_12)
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_12 /= bin_width
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_12, bin_edges_12, color="black", 
          label=r"$m_{a}=12$", linewidth=1.2)
ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\Delta R_{bb}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 6])
ax.set_ylim([0., 1.2*max(bin_values_12)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)

ax.text(
    0.68, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + "bb pair (same a)",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)


###############################################################################
# 4th Plot: pt A
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "pt_gen:A"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T,a}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 500])
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.7, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 5th Plot: eta A
###############################################################################
Rebin_Factor = 2;
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "eta_gen:A"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\eta_{a}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_ylim([0, 1.2*max(bin_values_20)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 6th Plot: pt higgs
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "pt_gen:H"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T,h}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 500])
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 7th Plot: eta higgs
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "eta_gen:H"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\eta_{h}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 7th Plot: deltaR AA
###############################################################################
hist_12, stat_12 = Read_Root_File(gen_dir_12, Rebin_Factor)
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "dr_gen:AA"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_12, bin_values_12, _ = Bin_Edges_Values_Errors(hist_12, name)
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_12 /= sum(bin_values_12)
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_12 /= bin_width
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_12, bin_edges_12, color="black", 
          label=r"$m_{a}=12$", linewidth=1.2)
ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\Delta R_{aa}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 6])
ax.set_ylim([0., 1.2*max(bin_values_60)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)


ax.text(
    0.68, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 8th Plot: pt W
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "pt_gen:W"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T,W}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 500])
ax.set_ylim([0, 1.2*max(bin_values_60)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 9th Plot: eta W
###############################################################################
Rebin_Factor = 2;
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "eta_gen:W"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\eta_{W}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 10th Plot: pt lepton
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "pt_gen:lepton"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T,lepton}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 500])
ax.set_ylim([0, 1.2*max(bin_values_60)])

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 11th Plot: MET
###############################################################################
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "pt_gen:neutrino"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$E_{T}^{miss}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 500])
ax.set_ylim([0, 1.2*max(bin_values_60)])

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
# 12th Plot: eta lepton
###############################################################################
Rebin_Factor = 2;
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)
name = "eta_gen:lepton"

fig, ax = plt.subplots(figsize=(8, 8))
bin_edges_20, bin_values_20, _ = Bin_Edges_Values_Errors(hist_20, name)
bin_edges_60, bin_values_60, _ = Bin_Edges_Values_Errors(hist_60, name)

#normalize to area
bin_values_20 /= sum(bin_values_20)
bin_values_60 /= sum(bin_values_60)

#divide with bin
bin_width=bin_edges_20[1:]-bin_edges_20[:-1]
bin_values_20 /= bin_width
bin_values_60 /= bin_width

ax.stairs(bin_values_20, bin_edges_20, color="blue", 
          label=r"$m_{a}=20$", linewidth=1.2)
ax.stairs(bin_values_60, bin_edges_60, color="red", 
          label=r"$m_{a}=60$", linewidth=1.2)

ax.set_ylabel(r"$\frac{1}{N} \frac{dN}{dx}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\eta_{lepton}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)

ax.set_ylim([0, 1.2*max(bin_values_60)])

leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$' + " " +  'Simulation' + " " + r'$\it{Work~in~Progress}$', 
             fontsize=18, loc="left")
ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)


