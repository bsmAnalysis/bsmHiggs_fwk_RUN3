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
            bin_errors = [hist.GetBinError(i) for i in range(1, hist.GetNbinsX() + 1)]

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
#Generator Plots
###############################################################################

Rebin_Factor = 2;

'''
Read Signal and Background paths
'''
file_path_12 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-12_2024.root"
f12 = ROOT.TFile.Open(file_path_12)
gen_dir_12 = f12.Get("gen")
hist_12, stat_12 = Read_Root_File(gen_dir_12, Rebin_Factor)

file_path_20 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-20_2024.root"
f20 = ROOT.TFile.Open(file_path_20)
gen_dir_20 = f20.Get("gen")
hist_20, stat_20 = Read_Root_File(gen_dir_20, Rebin_Factor)

file_path_60 = \
    "/Users/lizapenny/Desktop/PHD/NanoSetUP/Rootfiles/WH_WToAll_HToAATo4B_M-60_2024.root"
f60 = ROOT.TFile.Open(file_path_60)
gen_dir_60 = f60.Get("gen")
hist_60, stat_60 = Read_Root_File(gen_dir_60, Rebin_Factor)

#plt.style.use(hep.style.CMS)
#extract bins and values

###############################################################################
#First Plot: eta b-quarks
###############################################################################

names  = ["eta_gen:b1", "eta_gen:b2", "eta_gen:b3", "eta_gen:b4"]
colors = ["green", "darkkhaki", "darkorange", "gold"]
labels = [r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$"]
fig, ax = plt.subplots(figsize=(8, 8))
for index, name in enumerate(names):
    bin_edges, bin_values, _ = Bin_Edges_Values_Errors(hist_60, name)
    ax.stairs(bin_values, bin_edges, color=colors[index], label=labels[index], linewidth=1.2)
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(0.95, 0.95))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)

ax.set_ylabel(r"$N_{Events}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$\eta$", fontsize=22, loc="right",  fontweight="bold")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.5, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.3, direction='in', 
                right=True, top=True, labelsize=12)
bin_edges, bin_values, _ = Bin_Edges_Values_Errors(hist_60, "eta_gen:b1")
ax.set_ylim([0, 1.15*max(bin_values)])
ax.grid(linestyle=':', color='gray')
#ax.set_title("(13.6 TeV)", loc="right", fontsize=18)
#ax.set_title(r'$\bf{CMS}$' + " " +  r'$\it{Simulation}$', fontsize=18, loc="left")
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)

ax.text(
    0.05, 0.95,  
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + r"$m_{a}=60$",
    #transform=plt.gca().transAxes,  # Use Axes coordinates (0,0 bottom-left, 1,1 top-right)
    transform=ax.transAxes,
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)


###############################################################################
#Second Plot: pt b-quarks
###############################################################################

names = ["pt_gen:b1", "pt_gen:b2", "pt_gen:b3", "pt_gen:b4"]
colors = ["green", "darkkhaki", "darkorange", "gold"]
labels = [r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$"]
fig, ax = plt.subplots(figsize=(8, 8))
for index, name in enumerate(names):
    bin_edges, bin_values, _ = Bin_Edges_Values_Errors(hist_60, name)
    ax.stairs(bin_values, bin_edges, color=colors[index], label=labels[index], linewidth=1.7)
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(0.95, 0.95))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)

ax.set_ylabel(r"$N_{Events}$", loc="top", fontweight="bold", fontsize=22)
ax.set_xlabel(r"$p_{T}$" + "[GeV]", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.5, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.3, direction='in', 
                right=True, top=True, labelsize=12)
bin_edges, bin_values, _ = Bin_Edges_Values_Errors(hist_60, "pT_gen:b4")
ax.set_ylim([0, 1.15*max(bin_values)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)

ax.text(
    0.05, 0.95,  
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + r"$m_{a}=60$",
    #transform=plt.gca().transAxes,  # Use Axes coordinates (0,0 bottom-left, 1,1 top-right)
    transform=ax.transAxes,
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)
ax.set_xlim([0,300])


###############################################################################
#Third Plot: pt b1 
###############################################################################

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
ax.set_xlim([0, 300])
ax.set_ylim([0, 1.2*max(bin_values_20)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)

ax.text(
    0.7, 0.98, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + r"$b_{1}:\max ~ p_{T}$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Fourth Plo: pt b4
###############################################################################

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
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.7, 0.98, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + r"$b_{4}:\min ~ p_{T}$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Fifth Plot: deltaR bb1
###############################################################################

name = "dr_gen:bb1"

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
ax.set_xlabel(r"$\Delta R_{bb}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 8])
ax.set_ylim([0., 1.2*max(bin_values_20)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)

ax.text(
    0.68, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n" + "bb pair (same a)",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)


###############################################################################
#Sixth Plot: pt A
###############################################################################

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
ax.set_xlim([0, 350])
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.7, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Seventh Plot: eta A
###############################################################################

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
ax.set_xlabel(r"$\eta$(a)", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_ylim([0, 1.2*max(bin_values_20)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Eighth Plot: pt higgs
###############################################################################

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
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Ninth Plot: eta higgs
###############################################################################

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
ax.set_xlabel(r"$\eta(h)$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Fifth Plot: deltaR AA
###############################################################################

name = "dr_gen:AA"

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
ax.set_xlabel(r"$\Delta R_{aa}$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_xlim([0, 8])
ax.set_ylim([0., 1.2*max(bin_values_60)])
leg = ax.legend(
    fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for i in range(len(leg.get_lines())):
    leg.get_lines()[i].set_linewidth(3)
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)

ax.text(
    0.68, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$" + "\n",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Eleventh Plot: pt W
###############################################################################

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
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Twelveth Plot: eta W
###############################################################################

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
ax.set_xlabel(r"$\eta(W)$", fontsize=22, loc="right")
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.3, direction='in', 
                right=True, top=True, labelsize=14)
ax.tick_params(which='minor', length=3, width=1.1, direction='in', 
                right=True, top=True, labelsize=12)
ax.set_ylim([0, 1.2*max(bin_values_60)])
ax.grid(linestyle=':', color='gray')
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Thirteenth Plot: pt lepton
###############################################################################

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
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#Fourteenth Plot: MET
###############################################################################

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
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)

###############################################################################
#FIfteenth Plot: eta lepton
###############################################################################

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
ax.set_xlabel(r"$\eta(lepton)$", fontsize=22, loc="right")
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
ax.set_title(r'$\bf{CMS}$  $\it{Simulation}$', fontsize=18, loc="left")
ax.text(1.0, 1.02, "(13.6 TeV)", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18)
ax.text(
    0.72, 0.96, 
    r"$Wh\rightarrow aa \rightarrow 4b$",
    transform=plt.gca().transAxes, 
    fontsize=17,  # Text font size
    verticalalignment='top',  # Align the text vertically to the top
    horizontalalignment='left',  # Align the text horizontally to the left
)