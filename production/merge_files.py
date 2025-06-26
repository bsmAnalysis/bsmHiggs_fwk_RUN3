import os
#if eg: you have 1500 files of 500 events in  each, and want to hadd files to have 50k events in each:
for i in range(15):
    start = i * 100
    end = start + 99
    output = f"ZH_ZToAll_HToAATo4B_M-15_{i+1}.root"
    inputs = " ".join(
        f"ZH_ZToAll_HToAATo4B_M-15_TuneCP5_13p6TeV-madgraph_pythia8_cff_{j}.root"
        for j in range(start, end + 1)
    )
    cmd = f"python3 haddNano.py {output} {inputs}"
    print(cmd)
    
    # Uncomment the following line to actually run the command
    os.system(cmd)
