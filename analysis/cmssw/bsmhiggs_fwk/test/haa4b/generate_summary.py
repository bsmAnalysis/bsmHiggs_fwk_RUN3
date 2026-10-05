import os
import glob

# Set the year 
year = "2017"  

# Define input and output paths
input_dir = "evt-total-files-{}".format(year)
summary_filename = "summary-{}-totals.txt".format(year)

# Check if input directory exists
if not os.path.exists(input_dir):
    print "Error: Directory {} does not exist. Run getEvtTotals.py first.".format(input_dir)
    exit(1)

summary_data = {}

# Process each dataset file
for txt_file in glob.glob(os.path.join(input_dir, "*.txt")):
    with open(txt_file, "r") as f:
        lines = f.readlines()

    if not lines:
        print "Warning: {} is empty.".format(txt_file)
        continue

    # Extract dataset name from filename
    dtag = os.path.basename(txt_file).replace(".txt", "")

    # Find nTot line
    for line in lines:
        if "nTot:" in line:
            parts = line.replace(",", " ").split()
            try:
                nTot_index = parts.index("nTot:") + 1
                nTot_value = parts[nTot_index]

                # Convert float values to integer
                if "." in nTot_value:
                    nTot = int(float(nTot_value))  # Convert float to int
                else:
                    nTot = int(nTot_value)

                summary_data[dtag] = nTot
            except (ValueError, IndexError):
                print "*** Warning: Malformed 'nTot' line in {}: '{}'".format(txt_file, line)
                continue

# Write the summary file
with open(summary_filename, "w") as f:
    for dtag, nTot in summary_data.items():
        f.write("{} {}\n".format(dtag, nTot))

print "\nSummary file created: {}".format(summary_filename)
