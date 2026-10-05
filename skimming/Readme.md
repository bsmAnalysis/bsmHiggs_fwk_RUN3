## Prepare dataset JSONs with PocketCoffea

Find the NanoAOD datasets in CMS DAS. PocketCoffea turns their DAS names into JSON file lists for skimming. CVMFS provides the container image; the generated lists point to remote ROOT files.
After setting up the proxy above, start from your skimming directory and clone PocketCoffea alongside it:

```bash
export SKIM_DIR="$PWD"
cd ..
git clone git@github.com:PocketCoffea/PocketCoffea.git
cd PocketCoffea
mkdir -p datasets
```
 Keep the checkout outside `skimming/` so it is not archived with job resources.
Create `datasets/datasets_definitions.json` using the [PocketCoffea definition format](https://pocketcoffea.readthedocs.io/en/stable/datasets.html). 
Include `sample`, `json_output`, `files[].das_names` and the relevant metadata: `year`, `isMC`, `xsec` for MC, and `era` for data.
You do not need to list individual ROOT files, this will be done by the pocket-coffea build-datasets . 


A valid CMS VOMS proxy is required for the remote CMS input files. eg
```bash
voms-proxy-init --voms cms --valid 192:00 --out "$PWD/x509up"
chmod 600 "$PWD/x509up"
export X509_USER_PROXY="$PWD/x509up"
voms-proxy-info --file "$X509_USER_PROXY" --all
```
Enter the PocketCoffea container from the checkout directory:

```bash
apptainer shell \
    --bind /afs \
    --bind /cvmfs \
    --bind /tmp \
    --bind /eos/cms/ \
    --bind /etc/sysconfig/ngbauth-submit \
    --bind "$XDG_RUNTIME_DIR" \
    --env "KRB5CCNAME=FILE:${XDG_RUNTIME_DIR}/krb5cc" \
    --env "X509_USER_PROXY=$X509_USER_PROXY" \
    --pwd "$PWD" \
    /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable
```

Inside the container, verify that the proxy is selected :

```bash
echo "$X509_USER_PROXY"
voms-proxy-info -path
```

Continue after the check passes. If it fails, renew the proxy on lxplus and restart the container.

Build the file lists:

```bash
pocket-coffea build-datasets \
    --cfg datasets/datasets_definitions.json \
    -o \
    -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+' \
    -ir
```

Replace the `--cfg` path with your definition filename.

- `-o` overwrites existing generated JSONs.
- `-rs` restricts replicas to the selected European sites.
- `-ir` allows a redirector fallback when none of those sites has a file.
