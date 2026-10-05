import FWCore.ParameterSet.Config as cms
import os, sys, re, json, subprocess


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


"""
lists the files available in castor/eos/local
"""
def fillFromStore(dir, ffile=0, step=-1, generatePfn=True):
    localdataset = cms.untracked.vstring()
    if len(dir) == 0:
        return localdataset

    prefix = 'singlefile'
    lsout = [dir]

    if 'path=' in dir:
        print(f'Using dbs to query {dir}')
        prefix = 'eoscms'
        status, out = subprocess.getstatusoutput('dbs lsf --' + dir)
        lsout = out.split()

    elif 'castor' in dir:
        prefix = 'rfio'
        lscommand = 'rfdir ' + dir + " | awk '{print $9}'"
        status, out = subprocess.getstatusoutput(lscommand)
        lsout = out.split()

    elif dir.startswith('/store/'):
        prefix = 'eoscms'
        lscommand = 'cmsLs -R ' + dir + " | grep root | awk '{print $5}'"
        status, out = subprocess.getstatusoutput(lscommand)
        lsouttmp = out.split()

        lsout = []
        for l in lsouttmp:
            if not l:
                continue
            if l in lsout:
                continue
            basename = os.path.basename(l)
            if basename.startswith('tree_'):
                continue
            if basename.startswith('histogram'):
                continue
            lsout.append(l)

    elif not dir.endswith('.root'):
        prefix = 'file'
        lscommand = 'ls ' + dir
        status, out = subprocess.getstatusoutput(lscommand)
        lsout = out.split()

    # select files
    ifile = 0
    for line in lsout:
        if not isinstance(line, str):
            continue
        if len(line) == 0:
            continue
        if 'root' not in line:
            continue
        if ifile < ffile:
            continue

        if step < 0 or (step > 0 and ifile < ffile + step):
            sline = ''
            if prefix == 'eoscms':
                if generatePfn:
                    _, sline = subprocess.getstatusoutput('cmsPfn ' + line)
                else:
                    sline = line
            elif prefix == 'singlefile':
                sline = 'file://' + line
            else:
                sline = f'{prefix}://{dir}/{line.split()[0]}'
                if len(sline) == 0:
                    continue
            sline = sline.replace('?svcClass=default', '')
            localdataset.extend([sline])
        ifile += 1

    return natural_sort(localdataset)


"""
check that a file exist and is not corrupted
"""
def checkInputFile(url):
    if url.startswith('/store'):
        url = 'root://eoscms//eos/cms' + url
    status, out = subprocess.getstatusoutput("root -l -b -q " + url)
    if "Error" in out or "probably not closed" in out or "Corrupted" in out:
        return False
    return True


"""
check store for duplicates
"""
def checkStoreForDuplicates(outdir):
    ls_cms = "ls " + outdir
    isEOS = False
    isCastor = False

    if outdir.startswith('/store/'):
        isEOS = True
        ls_cms = "cmsLs " + outdir + " | grep root | awk '{print $5}'"
    elif 'castor' in outdir:
        isCastor = True
        ls_cms = "rfdir " + outdir + " | grep root"

    nOutFile = 0
    status, out = subprocess.getstatusoutput(ls_cms)
    jobNumbers, duplicatedJobs, origFiles, duplicatedFiles = [], [], [], []

    if status == 0:
        for fileLine in out.split("\n"):
            if "root" not in fileLine:
                continue
            fileName = fileLine
            if isCastor:
                fileName = fileLine.split()[8]

            if checkInputFile(fileName):
                try:
                    fileBaseName = os.path.basename(fileName)
                    jobNumber = int(fileBaseName.split("_")[1])
                except Exception:
                    continue

                if jobNumber in jobNumbers:
                    if jobNumber not in duplicatedJobs:
                        duplicatedJobs.append(jobNumber)
                    duplicatedFiles.append(fileName)
                else:
                    jobNumbers.append(jobNumber)
                    origFiles.append(fileName)
                    nOutFile += 1
            else:
                print(f"   #corrupted file found : {fileName}")
                duplicatedFiles.append(fileName)
    return natural_sort(duplicatedFiles)


"""
clean up duplicates
"""
def removeDuplicates(dir):
    duplicatedFiles = checkStoreForDuplicates(dir)
    print(f'Removing {len(duplicatedFiles)} duplicated files in {dir}')

    isNCG = dir.startswith('/lustre/ncg.ingrid.pt/')
    isEOS = dir.startswith('/store/')
    isCastor = 'castor' in dir

    for f in duplicatedFiles:
        print(f)
        if isNCG:
            subprocess.getstatusoutput('')  # TODO: implement
        elif isEOS:
            subprocess.getstatusoutput('cmsRm ' + f)
        elif isCastor:
            subprocess.getstatusoutput('rfrm ' + dir + '/' + f)
        else:
            subprocess.getstatusoutput('rm ' + dir + '/' + f)


"""
wrapper to read the configuration from command line
"""
def configureSourceFromCommandLine():
    storeDir = ''
    outputFile = 'Events.root'
    ffile = 0
    step = -1
    try:
        if len(sys.argv) > 2:
            if '/' in sys.argv[2] or '.root' in sys.argv[2]:
                storeDir = sys.argv[2]
                if len(sys.argv) > 3:
                    if '.root' in sys.argv[3]:
                        outputFile = sys.argv[3]
                    if len(sys.argv) > 4:
                        if sys.argv[4].isdigit():
                            ffile = int(sys.argv[4])
                        if len(sys.argv) > 5:
                            if sys.argv[5].isdigit():
                                step = int(sys.argv[5])
    except Exception:
        print('[storeTools_cff] Could not configure from command line, will return default values')

    return outputFile, fillFromStore(storeDir, ffile, step)


def addPrefixSuffixToFileList(Prefix, fileList, Suffix):
    outList = [Prefix + s + Suffix for s in fileList]
    return natural_sort(outList)


def keepOnlyFilesFromGoodRun(fileList, jsonPath):
    if jsonPath == '':
        return fileList

    with open(jsonPath, 'r') as jsonFile:
        runList = json.load(jsonFile)

    goodLumis = {}
    for run, lumis in sorted(runList.items()):
        goodLumis[int(run)] = [l for l in lumis]

    outFileList = []
    for F in fileList:
        try:
            if '/00000/' in F:
                Fsplit = F.split('/00000/')[0].split('/')
                run = int(Fsplit[-2]) * 1000 + int(Fsplit[-1])
                if run in goodLumis:
                    outFileList.append(F)
            else:
                print(f'das_client.py --limit=0 --query "lumi file={F} | grep lumi.run_number,lumi.number"')
                outFileList.append(F)
        except Exception:
            outFileList.append(F)

    return outFileList


"""
list EOS directory
"""
def getLslist(directory, mask='', prepend='root://eoscms//eos/cms', local=False):
    from subprocess import Popen, PIPE
    print('looking into: ' + directory + '...')

    eos_cmd = '/afs/cern.ch/project/eos/installation/0.2.41/bin/eos.select'

    if local:
        data = Popen(['ls', directory], stdout=PIPE)
    else:
        data = Popen([eos_cmd, 'ls', '/eos/cms/' + directory], stdout=PIPE)

    out, _ = data.communicate()
    out = out.decode()

    full_list = []

    if directory.endswith('.root'):
        if len(out.split('\n')[0]) > 0:
            return [prepend + directory]

    for line in out.split('\n'):
        if len(line.split()) == 0:
            continue
        full_list.append(prepend + directory + '/' + line)

    if mask != '':
        return [x for x in full_list if mask in x]

    print(full_list)
    return full_list
