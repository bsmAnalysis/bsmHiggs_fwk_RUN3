import ROOT as r
import glob
import sys
import commands
import os

import argparse

def help():
   print '\n\n This is the help.\n\n'

parser = argparse.ArgumentParser()

parser.add_argument( "year_to_run", type=str, help="Year to run jobs for: 2016, 2017, 2018") ;
parser.add_argument( "--check_dirs_only", dest='check_dirs_only', help='Check the ntuple directory contents only.', action='store_true' )

args = parser.parse_args()

print "\n\n args:\n"
print args
print "\n\n"

year = args.year_to_run
check_dirs_only = args.check_dirs_only

file_count = {}

out_dir = 'evt-total-files'
dtfile = 'none.txt'
ntuple_base_dir_list = []


if year == '2016' :
   out_dir = 'evt-total-files-2016'
   dtfile = 'dtags-2016-legacy-mc.txt'
 #-- 2016 Data and MC
   ntuple_base_dir_list.append("/eos/user/g/georgia/results_2016_2020_06_19")

if year == '2017' :
   out_dir = 'evt-total-files-2017'
   dtfile = 'dtags-2017-mc.txt'
 #-- 2017 Data and MC
   ntuple_base_dir_list.append("/eos/user/l/lrouseli/Ntuples")
 #-- 2017 DY NJets and W NJets samples

if year == '2018' :
   out_dir = 'evt-total-files-2018'
   dtfile = 'dtags-2018-mc.txt'
<<<<<<< HEAD
   ntuple_base_dir_list.append("/eos/user/k/kpaschos/NTuples_2018/")
=======
 #-- 2018 Data and MC
   ntuple_base_dir_list.append("/eos/cms/store/user/georgia/results_2018_2020_02_05")
>>>>>>> db917c897aec67c8a6b04bd935bbea7e7fd7ee45

adtfile = open(dtfile,"r")
dtags = adtfile.readlines()


if not os.path.exists( out_dir ) :
   os.mkdir( out_dir )


for ntuple_base_dir in ntuple_base_dir_list :

   num=1

   for tag in dtags:

      tag = tag.rstrip("\n")
      print(str(num)+". Processing the tag: "+tag)

      num += 1

      ntplpath = ntuple_base_dir + '/*/crab_' + tag + '*/*/*/'

      nTot = 0
      nPos = 0
      nNeg = 0

      ntuple_files = glob.glob(ntplpath+'*.root')

      if len( ntuple_files ) > 0 :

         outfile = out_dir + "/" + tag + ".txt"
         if os.path.exists( outfile ) :
            os.system( "mv {0} {0}-old".format( outfile ) )



         print "   Found {0} files for {1} in {2}".format( len( ntuple_files ), tag, ntplpath )
         file_count[ tag ] = len( ntuple_files )

         if check_dirs_only : continue

         ofp = open( outfile, "w")

         for nt_fname in ntuple_files :

            tf = r.TFile( nt_fname )
            if (tf.IsZombie() or (not tf.IsOpen())):
               print "*** Problem with file {0} from dtag {1} in {2}".format( nt_fname, tag, ntuple_base_dir )
               continue

            this_file_nTot = tf.Get("mainNtuplizer/nevents").GetBinContent(1)
            this_file_nPos = tf.Get("mainNtuplizer/n_posevents").GetBinContent(1)
            this_file_nNeg = tf.Get("mainNtuplizer/n_negevents").GetBinContent(1)
            nTot += this_file_nTot
            nPos += this_file_nPos
            nNeg += this_file_nNeg
            ofp.write( "{:200}  {:15}  {:15}  {:15}\n".format(nt_fname, this_file_nTot, this_file_nPos, this_file_nNeg) )
            tf.Close()

#         ofp.write("Totals {:15}  {:15}  {:15}\n".format(nTot, nPos, nNeg))
         ofp.write("nTot: {0:15}, nPos: {1:15}, nNeg: {2:15}, nPos-nNeg: {3:15}\n".format(nTot,nPos,nNeg,nPos-nNeg))  
      else :
         print "   * No files matching {0} in {1}".format( tag, ntplpath )



print "\n\n\n File count summary\n"

for tag in dtags:
   tag = tag.rstrip("\n")
   if tag in file_count :
      print "  {0:5} files for {1}".format( file_count[tag], tag )
   else:
      print "  *** Found no files for " + tag

### The do somthing like:
### >  grep nTot *.txt | sort | awk '{printf(" %70s %15s %15s \n",$1, $7,$8)}'



  
