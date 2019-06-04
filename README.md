*************
-------------
 COMPILATION
-------------
*************

Pass -DDEBUG_PRINTF if detailed diagonostics is required along
program run. This program requires OpenMP and C++11 support,
so pass -fopenmp (for g++)/-qopenmp (for icpc) and -std=c++11/
-std=c++0x.

Pass -DUSE_32_BIT_GRAPH if number of nodes in the graph are 
within 32-bit range (2 x 10^9), else 64-bit range is assumed.

Pass -DDONT_CREATE_DIAG_FILES if you dont want to create 2 files
per process with detail diagonostics.

Upon building, the program will generate two binaries:
bin/graphClustering (parallel) and bin/fileConvertDist (serial).

Please use bin/fileConvertDist for input graph conversion from 
native formats to a binary format that bin/graphClustering will
be able to read. The bin/fileConvertDist binary can also be used
for generating scale free graphs.

cmake build option: 

$ mkdir build install
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=~/vite-03-2018/install
$ make -j4
$ make install

************************
------------------------
 INPUT GRAPH CONVERSION
------------------------
************************

Convert input files to binary from various formats. 
Possible options (can be combined):

1. -n               : Input graph in SNAP format
2. -m               : Input graph in Matrix-market format
3. -e               : Input graph in METIS format
4. -p               : Input graph in Pajek format
5. -d               : Input graph in DIMACS format. Pass 0 or 1
                      to indicate undirected/directed graph
6. -s               : Input graph is directed edge list
7. -u               : Input graph is undirected edge list 
8. -r               : Create random edge weights
9. -w               : Initialize edge weights to 1
10. -o              : Output binary file with full path
11. -z              : If the index of input graph is 1-based,
                      then this option makes it 0-based
12. -f              : Option preceding input file                      

------------------------
File conversion related
------------------------

Typical example: 
bin/./fileConvertDist -n -f karate.txt -o karate.bin

If your input file does not have edge weights, and you 
want random edge weights, then pass option "-r". 

bin/./fileConvertDist -m -r -f karate.mtx -o karate.bin

Note: DIMACS file could be directed or undirected, so
pass 0 if input is directed, or a number > 0 input
graph is undirected. Also, we expect DIMACS inputs to
have weights, therefore passing -r has no effect.

bin/./fileConvertDist -d 0 -f sample.gr -o sample.bin

Same as the one before, except initializes all weights
to 1, no matter what is in the input file.

bin/./fileConvertDist -d 0 -w -f sample.gr -o sample.bin

Note: If the input file is index 1-based, then indicate
that by passing -z, such that it will be converted to
0-based internally.

bin/./fileConvertDist -s -z -f network.dat -o lfrtest.bin

Note: We consider files containing edge list, as 'simple' 
format files (option "-s" for directed, and option "-u" for
undirected). If you have a simple format file (i.e., an 
edge list) that does not have weights, then please pass 
"-w" or "-r" in addition to "-s" or "-u" such that we can 
assign weights internally. By default, we assume edge list
files to have weights.

bin/./fileConvertDist -f uk2007-edges.txt -s -w -o uk2007.bin

****************************
----------------------------
 SYNTHETIC GRAPH GENERATION
----------------------------
****************************

We have bindings to NetworKit[*], that can help generate random and scale
free graphs. This is mostly serial, but NetworKit internally may use OpenMP. 
We have only tested with NetworKit v4.2, and have no plans on supporting 
other NetworKit versions.

[*] https://networkit.iti.kit.edu/

Install C++ module of NetworKit, by following instructions from:
https://networkit.iti.kit.edu/get_started.html#build-networkit-from-source

Possible options (can be combined):
1.  -e               : Generate ER graph
2.  -r               : Generate RGG graph
3.  -b               : Generate Barabasi-Albert graph
4.  -h               : Generate hyperbolic graph
5.  -n <...>         : Number of (max) nodes/vertices
6.  -p <...>         : Probability for edge creation (for ER graph)
7.  -k <...>         : Number of clusters for RGG (each unit square is divided into
                      clusters where edges are created within) OR attachments per
                      node for Barabasi-Albert graph
8.  -m <...>         : Initial nodes attached for Barabasi-Albert                      
9.  -z               : Visualize RGG graph (EPS file with graph visualization created)
10. -r               : Create random edge weights 

***********************
-----------------------
 EXECUTING THE PROGRAM
-----------------------
***********************

0. Convert input file to binary format using fileConvertDist.
This program only accepts a binary file.

1. Set the desired number of threads using OMP_NUM_THREADS
OpenMP environment variable.  

2. Run distributed parallel Louvain algorithm using the binary 
file created in Step #1:

mpiexec -n 2 bin/./graphClustering -c -i -f karate.bin

Possible options (can be combined):
1. -c <ncolors>  : Enable coloring, adjacent vertices are processed
                   in different order. Synchronization happens once
                   per <ncolors>.
2. -d <ncolors>  : Enable coloring, adjacent vertices are processes
                   in different order. No synchronization.
3. -i            : Threshold cycling, threshold changes dynamically
                   in every phase of Louvain algorithm.
4. -t 1          : If a vertex is in the same community for past 3
                   iterations, then consider it inactive.
5. -t 3          : If a vertex is in the same community for past 3
                   iterations, then consider it inactive. Also,
                   adds a communication step to gather inactive 
                   vertices and terminate Louvain if >= 90% vertices
                   at an iteration are inactive.
6. -t 2 -a <0-1> : Early termination using probability alpha.
7. -t 4 -a <0-1> : Early termination using probability alpha. Also,
                   adds a communication step to gather inactive 
                   vertices and terminate Louvain if >= 90% vertices
                   at an iteration are inactive.
8. -b            : Only valid for real-world inputs. Attempts to 
                   distribute approximately equal number of edges among 
                   processes. Irregular number of vertices owned by a 
                   particular process. Increases the distributed graph 
                   creation time due to serial overheads, but may improve 
                   overall execution time.
9. -o            : Output communities into a file. This option will result 
                   in Vite dumping the communities (community-per-vertex in 
                   each line, total number of lines == number of vertices) 
                   in a text file named <input-binary-file>.communities in 
                   the same path as the input binary file.
                  
Note: Option to update inactive vertices percentage is defined as
macro ET_CUTOFF in louvain.hpp, and the default is 2%.

**********************************************
----------------------------------------------
 COMPARING COMMUNITIES WITH GROUND TRUTH DATA
----------------------------------------------
**********************************************

1. Generate benchmark graphs with ground truth
community information. For e.g., we have tested
with LFR benchmark. 
(https://sites.google.com/site/santofortunato/inthepress2)
LFR generates an edge list (index 1-based), along with community
membership of each vertex.

2. Convert the generated network to binary using 
fileConvertDist. 

3. Execute the program as shown in the e.g. below:
mpiexec -n 2 bin/./graphClustering -r 1 -f lfrtest.bin -z -g community.dat

The difference between standard way to execute the program and this way
is that one needs to pass the ground truth vertex-community mapping 
using the "-g" option. Also, the "-z" option tells the program that
the input ground truth file -- `community.dat` is 1-based. If "-z" is
not passed, then we assume that the file is 0-based.

We return some statistics such as F-score, Gini-coefficient, 
false positives/negatives and true positives.

However, extra communication per phase is required when these metrics
are to be computed, so the program execution time will increase
significantly. Plus, the community comparison part is currently
multithreaded (uses OpenMP), so everything is brought to a single 
node/PE and then communities are compared. 

***************
---------------
 OUTPUT RESULT
---------------
***************

If -DDONT_CREATE_DIAG_FILES is passed during compilation,
then output is send to stdout.
Otherwise, the output result is dumped per process on files 
named as dat.out.<process-id>. Check dat.out.0 to review 
program diagonostics. 
Output files are cleared with: `make clean`.
