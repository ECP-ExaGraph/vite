// ***********************************************************************
//
//            Vite: A C++ library for distributed-memory graph clustering 
//                  using MPI+OpenMP
// 
//               Daniel Chavarria (daniel.chavarria@pnnl.gov)
//               Antonino Tumeo (antonino.tumeo@pnnl.gov)
//               Mahantesh Halappanavar (hala@pnnl.gov)
//               Pacific Northwest National Laboratory	
//
//               Hao Lu (luhowardmark@wsu.edu)
//               Sayan Ghosh (sayan.ghosh@wsu.edu)
//               Ananth Kalyanaraman (ananth@eecs.wsu.edu)
//               Washington State University
//
// ***********************************************************************
//
//       Copyright (2017) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <omp.h>
#include <mpi.h>

#include "rebuild.hpp"
#include "louvain.hpp"
#include "compare.hpp"
#include "utils.hpp"

// for coloring
#define MAX_COVG    (70)

std::ofstream ofs;
std::ofstream ofcks;

static std::string inputFileName;
static std::string groundTruthFileName;
static int me, nprocs;

// coloring related
static bool   coloring                  = false;
static int    maxColors                 = 8;
static bool   vertexOrdering            = false;
static bool   runOnePhase               = false;

static bool   readBalanced              = false;
static int    ranksPerNode              = 1;
static bool   outputFiles               = false;
static bool   thresholdScaling          = false;

// early termination related
static bool   earlyTerm                 = false;
static int    ETType                    = 0;
static GraphWeight ETDelta                   = 1.0;

// community comparison with ground-truth
static bool  compareCommunities         = false;
static bool  isGroundTruthZeroBased     = true;

// parse command line parameters
static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  double t0, t1, t2, t3;
  
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  parseCommandLine(argc, argv);

  // read ground truth info if compareCommunities
  // is turned ON, serial
  
  // Ground Truth file is expected to be of the format generated 
  // by LFR-gen by Fortunato, et al.
  // https://sites.google.com/site/santofortunato/inthepress2

  std::vector<GraphElem> commGroundTruth;
  if (me == 0 /* root */ && compareCommunities) {
      t1 = MPI_Wtime();
      loadGroundTruthFile(commGroundTruth, groundTruthFileName, isGroundTruthZeroBased);
      t2 = MPI_Wtime();
#if defined(DONT_CREATE_DIAG_FILES)
      std::cout << "Time taken to load ground truth file (" 
          << groundTruthFileName << "): " << (t2-t1) << std::endl;
#else
      ofs << "Time taken to load ground truth file (" 
          << groundTruthFileName << "): " << (t2-t1) << std::endl;
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD); // wait till GT file is loaded

#if defined(DONT_CREATE_DIAG_FILES)
#else
  std::ostringstream oss;
  std::ostringstream oss2;
  oss2<< "check.out." <<me;
  oss << "dat.out." << me;

  ofs.open(oss.str());
  ofcks.open(oss2.str());
#endif

  rusage rus;

  createCommunityMPIType();
  createEdgeMPIType();
  double td0, td1;

  td0 = MPI_Wtime();
  DistGraph *dg = nullptr;

  GraphElem teps = 0;

  // load the input data file and distribute data    
  if (readBalanced == true)
      loadDistGraphMPIIOBalanced(me, nprocs, ranksPerNode, dg, inputFileName);
  else
      loadDistGraphMPIIO(me, nprocs, ranksPerNode, dg, inputFileName);
 
  MPI_Barrier(MPI_COMM_WORLD);
 
  assert(dg);
#ifdef PRINT_DIST_STATS 
  dg->printStats();
#endif
  
  td1 = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rus);
  
#ifdef DETAILOP
#if defined(DONT_CREATE_DIAG_FILES)
  std::cout << "Time to create distributed graph: " << (td1 - td0) << std::endl;
  std::cout << "Memory used: " << (static_cast<double>(rus.ru_maxrss) * 1024.0) / 1048576.0 <<std::endl;
#else
  ofs << "Time to create distributed graph: " << (td1 - td0) << std::endl;
  ofs << "Memory used: " << (static_cast<double>(rus.ru_maxrss) * 1024.0) / 1048576.0 <<std::endl;
#endif
#else
  if(me == 0)
    ofs << "Time to create distributed graph: " << (td1 - td0) << std::endl;
#endif  
#ifdef USE_OMP
  if(me == 0)
#if defined(DONT_CREATE_DIAG_FILES)
    std::cout << "Threads Enabled : " << omp_get_max_threads() << std::endl;
#else
    ofs << "Threads Enabled : " << omp_get_max_threads() << std::endl;
#endif
#endif

  GraphWeight currMod = -1.0, prevMod = -1.0;
  double total=0;
  int phase = 0, short_phase = 0;

  int iters = 0, tot_iters = 0;
  ColorElem numColors;
  ColorVector colors;
  CommunityVector cvect;
  GraphWeight threshold;

  std::vector<GraphElem> ssizes, rsizes, svdata, rvdata;
  size_t ssz = 0U, rsz = 0U;
  const GraphElem nv = dg->getTotalNumVertices();
    
  // gather all communities in root
  std::vector<GraphElem> commAll, cvectAll;
  
  // initialize map
  if (me == 0 && (outputFiles || compareCommunities))
      commAll.resize(nv, -1);

  MPI_Barrier(MPI_COMM_WORLD);

  // outermost loop
  while(1) {
    double ptotal = 0;
    
    // if threshold-scaling is ON, then
    // use larger threshold towards the
    // beginning, and smaller one towards
    // the end
    if (thresholdScaling && !runOnePhase) {
        if (short_phase > 12)
            short_phase = 0;

        if (short_phase >= 0 && short_phase <= 2)
            threshold = 1.0E-3;
        else if (short_phase >= 3 && short_phase <= 6)
            threshold = 1.0E-4;
        else if (short_phase >= 7 && short_phase <= 9)
            threshold = 1.0E-5;
        else if (short_phase >= 10 && short_phase <= 12)
            threshold = 1.0E-6;
    }
    else
        threshold = 1.0E-6;

    t1 = MPI_Wtime();
    // only invoke coloring for first phase when the graph is the largest
    if (coloring && (phase == 0)) { 
        t1 = MPI_Wtime();
        numColors = distColoringMultiHashMinMax(me, nprocs, *dg, colors, (maxColors/2), MAX_COVG, false);
#if defined(DONT_CREATE_DIAG_FILES)
        if (me == 0) std::cout << "Number of colors (2*nHash): " << numColors << std::endl;
#else
        if (me == 0) ofs << "Number of colors (2*nHash): " << numColors << std::endl;
#endif
        t0 = MPI_Wtime();
        if(me == 0) 
#if defined(DONT_CREATE_DIAG_FILES)
            std::cout<< "Coloring Time: "<<t0-t1<<std::endl;
#else
            ofs<< "Coloring Time: "<<t0-t1<<std::endl;
#endif
        if (earlyTerm && ETType == 1) {
            currMod = distLouvainMethodWithColoring(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, true);
        }
        else if (earlyTerm && ETType == 2) {
            currMod = distLouvainMethodWithColoring(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, ETDelta, true);
        }
        else if (earlyTerm && ETType == 3) {
            currMod = distLouvainMethodWithColoring(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, false);
        }
        else if (earlyTerm && ETType == 4) {
            currMod = distLouvainMethodWithColoring(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, ETDelta, false);
        }
        else {
            currMod = distLouvainMethodWithColoring(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters);
        }
    }
    else if (vertexOrdering && (phase == 0)) {
        t1 = MPI_Wtime();
        numColors = distColoringMultiHashMinMax(me, nprocs, *dg, colors, (maxColors/2), MAX_COVG, false);
#if defined(DONT_CREATE_DIAG_FILES)
        if (me == 0) std::cout << "Number of colors (2*nHash): " << numColors << std::endl;
#else
        if (me == 0) ofs << "Number of colors (2*nHash): " << numColors << std::endl;
#endif
        t0 = MPI_Wtime();
        if(me == 0) 
#if defined(DONT_CREATE_DIAG_FILES)
            std::cout<< "Coloring Time: "<<t0-t1<<std::endl;
#else
            ofs<< "Coloring Time: "<<t0-t1<<std::endl;
#endif
        if (earlyTerm && ETType == 1) {
            currMod = distLouvainMethodVertexOrder(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, true);
        }
        else if (earlyTerm && ETType == 2) {
            currMod = distLouvainMethodVertexOrder(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, ETDelta, true);
        }
        else if (earlyTerm && ETType == 3) {
            currMod = distLouvainMethodVertexOrder(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, false);
        }
        else if (earlyTerm && ETType == 4) {
            currMod = distLouvainMethodVertexOrder(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters, ETDelta, false);
        }
        else {
            currMod = distLouvainMethodVertexOrder(me, nprocs, *dg, numColors+1, colors, 
                    ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, currMod, threshold, 
                    iters);
        }
    }
    else {
        if (earlyTerm && ETType == 1) {
            currMod = distLouvainMethod(me, nprocs, *dg, ssz, rsz, ssizes, rsizes, 
                    svdata, rvdata, cvect, currMod, threshold, iters, true);
        }
        else if (earlyTerm && ETType == 2) {
            currMod = distLouvainMethod(me, nprocs, *dg, ssz, rsz, ssizes, rsizes, 
                    svdata, rvdata, cvect, currMod, threshold, iters, ETDelta, true);
        }
        else if (earlyTerm && ETType == 3) {
            currMod = distLouvainMethod(me, nprocs, *dg, ssz, rsz, ssizes, rsizes, 
                    svdata, rvdata, cvect, currMod, threshold, iters, false);
        }
        else if (earlyTerm && ETType == 4) {
            currMod = distLouvainMethod(me, nprocs, *dg, ssz, rsz, ssizes, rsizes, 
                    svdata, rvdata, cvect, currMod, threshold, iters, ETDelta, false);
        }
        else {
            currMod = distLouvainMethod(me, nprocs, *dg, ssz, rsz, ssizes, rsizes, 
                    svdata, rvdata, cvect, currMod, threshold, iters);
        }
    }
    t0 = MPI_Wtime();

    tot_iters += iters;
         
    ptotal += (t0-t1);
     
    if((currMod - prevMod) > threshold) {
               
        /// Store communities in every phase
        if (outputFiles || compareCommunities) { 

            // gather cvect into root
            gatherAllComm(0 /*root*/, me, nprocs, cvectAll, cvect);

            // update community info
            if (me == 0) {
#ifdef DEBUG_PRINTF    
                std::cout << "Updated community list per level into global array at root..." << std::endl;
#endif
                // ensure clusters are contiguous
                const GraphElem Nc = cvectAll.size();
                std::vector<GraphElem> cvectRen(Nc, 0);
                std::map<GraphElem, GraphElem> kval;
                GraphElem nidx = 0;

                std::copy(cvectAll.begin(), cvectAll.end(), cvectRen.begin());
                std::sort(cvectRen.begin(), cvectRen.end()); // default < cmp

                // create a dictionary and map to new indices
                for (GraphElem n = 0; n < Nc; n++)
                {     
                    if (kval.find(cvectRen[n]) == kval.end())
                    {
                        kval.insert(std::pair<GraphElem,GraphElem>(cvectRen[n], nidx));
                        nidx++;
                    }
                }

                // update cvectAll
                for (GraphElem c = 0; c < Nc; c++)
                    cvectAll[c] = kval[cvectAll[c]];

                cvectRen.clear();
                kval.clear();
                if (phase == 0) 
                    std::copy(cvectAll.begin(), cvectAll.end(), commAll.begin());
                else {
                    for (GraphElem p = 0; p < commAll.size(); p++)
                        commAll[p] = cvectAll[commAll[p]];
                }
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        /// Create new graph and rebuild 
        if (!runOnePhase) {
            t3 = MPI_Wtime();

            distbuildNextLevelGraph(nprocs, me, dg, ssz, rsz, 
                    ssizes, rsizes, svdata, rvdata, cvect);

            t2 = MPI_Wtime();

            if(me == 0) 
#if defined(DONT_CREATE_DIAG_FILES)
                std::cout<< "Rebuild Time: "<<t2-t3<<std::endl;
#else
                ofs<< "Rebuild Time: "<<t2-t3<<std::endl;
#endif
            ptotal+=(t2-t3);
        }
    }
    else { // modularity gain is not enough, exit phase.

        // if thresholdScaling is ON, and the program exits
        // in between the cycle, make sure it at least runs 
        // ONCE with 1E-6 as threshold before exiting
        GraphWeight tex_1 = 0.0, tex_2 = 0.0;
        if (thresholdScaling && !runOnePhase && phase < 10) {
            tex_1 = MPI_Wtime();
            currMod = distLouvainMethod(me, nprocs, *dg, ssz, rsz, ssizes, rsizes, 
                    svdata, rvdata, cvect, currMod, 1.0E-6, iters);
            tex_2 = MPI_Wtime();
            ptotal += (tex_2 - tex_1);
        }
        
        break;
    }

    if(me == 0 ) {
        teps += dg->getTotalNumEdges() * tot_iters;
#if defined(DONT_CREATE_DIAG_FILES)
        std::cout << " **************************" << std::endl;
        std::cout << "Level "<< phase << std::endl << "Modularity: " << currMod <<", Time: "<<t0-t1<< ", Iterations: " 
            << tot_iters << std::endl;
#else
        ofs << " **************************" << std::endl;
        ofs << "Level "<< phase << std::endl << "Modularity: " << currMod <<", Time: "<<t0-t1<< ", Iterations: " 
            << tot_iters << std::endl;
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    prevMod = currMod;
    currMod = -1.0;
 
    ptotal+=(t0-t1);

    if(me == 0) {
#if defined(DONT_CREATE_DIAG_FILES)
        std::cout<< "Phase Total time: " << ptotal << std::endl;
#else
        ofs<< "Phase Total time: " << ptotal << std::endl;
        ofcks << "**************************" << std::endl;
#endif
    }

    total+=ptotal;
    phase++;

    // Exit if only running for a single phase
    if (runOnePhase) {
        if (me == 0)
            std::cout << "Exiting at the end of first phase." << std::endl;
        break;
    }

    if (thresholdScaling && !runOnePhase)
        short_phase++;

    // high phase count, break anyway
    if (phase == TERMINATION_PHASE_COUNT || (tot_iters > 10000)) {
        if(me == 0) {
            std::cout << "Program ran for " << TERMINATION_PHASE_COUNT 
                << " phases or 10K iterations, exiting..." << std::endl;
            std::cout << "HINT: Check input graph weights..." << std::endl;
            std::cout << "HINT: Compile with -DDEBUG_PRINTF..." << std::endl;
        }
        break;
    }
  } // end of phases
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  double tot_time = 0.0;
  MPI_Reduce(&total, &tot_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(me == 0) {
#if defined(DONT_CREATE_DIAG_FILES)
      std::cout << "TERMINATE TIME: " << (tot_time/nprocs) << std::endl;
      std::cout << "TEPS: " << (double)teps/(double)(tot_time/nprocs) << std::endl;
#else
      ofs<< "TERMINATE TIME: "<< total<<std::endl;
      ofs<< "TEPS: " << (double)teps/(double)(tot_time/nprocs) <<std::endl;
#endif
  }

  // dump community information in a file    
  if (outputFiles) {
      
      if(me == 0) {
          std::string outFileName = inputFileName;
          outFileName += ".communities";

          std::ofstream ofs;
          ofs.open(outFileName.c_str(), std::ios::out);

          if (!ofs) {
              std::cerr << "Error creating community file: " << outFileName << std::endl;
              exit(EXIT_FAILURE);
          }

          // write the data
          for (GraphElem i = 0; i < nv; i++) {
              ofs << commAll[i] << "\n";
          }

          ofs.close();

          std::cout << std::endl;
          std::cout << "-----------------------------------------------------" << std::endl;
          std::cout << "Saved community information (" << commAll.size() 
              << " vertices)  on file: " << outFileName << std::endl;
          std::cout << "-----------------------------------------------------" << std::endl;
      }
      
      MPI_Barrier(MPI_COMM_WORLD);
  }

  // compare computed communities with ground truth
  if(compareCommunities) {
      if(me == 0)
          compare_communities(commGroundTruth, commAll);

      // rest of the processes wait on barrier    
      MPI_Barrier(MPI_COMM_WORLD);
  }

  commAll.clear();
  cvectAll.clear();
  commGroundTruth.clear();

  destroyCommunityMPIType();
  destroyEdgeMPIType();

  colors.clear();
  delete dg;

#if defined(DONT_CREATE_DIAG_FILES)
#else
#if defined(DETAILOP)
  ofs.close();
  ofcks.close();
#else
  if(me == 0)
    ofs.close();
#endif
#endif

  MPI_Finalize();

  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "f:bc:od:r:t:a:ig:zp")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
      break;
    case 'b':
      readBalanced = true;
      break;
    case 'c':
      maxColors = atoi(optarg);
      coloring = true;
      break;
    case 'o':
      outputFiles = true;
      break;
    case 'd':
      maxColors = atoi(optarg);
      vertexOrdering = true;
      break;
    case 'r':
      ranksPerNode = atoi(optarg);
      break;
    case 't':
      earlyTerm = true;
      ETType = atoi(optarg);
      break;
    case 'a':
      ETDelta = atof(optarg);
      break;
    case 'i':
      thresholdScaling = true;
      break;
    case 'g':
      groundTruthFileName.assign(optarg);
      compareCommunities = true;
      break;
    case 'z':
      isGroundTruthZeroBased = false;
      break;
    case 'p':
      runOnePhase = true;
      break;
    default:
      assert(0 && "Should not reach here!!");
      break;
    }
  }

  if (me == 0 && earlyTerm && (ETType < 1 || ETType > 4)) {
      std::cerr << "early-term-type parameter is between 1-4 : -t 1 OR -t 2 OR -t 3 OR -t 4" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }

  if (me == 0 && earlyTerm && (ETType == 2 || ETType == 4) && (ETDelta > 1.0 || ETDelta < 0.0)) {
      std::cerr << "early-term-alpha must be between 0 and 1: -t 2 -a 0.5" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && inputFileName.empty()) {
      std::cerr << "Must specify a binary file name with -f" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
   
  if (me == 0 && compareCommunities && groundTruthFileName.empty()) {
      std::cerr << "Must specify a ground-truth file name with -g" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
  
  if (me == 0 && (coloring && vertexOrdering)) {
      std::cerr << "Cannot turn on both coloring (-c) and vertex ordering (-d). Re-run using one of the options." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }

  if (me == 0 && (runOnePhase && thresholdScaling)) {
      std::cerr << "Cannot turn on threshold-cycling (-i) with a single phase run (-p). Re-run using one of the options." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
} // parseCommandLine
