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
#include <cstdint>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "../graph.hpp"
#include "../utils.hpp"

#include "parallel-shards.hpp"

static std::string inputFileName, outputFileName, shardedFileArgs;
static bool indexOneBased = false;
static int nAggrPEs = 1;

// this option will override whatever 
// weights there are in the file, and 
// make weights 1.0 for ALL edges
static bool makeWeightsOne = false;
static bool randomEdgeWeight = false;
static bool origEdgeWeight = false;

static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  int me, size;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  parseCommandLine(argc, argv);

  // fetch shard specific arguments, expecting four
  std::stringstream ss(shardedFileArgs);
  std::string s;
  std::vector<std::string> args;

  while (std::getline(ss, s, ' ')) {
      args.push_back(s);
  }

  if (args.size() != 4) {
      std::cerr << "Expected arguments to '-x' (in this order): <num-files> <start-chunk> <end-chunk> <shard-count>" 
          << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }

  GraphElem numFiles = std::stol(args[0]);
  GraphElem startChunk = std::stol(args[1]);
  GraphElem endChunk = std::stol(args[2]);
  GraphElem shardCount = std::stol(args[3]);

  args.clear();
 
  double t0, t1, t2, rt;
  t0 = mytimer();

  if (randomEdgeWeight)
      loadParallelFileShards(me, size, nAggrPEs, inputFileName, outputFileName, 
              startChunk, endChunk, indexOneBased, RND_WEIGHT, shardCount);
  else if (makeWeightsOne)
      loadParallelFileShards(me, size, nAggrPEs, inputFileName, outputFileName, 
              startChunk, endChunk, indexOneBased, ONE_WEIGHT, shardCount);
  else if (origEdgeWeight) 
      loadParallelFileShards(me, size, nAggrPEs, inputFileName, outputFileName, 
              startChunk, endChunk, indexOneBased, ORG_WEIGHT, shardCount);
  else
      loadParallelFileShards(me, size, nAggrPEs, inputFileName, outputFileName, 
              startChunk, endChunk, indexOneBased, ABS_WEIGHT, shardCount);

  t1 = mytimer();
  t2 = t1 - t0;
  MPI_Reduce(&t2, &rt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (me == 0) {
      std::cout << "Average time reading " << numFiles << " sharded files and writing binary file (in secs): " << rt / size << std::endl;
  }

  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "f:o:n:x:zw")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
      break;
    case 'o':
      outputFileName.assign(optarg);
      break;
    case 'x':
      shardedFileArgs.assign(optarg);
      break;
    case 'n':
      nAggrPEs = atoi(optarg);
      break;
    case 'z':
      indexOneBased = true;
      break;
    case 'w':
      makeWeightsOne = true;
      break;
    default:
      assert(0 && "Should not reach here!!");
      break;
    }
  }

  if (inputFileName.empty()) {
    std::cerr << "Must specify an input file name with -f" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (outputFileName.empty()) {
    std::cerr << "Must specify an output file name with -o" << std::endl;
    exit(EXIT_FAILURE);
  }
} // parseCommandLine
