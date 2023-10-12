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

#include "nkit-er.hpp"
#include "nkit-rgg.hpp"
#include "nkit-ba.hpp"
#include "nkit-hg.hpp"

static std::string outputFileName;

// G = (N,p) for ER
static unsigned int N = 0; 
static double p = 0.0;      // probability
// G = (N,k) for RGG
static unsigned int k = 0;  // nclusters
static unsigned int m0 = 0;  // initial attached vertices

static bool erGen       = false;  // ER
static bool rggGen      = false;  // RGG
static bool baGen       = false;  // Barabasi-Albert
static bool hgGen       = false;  // Hyperbolic

// create random weights (can be used for matching)
static bool randomEdgeWeight = false;

static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  parseCommandLine(argc, argv);

  // Only the following generators supported for now
  assert(erGen || rggGen || baGen || hgGen);

  Graph *g = NULL;
  rusage rus;
  
  if (erGen)
      generateER(g, N, p, randomEdgeWeight);
  else if (rggGen)
      generateRGG(g, N, k, randomEdgeWeight);
  else if (baGen)
      generateBA(g, N, m0, k, randomEdgeWeight);
  else if (hgGen)
      generateHG(g, N, randomEdgeWeight);
  else {
      std::cerr << "Generator not specified correctly!" << std::endl;
      if (g)
          delete g;
      return 1;
  }

  getrusage(RUSAGE_SELF, &rus);
  std::cout << "Generated file of size " << N << " x " << N << std::endl;
  std::cout << "Memory used: " << (static_cast<double>(rus.ru_maxrss) * 1024.0) / 1048576.0 <<
    std::endl;

  double t0, t1;

  t0 = mytimer();

  std::ofstream ofs(outputFileName.c_str(), std::ofstream::out | std::ofstream::binary |
		    std::ofstream::trunc);
  if (!ofs) {
    std::cerr << "Error opening output file: " << outputFileName << std::endl;
    exit(EXIT_FAILURE);
  }

  GraphElem nv, ne;

  nv = g->getNumVertices();
  ne = g->getNumEdges();

  ofs.write(reinterpret_cast<char *>(&nv), sizeof(GraphElem));
  ofs.write(reinterpret_cast<char *>(&ne), sizeof(GraphElem));

  ofs.write(reinterpret_cast<char *>(&g->edgeListIndexes[0]), (nv+1)*sizeof(GraphElem));

  for (GraphElem v = 0; v < nv; v++) {
      GraphElem e0, e1;

      g->getEdgeRangeForVertex(v, e0, e1);

      for (GraphElem j = e0; j < e1; j++) {
          const Edge &edge = g->getEdge(j);

          ofs.write(reinterpret_cast<const char *>(&edge), sizeof(Edge));
      }
  }

  ofs.close();
  delete g;

  t1 = mytimer();

  std::cout << "Time writing binary file: " << (t1 - t0) << std::endl;
      
  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "o:en:p:bhgm:k:r")) != -1) {
    switch (ret) {
    case 'o':
      outputFileName.assign(optarg);
      break;
    case 'e':
      erGen = true;
      break;
    case 'g':
      rggGen = true;
      break;
    case 'b':
      baGen = true;
      break;
    case 'h':
      hgGen = true;
      break;
    case 'n':
      N = atoi(optarg);
      break;
    case 'p':
      p = atof(optarg);
      break;
    case 'k':
      k = atoi(optarg);
      break;
    case 'r': 
      randomEdgeWeight = true;
      break;
    case 'm':
      m0 = atoi(optarg);
      break;
    default:
      assert(0 && "Incorrect parameter!!");
      break;
    }
  }

  if ((erGen || rggGen || baGen || hgGen) == false) {
    std::cerr << "Must select a generator for the input file!" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  if (outputFileName.empty()) {
    std::cerr << "Must specify an output file name with -o" << std::endl;
    exit(EXIT_FAILURE);
  }
} // parseCommandLine
