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

#include "dimacs.hpp"
#include "matrix-market.hpp"
#include "metis.hpp"
#include "simple.hpp"
#include "snap.hpp"

static std::string inputFileName, outputFileName;
static int hdrSize = 0;

static bool matrixMarketFormat = false;

static bool dimacsFormat = false;
static bool dimacs_directed = false;

static bool metisFormat = false;
static bool simpleFormat = false;
static bool simpleFormat2 = false;
static bool snapFormat = false;

static bool output = false;
static bool indexOneBased = false;

// this option will override whatever 
// weights there are in the file, and 
// make weights 1.0 for ALL edges
static bool makeWeightsOne = false;
static bool randomEdgeWeight = false;
static bool origEdgeWeight = false;

static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  parseCommandLine(argc, argv);

  // Only the following formats supported for now
  assert(dimacsFormat || metisFormat || simpleFormat || matrixMarketFormat|| simpleFormat2 || snapFormat);

  Graph *g = NULL;

  rusage rus;

  getrusage(RUSAGE_SELF, &rus);
  std::cout << "Memory used: " << (static_cast<double>(rus.ru_maxrss) * 1024.0) / 1048576.0 <<
    std::endl;

  if (dimacsFormat)
  {
      if (dimacs_directed) {
          if (randomEdgeWeight)
              loadDimacsFile(g, inputFileName, RND_WEIGHT);
          else if (makeWeightsOne)
              loadDimacsFile(g, inputFileName, ONE_WEIGHT);
          else if (origEdgeWeight) 
              loadDimacsFile(g, inputFileName, ORG_WEIGHT);
          else
              loadDimacsFile(g, inputFileName, ABS_WEIGHT);
      }
      else {
          if (randomEdgeWeight)
              loadDimacsFileUn(g, inputFileName, RND_WEIGHT);
          else if (makeWeightsOne)
              loadDimacsFileUn(g, inputFileName, ONE_WEIGHT);
          else if (origEdgeWeight) 
              loadDimacsFileUn(g, inputFileName, ORG_WEIGHT);
          else
              loadDimacsFileUn(g, inputFileName, ABS_WEIGHT);
      }
  }
  else if (metisFormat) {
      if (randomEdgeWeight)
          loadMetisFile(g, inputFileName, RND_WEIGHT);
      else if (makeWeightsOne)
          loadMetisFile(g, inputFileName, ONE_WEIGHT);
      else if (origEdgeWeight) 
          loadMetisFile(g, inputFileName, ORG_WEIGHT);
      else
          loadMetisFile(g, inputFileName, ABS_WEIGHT);
  }
  else if (simpleFormat) {
      if (randomEdgeWeight)
          loadSimpleFile(g, inputFileName, indexOneBased, RND_WEIGHT);
      else if (makeWeightsOne)
          loadSimpleFile(g, inputFileName, indexOneBased, ONE_WEIGHT);
      else if (origEdgeWeight) 
          loadSimpleFile(g, inputFileName, indexOneBased, ORG_WEIGHT);
      else
          loadSimpleFile(g, inputFileName, indexOneBased, ABS_WEIGHT);
  }
  else if (matrixMarketFormat) {
      if (randomEdgeWeight)
          loadMatrixMarketFile(g, inputFileName, RND_WEIGHT);
      else if (makeWeightsOne)
          loadMatrixMarketFile(g, inputFileName, ONE_WEIGHT);
      else if (origEdgeWeight) 
          loadMatrixMarketFile(g, inputFileName, ORG_WEIGHT);
      else
          loadMatrixMarketFile(g, inputFileName, ABS_WEIGHT);
  }
  else if (simpleFormat2) {
      if (randomEdgeWeight)
          loadSimpleFileUn(g, inputFileName, RND_WEIGHT);
      else if (makeWeightsOne)
          loadSimpleFileUn(g, inputFileName, ONE_WEIGHT);
      else if (origEdgeWeight) 
          loadSimpleFileUn(g, inputFileName, ORG_WEIGHT);
      else
          loadSimpleFileUn(g, inputFileName, ABS_WEIGHT);
  }
  else if (snapFormat) {
      if (randomEdgeWeight)
          loadSNAPFile(g, inputFileName, RND_WEIGHT);
      else if (makeWeightsOne)
          loadSNAPFile(g, inputFileName, ONE_WEIGHT);
      else if (origEdgeWeight) 
          loadSNAPFile(g, inputFileName, ORG_WEIGHT);
      else
          loadSNAPFile(g, inputFileName, ABS_WEIGHT);
  }
  else {
      std::cerr << "Format not specified correctly!" << std::endl;
      if (g)
          delete g;
      return 1;
  }

  getrusage(RUSAGE_SELF, &rus);
  std::cout << "Loaded graph from file: " << inputFileName << std::endl;
  std::cout << "Memory used: " << 
      (static_cast<double>(rus.ru_maxrss) * 1024.0) / 1048576.0 
      << std::endl;

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

  while ((ret = getopt(argc, argv, "f:o:md:uesnrizwh:")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
      break;
    case 'o':
      outputFileName.assign(optarg);
      break;
    case 'm':
      matrixMarketFormat = true;
      break;
    case 'd':
      dimacsFormat = true;
      dimacs_directed = (atoi(optarg) == 1 ? true : false);
      break;
    case 'e':
      metisFormat = true;
      break;
    case 's':
      simpleFormat = true;
      break;
    case 'n':
      snapFormat = true;
      break;
    case 'u':
      simpleFormat2 = true;
      break;
    case 'r':
      randomEdgeWeight = true;
      break;
    case 'i':
      origEdgeWeight = true;
      break;
    case 'z':
      indexOneBased = true;
      break;
    case 'w':
      makeWeightsOne = true;
      break;
    case 'h': 
      hdrSize = atoi(optarg);
      assert(hdrSize == 32 || hdrSize == 64);
      break;
    default:
      assert(0 && "Should not reach here!!");
      break;
    }
  }

  if ((matrixMarketFormat || dimacsFormat || metisFormat || simpleFormat 
              || simpleFormat2 || snapFormat) == false) {
    std::cerr << "Must select a file format for the input file!" << std::endl;
    exit(EXIT_FAILURE);
  }
  const bool fileFormat[] = { matrixMarketFormat, dimacsFormat, metisFormat, 
      simpleFormat, simpleFormat2, snapFormat };
  const int numFormats = sizeof(fileFormat) / sizeof(fileFormat[0]);

  int numTrue = 0;
  for (int i = 0; i < numFormats; i++)
    if (numTrue == 0 && fileFormat[i])
      numTrue++;
    else if (numTrue > 0 && fileFormat[i]) {
      std::cerr << "Must select a SINGLE file format for the input file!" << std::endl;
      exit(EXIT_FAILURE);
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
