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
#include <cctype>
#include <cmath>
#include <cstring>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "matrix-market.hpp"

/// Note: Trying out the new random number generator with static seed generator,
//  instead of first generating the seed (based on input filename) and then passing
//  it to the random number generator...

void loadMatrixMarketFile(Graph *&g, const std::string &fileName, Weight_t wtype)
{
  std::ifstream ifs;

  ifs.open(fileName.c_str());
  if (!ifs) {
    std::cerr << "Error opening Matrix Market format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  bool isComplex = false, isPattern = false, isSymmetric = false, isGeneral = false;

  std::getline(ifs, line);
  {
    std::string ls[5];

    std::istringstream iss(line);
    iss >> ls[0] >> ls[1] >> ls[2] >> ls[3] >> ls[4];

    if (!iss) {
      std::cerr << "Error parsing matrix market file: " << fileName << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "Processing: " << ls[0] << " " << ls[1] << " " << ls[2] << " " << ls[3] <<
      " " << ls[4] << std::endl;
    for (int i = 1; i < 5; i++)
      std::transform(ls[i].begin(), ls[i].end(), ls[i].begin(), toupper);

    if (ls[0] != "%%MatrixMarket") {
      std::cerr << "Error in the first line of: " << fileName << " " << ls[0] <<
	", expected: %%MatrixMarket" << std::endl;
      exit(EXIT_FAILURE);
    }

    if (ls[1] != "MATRIX") {
      std::cerr << "The object type should be matrix for file: " << fileName << std::endl;
      exit(EXIT_FAILURE);
    }

    if (ls[2] != "COORDINATE") {
      std::cerr << "The object type should be coordinate for file: " << fileName << std::endl;
      exit(EXIT_FAILURE);
    }

    if (ls[3] == "COMPLEX") {
      std::cout << "Warning will only process the real part" << std::endl;
      isComplex = true;
    }

    if (ls[3] == "PATTERN") {
      std::cout << "Note: matrix type is pattern.  All weights will be set to 1.0 unless -r is passed" <<
	std::endl;
      isPattern = true;
    }

    if (ls[4] == "GENERAL")
      isGeneral = true;
    else if (ls[4] == "SYMMETRIC") {
      std::cout << "Note: matrix type is symmetric" << std::endl;
      isSymmetric = true;
    }
  }

  if (!isSymmetric && !isGeneral) {
    std::cerr << "Error: matrix market type should be SYMMETRIC or GENERAL for file: " << fileName <<
      std::endl;
    exit(EXIT_FAILURE);
  }

  do {
    std::getline(ifs, line);
  } while (line[0] == '%');

  GraphElem numVertices, numEdges;
  GraphElem ns, nt, ne;

  std::istringstream iss(line);

  iss >> ns >> nt >> ne;
  if (!iss || iss.fail()) {
      std::cerr << "Error parsing Matrix Market format: " << line << std::endl;
      exit(EXIT_FAILURE);
  }

  numVertices = ns;
  numEdges = ne;

  std::cout << "Loading Matrix Market file: " << fileName << ", numvertices: " << numVertices <<
      ", numEdges: " << numEdges << std::endl;
  
  std::string crd;
  GraphElem source, dest;
  GraphWeight weight = 1.0;
  std::vector<GraphElem> edgeCount(numVertices+1);
  std::vector<GraphElemTuple> edgeList;

  // weights will be converted to positive numbers
  if (isSymmetric) {

      for (GraphElem i = 0; i < numEdges; i++) {

          std::getline(ifs, crd);
          std::istringstream iss(crd);

          if (isPattern)
              iss >> source >> dest;
          else {
              iss >> source >> dest >> weight;
              if (wtype == ABS_WEIGHT)
                  weight = std::fabs(weight);
          }

          source--; // Matrix market has 1-based indexing
          dest--;

          assert((source >= 0) && (source < numVertices));
          assert((dest >= 0) && (dest < numVertices));

          if (wtype == ONE_WEIGHT)
              weight = 1.0;

          if (wtype == RND_WEIGHT)
              weight = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

          if (source != dest) {
              edgeList.emplace_back(source, dest, weight);
              edgeList.emplace_back(dest, source, weight);
              edgeCount[source+1]++;
              edgeCount[dest+1]++;
          }
          else {
              edgeList.emplace_back(source, dest, weight);
              edgeCount[source+1]++;
          }
      }
  }
  else { // if General type (directed)
      
      for (GraphElem i = 0; i < numEdges; i++) {

          std::getline(ifs, crd);
          std::istringstream iss(crd);

          if (isPattern)
              iss >> source >> dest;
          else {
              iss >> source >> dest >> weight;
              if (wtype == ABS_WEIGHT)
                  weight = std::fabs(weight);
          }

          source--; // Matrix market has 1-based indexing
          dest--;

          assert((source >= 0) && (source < numVertices));
          assert((dest >= 0) && (dest < numVertices));

          if (wtype == ONE_WEIGHT)
              weight = 1.0;

          if (wtype == RND_WEIGHT)
              weight = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

          edgeList.emplace_back(source, dest, weight);
          edgeCount[source+1]++;
      }
  }

  ifs.close();

  numEdges = edgeList.size();

  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
} // loadMatrixMarketFile
