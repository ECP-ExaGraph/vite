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
#include <cassert>
#include <cstdint>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <utility>

#include "binary.hpp"
#include "../utils.hpp"

void loadBinaryFile(Graph *&g, const std::string &fileName, PartitionVector *parts)
{
  std::ifstream ifs;

  double t0, t1, t2, t3;

  t0 = mytimer();

  ifs.open(fileName.c_str(), std::ifstream::in | std::ifstream::binary);
  if (!ifs) {
    std::cerr << "Error opening binary format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  long int nv, ne;
  int32_t header, nvi, nei;

  ifs.read(reinterpret_cast<char *>(&header), sizeof(header));

  assert((header == 32) || (header == 64));

  if (header == 32) {
    ifs.read(reinterpret_cast<char *>(&nvi), sizeof(nvi));
    ifs.read(reinterpret_cast<char *>(&nei), sizeof(nei));

    nv = nvi;
    ne = nei;
  }
  else {
    ifs.read(reinterpret_cast<char *>(&nv), sizeof(nv));
    ifs.read(reinterpret_cast<char *>(&ne), sizeof(ne));
  }

  std::cout << "Loading " << header << "-bit binary file: " << fileName << ", numvertices: " <<
    nv << ", numEdges: " << ne << std::endl;

  std::vector<GraphElem> edgeCount(nv + 1L);
  std::vector<GraphElemPair> edgeList;

  if (header == 32) {
    int32_t *edgeListRaw = new int32_t[ne * 2L];

    ifs.read(reinterpret_cast<char *>(edgeListRaw), sizeof(int32_t) * (ne * 2L));

    edgeList.resize(ne);
    for (long int i =0L; i < ne; i++) {
      edgeList[i].first = edgeListRaw[(2L * i)];
      edgeList[i].second = edgeListRaw[(2L * i) + 1L];

      edgeCount[edgeList[i].first + 1L]++;
    }
    delete [] edgeListRaw;
  }
  else {
    int64_t *edgeListRaw = new int64_t[ne * 2L];

    ifs.read(reinterpret_cast<char *>(edgeListRaw), sizeof(int64_t) * (ne * 2L));

    edgeList.resize(ne);
    for (long int i = 0L; i < ne; i++) {
      edgeList[i].first = edgeListRaw[(2L * i)];
      edgeList[i].second = edgeListRaw[(2L * i) + 1L];

      edgeCount[edgeList[i].first + 1L]++;
    }
    delete [] edgeListRaw;
  }
  if (parts) {
    char *partsRaw = new char[nv];

    ifs.read(partsRaw, sizeof(char) * nv);
    parts->resize(nv);
    std::copy(partsRaw, partsRaw + nv, parts->begin());
    delete [] partsRaw;
  }
  ifs.close();

  t2 = mytimer();

  std::cout << "File read time: " << (t2 - t0) << std::endl;

  t2 = mytimer();

  std::cout << "Before allocating graph" << std::endl;

  g = new Graph(nv, ne);

  assert(g);

  std::cout << "Allocated graph: " << g << std::endl;

  processGraphData(*g, edgeCount, edgeList, nv, ne);

  t3 = mytimer();  

  std::cout << "Graph populate time: " << (t3 - t2) << std::endl;
  std::cout << "Total I/O time: " << (t3 - t0) << std::endl;
} // loadBinaryFile
