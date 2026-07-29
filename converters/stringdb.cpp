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

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <utility>
#include <random>
#include <unordered_map>

#include "stringdb.hpp"

/// This loader assumes undirected input file to have the following format:
/// First line: a b c 
/// Second line onward: str-u str-v int-weight
void loadStringDBFile(Graph *&g, const std::string &fileName)
{
  std::ifstream ifs;

  double t0, t1;

  t0 = mytimer();

  ifs.open(fileName.c_str(), std::ifstream::in);
  if (!ifs) {
    std::cerr << "Error opening StringDB format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  GraphElem numEdges = 0, numVertices = 0;
  bool is_score_missing = true;
  
  std::string line;
  std::getline(ifs, line);
       
  if (line.find("_score") != std::string::npos) 
     is_score_missing = false;

  if (!is_score_missing) {
       std::cout << "Reading edges from StringDB format file (assuming undirected): " << fileName << std::endl;
  }
  else {
    std::cerr << "Unable to find `_score in the first line of StringDB format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }
 
  std::vector<GraphElemTuple> edgeList;
  std::vector<GraphElem> edgeCount, intWeights;
  std::unordered_map<std::string, GraphElem> vmap;

  do {
    std::getline(ifs, line);
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    std::string v0, v1;
    GraphElem w = 1;
    
    std::istringstream iss(line);
    iss >> v0 >> v1 >> w;

    auto it = vmap.find(v0);

    if (it == vmap.end()) {
      vmap.emplace(v0, numVertices);
      numVertices++;
    }

    numEdges++;
    intWeights.push_back(w);
  } while (!ifs.eof());

  GraphElem max_weight = *std::max_element(intWeights.begin(), intWeights.end()); 
  edgeCount.resize(numVertices+1, 0);

  ifs.clear();
  ifs.seekg(0, std::ios::beg);  
  
  // dummy first line before writing data
  std::getline(ifs, line);

  do {
    std::getline(ifs, line);

    std::string v0, v1;
    GraphElem w = 1;
    
    std::istringstream iss(line);
    iss >> v0 >> v1 >> w;

    const GraphElem i_v0 = vmap[v0];
    const GraphElem i_v1 = vmap[v1];
    const GraphWeight d_w = ((GraphWeight) w / (GraphWeight) max_weight);
    
    edgeList.push_back({i_v0, i_v1, d_w});
    edgeCount[i_v0+1]++;

  } while (!ifs.eof());
  
  ifs.close();

  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
 
  /// write the mapping
  std::ofstream ofs;
  std::string fileNameExt = fileName + ".mapping." + std::to_string(numVertices);
  ofs.open(fileNameExt.c_str(), std::ofstream::out);

  if (!ofs) {
    std::cerr << "Error creating file to store mappings: " << fileNameExt << std::endl;
    exit(EXIT_FAILURE);
  }

  for (auto vm: vmap)
    ofs << vm.first << " " << vm.second << std::endl;
  ofs.close();
  std::cout << "String to Integer vertex ID mappings are written in file: " << fileNameExt << std::endl;

  t1 = mytimer();  
  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadStringDBFile
