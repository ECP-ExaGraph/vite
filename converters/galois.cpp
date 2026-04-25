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

#include <filesystem>
namespace fs = std::filesystem;

#include "galois.hpp"

//TODO FIXME this converter has unresolved bugs!!!

template <typename T>
void read_file(std::string const& fname, const size_t count, std::vector<T>& buffer) 
{
  std::ifstream file(fname.c_str(), std::ios::binary);
  buffer.resize(count); 
  if (!file.good()) {
    std::cerr << "Failed to open file: " << fname << "\n";
    exit(1);
  }
  file.read(reinterpret_cast<char*>(buffer.data()), sizeof(T) * count);
  file.close();
}

void loadGaloisFileUn(Graph *&g, const std::string &filePrefix)
{
  std::ifstream ifs;

  double t0, t1;

  t0 = mytimer();
 
  fs::path fp(filePrefix);
  fs::path dir = fp.parent_path();

  if (!fs::exists(dir)) {
    std::cerr << "Error opening Galois prefix file path: " << filePrefix << std::endl;
    exit(EXIT_FAILURE);
  }

  // taken from GraphMinerBench: https://github.com/chenxuhao/GraphMiner.git
  // read meta information
  std::ifstream f_meta((filePrefix + ".meta.txt").c_str());
  assert(f_meta);
  int vid_size = 0, eid_size = 0, vlabel_size = 0, elabel_size = 0, max_degree, feat_len, num_vertex_classes, num_edge_classes;
  GraphElem n_vertices, n_edges;

  f_meta >> n_vertices >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
         >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(max_degree > 0 && max_degree < n_vertices);
  f_meta.close();

  std::cout << "Galois meta file reports vertices, edges: " << n_vertices << ", " << n_edges << std::endl;
  
  // read row pointers and column indices
  // using vid_size and eid_size from meta file causes garbage reads,
  // most probably both are int32_t 
  std::vector<int32_t> vbuf;
  std::vector<int32_t> ebuf; 
  read_file<int32_t>(filePrefix + ".vertex.bin", (n_vertices+1), vbuf);
  read_file<int32_t>(filePrefix + ".edge.bin", n_edges, ebuf);
  // why are there zeros in rowptr?
  vbuf.erase(std::remove(vbuf.begin()+1, vbuf.end(), 0), vbuf.end());
  
  // combine into Vite binary format
  g = new Graph(n_vertices, n_edges);
  g->setEdgeStartForVertex(0, 0);
  for (GraphElem i = 0; i < n_vertices; i++)
    g->setEdgeStartForVertex(i + 1, (GraphElem)vbuf[i + 1]);

  GraphElem ePos = 0;
  for (GraphElem i = 0; i < n_vertices; i++) {
	  GraphElem e0, e1;
	  g->getEdgeRangeForVertex(i, e0, e1);

	  if ((i % 100000) == 0)
		  std::cout << "Processing edges for vertex: " << i << ", range(" << e0 << ", " << e1 <<
			  ")" << std::endl;

	  for (GraphElem j = e0; j < e1; j++) {
		  Edge& edge = g->getEdge(j);
		  
      assert(ePos == j);
      
      edge.tail = (GraphElem)ebuf[ePos];
		  edge.weight = 1.0;
		  
      ePos++;
	  }
  }

  t1 = mytimer();  
  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadGaloisFileUn
