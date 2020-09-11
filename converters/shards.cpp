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
#include <map>

#include "shards.hpp"

/// Pairwise read files between fileStartIndex and fileEndIndex and store 
/// it into a single file, also selectively convert to binary

/// fileInShardsPath expects the input directory where the shards are ('/' is not allowed at the end)

/// the input file is expected to contain 4 data points in every line:
/// ----ai, aj, common, jaccard---- (common is not processed at present)

/// We expect the shards to contain the 'upper triangle' of the adjacency...during binary 
/// conversion, we consider the data to be an undirected graph, and store both combinations of edge pairs

void loadFileShards(Graph *&g, const std::string &fileInShardsPath, 
		const int fileStartIndex, const int fileEndIndex, bool indexOneBased, 
		Weight_t wtype, GraphElem shardCount)
{
  double t0, t1;
  t0 = mytimer();

  assert(fileStartIndex >= 0);
  assert(fileEndIndex >= 0);
  assert(fileEndIndex >= fileStartIndex);
  
  GraphElem numFiles = 0, numEdges, numVertices, v_idx = 0, maxVertex = -1;
  std::vector<GraphElemTuple> edgeList;

  /// Part 1: Read the file shards 
  // start reading 
  for (GraphElem ci = fileStartIndex; ci < fileEndIndex + 1; ci++) {
	  for (GraphElem cj = fileStartIndex; cj < fileEndIndex + 1; cj++) {

		  // construct file name for each shard
		  std::string fileNameShard = std::to_string(ci) + "__" + std::to_string(cj) + ".csv";
		  std::string fileName = fileInShardsPath + "/" + fileNameShard;
	  
		  // open file shard
		  std::ifstream ifs;
		  ifs.open(fileName.c_str(), std::ifstream::in);
		  
		  if (ifs.fail()) 
			continue;
	
		  // actual chunk lo/hi range
		  GraphElem v_lo = (ci - 1)*shardCount;
		  GraphElem v_hi = (cj - 1)*shardCount;
		  std::cout << "File processing: " << fileNameShard << "; Ranges: " << v_lo  << ", " << v_hi << std::endl;
		  numFiles++;

		  // start reading shard
		  std::string line, dummyline;
		  std::getline(ifs, dummyline);

		  while(std::getline(ifs, line)) {

			  GraphElem v0, v1, info;
			  GraphWeight w;
			  char ch;

			  std::istringstream iss(line);

			  // read from current shard 
			  if (wtype == ORG_WEIGHT || wtype == ABS_WEIGHT)
				  iss >> v0 >> ch >> v1 >> ch >> info >> ch >> w;
			  else
				  iss >> v0 >> ch >> v1 >> ch >> info;

			  if (!indexOneBased) {
				  v0--; 
				  v1--;
			  }

			  // normalize v0/v1 by adding lo/hi shard ID
			  v0 += v_lo;
			  v1 += v_hi;

			  if (v0 == v1)
				  continue;
					  
                          if (v0 > maxVertex)
				  maxVertex = v0;
			  if (v1 > maxVertex)
				  maxVertex = v1;
	  
			  if (wtype == ONE_WEIGHT)
				  w = 1.0;

			  if (wtype == RND_WEIGHT)
				  w = (GraphWeight)genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

			  if (wtype == ABS_WEIGHT)
				  w = std::fabs(w);

			  edgeList.emplace_back(v0, v1, w);
			  edgeList.emplace_back(v1, v0, w);
		  }

		  // close current shard
		  ifs.close();
	  }
  }

  if (!indexOneBased)
	  numVertices = maxVertex + 1;
  else
	  numVertices = maxVertex;
  numEdges = edgeList.size();
  double t2 = mytimer();  
  
  std::cout << "Read " << numFiles << " file shards (undirected data)." << std::endl;
  std::cout << "Unadjusted #vertices: " << numVertices << " and #edges: " << numEdges << std::endl;
  std::cout << "Time taken for serial file I/O: " << (t2 - t0) << " secs." << std::endl;

  /// Part 1.5: remap vertex IDs   

  std::vector< GraphElem > vertexMap(numVertices, 0);
  for (GraphElem i = 0; i < numEdges; i++) {

	  if (vertexMap[edgeList[i].i_] == 0) 
              vertexMap[edgeList[i].i_] = 1;
	  if (vertexMap[edgeList[i].j_] == 0) 
              vertexMap[edgeList[i].j_] = 1;
  }

  for (GraphElem i = 0; i < numVertices; i++) {

	  if (vertexMap[i]) {
              vertexMap[i] = v_idx;
              v_idx += 1;
          }
  }

  /// Part 2: perform edge counts and generate the binary file    

  numVertices = v_idx;
  std::cout << "Updated number of vertices: " << numVertices << std::endl;
  for (GraphElem i = 0; i < numEdges; i++) {
	  
	  if (edgeList[i].i_ != vertexMap[edgeList[i].i_])
              edgeList[i].i_ = vertexMap[edgeList[i].i_];
	  if (edgeList[i].j_ != vertexMap[edgeList[i].j_])
              edgeList[i].j_ = vertexMap[edgeList[i].j_];
  }
  std::cout << "Remapped edge IDs..." << std::endl;
  std::vector< GraphElem > edgeCount(numVertices + 1, 0);
  for (GraphElem i = 0; i < numEdges; i++) {

          assert(edgeList[i].i_ >= 0 && edgeList[i].i_ < numVertices);	  
          assert(edgeList[i].j_ >= 0 && edgeList[i].j_ < numVertices);	  

	  edgeCount[edgeList[i].i_+1]++;
	  edgeCount[edgeList[i].j_+1]++;
  }
  std::cout << "Counted the #edges..." << std::endl;
  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);

  t1 = mytimer();  

  std::cout << "Total time (processing shards + binary generation): " << (t1 - t2) << " secs." << std::endl;
} // loadFileShards
