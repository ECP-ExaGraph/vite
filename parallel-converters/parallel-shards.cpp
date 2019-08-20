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
#include <climits>

#include "parallel-shards.hpp"

/// Pairwise read files between fileStartIndex and fileEndIndex and store 
/// it into a single file, also selectively convert to binary

/// fileInShardsPath expects the input directory where the shards are ('/' is not allowed at the end)

/// the input file is expected to contain 4 data points in every line:
/// ----ai, aj, common, jaccard---- (common is not processed at present)

/// We expect the shards to contain the 'upper triangle' of the adjacency...during binary 
/// conversion, we consider the data to be an undirected graph, and store both combinations of edge pairs

void loadParallelFileShards(int rank, int nprocs, int naggr, Graph* g, 
        const std::string &fileInShardsPath, const std::string &fileOutPath, 
        const int fileStartIndex, const int fileEndIndex, bool indexOneBased, 
        Weight_t wtype, GraphElem shardCount)
{
  assert(fileStartIndex >= 0);
  assert(fileEndIndex >= 0);
  assert(fileEndIndex >= fileStartIndex);
  
  GraphElem maxVertex = -1, numEdges = 0, numVertices;
  int file_open_error;
  MPI_File fh;
  MPI_Status status;

  /// Part 1: Read the file shards into edge list  
  
  std::vector<GraphElemTuple> edgeList;
  std::map<std::string,GraphElem> fileProc;

  // make a list of the files and processes
  GraphElem proc = 0;
  for (GraphElem ci = fileStartIndex; ci < fileEndIndex + 1; ci++) {
      for (GraphElem cj = fileStartIndex; cj < fileEndIndex + 1; cj++) {

          // construct file name
          std::string fileNameShard = std::to_string(ci) + "__" + std::to_string(cj) + ".csv";
          std::string fileName = fileInShardsPath + "/" + fileNameShard;

          // check if file exists
          std::ifstream ifs;
          ifs.open(fileName.c_str(), std::ifstream::in);

          if (ifs.fail())
              continue;

          // push in the dictionary
          fileProc.insert(std::pair<std::string,GraphElem>(fileName, proc));

          proc++;

          ifs.close();
      }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read the files 
  for (std::map<std::string, GraphElem>::iterator mpit = fileProc.begin(); mpit != fileProc.end(); ++mpit) {

      if (rank == mpit->second) {
         
          // retrieve lo/hi range from file name string
          std::string fileName_full = (mpit->first).substr((mpit->first).find_last_of("/") + 1);
          std::string fileName_noext = fileName_full.substr(0, fileName_full.find("."));
          std::string fileName_right = fileName_noext.substr(fileName_full.find("__") + 2);
          std::string fileName_left = fileName_noext.substr(0, fileName_full.find("__"));
          
          GraphElem v_lo = (std::stol(fileName_left) - 1)*shardCount;
          GraphElem v_hi = (std::stol(fileName_right) - 1)*shardCount;
#if defined(DEBUG_PRINTF)
          std::cout << "File processing: " << fileNameShard << "; Ranges: " << v_lo  << ", " << v_hi << std::endl;
#endif
          // open file shard and start reading
          std::ifstream ifs;
          ifs.open((mpit->first).c_str(), std::ifstream::in);
 
          std::string line;
          std::getline(ifs, line); // ignore first line

          while(std::getline(ifs, line)) {

              GraphElem v0, v1, info;
              GraphWeight w;
              char ch;

              std::istringstream iss(line);

              // read from current shard 
              if (wtype == ORG_WEIGHT)
                  iss >> v0 >> ch >> v1 >> ch >> info >> ch >> w;
              else
                  iss >> v0 >> ch >> v1 >> ch >> info;

              if (indexOneBased) {
                  v0--; 
                  v1--;
              }

              // normalize v0/v1 by adding lo/hi shard ID
              v0 += v_lo;
              v1 += v_hi;

              edgeList.push_back({v0, v1, w});
              edgeList.push_back({v1, v0, w});

              if (v0 > maxVertex)
                  maxVertex = v0;
              if (v1 > maxVertex)
                  maxVertex = v1;

              numEdges++;
          }

          // close current shard
          ifs.close();
      }
      else
          continue;
  }

  fileProc.clear();
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (!indexOneBased)
	numVertices = maxVertex + 1;
  else
	numVertices = maxVertex;
	
  /// Part 2: Count number of edges and sort edges
  std::vector<GraphElem> edgeCount(numVertices+1);
  numEdges = edgeList.size();
  
  for (GraphElem i = 0; i < numEdges; i++) {
	edgeCount[edgeList[i].i_+1]++;
	edgeCount[edgeList[i].j_+1]++;
  }

  // local sorting of edge list
  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
 
  GraphElem globalNumVertices, globalNumEdges;
  MPI_Reduce(&numVertices, &globalNumVertices, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&numEdges, &globalNumEdges, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

  /// Part 3: Dump the data to a binary file   
  // number of aggregates to improve MPI IO
  MPI_Info info;
  MPI_Info_create(&info);
  if (naggr >= nprocs)
      naggr = 1;
  std::stringstream tmp_str;
  tmp_str << naggr;
  std::string str = tmp_str.str();
  MPI_Info_set(info, "cb_nodes", str.c_str());

  file_open_error = MPI_File_open(MPI_COMM_WORLD, fileOutPath.c_str(), MPI_MODE_WRONLY, info, &fh); 
  MPI_Info_free(&info);

  if (file_open_error != MPI_SUCCESS) {
      std::cout<< " Error opening output file! " << std::endl;
      MPI_Abort(-99, MPI_COMM_WORLD);
  }

  // write the number of vertices/edges first
  if (rank == 0) {
      MPI_File_write(fh, &globalNumVertices, sizeof(GraphElem), MPI_BYTE, &status);
      MPI_File_write(fh, &globalNumEdges, sizeof(GraphElem), MPI_BYTE, &status);
  }

  GraphElem v_offset = 0;
  MPI_Exscan(&numVertices, &v_offset, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  
  // write the edge prefix counts first (required for CSR 
  // construction during reading)
  uint64_t tot_bytes = (numVertices+1)*sizeof(GraphElem);
  MPI_Offset offset = 2*sizeof(GraphElem) + v_offset*sizeof(GraphElem);
    
  if (tot_bytes < INT_MAX)
      MPI_File_write_at(fh, offset, &g->edgeListIndexes[0], tot_bytes, MPI_BYTE, &status);
    else {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*) &g->edgeListIndexes[0];
        uint64_t transf_bytes=0;

        while (transf_bytes<tot_bytes)
        {
            MPI_File_write_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
            transf_bytes+=chunk_bytes;
            offset+=chunk_bytes;
            curr_pointer+=chunk_bytes;

            if (tot_bytes-transf_bytes<INT_MAX)
                chunk_bytes=tot_bytes-transf_bytes;
        } 
    }  
  
    MPI_Barrier(MPI_COMM_WORLD);
    
    // write the edge list next
    g->setNumEdges(numEdges);
    tot_bytes = numEdges*(sizeof(Edge));

    //GraphElem e_offset = 0;
    //MPI_Exscan(&numEdges, &e_offset, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);
    //offset = 2*sizeof(GraphElem) + (globalNumVertices+1)*sizeof(GraphElem) + e_offset*(sizeof(Edge));
    
    offset = 2*sizeof(GraphElem) + (globalNumVertices+1)*sizeof(GraphElem) + g->edgeListIndexes[0]*(sizeof(Edge));

    if (tot_bytes<INT_MAX)
        MPI_File_write_at(fh, offset, &g->edgeList[0], tot_bytes, MPI_BYTE, &status);
    else {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*)&g->edgeList[0];
        uint64_t transf_bytes=0;

        while (transf_bytes<tot_bytes)
        {
            MPI_File_write_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
            transf_bytes+=chunk_bytes;
            offset+=chunk_bytes;
            curr_pointer+=chunk_bytes;

            if (tot_bytes-transf_bytes<INT_MAX)
                chunk_bytes=tot_bytes-transf_bytes;
        } 
    }    

    MPI_File_close(&fh);

    if (rank == 0) {
        std::cout << "Graph #Vertices: " << globalNumVertices << ", #Edges: " << globalNumEdges << std::endl;
    }

    delete g;
} // loadParallelFileShards
