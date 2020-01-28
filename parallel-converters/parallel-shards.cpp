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
#include <unordered_map>
#include <climits>

#include "parallel-shards.hpp"

/// Pairwise read files between fileStartIndex and fileEndIndex and store 
/// it into a single file, also selectively convert to binary

/// fileInShardsPath expects the input directory where the shards are ('/' is not allowed at the end)

/// the input file is expected to contain 4 data points in every line:
/// ----ai, aj, common, jaccard---- (common is not processed at present)

/// We expect the shards to contain the 'upper triangle' of the adjacency...during binary 
/// conversion, we consider the data to be an undirected graph, and store both combinations of edge pairs

void loadParallelFileShards(int rank, int nprocs, int naggr, 
	const std::string &fileInShardsPath, const std::string &fileOutPath, 
        const int fileStartIndex, const int fileEndIndex, bool indexOneBased, 
        Weight_t wtype, GraphElem shardCount)
{
  assert(fileStartIndex >= 0);
  assert(fileEndIndex >= 0);
  assert(fileEndIndex >= fileStartIndex);

  GraphElem numEdges = 0, numVertices = 0;
  int file_open_error;
  MPI_File fh;

  /// Part 1: Read the file shards into edge list  
  
  std::vector<GraphElemTuple> edgeList;
  std::map<GraphElem, std::vector<std::string> > fileProc;

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
          if (fileProc.find(proc) == fileProc.end()) { // not found
              fileProc.insert({proc, std::vector<std::string>()});
              fileProc[proc].push_back(fileName);
          } else { // found
              fileProc[proc].push_back(fileName);
          }

          proc = (proc + 1)%nprocs;

          ifs.close();
      }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read the files only if I can
  std::map<GraphElem, std::vector<std::string> >::iterator mpit = fileProc.find(rank);
  GraphElem minmax_v[2] = {-1,0};

  if (mpit != fileProc.end()) {

      for (int k = 0; k < mpit->second.size(); k++) {

          // retrieve lo/hi range from file name string
          std::string fileName_full = (mpit->second[k]).substr((mpit->second[k]).find_last_of("/") + 1);
          std::string fileName_noext = fileName_full.substr(0, fileName_full.find("."));
          std::string fileName_right = fileName_noext.substr(fileName_full.find("__") + 2);
          std::string fileName_left = fileName_noext.substr(0, fileName_full.find("__"));

          GraphElem v_lo = (GraphElem)(std::stoi(fileName_left) - 1)*shardCount;
          GraphElem v_hi = (GraphElem)(std::stoi(fileName_right) - 1)*shardCount;

          // open file shard and start reading
          std::ifstream ifs;
          ifs.open((mpit->second[k]).c_str(), std::ifstream::in);

          std::string line;

          while(!ifs.eof()) {

              GraphElem v0, v1, info;
              GraphWeight w;
              char ch;

              std::getline(ifs, line);
              std::istringstream iss(line);

              // read from current shard 
              if (wtype == ORG_WEIGHT || wtype == ABS_WEIGHT)
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

              numEdges++;

              if (v0 > numVertices)
                  numVertices = v0;
              if (v1 > numVertices)
                  numVertices = v1;

              // assuming vertices in each file are presorted
              // multiple files could have the *same* min/max vertex ID
              if (minmax_v[0] == -1)
                  minmax_v[0] = v0;
              minmax_v[1] = v0;
          }

          // close current shard
          ifs.close();
      }
  }

  numEdges *= 2;

  // idle processes wait at the barrier
  MPI_Barrier(MPI_COMM_WORLD);
  
  // get the maximum and minimum vertex ID per file/process
  std::vector<GraphElem> minmax_vs(nprocs*2);
  MPI_Alltoall(minmax_v, 2, MPI_GRAPH_TYPE, minmax_vs.data(), 2, MPI_GRAPH_TYPE, MPI_COMM_WORLD);
 
  const int elprocs = fileProc.size();
  
  if (rank == 0)
      std::cout << "Read the files using " << elprocs << " processes." << std::endl;      

  // numEdges/numVertices to be written by process 0
  GraphElem globalNumVertices = 0, globalNumEdges = 0;
  
  MPI_Allreduce(&numEdges, &globalNumEdges, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&numVertices, &globalNumVertices, 1, MPI_GRAPH_TYPE, MPI_MAX, MPI_COMM_WORLD);
  
  if (!indexOneBased)
      globalNumVertices += 1;

  if (rank == 0) {
      std::cout << "Graph #nvertices: " << globalNumVertices << ", #edges: " << globalNumEdges << std::endl;
      std::cout << "Starting to store the file data..." << std::endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  /// Part 1.5: Read the files again, this time store the data
  std::vector<GraphElem> parts(nprocs+1); 
  parts[0] = 0;

  for (GraphElem i = 1; i < nprocs+1; i++)
      parts[i] = ((globalNumVertices * i) / nprocs); 
  
  GraphElem localNumVertices = ((globalNumVertices*(rank + 1)) / nprocs) - ((globalNumVertices*rank) / nprocs); 

  std::vector<GraphElem> edgeCount(globalNumVertices+1), edgeCountTmp(globalNumVertices+1);
  std::vector<std::vector<GraphElemTuple>> outEdges(nprocs);
  int v = 0; // index for max|min_vs

  for (auto mpit = fileProc.begin(); mpit != fileProc.end(); ++mpit) {
      for (int k = 0; k < mpit->second.size(); k++) {

          // retrieve lo/hi range from file name string
          std::string fileName_full = (mpit->second[k]).substr((mpit->second[k]).find_last_of("/") + 1);
          std::string fileName_noext = fileName_full.substr(0, fileName_full.find("."));
          std::string fileName_right = fileName_noext.substr(fileName_full.find("__") + 2);
          std::string fileName_left = fileName_noext.substr(0, fileName_full.find("__"));

          GraphElem v_lo = (GraphElem)(std::stoi(fileName_left) - 1)*shardCount;
          GraphElem v_hi = (GraphElem)(std::stoi(fileName_right) - 1)*shardCount;
#if defined(DEBUG_PRINTF)
          std::cout << "File processing: " << fileName_full << "; Ranges: " << v_lo  << ", " << v_hi << std::endl;
#endif

          if ((parts[rank] >= minmax_vs[v]) && (parts[rank+1] <= minmax_vs[v+1])) {

              // open file shard and start reading
              std::ifstream ifs;
              ifs.open((mpit->second[k]).c_str(), std::ifstream::in);

              std::string line;
              int owner = -1;
              bool checkedFile = false;
              GraphElem past_v = -1;

              while(!ifs.eof()) {

                  GraphElem v0, v1, info;
                  GraphWeight w;
                  char ch;

                  std::getline(ifs, line);
                  std::istringstream iss(line);

                  // read from current shard 
                  if (wtype == ORG_WEIGHT)
                      iss >> v0 >> ch >> v1 >> ch >> info >> ch >> w;
                  if (wtype == ABS_WEIGHT) {
                      iss >> v0 >> ch >> v1 >> ch >> info >> ch >> w;
                      w = std::fabs(w);
                  }
                  else
                      iss >> v0 >> ch >> v1 >> ch >> info;

                  if (indexOneBased) {
                      v0--; 
                      v1--;
                  }

                  // normalize v0/v1 by adding lo/hi shard ID
                  v0 += v_lo;
                  v1 += v_hi;

                  // recompute owner when needed
                  if (past_v != v0) { 
                      auto iter = std::upper_bound(parts.begin(), parts.end(), v0);
                      owner = (iter - parts.begin() - 1);
                  }

                  // populate edge list
                  if (owner == rank) {
                      edgeList.push_back({v0, v1, w});

                      // edge count
                      edgeCount[v0+1]++; 
                      edgeCount[v1+1]++; 

                      // search ghost owner and push it to outgoing edge list
                      auto iter = std::upper_bound(parts.begin(), parts.end(), v1);
                      int ghost_owner = (iter - parts.begin() - 1);
                      outEdges[ghost_owner].push_back({v1, v0, w});

                      if (!checkedFile)
                          checkedFile = true;
                  }

                  // do I need to continue reading or am I done
                  if (checkedFile && (owner != rank))
                      break;

                  past_v = v0;
              }

              // close current shard
              ifs.close();
          }

          v += 2;
      }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  fileProc.clear();

  /// Part 2: Assuming a vertex-based distribution, distribute edges and build edgeCount
 
  // build MPI edge tuple datatype
  GraphElemTuple et;
  MPI_Datatype ettype;
  MPI_Aint displ[3], base;
  int blens[] = { 1, 1, 1 };
  MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

  MPI_Get_address(&et, displ);
  MPI_Get_address(&et.j_, displ+1);
  MPI_Get_address(&et.w_, displ+2);
  base = displ[0];

  for (int i = 0; i < 3; i++)
      displ[i] = MPI_Aint_diff(displ[i], base);

  MPI_Type_create_struct(3, blens, displ, types, &ettype);
  MPI_Type_commit(&ettype);
  
  if (rank == 0)
      std::cout << "Filled outgoing (undirected) edge lists." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  // exchange count information
  std::vector<int> ssize(nprocs), rsize(nprocs), sdispls(nprocs), rdispls(nprocs);

  // outEdges contains outgoing {edges,weight} information
  int spos = 0;
  for (int p = 0; p < nprocs; p++) {
      ssize[p] = (int)outEdges[p].size();
      sdispls[p] = spos;
      spos += ssize[p];
  }

  std::vector<GraphElemTuple> sredata(spos);
  spos = 0;
  for (int p = 0; p < nprocs; p++) {
      std::memcpy(&sredata[spos], outEdges[p].data(), sizeof(GraphElemTuple)*outEdges[p].size());
      spos += outEdges[p].size();
  }

  for (int p = 0; p < nprocs; p++)
      outEdges[p].clear();
  outEdges.clear();
  
  MPI_Alltoall(ssize.data(), 1, MPI_INT, rsize.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  int rpos = 0;
  for (int p = 0; p < nprocs; p++) {
      rdispls[p] = rpos;
      rpos += rsize[p];
  }

  std::vector<GraphElemTuple> rredata(rpos);
  MPI_Alltoallv(sredata.data(), ssize.data(), sdispls.data(), 
          ettype, rredata.data(), rsize.data(), rdispls.data(), 
          ettype, MPI_COMM_WORLD);
  
  if (rank == 0)
      std::cout << "Exchanged ghost vertices across processes." << std::endl;

  // insert ghosts into the local edgeList
  GraphElem j = 0;
  for (int p = 0; p < nprocs; p++) {
      for (GraphElem i = 0; i < rsize[p]; i++) {
          edgeList.push_back(rredata[j+i]);
      }
      j += rsize[p];
  }
  
  // updated #edges
  numEdges = edgeList.size();

  // reduction on edge counts
  MPI_Reduce(edgeCount.data(), edgeCountTmp.data(), globalNumVertices, 
          MPI_GRAPH_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
   
  if (rank == 0) {
      edgeCount = edgeCountTmp;

      // local prefix sum
      for (GraphElem i = 1; i < globalNumVertices; i++)
          edgeCount[i] += edgeCount[i-1];
      
      std::cout << "Redistributed edges and performed reduction on edge counts." << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // local sorting of edge list
  auto ecmp = [] (GraphElemTuple const& e0, GraphElemTuple const& e1)
  { return ((e0.i_ < e1.i_) || ((e0.i_ == e1.i_) && (e0.j_ < e1.j_))); };

  if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp)) {
#if defined(DEBUG_PRINTF)
	  std::cout << "Edge list is not sorted" << std::endl;
#endif
	  std::sort(edgeList.begin(), edgeList.end(), ecmp);
  }
  else {
#if defined(DEBUG_PRINTF)
	  std::cout << "Edge list is sorted!" << std::endl;
#endif
  }

  if (rank == 0)
      std::cout << "Sorted the edge list." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /// Part 3: Dump the data to a binary file   
  file_open_error = MPI_File_open(MPI_COMM_WORLD, fileOutPath.c_str(), 
          MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); 

  if (file_open_error != MPI_SUCCESS) {
      std::cout<< " Error opening output file! " << std::endl;
      MPI_Abort(-99, MPI_COMM_WORLD);
  }
  
  // process 0 writes the #vertices/edges first, followed by edgeCount
  
  uint64_t tot_bytes;
  MPI_Offset offset;

  if (rank == 0) {
      std::cout << "Processing complete, about to write the binary file." << std::endl;
      MPI_File_write_at(fh, 0, &globalNumVertices, sizeof(GraphElem), MPI_BYTE, MPI_STATUS_IGNORE);
      MPI_File_write_at(fh, sizeof(GraphElem), &globalNumEdges, sizeof(GraphElem), MPI_BYTE, MPI_STATUS_IGNORE);

      // write the edge prefix counts first (required for CSR 
      // construction during reading) 
      tot_bytes = (globalNumVertices+1) * sizeof(GraphElem);
      offset = 2*sizeof(GraphElem);

      if (tot_bytes < INT_MAX)
          MPI_File_write_at(fh, offset, edgeCount.data(), tot_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
      else {
          int chunk_bytes=INT_MAX;
          uint8_t *curr_pointer = (uint8_t*) edgeCount.data();
          uint64_t transf_bytes=0;

          while (transf_bytes < tot_bytes)
          {
              MPI_File_write_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
              transf_bytes+=chunk_bytes;
              offset+=chunk_bytes;
              curr_pointer+=chunk_bytes;

              if (tot_bytes-transf_bytes < INT_MAX)
                  chunk_bytes=tot_bytes-transf_bytes;
          } 
      }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
      std::cout << "Beginning to write the second part of the binary file (edges)." << std::endl;      

  // write the edge list next, prepare CSR format
  tot_bytes = numEdges * sizeof(Edge);
  std::vector<Edge> csrCols(numEdges);

  for (GraphElem i = 0; i < numEdges; i++) {
      csrCols.emplace_back(edgeList[i].j_, edgeList[i].w_);
  }

  GraphElem e_offset = 0;
  MPI_Exscan(&numEdges, &e_offset, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  
  // processes write edges
  offset = 2*sizeof(GraphElem) + (globalNumVertices+1)*sizeof(GraphElem) + e_offset*(sizeof(Edge));

  if (tot_bytes < INT_MAX)
      MPI_File_write_at(fh, offset, csrCols.data(), tot_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
  else {
      int chunk_bytes = INT_MAX;
      uint8_t *curr_pointer = (uint8_t*)csrCols.data();
      uint64_t transf_bytes=0;

      while (transf_bytes<tot_bytes)
      {
          MPI_File_write_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
          transf_bytes+=chunk_bytes;
          offset+=chunk_bytes;
          curr_pointer+=chunk_bytes;

          if (tot_bytes-transf_bytes < INT_MAX)
              chunk_bytes=tot_bytes-transf_bytes;
      } 
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_File_close(&fh);

  if (rank == 0)
      std::cout << "Completed writing the binary file: " << fileOutPath << std::endl;      

  minmax_vs.clear();
  edgeList.clear();
  edgeCount.clear();
  edgeCountTmp.clear();
  sredata.clear();
  rredata.clear();
  ssize.clear();
  rsize.clear();
  sdispls.clear();
  rdispls.clear();
  parts.clear();
} // loadParallelFileShards
