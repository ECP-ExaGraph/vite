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

#include <algorithm>
#include <fstream>
#include <numeric>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <mpi.h>
#include <climits>
#include <cstdio>

#include "distgraph.hpp"

extern std::ofstream ofs;
        
// find a distribution such that every 
// process own equal number of edges (serial)
void balanceEdges(int nprocs, std::string& fileName, std::vector<GraphElem>& mbins)
{
    FILE *fp;
    GraphElem nv, ne; // #vertices, #edges
    std::vector<GraphElem> nbins(nprocs,0);

    fp = fopen(fileName.c_str(), "rb");
    if (fp == NULL) {
        std::cout<< " Error opening file! " << std::endl;
        return;
    }

    // read nv and ne
    fread(&nv, sizeof(GraphElem), 1, fp);
    fread(&ne, sizeof(GraphElem), 1, fp);

    // bin capacity
    GraphElem nbcap = (ne / nprocs), ecount_idx, past_ecount_idx = 0;
    int p = 0;

    for (GraphElem m = 0; m < nv; m++)
    {
        fread(&ecount_idx, sizeof(GraphElem), 1, fp);

        // bins[p] >= capacity only for the last process
        if ((nbins[p] < nbcap) || (p == (nprocs - 1)))
            nbins[p] += (ecount_idx - past_ecount_idx);

        // increment p as long as p is not the last process
        // worst case: excess edges piled up on (p-1)
        if ((nbins[p] >= nbcap) && (p < (nprocs - 1)))
            p++;

        mbins[p+1]++;
        past_ecount_idx = ecount_idx;
    }

    fclose(fp);

    // prefix sum to store indices 
    for (int k = 1; k < nprocs+1; k++)
        mbins[k] += mbins[k-1]; 

    nbins.clear();
}

// MPI parallel-I/O read binary file
void loadDistGraphMPIIO(int me, int nprocs, int ranks_per_node, DistGraph *&dg, std::string &fileName)
{
    GraphElem globalNumEdges;
    GraphElem globalNumVertices;

    int file_open_error;
    MPI_File fh;
    MPI_Status status;

    // specify the number of aggregates
    // nprocs / ranks_per_node 
    MPI_Info info;
    MPI_Info_create(&info);
    int naggr = (ranks_per_node > 1) ? (nprocs/ranks_per_node) : ranks_per_node;
    if (naggr >= nprocs)
        naggr = 1;
    std::stringstream tmp_str;
    tmp_str << naggr;
    std::string str = tmp_str.str();
    MPI_Info_set(info, "cb_nodes", str.c_str());

    file_open_error = MPI_File_open(MPI_COMM_WORLD, fileName.c_str(), MPI_MODE_RDONLY, info, &fh); 

    MPI_Info_free(&info);

    if (file_open_error != MPI_SUCCESS) {
        std::cout<< " Error opening file! " << std::endl;
    }

    MPI_File_read_all(fh,&globalNumVertices,sizeof(GraphElem), MPI_BYTE, &status);
    MPI_File_read_all(fh,&globalNumEdges,sizeof(GraphElem), MPI_BYTE, &status);

    GraphElem localNumVertices = ((globalNumVertices * (me + 1)) / nprocs) - ((globalNumVertices * me) / nprocs); 

    // set to zero initially
    GraphElem localNumEdges = 0;

    std::vector<GraphElem> party(nprocs+1);

    dg = new DistGraph(globalNumVertices,globalNumEdges);

    party[0]=0;

    // account for hash in data structure...
    for (int i=1;i<nprocs+1; i++)
        party[i]=((globalNumVertices * i) / nprocs);  

    dg->createLocalGraph(localNumVertices, localNumEdges, &party);
    Graph &g = dg->getLocalGraph(); 

    //  Let N = array length and P = number of processors.
    //  From j = 0 to P-1,
    // Starting point of array on processor j = floor(N * j / P)
    // Length of array on processor j = floor(N * (j + 1) / P) – floor(N * j / P)

    // MPI LIMITS
    uint64_t tot_bytes=(localNumVertices+1)*sizeof(GraphElem);
    MPI_Offset offset = 2*sizeof(GraphElem) + ((globalNumVertices * me) / nprocs)*sizeof(GraphElem);

    // printf("Process: %d, Edge-list size: %d, elements: %d offset: %ld, offset elements: %d\n", me, g.edgeListIndexes.size(),  tot_bytes/sizeof(GraphElem), offset, offset/sizeof(GraphElem) - 2); 

    if (tot_bytes<INT_MAX)
        MPI_File_read_at(fh, offset, &g.edgeListIndexes[0], tot_bytes, MPI_BYTE, &status);
    else {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*) &g.edgeListIndexes[0];
        //GraphElem *curr_pointer = (GraphElem*) &g.edgeListIndexes[0];
        uint64_t transf_bytes=0;

        while (transf_bytes<tot_bytes)
        {
            MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
            transf_bytes+=chunk_bytes;
            offset+=chunk_bytes;
            curr_pointer+=chunk_bytes;

            if (tot_bytes-transf_bytes<INT_MAX)
                chunk_bytes=tot_bytes-transf_bytes;
        } 
    }    

    localNumEdges = g.edgeListIndexes[localNumVertices]-g.edgeListIndexes[0];

    g.setNumEdges(localNumEdges);

    tot_bytes=localNumEdges*(sizeof(Edge));

    offset = 2*sizeof(GraphElem) + (globalNumVertices+1)*sizeof(GraphElem) + g.edgeListIndexes[0]*(sizeof(Edge));

    if (tot_bytes<INT_MAX)
        MPI_File_read_at(fh, offset, &g.edgeList[0], tot_bytes, MPI_BYTE, &status);
    else {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*)&g.edgeList[0];
        //GraphElem *curr_pointer = (GraphElem*)&g.edgeList[0];
        uint64_t transf_bytes=0;

        while (transf_bytes<tot_bytes)
        {
            MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
            transf_bytes+=chunk_bytes;
            offset+=chunk_bytes;
            curr_pointer+=chunk_bytes;

            if (tot_bytes-transf_bytes<INT_MAX)
                chunk_bytes=tot_bytes-transf_bytes;
        } 
    }    


    MPI_File_close(&fh);

    for(GraphElem i=1;  i < localNumVertices+1; i++)
        g.edgeListIndexes[i]-=g.edgeListIndexes[0];   

    g.edgeListIndexes[0]=0;

    // TODO FIXME create a runtime option for setting edge weights to 1.0
#if defined(SET_EDGE_WEIGHTS_TO_ONE)
    g.setEdgeWeightstoOne();
#endif
}

// MPI parallel-I/O read binary file and make a balanced edge distribution
void loadDistGraphMPIIOBalanced(int me, int nprocs, int ranks_per_node, DistGraph *&dg, std::string &fileName)
{
    GraphElem globalNumEdges;
    GraphElem globalNumVertices;

    int file_open_error;
    MPI_File fh;
    MPI_Status status;
    std::vector<GraphElem> mbins(nprocs+1,0);

    // find #vertices per process such that 
    // each process roughly owns equal #edges
    if (me == 0)
    {
        balanceEdges(nprocs, fileName, mbins);
        std::cout << "Trying to achieve equal edge distribution across processes." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(mbins.data(), nprocs+1, MPI_GRAPH_TYPE, 0, MPI_COMM_WORLD);

    // specify the number of aggregates
    // nprocs / ranks_per_node 
    MPI_Info info;
    MPI_Info_create(&info);
    int naggr = (ranks_per_node > 1) ? (nprocs/ranks_per_node) : ranks_per_node;
    if (naggr >= nprocs)
        naggr = 1;
    std::stringstream tmp_str;
    tmp_str << naggr;
    std::string str = tmp_str.str();
    MPI_Info_set(info, "cb_nodes", str.c_str());

    file_open_error = MPI_File_open(MPI_COMM_WORLD, fileName.c_str(), MPI_MODE_RDONLY, info, &fh); 

    MPI_Info_free(&info);

    if (file_open_error != MPI_SUCCESS) {
        std::cout<< " Error opening file! " << std::endl;
    }

    MPI_File_read_all(fh,&globalNumVertices,sizeof(GraphElem), MPI_BYTE, &status);
    MPI_File_read_all(fh,&globalNumEdges,sizeof(GraphElem), MPI_BYTE, &status);

    GraphElem localNumVertices = mbins[me+1] - mbins[me];

    // set to zero initially
    GraphElem localNumEdges = 0;

    std::vector<GraphElem> party(nprocs+1);

    dg = new DistGraph(globalNumVertices,globalNumEdges);

    party[0]=0;
    // account for hash in data structure...
    for (int i=1;i<nprocs+1; i++)
        party[i]=mbins[i];  

    dg->createLocalGraph(localNumVertices, localNumEdges, &party);
    Graph &g = dg->getLocalGraph(); 

    //  Let N = array length and P = number of processors.
    //  From j = 0 to P-1,
    // Starting point of array on processor j = floor(N * j / P)
    // Length of array on processor j = floor(N * (j + 1) / P) – floor(N * j / P)

    // MPI LIMITS
    uint64_t tot_bytes=(localNumVertices+1)*sizeof(GraphElem);
    MPI_Offset offset = 2*sizeof(GraphElem) + mbins[me]*sizeof(GraphElem);

    // printf("Process: %d, Edge-list size: %d, elements: %d offset: %ld, offset elements: %d\n", me, g.edgeListIndexes.size(),  tot_bytes/sizeof(GraphElem), offset, offset/sizeof(GraphElem) - 2); 

    if (tot_bytes<INT_MAX)
        MPI_File_read_at(fh, offset, &g.edgeListIndexes[0], tot_bytes, MPI_BYTE, &status);
    else {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*) &g.edgeListIndexes[0];
        //GraphElem *curr_pointer = (GraphElem*) &g.edgeListIndexes[0];
        uint64_t transf_bytes=0;

        while (transf_bytes<tot_bytes)
        {
            MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
            transf_bytes+=chunk_bytes;
            offset+=chunk_bytes;
            curr_pointer+=chunk_bytes;

            if (tot_bytes-transf_bytes<INT_MAX)
                chunk_bytes=tot_bytes-transf_bytes;
        } 
    }    

    localNumEdges = g.edgeListIndexes[localNumVertices]-g.edgeListIndexes[0];

    g.setNumEdges(localNumEdges);

    tot_bytes=localNumEdges*(sizeof(Edge));

    offset = 2*sizeof(GraphElem) + (globalNumVertices+1)*sizeof(GraphElem) + g.edgeListIndexes[0]*(sizeof(Edge));

    if (tot_bytes<INT_MAX)
        MPI_File_read_at(fh, offset, &g.edgeList[0], tot_bytes, MPI_BYTE, &status);
    else {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*)&g.edgeList[0];
        uint64_t transf_bytes=0;

        while (transf_bytes<tot_bytes)
        {
            MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
            transf_bytes+=chunk_bytes;
            offset+=chunk_bytes;
            curr_pointer+=chunk_bytes;

            if (tot_bytes-transf_bytes<INT_MAX)
                chunk_bytes=tot_bytes-transf_bytes;
        } 
    }    

    MPI_File_close(&fh);

    for(GraphElem i=1;  i < localNumVertices+1; i++)
        g.edgeListIndexes[i]-=g.edgeListIndexes[0];   

    g.edgeListIndexes[0]=0;
    mbins.clear();

    // TODO FIXME create a runtime option for setting edge weights to 1.0
#if defined(SET_EDGE_WEIGHTS_TO_ONE)
    g.setEdgeWeightstoOne();
#endif
}
