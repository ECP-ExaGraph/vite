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
#include <array>
#include <unistd.h>

#include "distgraph.hpp"
#include "utils.hpp"

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

    if (me == 0)
	    std::cout << "Reading file of " << globalNumVertices 
		    << " vertices and " << globalNumEdges << " edges." << std::endl;
    
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

// generate graph
// 1D vertex distribution
void generateInMemGraph(int rank, int nprocs, DistGraph *&dg, GraphElem nv, int randomEdgePercent)
{
    GraphWeight rn;

    // calculate r(n)
    GraphWeight rc = sqrt((GraphWeight)log(nv)/(GraphWeight)(PI*nv));
    GraphWeight rt = sqrt((GraphWeight)2.0736/(GraphWeight)nv);
    rn = (rc + rt)/(GraphWeight)2.0;

    assert(((GraphWeight)1.0/(GraphWeight)nprocs) > rn);

    // generate distributed RGG in memory
    dg = generateRGG(rank, nprocs, nv, rn, randomEdgePercent);

    MPI_Barrier(MPI_COMM_WORLD);
}

// create RGG and returns Graph
// TODO FIXME use OpenMP wherever possible
// use Euclidean distance as edge weight
DistGraph* generateRGG(int rank, int nprocs, GraphElem nv, GraphWeight rn, int randomEdgePercent)
{
    int up, down;

    // neighbors
    up = down = MPI_PROC_NULL;
    if (nprocs > 1) {
        if (rank > 0 && rank < (nprocs - 1)) {
            up = rank - 1;
            down = rank + 1;
        }
        if (rank == 0)
            down = 1;
        if (rank == (nprocs - 1))
            up = rank - 1;
    }

    GraphElem n = nv / nprocs;

    // Generate random coordinate points
    std::vector<GraphWeight> X, Y, X_up, Y_up, X_down, Y_down;
    X.resize(2*n);
    Y.resize(n);

    if (up != MPI_PROC_NULL) {
        X_up.resize(n);
        Y_up.resize(n);
    }

    if (down != MPI_PROC_NULL) {
        X_down.resize(n);
        Y_down.resize(n);
    }

    // create local graph
    std::vector<GraphElem> party(nprocs+1);
    DistGraph* dg = new DistGraph(nv, 0);
    party[0] = 0;

    // account for hash in data structure...
    for (int i = 1;i < nprocs + 1; i++)
        party[i]=((nv * i) / nprocs);  

    // set #edges later
    dg->createLocalGraph(n, 0, &party);
    Graph &g = dg->getLocalGraph(); 

    // generate random number within range
    // X: 0, 1
    // Y: rank_*1/p, (rank_+1)*1/p,
    GraphWeight rec_np = (GraphWeight)(1.0/(GraphWeight)nprocs);
    GraphWeight lo = rank* rec_np; 
    GraphWeight hi = lo + rec_np;
    assert(hi > lo);

    // measure the time to generate random numbers
    MPI_Barrier(MPI_COMM_WORLD);
    double st = MPI_Wtime();

    // X | Y
    // e.g seeds: 1741, 3821
    // create LCG object
    // seed to generate x0
    LCG xr(1, X.data(), 2*n); 

    // generate random numbers between 0-1
    xr.generate();

    // rescale xr further between lo-hi
    // and put the numbers in Y taking
    // from X[n]
    xr.rescale(Y.data(), n, lo);

#if defined(PRINT_RANDOM_XY_COORD)
    for (int k = 0; k < nprocs; k++) {
        if (k == rank) {
            std::cout << "Random number generated on Process#" << k << " :" << std::endl;
            for (GraphElem i = 0; i < n; i++) {
                std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    double et = MPI_Wtime();
    double tt = et - st;
    double tot_tt = 0.0;
    MPI_Reduce(&tt, &tot_tt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double tot_avg = (tot_tt/nprocs);
        std::cout << "Average time to generate " << 2*n 
            << " random numbers using LCG (in s): " 
            << tot_avg << std::endl;
    }

    // ghost(s)

    // cross edges, each processor
    // communicates with up or/and down
    // neighbor only
    std::vector<EdgeTuple> sendup_edges, senddn_edges; 
    std::vector<EdgeTuple> recvup_edges, recvdn_edges;
    std::vector<EdgeTuple> edgeList;

    // counts, indexing: [2] = {up - 0, down - 1}
    // TODO can't we use MPI_INT 
    std::array<GraphElem, 2> send_sizes = {0, 0}, recv_sizes = {0, 0};
#if defined(CHECK_NUM_EDGES)
    GraphElem numEdges = 0;
#endif

    // local
    // TODO FIXME parallelize
    for (GraphElem i = 0; i < n; i++) {
        for (GraphElem j = i + 1; j < n; j++) {
            // euclidean distance:
            // 2D: sqrt((px-qx)^2 + (py-qy)^2)
            GraphWeight dx = X[i] - X[j];
            GraphWeight dy = Y[i] - Y[j];
            GraphWeight ed = sqrt(dx*dx + dy*dy);
            // are the two vertices within the range?
            if (ed <= rn) {
                // local to global index
                const GraphElem g_i = dg->localToGlobal(i, rank);
                const GraphElem g_j = dg->localToGlobal(j, rank);

                edgeList.emplace_back(i, g_j, ed);
                edgeList.emplace_back(j, g_i, ed);

#if defined(CHECK_NUM_EDGES)
                numEdges += 2;
#endif

                g.edgeListIndexes[i+1]++;
                g.edgeListIndexes[j+1]++;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // communicate ghost coordinates with neighbors
    const int x_ndown   = X_down.empty() ? 0 : n;
    const int y_ndown   = Y_down.empty() ? 0 : n;
    const int x_nup     = X_up.empty() ? 0 : n;
    const int y_nup     = Y_up.empty() ? 0 : n;

    MPI_Sendrecv(X.data(), n, MPI_WEIGHT_TYPE, up, SR_X_UP_TAG, 
            X_down.data(), x_ndown, MPI_WEIGHT_TYPE, down, SR_X_UP_TAG, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(X.data(), n, MPI_WEIGHT_TYPE, down, SR_X_DOWN_TAG, 
            X_up.data(), x_nup, MPI_WEIGHT_TYPE, up, SR_X_DOWN_TAG, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(Y.data(), n, MPI_WEIGHT_TYPE, up, SR_Y_UP_TAG, 
            Y_down.data(), y_ndown, MPI_WEIGHT_TYPE, down, SR_Y_UP_TAG, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(Y.data(), n, MPI_WEIGHT_TYPE, down, SR_Y_DOWN_TAG, 
            Y_up.data(), y_nup, MPI_WEIGHT_TYPE, up, SR_Y_DOWN_TAG, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // exchange ghost vertices / cross edges
    if (nprocs > 1) {
        if (up != MPI_PROC_NULL) {

            for (GraphElem i = 0; i < n; i++) {
                for (GraphElem j = i + 1; j < n; j++) {
                    GraphWeight dx = X[i] - X_up[j];
                    GraphWeight dy = Y[i] - Y_up[j];
                    GraphWeight ed = sqrt(dx*dx + dy*dy);

                    if (ed <= rn) {
                        const GraphElem g_i = dg->localToGlobal(i, rank);
                        const GraphElem g_j = j + up*n;
                        sendup_edges.emplace_back(j, g_i, ed);
                        edgeList.emplace_back(i, g_j, ed);

#if defined(CHECK_NUM_EDGES)
                        numEdges++;
#endif
                        g.edgeListIndexes[i+1]++;
                    }
                }
            }

            // send up sizes
            send_sizes[0] = sendup_edges.size();
        }

        if (down != MPI_PROC_NULL) {

            for (GraphElem i = 0; i < n; i++) {
                for (GraphElem j = i + 1; j < n; j++) {
                    GraphWeight dx = X[i] - X_down[j];
                    GraphWeight dy = Y[i] - Y_down[j];
                    GraphWeight ed = sqrt(dx*dx + dy*dy);

                    if (ed <= rn) {
                        const GraphElem g_i = dg->localToGlobal(i, rank);
                        const GraphElem g_j = j + down*n;

                        senddn_edges.emplace_back(j, g_i, ed);
                        edgeList.emplace_back(i, g_j, ed);
#if defined(CHECK_NUM_EDGES)
                        numEdges++;
#endif
                        g.edgeListIndexes[i+1]++;
                    }
                }
            }

            // send down sizes
            send_sizes[1] = senddn_edges.size();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // communicate ghost vertices with neighbors
    // send/recv buffer sizes

    MPI_Sendrecv(&send_sizes[0], 1, MPI_GRAPH_TYPE, up, SR_SIZES_UP_TAG, 
            &recv_sizes[1], 1, MPI_GRAPH_TYPE, down, SR_SIZES_UP_TAG, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_sizes[1], 1, MPI_GRAPH_TYPE, down, SR_SIZES_DOWN_TAG, 
            &recv_sizes[0], 1, MPI_GRAPH_TYPE, up, SR_SIZES_DOWN_TAG, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // resize recv buffers

    if (recv_sizes[0] > 0)
        recvup_edges.resize(recv_sizes[0]);
    if (recv_sizes[1] > 0)
        recvdn_edges.resize(recv_sizes[1]);

    // send/recv both up and down

    MPI_Sendrecv(sendup_edges.data(), send_sizes[0]*sizeof(struct EdgeTuple), MPI_BYTE, 
            up, SR_UP_TAG, recvdn_edges.data(), recv_sizes[1]*sizeof(struct EdgeTuple), 
            MPI_BYTE, down, SR_UP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(senddn_edges.data(), send_sizes[1]*sizeof(struct EdgeTuple), MPI_BYTE, 
            down, SR_DOWN_TAG, recvup_edges.data(), recv_sizes[0]*sizeof(struct EdgeTuple), 
            MPI_BYTE, up, SR_DOWN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // update local #edges

    // down
    if (down != MPI_PROC_NULL) {
        for (GraphElem i = 0; i < recv_sizes[1]; i++) {
#if defined(CHECK_NUM_EDGES)
            numEdges++;
#endif           
            edgeList.emplace_back(recvdn_edges[i].ij_[0], recvdn_edges[i].ij_[1], recvdn_edges[i].w_);
            g.edgeListIndexes[recvdn_edges[i].ij_[0]+1]++; 
        } 
    }

    // up
    if (up != MPI_PROC_NULL) {
        for (GraphElem i = 0; i < recv_sizes[0]; i++) {
#if defined(CHECK_NUM_EDGES)
            numEdges++;
#endif
            edgeList.emplace_back(recvup_edges[i].ij_[0], recvup_edges[i].ij_[1], recvup_edges[i].w_);
            g.edgeListIndexes[recvup_edges[i].ij_[0]+1]++; 
        }
    }

    // add random edges based on 
    // randomEdgePercent 
    if (randomEdgePercent > 0) {
        const GraphElem pnedges = (edgeList.size()/2);
        GraphElem tot_pnedges = 0;

        MPI_Allreduce(&pnedges, &tot_pnedges, 1, MPI_GRAPH_TYPE, 
                MPI_SUM, MPI_COMM_WORLD);

        // extra #edges per process
        const GraphElem nrande = (((GraphElem)randomEdgePercent * tot_pnedges)/100);
        GraphElem pnrande;

        // TODO FIXME try to ensure a fair edge distibution
        if (nrande < nprocs) {
            if (rank == (nprocs - 1))
                pnrande += nrande;
        }
        else {
            pnrande = nrande / nprocs;
            const GraphElem pnrem = nrande % nprocs;
            if (pnrem != 0) {
                if (rank == (nprocs - 1))
                    pnrande += pnrem;
            }
        }

        // add pnrande edges 

        // send/recv buffers
        std::vector<std::vector<EdgeTuple>> rand_edges(nprocs); 
        std::vector<EdgeTuple> sendrand_edges, recvrand_edges;

        // outgoing/incoming send/recv sizes
        // TODO FIXME if number of randomly added edges are above
        // INT_MAX, weird things will happen, fix it
        std::vector<int> sendrand_sizes(nprocs), recvrand_sizes(nprocs);

#if defined(PRINT_EXTRA_NEDGES)
        int extraEdges = 0;
#endif

#if defined(DEBUG_PRINTF)
        for (int i = 0; i < nprocs; i++) {
            if (i == rank) {
                std::cout << "[" << i << "]Target process for random edge insertion between " 
                    << lo << " and " << hi << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif

        // make sure each process has a 
        // different seed this time since
        // we want random edges
        unsigned rande_seed = (unsigned)(time(0)^getpid());
        GraphWeight weight = 1.0;
        std::hash<GraphElem> reh;

        // cannot use genRandom if it's already been seeded
        std::default_random_engine re(rande_seed); 
        std::uniform_int_distribution<GraphElem> IR, JR; 
        std::uniform_real_distribution<GraphWeight> IJW; 

        for (GraphElem k = 0; k < pnrande; k++) {

            // randomly pick start/end vertex and target from my list
            const GraphElem i = (GraphElem)IR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (n - 1)});
            const GraphElem g_j = (GraphElem)JR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (nv - 1)});
            const int target = dg->getOwner(g_j);
            const GraphElem j = dg->globalToLocal(g_j, target); // local

            if (i == j) 
                continue;

            const GraphElem g_i = dg->localToGlobal(i, rank);

            // check for duplicates prior to edgeList insertion
            auto found = std::find_if(edgeList.begin(), edgeList.end(), 
                    [&](EdgeTuple const& et) 
                    { return ((et.ij_[0] == i) && (et.ij_[1] == g_j)); });

            // OK to insert, not in list
            if (found == std::end(edgeList)) { 

                // calculate weight
                // TODO FIXME when target isn't myself or
                // my neighbors then edge weight is random
                // (it's not purely RGG at this point anyway)
                if (target == rank) {
                    GraphWeight dx = X[i] - X[j];
                    GraphWeight dy = Y[i] - Y[j];
                    weight = sqrt(dx*dx + dy*dy);
                }
                else if (target == up) {
                    GraphWeight dx = X[i] - X_up[j];
                    GraphWeight dy = Y[i] - Y_up[j];
                    weight = sqrt(dx*dx + dy*dy);
                }
                else if (target == down) {
                    GraphWeight dx = X[i] - X_down[j];
                    GraphWeight dy = Y[i] - Y_down[j];
                    weight = sqrt(dx*dx + dy*dy);
                }
                else {
                    unsigned randw_seed = reh((GraphElem)(g_i*nv+g_j));
                    std::default_random_engine rew(randw_seed); 
                    weight = (GraphWeight)IJW(rew, std::uniform_real_distribution<GraphWeight>::param_type{0.01, 1.0});
                }

                rand_edges[target].emplace_back(j, g_i, weight);
                sendrand_sizes[target]++;

#if defined(PRINT_EXTRA_NEDGES)
                extraEdges++;
#endif
#if defined(CHECK_NUM_EDGES)
                numEdges++;
#endif                       
                edgeList.emplace_back(i, g_j, weight);
                g.edgeListIndexes[i+1]++;
            }
        }

#if defined(PRINT_EXTRA_NEDGES)
        int totExtraEdges = 0;
        MPI_Reduce(&extraEdges, &totExtraEdges, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "Adding extra " << totExtraEdges << " edges while trying to incorporate " 
                << randomEdgePercent << "%" << " extra edges globally." << std::endl;
#endif

        MPI_Barrier(MPI_COMM_WORLD);

        // communicate ghosts edges
        MPI_Request rande_sreq;

        MPI_Ialltoall(sendrand_sizes.data(), 1, MPI_INT, 
                recvrand_sizes.data(), 1, MPI_INT, 
                MPI_COMM_WORLD, &rande_sreq);

        // send data if outgoing size > 0
        for (int p = 0; p < nprocs; p++) {
            sendrand_edges.insert(sendrand_edges.end(), 
                    rand_edges[p].begin(), rand_edges[p].end());
        }

        MPI_Wait(&rande_sreq, MPI_STATUS_IGNORE);

        // total recvbuffer size
        const int rcount = std::accumulate(recvrand_sizes.begin(), recvrand_sizes.end(), 0);
        recvrand_edges.resize(rcount);

        // alltoallv for incoming data
        // TODO FIXME make sure size of extra edges is 
        // within INT limits

        int rpos = 0, spos = 0;
        std::vector<int> sdispls(nprocs), rdispls(nprocs);

        for (int p = 0; p < nprocs; p++) {

            sendrand_sizes[p] *= sizeof(struct EdgeTuple);
            recvrand_sizes[p] *= sizeof(struct EdgeTuple);

            sdispls[p] = spos;
            rdispls[p] = rpos;

            spos += sendrand_sizes[p];
            rpos += recvrand_sizes[p];
        }

        MPI_Alltoallv(sendrand_edges.data(), sendrand_sizes.data(), sdispls.data(), 
                MPI_BYTE, recvrand_edges.data(), recvrand_sizes.data(), rdispls.data(), 
                MPI_BYTE, MPI_COMM_WORLD);

        // update local edge list
        for (int i = 0; i < rcount; i++) {
#if defined(CHECK_NUM_EDGES)
            numEdges++;
#endif
            edgeList.emplace_back(recvrand_edges[i].ij_[0], recvrand_edges[i].ij_[1], recvrand_edges[i].w_);
            g.edgeListIndexes[recvrand_edges[i].ij_[0]+1]++; 
        }

        sendrand_edges.clear();
        recvrand_edges.clear();
        rand_edges.clear();
    } // end of (conditional) random edges addition

    MPI_Barrier(MPI_COMM_WORLD);

    // set graph edge indices and prepare
    // graph data structure

    std::vector<GraphElem> ecTmp(n+1);
    std::partial_sum(g.edgeListIndexes.begin(), g.edgeListIndexes.end(), ecTmp.begin());
    g.edgeListIndexes = ecTmp;

    for(GraphElem i = 1; i < n+1; i++)
        g.edgeListIndexes[i] -= g.edgeListIndexes[0];   
    g.edgeListIndexes[0] = 0;

    g.setEdgeStartForVertex(0, 0);
    for (GraphElem i = 0; i < n; i++)
        g.setEdgeStartForVertex(i+1, g.edgeListIndexes[i+1]);

    const GraphElem nedges = g.edgeListIndexes[n] - g.edgeListIndexes[0];
    g.setNumEdges(nedges);

    // set graph edge list
    // sort edge list
    auto ecmp = [] (EdgeTuple const& e0, EdgeTuple const& e1)
    { return ((e0.ij_[0] < e1.ij_[0]) || ((e0.ij_[0] == e1.ij_[0]) && (e0.ij_[1] < e1.ij_[1]))); };

    if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp)) {
#if defined(DEBUG_PRINTF)
        std::cout << "Edge list is not sorted." << std::endl;
#endif
        std::sort(edgeList.begin(), edgeList.end(), ecmp);
    }
#if defined(DEBUG_PRINTF)
    else
        std::cout << "Edge list is sorted!" << std::endl;
#endif

    GraphElem ePos = 0;
    for (GraphElem i = 0; i < n; i++) {
        GraphElem e0, e1;

        g.getEdgeRangeForVertex(i, e0, e1);

#if defined(DEBUG_PRINTF)
        if ((i % 100000) == 0)
            std::cout << "Processing edges for vertex: " << i << ", range(" << e0 << ", " << e1 <<
                ")" << std::endl;
#endif
        for (GraphElem j = e0; j < e1; j++) {
            Edge &edge = g.getEdge(j);

            assert(ePos == j);
            assert(i == edgeList[ePos].ij_[0]);

            edge.tail = edgeList[ePos].ij_[1];
            edge.weight = edgeList[ePos].w_;

            ePos++;
        }
    }

#if defined(CHECK_NUM_EDGES)
    GraphElem tot_numEdges = 0;
    MPI_Allreduce(&numEdges, &tot_numEdges, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);
    const GraphElem tne = dg->getTotalNumEdges();
    assert(tne == tot_numEdges);
#endif
    edgeList.clear();

    X.clear();
    Y.clear();
    X_up.clear();
    Y_up.clear();
    X_down.clear();
    Y_down.clear();

    sendup_edges.clear();
    senddn_edges.clear();
    recvup_edges.clear();
    recvdn_edges.clear();

    return dg;
}
