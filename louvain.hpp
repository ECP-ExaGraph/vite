#ifndef __LOUVAIN_H
#define __LOUVAIN_H

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <mpi.h>
#include <omp.h>

#include "distgraph.hpp"
#include "coloring.hpp"

#if defined(__CRAY_MIC_KNL) && defined(USE_AUTOHBW_MEMALLOC)
#include <hbw_allocator.h>

typedef std::vector<GraphElem, hbw::allocator<GraphElem> > CommunityVector;
typedef std::vector<GraphWeight, hbw::allocator<GraphWeight> > GraphWeightVector;
typedef std::vector<GraphElem, hbw::allocator<GraphElem> > GraphElemVector;

typedef std::unordered_map<GraphElem, GraphElem, std::hash<GraphElem>, std::equal_to<GraphElem>,
        hbw::allocator< std::pair< const GraphElem, GraphElem > > > VertexCommMap;

typedef std::unordered_set<GraphElem, std::hash<GraphElem>, std::equal_to<GraphElem>,
        hbw::allocator<GraphElem> > RemoteVertexList;
typedef std::vector<RemoteVertexList, hbw::allocator<RemoteVertexList> > PartnerArray;

typedef std::vector<Comm, hbw::allocator<Comm> > CommVector;
typedef std::map<GraphElem, Comm, std::less<GraphElem>,
        hbw::allocator< std::pair< const GraphElem, Comm > > > CommMap;
typedef std::unordered_map<GraphElem, GraphElem, std::hash<GraphElem>, std::equal_to<GraphElem>,
        hbw::allocator< std::pair< const GraphElem, GraphElem > > > ClusterLocalMap;
#else
typedef std::vector<GraphElem> CommunityVector;
typedef std::vector<GraphWeight> GraphWeightVector;
typedef std::vector<GraphElem> GraphElemVector;

typedef std::unordered_map<GraphElem, GraphElem> VertexCommMap;

typedef std::unordered_set<GraphElem> RemoteVertexList;
typedef std::vector<RemoteVertexList> PartnerArray;

typedef std::vector<Comm> CommVector;
typedef std::map<GraphElem, Comm> CommMap;

typedef std::unordered_map<GraphElem, GraphElem> ClusterLocalMap;
#endif

const int SizeTag = 1;
const int VertexTag = 2;
const int CommunityTag = 3;
const int CommunitySizeTag = 4;
const int CommunityDataTag = 5;

struct CommInfo {
    GraphElem community;
    GraphElem size;
    GraphWeight degree;
};

// early termination cutoff for frozen 
// vertices is 90%
#define ET_CUTOFF 90
// early termination cutoff for 
// probabilistically freezing vertices 
// is 20%
#define P_CUTOFF 0.02

typedef std::vector<CommInfo> CommInfoVector;
extern std::ofstream ofs;
static MPI_Datatype commType;

GraphWeight distLouvainMethod(const int me, const int nprocs, const DistGraph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, 
        std::vector<GraphElem> &rsizes, std::vector<GraphElem> &svdata, 
        std::vector<GraphElem> &rvdata, CommunityVector &cvect, const GraphWeight lower,
        const GraphWeight thresh, int& iters);

GraphWeight distLouvainMethod(const int me, const int nprocs, const DistGraph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, 
        std::vector<GraphElem> &rsizes, std::vector<GraphElem> &svdata, 
        std::vector<GraphElem> &rvdata, CommunityVector &cvect, const GraphWeight lower,
        const GraphWeight thresh, int& iters, bool ETLocalOrRemote);

GraphWeight distLouvainMethod(const int me, const int nprocs, const DistGraph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, 
        std::vector<GraphElem> &rsizes, std::vector<GraphElem> &svdata, 
        std::vector<GraphElem> &rvdata, CommunityVector &cvect, const GraphWeight lower,
        const GraphWeight thresh, int& iters, GraphWeight ETDelta, bool ETLocalOrRemote);

GraphWeight distLouvainMethodWithColoring(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, int& iters);

GraphWeight distLouvainMethodWithColoring(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, bool ETLocalOrRemote);

GraphWeight distLouvainMethodWithColoring(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, 
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, 
        std::vector<GraphElem> &rsizes, std::vector<GraphElem> &svdata, 
        std::vector<GraphElem> &rvdata, CommunityVector &cvect, 
        const GraphWeight lower, const GraphWeight thresh, int& iters, GraphWeight ETDelta, 
        bool ETLocalOrRemote);

GraphWeight distLouvainMethodVertexOrder(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,                        
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, int& iters);

GraphWeight distLouvainMethodVertexOrder(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,                        
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, int& iters, 
        bool ETLocalOrRemote);

GraphWeight distLouvainMethodVertexOrder(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,                        
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, int& iters, 
        GraphWeight ETDelta, bool ETLocalOrRemote);

static void distInitLouvain(const DistGraph &dg, CommunityVector &pastComm, 
        CommunityVector &currComm, GraphWeightVector &vDegree, 
        GraphWeightVector &clusterWeight, CommVector &localCinfo, 
        CommVector &localCupdate, GraphWeight &constantForSecondTerm, 
        const int me);

static void distExecuteLouvainIteration(const GraphElem i, const DistGraph &dg,
        const CommunityVector &currComm, CommunityVector &targetComm,
        const GraphWeightVector &vDegree, CommVector &localCinfo, CommVector &localCupdate, 
        const VertexCommMap &remoteComm, const CommMap &remoteCinfo, CommMap &remoteCupdate,
        const GraphWeight constantForSecondTerm, GraphWeightVector &clusterWeight, const int me);

static void distSumVertexDegree(const Graph &g, GraphWeightVector &vDegree, CommVector &localCinfo);

static GraphWeight distCalcConstantForSecondTerm(const GraphWeightVector &vDegree);

static GraphElem distGetMaxIndex(const ClusterLocalMap &clmap, const GraphWeightVector &counter,
        const GraphWeight selfLoop, const CommVector &localCinfo, const CommMap &remoteCinfo,
        const GraphWeight vDegree, const GraphElem currSize, const GraphElem currDegree, 
        const GraphElem currComm, const GraphElem base, const GraphElem bound, const GraphWeight constant);

static GraphWeight distBuildLocalMapCounter(const GraphElem e0, const GraphElem e1,
        ClusterLocalMap &clmap, GraphWeightVector &counter, const Graph &g, const CommunityVector &currComm,
        const VertexCommMap &remoteComm, const GraphElem vertex, const GraphElem base, const GraphElem bound);

static GraphWeight distComputeModularity(const Graph &g, CommVector &localCinfo,
        const GraphWeightVector &clusterWeight, 
        const GraphWeight constantForSecondTerm, const int me);

static void distUpdateLocalCinfo(CommVector &localCinfo, const CommVector &localCupdate);

static void distCleanCWandCU(const GraphElem nv, GraphWeightVector &clusterWeight,
        CommVector &localCupdate);

static void distInitComm(CommunityVector &pastComm, CommunityVector &currComm,
        const GraphElem base);

static void updateRemoteCommunities(const DistGraph &dg, CommVector &localCinfo,
        const CommMap &remoteCupdate,
        const int me, const int nprocs);

static void fillRemoteCommunities(const DistGraph &dg, const int me, 
        const int nprocs, const size_t &ssz, const size_t &rsz,
        const std::vector<GraphElem> &ssizes, const std::vector<GraphElem> &rsizes, 
        const std::vector<GraphElem> &svdata, const std::vector<GraphElem> &rvdata,
        const CommunityVector &currComm, const CommVector &localCinfo, 
        CommMap &remoteCinfo, VertexCommMap &remoteComm, CommMap &remoteCupdate);

static void exchangeVertexReqs(const DistGraph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        const int me, const int nprocs);

void createCommunityMPIType();
void destroyCommunityMPIType();

// load ground truth file (vertex id --- community id) into a vector

// Ground Truth file is expected to be of the format generated 
// by LFR-gen by Fortunato, et al.
// https://sites.google.com/site/santofortunato/inthepress2

void loadGroundTruthFile(std::vector<GraphElem>& commGroundTruth, 
        std::string const& groundTruthFileName, 
        bool isGroundTruthZeroBased = true);

// gather current community info to root
void gatherAllComm(int root, int me, int nprocs, 
        std::vector<GraphElem>& commAll, 
        std::vector<GraphElem> const& localComm);

#endif
