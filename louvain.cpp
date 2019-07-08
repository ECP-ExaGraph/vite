#include "louvain.hpp"
#include <cstring>
#include <sstream>

#include "locks.hpp"

GraphWeight distLouvainMethod(const int me, const int nprocs, const DistGraph &dg,
        size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, bool ETLocalOrRemote)
{
  CommunityVector pastComm, currComm, targetComm;
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight, frozenClusterWeight;
  CommVector localCinfo, localCupdate;
 
  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  // initializations
  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);
  
  // for frozen vertices
  frozenClusterWeight.resize(nv);
  std::vector<bool> vActive;
  vActive.resize(nv);

  std::fill(vActive.begin(), vActive.end(), true);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);

#ifdef DEBUG_PRINTF  
  double t0, t1;
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif
  
  while(true) {
#ifdef DEBUG_PRINTF  
    const double t2 = MPI_Wtime();
    ofs << "Starting iteration: " << numIters << std::endl;
#endif
    numIters++;
    long vc_count = 0;

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Remote community map size: " << remoteComm.size() << std::endl;
    ofs << "Iteration communication time: " << (t1 - t0) << std::endl;
#endif

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

#pragma omp parallel default(none), shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        frozenClusterWeight, vActive), firstprivate(constantForSecondTerm) \
    reduction(+: vc_count)
    {
        distCleanCWandCU(nv, clusterWeight, localCupdate);

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided)
#endif
        for (GraphElem i = 0; i < nv; i++) {
            if (!vActive[i]) {
                clusterWeight[i] = frozenClusterWeight[i];
                vc_count++;
                continue;
            }
            else
            {
                distExecuteLouvainIteration(i, dg, currComm, targetComm, vDegree, localCinfo, 
                        localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                        constantForSecondTerm, clusterWeight, me);
                frozenClusterWeight[i] = clusterWeight[i];
            }
        }
    }
    if (!ETLocalOrRemote) {
        MPI_Allreduce(MPI_IN_PLACE, &vc_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        GraphWeight perc = 100.0 * ((GraphWeight)(tnv-vc_count)/(GraphWeight)tnv);
        if (vc_count >= ET_CUTOFF) {
            currMod = -1;
            break;
        }
    }

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Iteration computation time: " << (t1 - t0) << std::endl;
    t0 = MPI_Wtime();
#endif

#pragma omp parallel default(none), shared(localCinfo, localCupdate)
    {
        distUpdateLocalCinfo(localCinfo, localCupdate);
    }

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif
    updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Update remote communities communication time: " << (t1 - t0) << std::endl;
    t0 = MPI_Wtime();
#endif

    currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Modularity computation + communication time: " << (t1 - t0) << std::endl;
#endif

    if ((currMod - prevMod) < threshMod){
#ifdef DEBUG_PRINTF  
        ofs << "Break here - no updates " << std::endl;
#endif
        break;
    }

    prevMod = currMod;

    if (prevMod < lower)
        prevMod = lower;

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive) \
    firstprivate(numIters) schedule(runtime)
#else
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive) \
    firstprivate(numIters) schedule(static)
#endif
    for (GraphElem i = 0; i < nv; i++) {
        if (numIters > 2 
                && (targetComm[i] == currComm[i] == pastComm[i]))
            vActive[i] = false;
        else {
            GraphElem tmp = pastComm[i];
            pastComm[i] = currComm[i];
            currComm[i] = targetComm[i];
            targetComm[i] = tmp;
        }
    }
#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Update local communities time: " << (t1 - t0) << std::endl;
    ofs << "Total iteration time: " << (t1 - t2) << std::endl;
#endif
  };

  cvect = pastComm;
  iters = numIters;
  
  vActive.clear();
  frozenClusterWeight.clear();
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethod with early termination

GraphWeight distLouvainMethod(const int me, const int nprocs, 
        const DistGraph &dg, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, GraphWeight ETDelta, bool ETLocalOrRemote)
{
  CommunityVector pastComm, currComm, targetComm;
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight, frozenClusterWeight;
  CommVector localCinfo, localCupdate;
 
  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  // initializations
  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);
  
  // for frozen vertices
  frozenClusterWeight.resize(nv);
  std::vector<bool> vActive;
  vActive.resize(nv);
  
  // vertex probabilities for current and previous iteration
  std::vector<GraphWeight> p_curr, p_prev;
  p_curr.resize(nv); 
  p_prev.resize(nv);

  std::fill(vActive.begin(), vActive.end(), true);
  std::fill(p_prev.begin(), p_prev.end(), 1.0);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);

#ifdef DEBUG_PRINTF  
  double t0, t1;
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif
  
  while(true) {
#ifdef DEBUG_PRINTF  
    const double t2 = MPI_Wtime();
    ofs << "Starting iteration: " << numIters << std::endl;
#endif
    numIters++;
    long vc_count = 0;

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Remote community map size: " << remoteComm.size() << std::endl;
    ofs << "Iteration communication time: " << (t1 - t0) << std::endl;
#endif

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

#pragma omp parallel default(none), shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        frozenClusterWeight, vActive), firstprivate(constantForSecondTerm) \
    reduction(+: vc_count)
    {
        distCleanCWandCU(nv, clusterWeight, localCupdate);

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime) 
#else
#pragma omp for schedule(guided) 
#endif
        for (GraphElem i = 0; i < nv; i++) {
            if (!vActive[i]) {
                clusterWeight[i] = frozenClusterWeight[i];
                vc_count++;
                continue;
            }
            else
            {
                distExecuteLouvainIteration(i, dg, currComm, targetComm, vDegree, localCinfo, 
                        localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                        constantForSecondTerm, clusterWeight, me);
                frozenClusterWeight[i] = clusterWeight[i];
            }
        }
    }
    if (!ETLocalOrRemote) {
        MPI_Allreduce(MPI_IN_PLACE, &vc_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        GraphWeight perc = 100.0 * ((GraphWeight)(tnv-vc_count)/(GraphWeight)tnv);
        if (vc_count >= ET_CUTOFF) {
            currMod = -1;
            break;
        }
    }

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Iteration computation time: " << (t1 - t0) << std::endl;
    t0 = MPI_Wtime();
#endif

#pragma omp parallel default(none), shared(localCinfo, localCupdate)
    {
        distUpdateLocalCinfo(localCinfo, localCupdate);
    }

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif
    updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Update remote communities communication time: " << (t1 - t0) << std::endl;
    t0 = MPI_Wtime();
#endif

    currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Modularity computation + communication time: " << (t1 - t0) << std::endl;
#endif

    if ((currMod - prevMod) < threshMod){
#ifdef DEBUG_PRINTF  
        ofs << "Break here - no updates " << std::endl;
#endif
        break;
    }

    prevMod = currMod;

    if (prevMod < lower)
        prevMod = lower;

    const GraphWeight one_minus_delta = (1.0 - ETDelta);
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive, p_curr, p_prev) \
    firstprivate(numIters, one_minus_delta) schedule(runtime)
#else
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive, p_curr, p_prev) \
    firstprivate(numIters, one_minus_delta) schedule(static)
#endif
    for (GraphElem i = 0; i < nv; i++) {
        if (!vActive[i])
            continue;
        
        if (numIters > 2 && (currComm[i] == pastComm[i])) {
            p_curr[i] = p_prev[i]*one_minus_delta;
            if (p_curr[i] <= P_CUTOFF)
                vActive[i] = false;
        }

        if (vActive[i]) 
        {
            GraphElem tmp = pastComm[i];
            pastComm[i] = currComm[i];
            currComm[i] = targetComm[i];
            targetComm[i] = tmp;
        }
    }
    
    // swap p_active k, k-1 iteration
    std::swap(p_curr, p_prev);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Update local communities time: " << (t1 - t0) << std::endl;
    ofs << "Total iteration time: " << (t1 - t2) << std::endl;
#endif
  };

  cvect = pastComm;
  iters = numIters;
  
  p_curr.clear();
  p_prev.clear();
  vActive.clear();
  frozenClusterWeight.clear();
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethod with early termination probabilities

GraphWeight distLouvainMethod(const int me, const int nprocs, const DistGraph &dg,
        size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters)
{
  CommunityVector pastComm, currComm, targetComm;
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight;
  CommVector localCinfo, localCupdate;
  
  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;
 
  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);

#ifdef DEBUG_PRINTF  
  double t0, t1;
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif
  
  while(true) {
#ifdef DEBUG_PRINTF  
    const double t2 = MPI_Wtime();
    ofs << "Starting iteration: " << numIters << std::endl;
#endif
    numIters++;

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Remote community map size: " << remoteComm.size() << std::endl;
    ofs << "Iteration communication time: " << (t1 - t0) << std::endl;
#endif

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif
#pragma omp parallel default(none), shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate), \
    firstprivate(constantForSecondTerm)
    {
        distCleanCWandCU(nv, clusterWeight, localCupdate);

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided) 
#endif
        for (GraphElem i = 0; i < nv; i++) {
            distExecuteLouvainIteration(i, dg, currComm, targetComm, vDegree, localCinfo, 
                    localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                    constantForSecondTerm, clusterWeight, me);
        }
    }

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Iteration computation time: " << (t1 - t0) << std::endl;
    t0 = MPI_Wtime();
#endif

#pragma omp parallel default(none), shared(localCinfo, localCupdate)
    {
        distUpdateLocalCinfo(localCinfo, localCupdate);
    }

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif
    updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Update remote communities communication time: " << (t1 - t0) << std::endl;
    t0 = MPI_Wtime();
#endif

    currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Modularity computation + communication time: " << (t1 - t0) << std::endl;
#endif

    if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
        ofs << "Break here - no updates " << std::endl;
#endif
        break;
    }

    prevMod = currMod;

    if (prevMod < lower)
        prevMod = lower;

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none) \
    shared(pastComm, currComm, targetComm) \
    schedule(runtime)
#else
#pragma omp parallel for default(none) \
    shared(pastComm, currComm, targetComm) \
    schedule(static)
#endif
    for (GraphElem i = 0; i < nv; i++) {
        GraphElem tmp = pastComm[i];
        pastComm[i] = currComm[i];
        currComm[i] = targetComm[i];
        targetComm[i] = tmp;
    }

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    ofs << "Update local communities time: " << (t1 - t0) << std::endl;
    ofs << "Total iteration time: " << (t1 - t2) << std::endl;
#endif
  };

  cvect = pastComm;
  iters = numIters;
  
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethod plain

GraphWeight distLouvainMethodWithColoring(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters)
{
  // if no colors, then fall back to original distLouvain  
  if (numColor == 1) {
      ofs << "No color specified, executing non-color Louvain..." << std::endl;
      return distLouvainMethod(me, nprocs, dg, ssz, rsz, ssizes, 
              rsizes, svdata, rvdata, cvect, lower, thresh, iters);
  }
    
  CommunityVector pastComm, currComm, targetComm; 
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight;
  CommVector localCinfo, localCupdate;

  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  double t0, t1;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif

  /*** Create a CSR-like datastructure for vertex-colors ***/
  std::vector<long> colorPtr(numColor+1,0);
  std::vector<long> colorIndex(nv,0);
  std::vector<long> colorAdded(numColor,0);

  // Count the size of each color	
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for(long i=0; i < nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      __sync_fetch_and_add(&colorPtr[(long)start+1],1);
  }

  //Prefix sum:
  for(long i=0; i<numColor; i++) {
      colorPtr[i+1] += colorPtr[i];
  }	
  
  //Group vertices with the same color in particular order
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for (long i=0; i<nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      const long vindex = colorPtr[start] + __sync_fetch_and_add(&(colorAdded[start]), 1);
      colorIndex[vindex] = i;
  }

  while(true) {
      
#ifdef DEBUG_PRINTF  
      const double t2 = MPI_Wtime();
      ofs << "Starting iteration: " << numIters << std::endl;
#endif      
      numIters++;	
#ifdef DEBUG_PRINTF  
      t0 = MPI_Wtime();
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight) schedule(runtime)
#else
#pragma omp parallel for schedule(static) shared(clusterWeight)
#endif
      for (GraphElem i = 0L; i < nv; i++)
          clusterWeight[i] = 0;

      // Color loop
      for(long ci = 0; ci < numColor; ci++) {

          fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
                  rsizes, svdata, rvdata, currComm, localCinfo, 
                  remoteCinfo, remoteComm, remoteCupdate);

          const long coloradj1 = colorPtr[ci];
          const long coloradj2 = colorPtr[ci+1];

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, pastComm), \
          firstprivate(constantForSecondTerm) schedule(runtime)
#else
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, pastComm), \
          firstprivate(constantForSecondTerm) schedule(static)
#endif
          for (long K = coloradj1; K < coloradj2; K++) {
              distExecuteLouvainIteration(colorIndex[K], dg, currComm, targetComm, 
                      vDegree, localCinfo, localCupdate, remoteComm, remoteCinfo, 
                      remoteCupdate, constantForSecondTerm, clusterWeight, me);
          }

          // update local cinfo
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(static)
#endif          
          for (GraphElem i = 0; i < nv; i++) {
              localCinfo[i].size += localCupdate[i].size;
              localCinfo[i].degree += localCupdate[i].degree;

              localCupdate[i].size = 0;
              localCupdate[i].degree = 0;
          }
          
          updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
      } // end of Color loop
#ifdef DEBUG_PRINTF  
      t1 = MPI_Wtime();
      ofs << "Color iteration computation time: " << (t1 - t0) << std::endl;
#endif      
      
      // global modularity
      currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);
      if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
          ofs << "Break here - no updates " << std::endl;
#endif
          break;
      }
      
      prevMod = currMod;
      if (prevMod < lower)
          prevMod = lower;   

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm) \
      schedule(runtime)
#else
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm) \
      schedule(static)
#endif
      for (GraphElem i = 0L; i < nv; i++) {    
          GraphElem tmp = pastComm[i];
          pastComm[i] = currComm[i];
          currComm[i] = targetComm[i];
          targetComm[i] = tmp;       
      }
  }; // end while loop

  cvect = pastComm;  
  iters = numIters;
  
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethodWithColoring plain

GraphWeight distLouvainMethodWithColoring(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, bool ETLocalOrRemote)
{
  // if no colors, then fall back to original distLouvain  
  if (numColor == 1) {
      ofs << "No color specified, executing non-color Louvain..." << std::endl;
      return distLouvainMethod(me, nprocs, dg, ssz, rsz, ssizes, 
              rsizes, svdata, rvdata, cvect, lower, thresh, iters);
  }
    
  CommunityVector pastComm, currComm, targetComm; 
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight, frozenClusterWeight;
  CommVector localCinfo, localCupdate;

  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);
  
  // for frozen vertices
  frozenClusterWeight.resize(nv);
  std::vector<bool> vActive;
  vActive.resize(nv);
  std::fill(vActive.begin(), vActive.end(), true);
 
#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  double t0, t1;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif

  /*** Create a CSR-like datastructure for vertex-colors ***/
  std::vector<long> colorPtr(numColor+1, 0);
  std::vector<long> colorIndex(nv, 0);
  std::vector<long> colorAdded(numColor, 0);

  // Count the size of each color	
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for(long i=0; i < nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      __sync_fetch_and_add(&colorPtr[(long)start+1],1);
  }

  //Prefix sum:
  for(long i=0; i<numColor; i++) {
      colorPtr[i+1] += colorPtr[i];
  }	
  
  //Group vertices with the same color in particular order
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for (long i=0; i<nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      const long vindex = colorPtr[start] + __sync_fetch_and_add(&(colorAdded[start]), 1);
      colorIndex[vindex] = i;
  }

  while(true) {
      
#ifdef DEBUG_PRINTF  
      const double t2 = MPI_Wtime();
      ofs << "Starting iteration: " << numIters << std::endl;
#endif      
      numIters++;	
      long vc_count = 0;
#ifdef DEBUG_PRINTF  
      t0 = MPI_Wtime();
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight) schedule(runtime)
#else
#pragma omp parallel for schedule(static) shared(clusterWeight)
#endif
      for (GraphElem i = 0L; i < nv; i++)
          clusterWeight[i] = 0;

      // Color loop
      for(long ci = 0; ci < numColor; ci++) {

          fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
                  rsizes, svdata, rvdata, currComm, localCinfo, 
                  remoteCinfo, remoteComm, remoteCupdate);

          const long coloradj1 = colorPtr[ci];
          const long coloradj2 = colorPtr[ci+1];

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, pastComm, \
        colorIndex, frozenClusterWeight, vActive), firstprivate(constantForSecondTerm), \
          reduction(+: vc_count) schedule(runtime)
#else
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, pastComm, \
        colorIndex, frozenClusterWeight, vActive), firstprivate(constantForSecondTerm), \
          reduction(+: vc_count) schedule(static)
#endif
          for (long K = coloradj1; K < coloradj2; K++) {
              if (!vActive[colorIndex[K]]) {
                  clusterWeight[colorIndex[K]] = frozenClusterWeight[colorIndex[K]];
                  vc_count++;
                  continue;
              }
              else
              {
                  distExecuteLouvainIteration(colorIndex[K], dg, currComm, targetComm, 
                          vDegree, localCinfo, localCupdate, remoteComm, remoteCinfo, 
                          remoteCupdate, constantForSecondTerm, clusterWeight, me);
                  frozenClusterWeight[colorIndex[K]] = clusterWeight[colorIndex[K]];
              }
          }

          // update local cinfo
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(static)
#endif
          for (GraphElem i = 0; i < nv; i++) {
              localCinfo[i].size += localCupdate[i].size;
              localCinfo[i].degree += localCupdate[i].degree;

              localCupdate[i].size = 0;
              localCupdate[i].degree = 0;
          }
          
          updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
      } // end of Color loop
#ifdef DEBUG_PRINTF  
      t1 = MPI_Wtime();
      ofs << "Color iteration computation time: " << (t1 - t0) << std::endl;
#endif      

      if (!ETLocalOrRemote) {
          MPI_Allreduce(MPI_IN_PLACE, &vc_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
          GraphWeight perc = 100.0 * ((GraphWeight)(tnv-vc_count)/(GraphWeight)tnv);
          if (vc_count >= ET_CUTOFF) {
              currMod = -1;
              break;
          }
      }

      // global modularity
      currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);
      if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
          ofs << "Break here - no updates " << std::endl;
#endif
          break;
      }
      prevMod = currMod;
      if (prevMod < lower)
          prevMod = lower;   

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm, vActive) \
      firstprivate(numIters) schedule(runtime)
#else
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm, vActive) \
      firstprivate(numIters) schedule(static)
#endif
      for (GraphElem i = 0L; i < nv; i++) {    
          if (numIters > 2 
                  && (targetComm[i] == currComm[i] == pastComm[i]))
              vActive[i] = false;
          else {
              GraphElem tmp = pastComm[i];
              pastComm[i] = currComm[i];
              currComm[i] = targetComm[i];
              targetComm[i] = tmp;
          }
      }
  }; // end while loop

  cvect = pastComm;  
  iters = numIters;

  vActive.clear();
  frozenClusterWeight.clear();
    
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethodWithColoring with early termination

GraphWeight distLouvainMethodWithColoring(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, GraphWeight ETDelta, bool ETLocalOrRemote)
{
  // if no colors, then fall back to original distLouvain  
  if (numColor == 1) {
      ofs << "No color specified, executing non-color Louvain..." << std::endl;
      return distLouvainMethod(me, nprocs, dg, ssz, rsz, ssizes, 
              rsizes, svdata, rvdata, cvect, lower, thresh, iters);
  }
    
  CommunityVector pastComm, currComm, targetComm; 
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight, frozenClusterWeight;
  CommVector localCinfo, localCupdate;

  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);
  
  // for frozen vertices
  frozenClusterWeight.resize(nv);
  std::vector<bool> vActive;
  vActive.resize(nv);
  
  // vertex probabilities for current and previous iteration
  std::vector<GraphWeight> p_curr, p_prev;
  p_curr.resize(nv); 
  p_prev.resize(nv); 

  std::fill(vActive.begin(), vActive.end(), true);
  std::fill(p_prev.begin(), p_prev.end(), 1.0);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  double t0, t1;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif

  /*** Create a CSR-like datastructure for vertex-colors ***/
  std::vector<long> colorPtr(numColor+1, 0);
  std::vector<long> colorIndex(nv, 0);
  std::vector<long> colorAdded(numColor, 0);

  // Count the size of each color	
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for(long i=0; i < nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      __sync_fetch_and_add(&colorPtr[(long)start+1],1);
  }

  //Prefix sum:
  for(long i=0; i<numColor; i++) {
      colorPtr[i+1] += colorPtr[i];
  }	
  
  //Group vertices with the same color in particular order
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for (long i=0; i<nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      const long vindex = colorPtr[start] + __sync_fetch_and_add(&(colorAdded[start]), 1);
      colorIndex[vindex] = i;
  }

  while(true) {
      
#ifdef DEBUG_PRINTF  
      const double t2 = MPI_Wtime();
      ofs << "Starting iteration: " << numIters << std::endl;
#endif      
      numIters++;	
      long vc_count = 0;
#ifdef DEBUG_PRINTF  
      t0 = MPI_Wtime();
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight) schedule(runtime)
#else
#pragma omp parallel for schedule(static) shared(clusterWeight)
#endif
      for (GraphElem i = 0L; i < nv; i++)
          clusterWeight[i] = 0;

      // Color loop
      for(long ci = 0; ci < numColor; ci++) {

          fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
                  rsizes, svdata, rvdata, currComm, localCinfo, 
                  remoteCinfo, remoteComm, remoteCupdate);

          const long coloradj1 = colorPtr[ci];
          const long coloradj2 = colorPtr[ci+1];

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, pastComm, \
        vActive, frozenClusterWeight, colorIndex), firstprivate(constantForSecondTerm), \
          reduction(+: vc_count) schedule(runtime)
#else
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, pastComm, \
        vActive, frozenClusterWeight, colorIndex), firstprivate(constantForSecondTerm), \
          reduction(+: vc_count) schedule(static)
#endif
          for (long K = coloradj1; K < coloradj2; K++) {
              if (!vActive[colorIndex[K]]) {
                  clusterWeight[colorIndex[K]] = frozenClusterWeight[colorIndex[K]];
                  vc_count++;
                  continue;
              }
              else
              {
                  distExecuteLouvainIteration(colorIndex[K], dg, currComm, targetComm, 
                          vDegree, localCinfo, localCupdate, remoteComm, remoteCinfo, 
                          remoteCupdate, constantForSecondTerm, clusterWeight, me);
                  frozenClusterWeight[colorIndex[K]] = clusterWeight[colorIndex[K]];
              }
          }

          // update local cinfo
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(static)
#endif
          for (GraphElem i = 0; i < nv; i++) {
              localCinfo[i].size += localCupdate[i].size;
              localCinfo[i].degree += localCupdate[i].degree;

              localCupdate[i].size = 0;
              localCupdate[i].degree = 0;
          }
          
          updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
      } // end of Color loop
#ifdef DEBUG_PRINTF  
      t1 = MPI_Wtime();
      ofs << "Color iteration computation time: " << (t1 - t0) << std::endl;
#endif      

      if (!ETLocalOrRemote) {
          MPI_Allreduce(MPI_IN_PLACE, &vc_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
          GraphWeight perc = 100.0 * ((GraphWeight)(tnv-vc_count)/(GraphWeight)tnv);
          if (vc_count >= ET_CUTOFF) {
              currMod = -1;
              break;
          }
      }

      // global modularity
      currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);
      if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
          ofs << "Break here - no updates " << std::endl;
#endif
          break;
      }
      prevMod = currMod;
      if (prevMod < lower)
          prevMod = lower;   

      const GraphWeight one_minus_delta = (1.0 - ETDelta);
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive, p_curr, p_prev) \
    firstprivate(numIters, one_minus_delta) schedule(runtime)
#else
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive, p_curr, p_prev) \
    firstprivate(numIters, one_minus_delta) schedule(static)
#endif
    for (GraphElem i = 0; i < nv; i++) {
        if (!vActive[i])
            continue;
        
        if (numIters > 2 && (currComm[i] == pastComm[i])) {
            p_curr[i] = p_prev[i]*one_minus_delta;
            if (p_curr[i] <= P_CUTOFF)
                vActive[i] = false;
        }

        if (vActive[i]) 
        {
            GraphElem tmp = pastComm[i];
            pastComm[i] = currComm[i];
            currComm[i] = targetComm[i];
            targetComm[i] = tmp;
        }
    }
    
    // swap p_active k, k-1 iteration
    std::swap(p_curr, p_prev);
  }; // end while loop

  cvect = pastComm;  
  iters = numIters;
  
  frozenClusterWeight.clear();
  vActive.clear();
  p_curr.clear();
  p_prev.clear();
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethodWithColoring with early termination + probability

// Invoke Louvain step according to a vertex ordering scheme implemented
// by coloring...the difference between this and the previous coloring 
// version is that communication is exactly same as in non-coloring version,
// only the Louvain step is arranged such that adjacent locally owned vertices 
// are not processed at a time 
GraphWeight distLouvainMethodVertexOrder(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters)
{
  // if no colors, then fall back to original distLouvain  
  if (numColor == 1) {
      ofs << "No color specified, executing non-color Louvain..." << std::endl;
      return distLouvainMethod(me, nprocs, dg, ssz, rsz, ssizes, 
              rsizes, svdata, rvdata, cvect, lower, thresh, iters);
  }
    
  CommunityVector pastComm, currComm, targetComm; 
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight;
  CommVector localCinfo, localCupdate;

  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  double t0, t1;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif

  /*** Create a CSR-like datastructure for vertex-colors ***/
  std::vector<long> colorPtr(numColor+1, 0);
  std::vector<long> colorIndex(nv, 0);
  std::vector<long> colorAdded(numColor, 0);

  // Count the size of each color	
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for(long i=0; i < nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      __sync_fetch_and_add(&colorPtr[(long)start+1],1);
  }

  //Prefix sum:
  for(long i=0; i<numColor; i++) {
      colorPtr[i+1] += colorPtr[i];
  }	
  
  //Group vertices with the same color in particular order
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for (long i=0; i<nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      const long vindex = colorPtr[start] + __sync_fetch_and_add(&(colorAdded[start]), 1);
      colorIndex[vindex] = i;
  }

  while(true) {
      
#ifdef DEBUG_PRINTF  
      const double t2 = MPI_Wtime();
      ofs << "Starting iteration: " << numIters << std::endl;
#endif      
      numIters++;	

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight) schedule(runtime)
#else
#pragma omp parallel for schedule(static) shared(clusterWeight)
#endif
      for (GraphElem i = 0L; i < nv; i++)
          clusterWeight[i] = 0;
          
      fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
                  rsizes, svdata, rvdata, currComm, localCinfo, 
                  remoteCinfo, remoteComm, remoteCupdate);
       
#ifdef DEBUG_PRINTF  
      t0 = MPI_Wtime();
#endif
      // Color loop
      for(long ci = 0; ci < numColor; ci++) {
        
          const long coloradj1 = colorPtr[ci];
          const long coloradj2 = colorPtr[ci+1];

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        colorIndex), firstprivate(constantForSecondTerm) schedule(runtime)
#else
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        colorIndex), firstprivate(constantForSecondTerm) schedule(static)
#endif
          for (long K = coloradj1; K < coloradj2; K++) {
              distExecuteLouvainIteration(colorIndex[K], dg, currComm, targetComm, 
                      vDegree, localCinfo, localCupdate, remoteComm, remoteCinfo, 
                      remoteCupdate, constantForSecondTerm, clusterWeight, me);
          }
      } // end of Color loop
#ifdef DEBUG_PRINTF  
      t1 = MPI_Wtime();
      ofs << "Color iteration computation time: " << (t1 - t0) << std::endl;
#endif      
      
      // update local cinfo
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(static)
#endif
      for (GraphElem i = 0; i < nv; i++) {
          localCinfo[i].size += localCupdate[i].size;
          localCinfo[i].degree += localCupdate[i].degree;

          localCupdate[i].size = 0;
          localCupdate[i].degree = 0;
      }

      updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
      
      // global modularity
      currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);
      if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
          ofs << "Break here - no updates " << std::endl;
#endif
          break;
      }
      prevMod = currMod;
      if (prevMod < lower)
          prevMod = lower;   
      
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm) \
      schedule(runtime)
#else
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm) \
      schedule(static)
#endif
      for (GraphElem i = 0; i < nv; i++) {
          GraphElem tmp = pastComm[i];
          pastComm[i] = currComm[i];
          currComm[i] = targetComm[i];
          targetComm[i] = tmp;       
      }
  }; // end while loop

  cvect = pastComm;
  iters = numIters;
  
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethodVertexOrder plain

GraphWeight distLouvainMethodVertexOrder(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, bool ETLocalOrRemote)
{
  // if no colors, then fall back to original distLouvain  
  if (numColor == 1) {
      ofs << "No color specified, executing non-color Louvain..." << std::endl;
      return distLouvainMethod(me, nprocs, dg, ssz, rsz, ssizes, 
              rsizes, svdata, rvdata, cvect, lower, thresh, iters);
  }
    
  CommunityVector pastComm, currComm, targetComm; 
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight, frozenClusterWeight;
  CommVector localCinfo, localCupdate;

  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);
  
  // for frozen vertices
  frozenClusterWeight.resize(nv);
  std::vector<bool> vActive;
  vActive.resize(nv);
  std::fill(vActive.begin(), vActive.end(), true);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  double t0, t1;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif

  /*** Create a CSR-like datastructure for vertex-colors ***/
  std::vector<long> colorPtr(numColor+1, 0);
  std::vector<long> colorIndex(nv, 0);
  std::vector<long> colorAdded(numColor, 0);

  // Count the size of each color	
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for(long i=0; i < nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      __sync_fetch_and_add(&colorPtr[(long)start+1],1);
  }

  //Prefix sum:
  for(long i=0; i<numColor; i++) {
      colorPtr[i+1] += colorPtr[i];
  }	
  
  //Group vertices with the same color in particular order
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for (long i=0; i<nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      const long vindex = colorPtr[start] + __sync_fetch_and_add(&(colorAdded[start]), 1);
      colorIndex[vindex] = i;
  }

  while(true) {
      
#ifdef DEBUG_PRINTF  
      const double t2 = MPI_Wtime();
      ofs << "Starting iteration: " << numIters << std::endl;
#endif      
      numIters++;	
      long vc_count = 0;

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight) schedule(runtime)
#else
#pragma omp parallel for schedule(static) shared(clusterWeight)
#endif
      for (GraphElem i = 0L; i < nv; i++)
          clusterWeight[i] = 0;

      fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
                  rsizes, svdata, rvdata, currComm, localCinfo, 
                  remoteCinfo, remoteComm, remoteCupdate);
       
#ifdef DEBUG_PRINTF  
      t0 = MPI_Wtime();
#endif
      // Color loop
      for(long ci = 0; ci < numColor; ci++) {

          const long coloradj1 = colorPtr[ci];
          const long coloradj2 = colorPtr[ci+1];

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        frozenClusterWeight, colorIndex), firstprivate(constantForSecondTerm) \
          reduction(+:vc_count) schedule(runtime)
#else
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        frozenClusterWeight, colorIndex), firstprivate(constantForSecondTerm) \
          reduction(+:vc_count) schedule(static)
#endif
          for (long K = coloradj1; K < coloradj2; K++) {
              if (!vActive[colorIndex[K]]) {
                  clusterWeight[colorIndex[K]] = frozenClusterWeight[colorIndex[K]];
                  vc_count++;
                  continue;
              }
              else
              {
                  distExecuteLouvainIteration(colorIndex[K], dg, currComm, targetComm, 
                          vDegree, localCinfo, localCupdate, remoteComm, remoteCinfo, 
                          remoteCupdate, constantForSecondTerm, clusterWeight, me);
                  frozenClusterWeight[colorIndex[K]] = clusterWeight[colorIndex[K]];
              }
          }
      } // end of Color loop
#ifdef DEBUG_PRINTF  
      t1 = MPI_Wtime();
      ofs << "Color iteration computation time: " << (t1 - t0) << std::endl;
#endif      
 
      if (!ETLocalOrRemote) {
          MPI_Allreduce(MPI_IN_PLACE, &vc_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
          GraphWeight perc = 100.0 * ((GraphWeight)(tnv-vc_count)/(GraphWeight)tnv);
          if (vc_count >= ET_CUTOFF) {
              currMod = -1;
              break;
          }
      }     
      
      // update local cinfo
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(static)
#endif
      for (GraphElem i = 0; i < nv; i++) {
          localCinfo[i].size += localCupdate[i].size;
          localCinfo[i].degree += localCupdate[i].degree;

          localCupdate[i].size = 0;
          localCupdate[i].degree = 0;
      }

      updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
      
      // global modularity
      currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);
      if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
          ofs << "Break here - no updates " << std::endl;
#endif
          break;
      }
      prevMod = currMod;
      if (prevMod < lower)
          prevMod = lower;   
      
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm, vActive) \
      firstprivate(numIters) schedule(runtime)
#else
#pragma omp parallel for default(none), \
      shared(pastComm, currComm, targetComm, vActive) \
      firstprivate(numIters) schedule(static)
#endif
      for (GraphElem i = 0L; i < nv; i++) {    
          if (numIters > 2 
                  && (targetComm[i] == currComm[i] == pastComm[i]))
              vActive[i] = false;
          else {
              GraphElem tmp = pastComm[i];
              pastComm[i] = currComm[i];
              currComm[i] = targetComm[i];
              targetComm[i] = tmp;
          }
      }
  }; // end while loop

  cvect = pastComm;
  iters = numIters;

  vActive.clear();
  frozenClusterWeight.clear();
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethodVertexOrder with early termination

GraphWeight distLouvainMethodVertexOrder(const int me, const int nprocs, const DistGraph &dg,
        const long numColor, const ColorVector &vertexColor, size_t &ssz, size_t &rsz, 
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, const GraphWeight lower, const GraphWeight thresh, 
        int& iters, GraphWeight ETDelta, bool ETLocalOrRemote)
{
  // if no colors, then fall back to original distLouvain  
  if (numColor == 1) {
      ofs << "No color specified, executing non-color Louvain..." << std::endl;
      return distLouvainMethod(me, nprocs, dg, ssz, rsz, ssizes, 
              rsizes, svdata, rvdata, cvect, lower, thresh, iters);
  }
    
  CommunityVector pastComm, currComm, targetComm; 
  GraphWeightVector vDegree;
  GraphWeightVector clusterWeight, frozenClusterWeight;
  CommVector localCinfo, localCupdate;

  VertexCommMap remoteComm;
  CommMap remoteCinfo, remoteCupdate;
  
  const Graph &g = dg.getLocalGraph();
  const GraphElem tnv = dg.getTotalNumVertices();
  const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
  const GraphWeight threshMod = thresh;

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;

  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);
  
  // for frozen vertices
  frozenClusterWeight.resize(nv);
  std::vector<bool> vActive;
  vActive.resize(nv);
   
  // vertex probabilities for current and previous iteration
  std::vector<GraphWeight> p_curr, p_prev;
  p_curr.resize(nv); 
  p_prev.resize(nv); 

  std::fill(vActive.begin(), vActive.end(), true);
  std::fill(p_prev.begin(), p_prev.end(), 1.0);

#ifdef DEBUG_PRINTF  
  ofs << "constantForSecondTerm: " << constantForSecondTerm << std::endl;
#endif
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  double t0, t1;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ofs << "Initial communication setup time: " << (t1 - t0) << std::endl;
#endif

  /*** Create a CSR-like datastructure for vertex-colors ***/
  std::vector<long> colorPtr(numColor+1, 0);
  std::vector<long> colorIndex(nv, 0);
  std::vector<long> colorAdded(numColor, 0);

  // Count the size of each color	
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for(long i=0; i < nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      __sync_fetch_and_add(&colorPtr[(long)start+1],1);
  }

  //Prefix sum:
  for(long i=0; i<numColor; i++) {
      colorPtr[i+1] += colorPtr[i];
  }	
  
  //Group vertices with the same color in particular order
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for firstprivate(numColor) schedule(runtime)
#else
#pragma omp parallel for firstprivate(numColor) schedule(static)
#endif
  for (long i=0; i<nv; i++) {
      const long start = (vertexColor[i] < 0)?(numColor-1):vertexColor[i];
      const long vindex = colorPtr[start] + __sync_fetch_and_add(&(colorAdded[start]), 1);
      colorIndex[vindex] = i;
  }

  while(true) {
      
#ifdef DEBUG_PRINTF  
      const double t2 = MPI_Wtime();
      ofs << "Starting iteration: " << numIters << std::endl;
#endif      
      numIters++;	
      long vc_count = 0;

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight) schedule(runtime)
#else
#pragma omp parallel for schedule(static) shared(clusterWeight)
#endif
      for (GraphElem i = 0L; i < nv; i++)
          clusterWeight[i] = 0;

      fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
                  rsizes, svdata, rvdata, currComm, localCinfo, 
                  remoteCinfo, remoteComm, remoteCupdate);
       
#ifdef DEBUG_PRINTF  
      t0 = MPI_Wtime();
#endif
      // Color loop
      for(long ci = 0; ci < numColor; ci++) {

          const long coloradj1 = colorPtr[ci];
          const long coloradj2 = colorPtr[ci+1];

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        frozenClusterWeight, colorIndex), firstprivate(constantForSecondTerm) \
          reduction(+:vc_count) schedule(runtime)
#else
#pragma omp parallel for shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate, \
        frozenClusterWeight, colorIndex), firstprivate(constantForSecondTerm) \
          reduction(+:vc_count) schedule(static)
#endif
          for (long K = coloradj1; K < coloradj2; K++) {
              if (!vActive[colorIndex[K]]) {
                  clusterWeight[colorIndex[K]] = frozenClusterWeight[colorIndex[K]];
                  vc_count++;
                  continue;
              }
              else
              {
                  distExecuteLouvainIteration(colorIndex[K], dg, currComm, targetComm, 
                          vDegree, localCinfo, localCupdate, remoteComm, remoteCinfo, 
                          remoteCupdate, constantForSecondTerm, clusterWeight, me);
                  frozenClusterWeight[colorIndex[K]] = clusterWeight[colorIndex[K]];
              }
          }
      } // end of Color loop
#ifdef DEBUG_PRINTF  
      t1 = MPI_Wtime();
      ofs << "Color iteration computation time: " << (t1 - t0) << std::endl;
#endif      
 
      if (!ETLocalOrRemote) {
          MPI_Allreduce(MPI_IN_PLACE, &vc_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
          GraphWeight perc = 100.0 * ((GraphWeight)(tnv-vc_count)/(GraphWeight)tnv);
          if (vc_count >= ET_CUTOFF) {
              currMod = -1;
              break;
          }
      }     
      
      // update local cinfo
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(localCinfo, localCupdate) schedule(static)
#endif
      for (GraphElem i = 0; i < nv; i++) {
          localCinfo[i].size += localCupdate[i].size;
          localCinfo[i].degree += localCupdate[i].degree;

          localCupdate[i].size = 0;
          localCupdate[i].degree = 0;
      }

      updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
      
      // global modularity
      currMod = distComputeModularity(g, localCinfo, clusterWeight, constantForSecondTerm, me);
      if ((currMod - prevMod) < threshMod) {
#ifdef DEBUG_PRINTF  
          ofs << "Break here - no updates " << std::endl;
#endif
          break;
      }
      prevMod = currMod;
      if (prevMod < lower)
          prevMod = lower;   

      const GraphWeight one_minus_delta = (1.0 - ETDelta);
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive, p_curr, p_prev) \
    firstprivate(numIters, one_minus_delta) schedule(runtime)
#else
#pragma omp parallel for default(none), \
    shared(pastComm, currComm, targetComm, vActive, p_curr, p_prev) \
    firstprivate(numIters, one_minus_delta) schedule(static)
#endif
    for (GraphElem i = 0; i < nv; i++) {
        if (!vActive[i])
            continue;
        
        if (numIters > 2 && (currComm[i] == pastComm[i])) {
            p_curr[i] = p_prev[i]*one_minus_delta;
            if (p_curr[i] <= P_CUTOFF)
                vActive[i] = false;
        }

        if (vActive[i]) 
        {
            GraphElem tmp = pastComm[i];
            pastComm[i] = currComm[i];
            currComm[i] = targetComm[i];
            targetComm[i] = tmp;
        }
    }
    
    // swap p_active k, k-1 iteration
    std::swap(p_curr, p_prev);
  }; // end while loop

  cvect = pastComm;
  iters = numIters;

  vActive.clear();
  frozenClusterWeight.clear();
  p_curr.clear();
  p_prev.clear();
  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethodVertexOrder with early termination

void distInitLouvain(const DistGraph &dg, CommunityVector &pastComm, 
        CommunityVector &currComm, GraphWeightVector &vDegree, 
        GraphWeightVector &clusterWeight, CommVector &localCinfo, 
        CommVector &localCupdate, GraphWeight &constantForSecondTerm,
        const int me)
{
  const Graph &g = dg.getLocalGraph();
  const GraphElem base = dg.getBase(me);
  const GraphElem nv = g.getNumVertices();

  vDegree.resize(nv);
  pastComm.resize(nv);
  currComm.resize(nv);
  clusterWeight.resize(nv);
  localCinfo.resize(nv);
  localCupdate.resize(nv);
 
  distSumVertexDegree(g, vDegree, localCinfo);
  constantForSecondTerm = distCalcConstantForSecondTerm(vDegree);

  distInitComm(pastComm, currComm, base);
} // distInitLouvain

void distSumVertexDegree(const Graph &g, GraphWeightVector &vDegree, CommVector &localCinfo)
{
  const GraphElem nv = g.getNumVertices();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(g, vDegree, localCinfo), schedule(runtime)
#else
#pragma omp parallel for default(none), shared(g, vDegree, localCinfo), schedule(guided)
#endif
  for (GraphElem i = 0; i < nv; i++) {
    GraphElem e0, e1;
    GraphWeight tw = 0.0;

    g.getEdgeRangeForVertex(i, e0, e1);

    for (GraphElem k = e0; k < e1; k++) {
      const Edge &edge = g.getEdge(k);
      tw += edge.weight;
    }

    vDegree[i] = tw;
   
    localCinfo[i].degree=tw;
    localCinfo[i].size=1L;
  }
} // distSumVertexDegree

GraphWeight distCalcConstantForSecondTerm(const GraphWeightVector &vDegree)
{
  GraphWeight totalEdgeWeightTwice = 0;
  GraphWeight localWeight = 0;
  int me = -1;

  const size_t vsz = vDegree.size();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(vDegree), reduction(+: localWeight) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(vDegree), reduction(+: localWeight) schedule(static)
#endif  
  for (GraphElem i = 0; i < vsz; i++)
    localWeight += vDegree[i]; // Local reduction

#ifdef DEBUG_PRINTF  
  ofs << "Degree size: " << vsz << std::endl;
  ofs << "Local: " << localWeight << std::endl;
#endif

  // Global reduction
  MPI_Allreduce(&localWeight, &totalEdgeWeightTwice, 1, MPI_WEIGHT_TYPE, 
          MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG_PRINTF  
  ofs << "Global: " << totalEdgeWeightTwice << std::endl;
#endif
  
  return (1.0 / static_cast<GraphWeight>(totalEdgeWeightTwice));
} // distCalcConstantForSecondTerm

GraphElem distGetMaxIndex(const ClusterLocalMap &clmap, const GraphWeightVector &counter,
			  const GraphWeight selfLoop, const CommVector &localCinfo, 
			  const CommMap &remoteCinfo,
			  const GraphWeight vDegree, 
                          const GraphElem currSize,
                          const GraphWeight currDegree, 
			  const GraphElem currComm,
			  const GraphElem base,
			  const GraphElem bound,
			  const GraphWeight constant)
{
  ClusterLocalMap::const_iterator storedAlready;
  GraphElem maxIndex = currComm;
  GraphWeight curGain = 0.0, maxGain = 0.0;
  GraphWeight eix = static_cast<GraphWeight>(counter[0]) - static_cast<GraphWeight>(selfLoop);

  GraphWeight ax = currDegree - vDegree;
  GraphWeight eiy = 0.0, ay = 0.0;

  GraphElem maxSize = currSize; 
  GraphElem size = 0;

  storedAlready = clmap.begin();
#ifdef DEBUG_PRINTF  
  assert(storedAlready != clmap.end());
#endif
  do {
      if (currComm != storedAlready->first) {

          // is_local, direct access local info
          if ((storedAlready->first >= base) && (storedAlready->first < bound)) {
              ay = localCinfo[storedAlready->first-base].degree;
              size = localCinfo[storedAlready->first - base].size;   
          }
          else {
              // is_remote, lookup map
              CommMap::const_iterator citer = remoteCinfo.find(storedAlready->first);
              ay = citer->second.degree;
              size = citer->second.size; 
          }

          eiy = counter[storedAlready->second];

          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant;

          if ((curGain > maxGain) ||
                  ((curGain == maxGain) && (curGain != 0.0) && (storedAlready->first < maxIndex))) {
              maxGain = curGain;
              maxIndex = storedAlready->first;
              maxSize = size;
          }
      }
      storedAlready++;
  } while (storedAlready != clmap.end());

  if ((maxSize == 1) && (currSize == 1) && (maxIndex > currComm))
    maxIndex = currComm;

  return maxIndex;
} // distGetMaxIndex

void distExecuteLouvainIteration(const GraphElem i, const DistGraph &dg,
				 const CommunityVector &currComm,
				 CommunityVector &targetComm,
			         const GraphWeightVector &vDegree,
                                 CommVector &localCinfo, 
                                 CommVector &localCupdate,
				 const VertexCommMap &remoteComm,
                                 const CommMap &remoteCinfo,
                                 CommMap &remoteCupdate,
                                 const GraphWeight constantForSecondTerm,
                                 GraphWeightVector &clusterWeight, 
				 const int me)
{
  GraphElem localTarget = -1;
  GraphElem e0, e1; 
  GraphWeight selfLoop = 0.0;
  ClusterLocalMap clmap;
  GraphWeightVector counter;

  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  const Graph &g = dg.getLocalGraph();
  const GraphElem cc = currComm[i];
  GraphWeight ccDegree;
  GraphElem ccSize;  
  bool currCommIsLocal=false; 
  bool targetCommIsLocal=false;

  // Current Community is local
  if (cc >= base && cc < bound) {
	ccDegree=localCinfo[cc-base].degree;
        ccSize=localCinfo[cc-base].size;
        currCommIsLocal=true;
  } else {
  // is remote
        CommMap::const_iterator citer = remoteCinfo.find(cc);
	ccDegree = citer->second.degree;
 	ccSize = citer->second.size;
	currCommIsLocal=false;
  }

  g.getEdgeRangeForVertex(i, e0, e1);

  if (e0 != e1) {
    clmap.insert(ClusterLocalMap::value_type(cc, 0));
    counter.push_back(0.0);

    selfLoop =  distBuildLocalMapCounter(e0, e1, clmap, counter, g, currComm, remoteComm, i, base, bound);

    clusterWeight[i] += counter[0];

    localTarget = distGetMaxIndex(clmap, counter, selfLoop, localCinfo, remoteCinfo, vDegree[i], ccSize, ccDegree, cc, base, bound, constantForSecondTerm);
  
  }
  else
    localTarget = cc;

   // is the Target Local?
   if (localTarget >= base && localTarget < bound) {
      targetCommIsLocal = true;
   }
  
  // current and target comm are local - atomic updates to vectors
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && targetCommIsLocal) {
        
#ifdef DEBUG_PRINTF  
        assert( base < localTarget < bound);
        assert( base < cc < bound);
	assert( cc - base < localCupdate.size()); 	
	assert( localTarget - base < localCupdate.size()); 	
#endif
        #pragma omp atomic update
        localCupdate[localTarget-base].degree += vDegree[i];
        #pragma omp atomic update
        localCupdate[localTarget-base].size++;
        #pragma omp atomic update
        localCupdate[cc-base].degree -= vDegree[i];
        #pragma omp atomic update
        localCupdate[cc-base].size--;
     }	

  // current is local, target is not - do atomic on local, accumulate in Maps for remote
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && !targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[cc-base].degree -= vDegree[i];
        #pragma omp atomic update
        localCupdate[cc-base].size--;
 
        // search target!     
        CommMap::iterator iter=remoteCupdate.find(localTarget);
 
        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
  }
        
   // current is remote, target is local - accumulate for current, atomic on local
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[localTarget-base].degree += vDegree[i];
        #pragma omp atomic update
        localCupdate[localTarget-base].size++;
       
        // search current 
        CommMap::iterator iter=remoteCupdate.find(cc);
  
        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   }
                    
   // current and target are remote - accumulate for both
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && !targetCommIsLocal) {
       
        // search current 
        CommMap::iterator iter=remoteCupdate.find(cc);
  
        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   
        // search target
        iter=remoteCupdate.find(localTarget);
  
        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
   }

#ifdef DEBUG_PRINTF  
  assert(localTarget != -1);
#endif
  targetComm[i] = localTarget;
} // distExecuteLouvainIteration

GraphWeight distBuildLocalMapCounter(const GraphElem e0, const GraphElem e1,
				   ClusterLocalMap &clmap, 
				   GraphWeightVector &counter,
				   const Graph &g, const CommunityVector &currComm,
				   const VertexCommMap &remoteComm,
				   const GraphElem vertex, 
				   const GraphElem base, const GraphElem bound)
{
  GraphElem numUniqueClusters = 1L;
  GraphWeight selfLoop = 0.0;
  ClusterLocalMap::const_iterator storedAlready;

  for (GraphElem j = e0; j < e1; j++) {
        
    const Edge &edge = g.getEdge(j);
    const GraphElem &tail = edge.tail;
    const GraphWeight &weight = edge.weight;
    GraphElem tcomm;

    if (tail == vertex + base)
      selfLoop += weight;


    // is_local, direct access local CommunityVector
    if ((tail >= base) && (tail < bound))
      tcomm = currComm[tail - base];
    else { // is_remote, lookup map
      VertexCommMap::const_iterator iter = remoteComm.find(tail);

#ifdef DEBUG_PRINTF  
      assert(iter != remoteComm.end());
#endif
      tcomm = iter->second;
    }

    storedAlready = clmap.find(tcomm);
    
    if (storedAlready != clmap.end())
      counter[storedAlready->second] += weight;
    else {
        clmap.insert(ClusterLocalMap::value_type(tcomm, numUniqueClusters));
        counter.push_back(weight);
        numUniqueClusters++;
    }
  }

  return selfLoop;
} // distBuildLocalMapCounter

GraphWeight distComputeModularity(const Graph &g, CommVector &localCinfo,
			     const GraphWeightVector &clusterWeight,
			     const GraphWeight constantForSecondTerm,
			     const int me)
{
  const GraphElem nv = g.getNumVertices();

  GraphWeight le_la_xx[2];
  GraphWeight e_a_xx[2] = {0.0, 0.0};
  GraphWeight le_xx = 0.0, la2_x = 0.0;

#ifdef DEBUG_PRINTF  
  assert((clusterWeight.size() == nv));
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(runtime)
#else
#pragma omp parallel for default(none), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(static)
#endif
  for (GraphElem i = 0L; i < nv; i++) {
    le_xx += clusterWeight[i];
    la2_x += static_cast<GraphWeight>(localCinfo[i].degree) * static_cast<GraphWeight>(localCinfo[i].degree); 
  } 
  le_la_xx[0] = le_xx;
  le_la_xx[1] = la2_x;

#ifdef DEBUG_PRINTF  
  const double t0 = MPI_Wtime();
#endif

  MPI_Allreduce(le_la_xx, e_a_xx, 2, MPI_WEIGHT_TYPE, MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
#endif

  GraphWeight currMod = ((e_a_xx[0] * constantForSecondTerm) - (e_a_xx[1] * constantForSecondTerm *
						     constantForSecondTerm));
#ifdef DEBUG_PRINTF  
  ofs << "le_xx: " << le_xx << ", la2_x: " << la2_x << std::endl;
  ofs << "e_xx: " << e_a_xx[0] << ", a2_x: " << e_a_xx[1] << ", currMod: " << currMod << std::endl;
  ofs << "Reduction time: " << (t1 - t0) << std::endl;
#endif

  return currMod;
} // distComputeModularity

void distUpdateLocalCinfo(CommVector &localCinfo, const CommVector &localCupdate)
{
    size_t csz = localCinfo.size();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(static)
#endif
    for (GraphElem i = 0L; i < csz; i++) {
        localCinfo[i].size += localCupdate[i].size;
        localCinfo[i].degree += localCupdate[i].degree;
    }
}

void distCleanCWandCU(const GraphElem nv, GraphWeightVector &clusterWeight,
        CommVector &localCupdate)
{
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(static)
#endif
    for (GraphElem i = 0L; i < nv; i++) {
        clusterWeight[i] = 0;
        localCupdate[i].degree = 0;
        localCupdate[i].size = 0;
    }
} // distCleanCWandCU

void distInitComm(CommunityVector &pastComm, CommunityVector &currComm, const GraphElem base)
{
  const size_t csz = currComm.size();

#ifdef DEBUG_PRINTF  
  assert(csz == pastComm.size());
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(pastComm, currComm), schedule(runtime)
#else
#pragma omp parallel for default(none), shared(pastComm, currComm), schedule(static)
#endif
  for (GraphElem i = 0L; i < csz; i++) {
    pastComm[i] = i + base;
    currComm[i] = i + base;
  }
} // distInitComm

void fillRemoteCommunities(const DistGraph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, const CommunityVector &currComm, 
        const CommVector &localCinfo, CommMap &remoteCinfo, VertexCommMap &remoteComm, 
        CommMap &remoteCupdate)
{
  std::vector<GraphElem> rcdata(rsz), scdata(ssz);
  GraphElem spos, rpos;
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector< std::vector< GraphElem > > rcinfo(nprocs);
#else
  std::vector<std::unordered_set<GraphElem> > rcinfo(nprocs);
#endif
#ifdef DEBUG_PRINTF  
  double t0, t1, ta = 0.0;
#endif

  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  const Graph &g = dg.getLocalGraph();
  const GraphElem nv = g.getNumVertices();
  const GraphElem tnv = dg.getTotalNumVertices();

  // Collects Communities of local Vertices for remote nodes
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(svdata, scdata, currComm) schedule(runtime)
#else
#pragma omp parallel for shared(svdata, scdata, currComm) schedule(static)
#endif
  for (GraphElem i = 0; i < ssz; i++) {
    const GraphElem vertex = svdata[i];
#ifdef DEBUG_PRINTF  
    assert((vertex >= base) && (vertex < bound));
#endif
    const GraphElem comm = currComm[vertex - base];
    scdata[i] = comm;
  }

  std::vector<GraphElem> rcsizes(nprocs), scsizes(nprocs);
  CommInfoVector sinfo, rinfo;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  spos = 0;
  rpos = 0;
#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs), rdispls(nprocs);
  for (int i = 0; i < nprocs; i++) {
      scnts[i] = ssizes[i];
      rcnts[i] = rsizes[i];
      sdispls[i] = spos;
      rdispls[i] = rpos;
      spos += scnts[i];
      rpos += rcnts[i];
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(scdata.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rcdata.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, MPI_COMM_WORLD);
#elif defined(USE_MPI_SENDRECV)
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Sendrecv(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

      spos += ssizes[i];
      rpos += rsizes[i];
  }
#else
  std::vector<MPI_Request> rreqs(nprocs), sreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
    if (i != me)
      MPI_Irecv(rcdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, 
              CommunityTag, MPI_COMM_WORLD, &rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;

    rpos += rsizes[i];
  }
  for (int i = 0; i < nprocs; i++) {
    if (i != me)
      MPI_Isend(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, 
              CommunityTag, MPI_COMM_WORLD, &sreqs[i]);
    else
      sreqs[i] = MPI_REQUEST_NULL;

    spos += ssizes[i];
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

  // reserve vectors
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  for (GraphElem i = 0; i < nprocs; i++) {
      rcinfo[i].reserve(rpos);
  }
#endif

  remoteComm.clear();
  for (GraphElem i = 0; i < rpos; i++) {
    const GraphElem comm = rcdata[i];

    remoteComm.insert(VertexCommMap::value_type(rvdata[i], comm));

    const int tproc = dg.getOwner(comm);

    if (tproc != me)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
      rcinfo[tproc].emplace_back(comm);
#else
      rcinfo[tproc].insert(comm);
#endif
  }

  for (GraphElem i = 0; i < nv; i++) {
    const GraphElem comm = currComm[i];
    const int tproc = dg.getOwner(comm);

    if (tproc != me)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
      rcinfo[tproc].emplace_back(comm);
#else
      rcinfo[tproc].insert(comm);
#endif
  }

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  GraphElem stcsz = 0, rtcsz = 0;
  
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(scsizes, rcinfo) \
  reduction(+:stcsz) schedule(runtime)
#else
#pragma omp parallel for shared(scsizes, rcinfo) \
  reduction(+:stcsz) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    scsizes[i] = rcinfo[i].size();
    stcsz += scsizes[i];
  }

  MPI_Alltoall(scsizes.data(), 1, MPI_GRAPH_TYPE, rcsizes.data(), 
          1, MPI_GRAPH_TYPE, MPI_COMM_WORLD);

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(rcsizes) \
  reduction(+:rtcsz) schedule(runtime)
#else
#pragma omp parallel for shared(rcsizes) \
  reduction(+:rtcsz) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    rtcsz += rcsizes[i];
  }

#ifdef DEBUG_PRINTF  
  ofs << "Total communities to receive: " << rtcsz << std::endl;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<GraphElem> rcomms(rtcsz), scomms(stcsz);
#else
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector<GraphElem> rcomms(rtcsz);
#else
  std::vector<GraphElem> rcomms(rtcsz), scomms(stcsz);
#endif
#endif
  sinfo.resize(rtcsz);
  rinfo.resize(stcsz);

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  spos = 0;
  rpos = 0;
#if defined(USE_MPI_COLLECTIVES)
  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
      }
      scnts[i] = scsizes[i];
      rcnts[i] = rcsizes[i];
      sdispls[i] = spos;
      rdispls[i] = rpos;
      spos += scnts[i];
      rpos += rcnts[i];
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(scomms.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rcomms.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, MPI_COMM_WORLD);

  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo, rdispls), \
          firstprivate(i), schedule(runtime) /*, if(rcsizes[i] >= 1000) */
#else
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo, rdispls), \
          firstprivate(i), schedule(guided) /*, if(rcsizes[i] >= 1000) */
#endif
          for (GraphElem j = 0; j < rcsizes[i]; j++) {
              const GraphElem comm = rcomms[rdispls[i] + j];
              sinfo[rdispls[i] + j] = {comm, localCinfo[comm-base].size, localCinfo[comm-base].degree};
          }
      }
  }
  
  MPI_Alltoallv(sinfo.data(), rcnts.data(), rdispls.data(), 
          commType, rinfo.data(), scnts.data(), sdispls.data(), 
          commType, MPI_COMM_WORLD);
#else
#if !defined(USE_MPI_SENDRECV)
  std::vector<MPI_Request> rcreqs(nprocs);
#endif
  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#if defined(USE_MPI_SENDRECV)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
          MPI_Sendrecv(rcinfo[i].data(), scsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
#else
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
          MPI_Sendrecv(scomms.data() + spos, scsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
#endif
#else
          MPI_Irecv(rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, 
                  CommunityTag, MPI_COMM_WORLD, &rreqs[i]);
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
          MPI_Isend(rcinfo[i].data(), scsizes[i], MPI_GRAPH_TYPE, i, 
                  CommunityTag, MPI_COMM_WORLD, &sreqs[i]);
#else
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
          MPI_Isend(scomms.data() + spos, scsizes[i], MPI_GRAPH_TYPE, i, 
                  CommunityTag, MPI_COMM_WORLD, &sreqs[i]);
#endif
#endif
      }
      else {
#if !defined(USE_MPI_SENDRECV)
          rreqs[i] = MPI_REQUEST_NULL;
          sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      rpos += rcsizes[i];
      spos += scsizes[i];
  }

  spos = 0;
  rpos = 0;
          
  // poke progress on last isend/irecvs
#if !defined(USE_MPI_COLLECTIVES) && !defined(USE_MPI_SENDRECV) && defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
  int tf = 0, id = 0;
  MPI_Testany(nprocs, sreqs.data(), &id, &tf, MPI_STATUS_IGNORE);
#endif

#if !defined(USE_MPI_COLLECTIVES) && !defined(USE_MPI_SENDRECV) && !defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif

  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#if defined(USE_MPI_SENDRECV)
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos), schedule(runtime) /*, if(rcsizes[i] >= 1000)*/
#else
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos), schedule(guided) /*, if(rcsizes[i] >= 1000)*/
#endif
          for (GraphElem j = 0; j < rcsizes[i]; j++) {
              const GraphElem comm = rcomms[rpos + j];
              sinfo[rpos + j] = {comm, localCinfo[comm-base].size, localCinfo[comm-base].degree};
          }
          
          MPI_Sendrecv(sinfo.data() + rpos, rcsizes[i], commType, i, CommunityDataTag, 
                  rinfo.data() + spos, scsizes[i], commType, i, CommunityDataTag, 
                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
#else
          MPI_Irecv(rinfo.data() + spos, scsizes[i], commType, i, CommunityDataTag, 
                  MPI_COMM_WORLD, &rcreqs[i]);

          // poke progress on last isend/irecvs
#if defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
          int flag = 0, done = 0;
          while (!done) {
              MPI_Test(&sreqs[i], &flag, MPI_STATUS_IGNORE);
              MPI_Test(&rreqs[i], &flag, MPI_STATUS_IGNORE);
              if (flag) 
                  done = 1;
          }
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos), schedule(runtime) /*, if(rcsizes[i] >= 1000)*/
#else
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos), schedule(guided) /*, if(rcsizes[i] >= 1000)*/
#endif
          for (GraphElem j = 0; j < rcsizes[i]; j++) {
              const GraphElem comm = rcomms[rpos + j];
              sinfo[rpos + j] = {comm, localCinfo[comm-base].size, localCinfo[comm-base].degree};
          }

          MPI_Isend(sinfo.data() + rpos, rcsizes[i], commType, i, CommunityDataTag, 
                  MPI_COMM_WORLD, &sreqs[i]);
#endif
      }
      else {
#if !defined(USE_MPI_SENDRECV)
          rcreqs[i] = MPI_REQUEST_NULL;
          sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      rpos += rcsizes[i];
      spos += scsizes[i];
  }

#if !defined(USE_MPI_SENDRECV)
  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rcreqs.data(), MPI_STATUSES_IGNORE);
#endif

#endif

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

  remoteCinfo.clear();
  remoteCupdate.clear();

  for (GraphElem i = 0; i < stcsz; i++) {
      const GraphElem ccomm = rinfo[i].community;

      Comm comm;

      comm.size = rinfo[i].size;
      comm.degree = rinfo[i].degree;

      remoteCinfo.insert(CommMap::value_type(ccomm, comm));
      remoteCupdate.insert(CommMap::value_type(ccomm, Comm()));
  }

#ifdef DEBUG_PRINTF  
  ofs << "Actual MPI time: " << ta << std::endl;
#endif
} // end fillRemoteCommunityMap

void createCommunityMPIType()
{
  CommInfo cinfo;

  MPI_Aint begin, community, size, degree;

  MPI_Get_address(&cinfo, &begin);
  MPI_Get_address(&cinfo.community, &community);
  MPI_Get_address(&cinfo.size, &size);
  MPI_Get_address(&cinfo.degree, &degree);

  int blens[] = { 1, 1, 1 };
  MPI_Aint displ[] = { community - begin, size - begin, degree - begin };
  MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

  MPI_Type_create_struct(3, blens, displ, types, &commType);
  MPI_Type_commit(&commType);
} // createCommunityMPIType

void destroyCommunityMPIType()
{ MPI_Type_free(&commType); } // destroyCommunityMPIType

void updateRemoteCommunities(const DistGraph &dg, CommVector &localCinfo,
			     const CommMap &remoteCupdate,
			     const int me, const int nprocs)
{
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);

  std::vector<CommInfoVector> remoteArray(nprocs);

#ifdef DEBUG_PRINTF  
  ofs << "Starting update remote communities" << std::endl;
#endif

  // FIXME TODO can we use TBB::concurrent_vector instead,
  // first we have to get rid of maps
  for (CommMap::const_iterator iter = remoteCupdate.begin(); iter != remoteCupdate.end(); iter++) {
      const GraphElem i = iter->first;
      const Comm &curr = iter->second;

      const int tproc = dg.getOwner(i);

#ifdef DEBUG_PRINTF  
      assert(tproc != me);
#endif
      CommInfo rcinfo;

      rcinfo.community = i;
      rcinfo.size = curr.size;
      rcinfo.degree = curr.degree;

      remoteArray[tproc].push_back(rcinfo);
  }

  std::vector<GraphElem> send_sz(nprocs), recv_sz(nprocs);

#ifdef DEBUG_PRINTF  
  double tc = 0.0;
  const double t0 = MPI_Wtime();
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for schedule(runtime)
#else
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    send_sz[i] = remoteArray[i].size();
  }

  MPI_Alltoall(send_sz.data(), 1, MPI_GRAPH_TYPE, recv_sz.data(), 
          1, MPI_GRAPH_TYPE, MPI_COMM_WORLD);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
  tc += (t1 - t0);
#endif

  GraphElem rcnt = 0, scnt = 0;
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(recv_sz, send_sz) \
  reduction(+:rcnt, scnt) schedule(runtime)
#else
#pragma omp parallel for shared(recv_sz, send_sz) \
  reduction(+:rcnt, scnt) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    rcnt += recv_sz[i];
    scnt += send_sz[i];
  }
#ifdef DEBUG_PRINTF  
  ofs << "Total number of remote communities to update: " << scnt << std::endl;
#endif

  GraphElem currPos = 0;
  CommInfoVector rdata(rcnt);

#ifdef DEBUG_PRINTF  
  const double t2 = MPI_Wtime();
#endif
#if defined(USE_MPI_SENDRECV)
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Sendrecv(remoteArray[i].data(), send_sz[i], commType, i, CommunityDataTag, 
                  rdata.data() + currPos, recv_sz[i], commType, i, CommunityDataTag, 
                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

      currPos += recv_sz[i];
  }
#else
  std::vector<MPI_Request> sreqs(nprocs), rreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
    if (i != me)
      MPI_Irecv(rdata.data() + currPos, recv_sz[i], commType, i, 
              CommunityDataTag, MPI_COMM_WORLD, &rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;

    currPos += recv_sz[i];
  }

  for (int i = 0; i < nprocs; i++) {
    if (i != me)
      MPI_Isend(remoteArray[i].data(), send_sz[i], commType, i, 
              CommunityDataTag, MPI_COMM_WORLD, &sreqs[i]);
    else
      sreqs[i] = MPI_REQUEST_NULL;
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif
#ifdef DEBUG_PRINTF  
  const double t3 = MPI_Wtime();
  tc += (t3 - t2);
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(rdata, localCinfo) schedule(runtime)
#else
#pragma omp parallel for shared(rdata, localCinfo) schedule(dynamic)
#endif
  for (GraphElem i = 0; i < rcnt; i++) {
    const CommInfo &curr = rdata[i];

#ifdef DEBUG_PRINTF  
    assert(dg.getOwner(curr.community) == me);
#endif
    localCinfo[curr.community-base].size += curr.size;
    localCinfo[curr.community-base].degree += curr.degree;
  }

#ifdef DEBUG_PRINTF  
  ofs << "Update remote community MPI time: " << (t3 - t2) << std::endl;
#endif
} // updateRemoteCommunities

void exchangeVertexReqs(const DistGraph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        const int me, const int nprocs)
{
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  const Graph &g = dg.getLocalGraph();
  const GraphElem nv = g.getNumVertices();

#ifdef USE_OPENMP_LOCK
  std::vector<omp_lock_t> locks(nprocs);
  for (int i = 0; i < nprocs; i++)
    omp_init_lock(&locks[i]);
#endif
  PartnerArray parray(nprocs);

#ifdef USE_OPENMP_LOCK
#pragma omp parallel default(none), shared(dg, g, locks, parray)
#else
#pragma omp parallel default(none), shared(dg, g, parray)
#endif
  {
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided)
#endif
    for (GraphElem i = 0; i < nv; i++) {
      GraphElem e0, e1;

      g.getEdgeRangeForVertex(i, e0, e1);

      for (GraphElem j = e0; j < e1; j++) {
	const Edge &edge = g.getEdge(j);
	const int tproc = dg.getOwner(edge.tail);

	if (tproc != me) {
#ifdef USE_OPENMP_LOCK
	  omp_set_lock(&locks[tproc]);
#else
          lock();
#endif
	  parray[tproc].insert(edge.tail);
#ifdef USE_OPENMP_LOCK
	  omp_unset_lock(&locks[tproc]);
#else
          unlock();
#endif
	}
      }
    }
  }

#ifdef USE_OPENMP_LOCK
  for (int i = 0; i < nprocs; i++) {
#ifdef DEBUG_PRINTF
    ofs << "Number of remote vertices from: " << i << ", " << parray[i].size() << std::endl;
#endif
    omp_destroy_lock(&locks[i]);
  }
#endif
  
  rsizes.resize(nprocs);
  ssizes.resize(nprocs);
  ssz = 0, rsz = 0;

  int pproc = 0;
  // TODO FIXME use OpenMP
  for (PartnerArray::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
    ssz += iter->size();
    ssizes[pproc] = iter->size();
    pproc++;
  }

  MPI_Alltoall(ssizes.data(), 1, MPI_GRAPH_TYPE, rsizes.data(), 
          1, MPI_GRAPH_TYPE, MPI_COMM_WORLD);

  GraphElem rsz_r = 0;
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(rsizes) \
  reduction(+:rsz_r) schedule(runtime)
#else
#pragma omp parallel for shared(rsizes) \
  reduction(+:rsz_r) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++)
    rsz_r += rsizes[i];
  rsz = rsz_r;
  
  svdata.resize(ssz);
  rvdata.resize(rsz);

  GraphElem cpos = 0, rpos = 0;
  pproc = 0;
#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs), rdispls(nprocs);
  
  for (PartnerArray::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
      std::copy(iter->begin(), iter->end(), svdata.begin() + cpos);
      
      scnts[pproc] = iter->size();
      rcnts[pproc] = rsizes[pproc];
      sdispls[pproc] = cpos;
      rdispls[pproc] = rpos;
      cpos += iter->size();
      rpos += rcnts[pproc];

      pproc++;
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(svdata.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rvdata.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, MPI_COMM_WORLD);
#else
  std::vector<MPI_Request> rreqs(nprocs), sreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Irecv(rvdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, VertexTag, MPI_COMM_WORLD,
                  &rreqs[i]);
      else
          rreqs[i] = MPI_REQUEST_NULL;

      rpos += rsizes[i];
  }

  for (PartnerArray::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
      std::copy(iter->begin(), iter->end(), svdata.begin() + cpos);

      if (me != pproc)
          MPI_Isend(svdata.data() + cpos, iter->size(), MPI_GRAPH_TYPE, pproc, VertexTag, MPI_COMM_WORLD,
                  &sreqs[pproc]);
      else
          sreqs[pproc] = MPI_REQUEST_NULL;

      cpos += iter->size();
      pproc++;
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif

  std::swap(svdata, rvdata);
  std::swap(ssizes, rsizes);
  std::swap(ssz, rsz);
} // exchangeVertexReqs

// load ground truth file (vertex-id  community-id) into a vector
// executed by process root

// Ground Truth file is expected to be of the format generated 
// by LFR-gen by Fortunato, et al.
// https://sites.google.com/site/santofortunato/inthepress2
void loadGroundTruthFile(std::vector<GraphElem>& commGroundTruth, 
        std::string const& groundTruthFileName,
        bool isGroundTruthZeroBased) {
  std::ifstream ifs;

  ifs.open(groundTruthFileName.c_str(), std::ifstream::in);
  if (!ifs) {
    std::cerr << "Error opening ground truth file: " << groundTruthFileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;

  // read the data
  while(std::getline(ifs, line)) {
      GraphElem v = -1, comm_id = -1;

      std::istringstream iss(line);
      iss >> v >> comm_id;

      if (!isGroundTruthZeroBased)
          comm_id--;

      commGroundTruth.push_back(comm_id); 
  }

  std::cout << "Loaded ground-truth file: " << groundTruthFileName 
      << ", containing community information for " 
      << commGroundTruth.size() << " vertices." << std::endl;

  ifs.close();
} // loadGroundTruthFile

// gather current community info to root
void gatherAllComm(int root, int me, int nprocs, 
        std::vector<GraphElem>& commAll, 
        std::vector<GraphElem> const& localComm) {
    
    const int lnv = localComm.size();
    int* rcounts = nullptr;
    int* rdispls = nullptr;

    // allocate communication params
    if (root == me)
    {
        rcounts = new int[nprocs];
        rdispls = new int[nprocs];
    }

    MPI_Gather(&lnv, 1, MPI_INT, rcounts, 1, MPI_INT, 
            root, MPI_COMM_WORLD);
    
    // communication params (at root)
    if (me == root)
    {
        GraphElem index = 0;
        for (int p = 0; p < nprocs; p++)
        {
            rdispls[p] = index;
            index += rcounts[p];
        }
        
        commAll.resize(index);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // data
    MPI_Gatherv(localComm.data(), lnv, MPI_GRAPH_TYPE, commAll.data(), rcounts, 
            rdispls, MPI_GRAPH_TYPE, root, MPI_COMM_WORLD);
  
    MPI_Barrier(MPI_COMM_WORLD);

    delete []rcounts;
    delete []rdispls;
} // gatherAllComm
