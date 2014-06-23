/** @file */
/**
 * C header for felzenszwalb segmentation.
 */
#pragma once
#include "ImageGraphs.hpp"
#include "DisjointSet.hpp"
#include "Felzenszwalb.hpp"

using namespace cv;

extern "C" {
  /**
   * Wrapper for the graph functions.
   */
  WeightedGraph c_gridGraph(const Mat_<Vec3f> *image, 
			    ConnectivityType conn, Mat_<float> *mask,
			    double (*simfunc)(const Mat*, const Mat*),
			    int bidirectional);

  WeightedGraph c_kNearestGraph(const Mat_<Vec3f> *image, 
				const Mat_<float> *mask,
				int k, 
				double (*simfunc)(const Mat*,const Mat*),
				int bidirectional);

  /**
   * Wrapper for the DisjointSetForest class.
   */
  DisjointSetForest new_DisjointSetForest(int nbelems);

  int find(DisjointSetForest *forest, int elem);

  int setUnion(DisjointSetForest *forest, int e1, int e2);

  int getNumberOfComponents(DisjointSetForest *forest);

  int getComponentSize(DisjointSetForest *forest, int elem);

  void fuseSmallComponents(DisjointSetForest *forest, 
			   WeightedGraph *graph,
			   int minSize, Mat_<float> *mask);

  /**
   * Wrapper for felzenszwalb's segmentation algorithm.
   */
  DisjointSetForest c_felzenszwalbSegment(int k, 
					  const WeightedGraph *graph,
					  int mincompsize, 
					  Mat_<float> *mask,
					  ScaleType st);
}
