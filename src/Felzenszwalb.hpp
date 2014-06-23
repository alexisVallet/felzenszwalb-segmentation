/** @file */
/**
 * Implemtnation of Felzenszwalb's segmentation method.
 */
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "Utils.hpp"
#include "ImageGraphs.hpp"
#include "SegmentationGraph.hpp"

using namespace std;
using namespace cv;

/**
 * Type specifying which type of scale measure to use for segment sizes, either
 * cardinality (the number of element for finite graphs) or volume (the sum of
 * degrees) of the vertex set.
 */
enum ScaleType { CARDINALITY, VOLUME };

/**
 * Segments a graph using Felzenszwalb's method. Returns the result as
 * a disjoint set forest data structure. Also goes through a post processing
 * phase to weed out small components.
 *
 * @param k scale parameter.
 * @param graph the graph to segment.
 * @param minCompSize minimum size of components.
 * @param scaleType type of scale measure to use for segment size in the threshold
 * function. This is defined as cardinality in the original paper by Felzenszwalb,
 * but we introduce volume as a way of making the algorithm sensitive to local
 * scale.
 * @return a segmentation of the graph.
 */
DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize, Mat_<float> mask, ScaleType scaleType = CARDINALITY);

extern "C" {
  /**
   * Wrapper for the graph functions.
   */
  WeightedGraph *c_gridGraph(const float *image, int rows, int cols, 
			     int fdim, int bidirectional);

  void *free_graph(WeightedGraph *graph);

  /**
   * Wrapper for the DisjointSetForest class.
   */
  // DisjointSetForest new_DisjointSetForest(int nbelems);

  // int find(DisjointSetForest *forest, int elem);

  // int setUnion(DisjointSetForest *forest, int e1, int e2);

  // int getNumberOfComponents(DisjointSetForest *forest);

  // int getComponentSize(DisjointSetForest *forest, int elem);

  // /**
  //  * Wrapper for felzenszwalb's segmentation algorithm.
  //  */
  // DisjointSetForest c_felzenszwalbSegment(int k, const WeightedGraph *graph,
  // 					  int mincompsize,
  // 					  int rows, int cols,
  // 					  ScaleType st);
}
