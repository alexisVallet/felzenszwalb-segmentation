/** @file */
#pragma once

#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"

using namespace cv;
using namespace std;

/**
 * Computes the border length between each pair of segments in the
 * segmentation.
 *
 * @param segmentation a segmentation of the image
 * @param gridGraph the grid graph of the image
 * @return an n by n matrix, where n is the number of segments, and the (i,j) entry
 * contains the border length between segments i and j.
 */
Mat_<int> computeBorderLengths(DisjointSetForest &segmentation, WeightedGraph &gridGraph);

/**
 * Computes the segmentation graph of an image segmentation, where
 * vertices are segments and there is an edge between two segments
 * iff these segments are connected in a 4-connexity sense. Every edge
 * is represented in only one adjacency list.
 *
 * @param image the segmented image
 * @param segmentation a segmentation of the image
 * @param grid grid graph of the image
 * @return the segmentation graph of this image.
 */
WeightedGraph segmentationGraph(DisjointSetForest &segmentation, const WeightedGraph &grid);

/**
 * Computes the center of gravity of each segment in an image.
 *
 * @param image image to compute the centers of gravity from.
 * @param segmentation a segmentation of the image.
 * @return center of gravity for each segment.
 */
vector<Vec<float,2> > segmentCenters(const Mat_<Vec<uchar,3> > &image, DisjointSetForest &segmentation);

/**
 * Adds a "ground" vertex to a labeled graph, labelled with the 0 matrix and with
 * an edge to every other vertex in the graph. The point of the ground vertex is to
 * encode some absolute information about the segments in the image, not just
 * relative to each other, so the grounded segmentation graphs can be compared more
 * meaningfully.
 *
 * @param unGrounded labeled graph where all labels are matrices of the same dimensions.
 * @return a graph with an additional ground vertices, with properties as specified
 * in the description above.
 */
LabeledGraph<Mat> groundGraph(const LabeledGraph<Mat> &unGrounded);