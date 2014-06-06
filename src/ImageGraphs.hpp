/** @file */
#pragma once

#include <opencv2/opencv.hpp>

#include "WeightedGraph.hpp"
#include "Utils.hpp"

using namespace cv;
using namespace std;

/**
 * Returns the grid graph of a color image. The grid graph is defined
 * as the graph where vertices are pixels of the image and vertices have an
 * edge (undirected) between them iff they are neighbor in an N-connectivity
 * sense. Does not repeat edges in both directions, every edge only appears
 * on the first pixel's adjacency list in row major order. Every edge is weighted
 * by an appropriate similarity function.
 *
 * @param image image to compute the grid graph from.
 * @param connectivity the type of connectivity to search for neighboring pixels,
 * cam be CONNECTIVITY_4 or CONNECTIVITY_8 for 4 and 8 neighbors respectively. It should
 * be noted that 8 connectivity does not in general yield a planar graph - one can prove
 * that the 5*5 vertices 8 connected graph is not planar using Euler's bound on the
 * number of edges in a planar graph.
 * @param bidirectional set to true so that edges are repeated in both directions in the
 * adjacency list representation. This is useful for more efficient listing of vertices
 * neighbors, but consumes more space.
 */
WeightedGraph gridGraph(const Mat_<Vec3f> &image, ConnectivityType connectivity, Mat_<float> mask, double (*simFunc)(const Mat&, const Mat&), bool bidirectional = false);

/**
 * Returns a graph where vertices are pixels in the image, and every vertex has an edge
 * to each of its K nearest neighbor in feature space. For a pixel at position (x,y) and
 * color (r,g,b), the associated vector in feature space is (x,y,r,g,b). Uses approximate
 * nearest neighbor search.
 *
 * @param image image to compute the nearest neighbor graph from.
 * @param mask mask indicating pixels to take into account.
 * @param k the number of nearest neighbors each pixel should have an edge
 * towards. Note that this does not mean that the graph is k-regular (seen as an
 * undirected graph) as the k nearest neighbor relation is not symmetric.
 * @return the nearest neighbor graph of the image.
 */
WeightedGraph kNearestGraph(const Mat_<Vec3f> &image, const Mat_<float> mask, int k, double (*simFunc)(const Mat&, const Mat&), bool bidirectional = false);
