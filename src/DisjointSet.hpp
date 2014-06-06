/** @file */
#pragma once
/**
 * Implementation of a disjoint set forest data structure with rank
 * and path compression.
 */

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "WeightedGraph.hpp"
#include "Utils.hpp"

using namespace std;
using namespace cv;

struct DisjointSet {
  int parent;
  int rank;
};

class DisjointSetForest {
private:
  vector<DisjointSet> forest;
  int numberOfComponents;
  vector<int> componentSizes;
  bool isModified;
  map<int,int> rootIndexes;

public:
  DisjointSetForest(); // should not be called
  /**
   * Initializes the forest with numberOfElements singleton disjoint
   * sets, numbered 0 to numberOfElements-1.
   *
   * @param numberOfElements the number of elements in the partition,
   * corresponding to the number of singletons at initialization.
   */
  DisjointSetForest(int numberOfElements);

  int constFind(int element) const;
  /**
   * Returns the representant of the set containing a specific element.
   *
   * @param element an element from a set.
   * @return the representant of the set containing the element.
   */
  int find(int element);
  /**
   * Unifies two sets in the forest into one, represented by an element
   * from each set, returning the new root (or representant) of the set.
   *
   * @param element1 an element from the first set.
   * @param element2 an element from the second set.
   * @return the new root (or representant) of the set.
   */
  int setUnion(int element1, int element2);
  /**
   * Computes a region image assuming each element in the set forest
   * is a pixel in row major order, labelled by their set representant.
   */
  Mat_<Vec<uchar,3> > toRegionImage(Mat_<Vec<uchar,3> > sourceImage, vector<Vec3b> colors_ = vector<Vec3b>());
  /**
   * Returns the number of components in the partition.
   *
   * @return the number of components in the partition.
   */
  int getNumberOfComponents() const;
  /**
   * Map associating a linear index in [0..getNumberOfComponents]
   * to each component root. Runs in O(n) time where n is the number
   * of leaves in the forest (ie. elements to partition).
   */
  map<int,int> getRootIndexes();
  /**
   * Returns the size of the component containing a specific element.
   */
  int getComponentSize(int element);
  /**
   * Returns the total number of elements this forest partitions.
   */
  int getNumberOfElements() const;

  friend ostream &operator<<(ostream &os, DisjointSetForest &forest);

  /**
   * Fuses components below a minimum size with their neighbors.
   *
   * @param segmentedGraph graph indicating neighborhoods between elements.
   * (e.g. a grid graph or nearest neighbor graph, in many cases).
   * @param minSize size below which components will get fused out.
   */
  void fuseSmallComponents(WeightedGraph &segmentedGraph, int minSize, const Mat_<float> &mask);
};

/**
 * Computes the gravity center for each segment of an image.
 *
 * @param image image to compute the gravity centers from.
 * @param mask mask for the image.
 * @param segmentation a segmentation of the image.
 * @param centers output vector containing the gravity center for each segment.
 */
void gravityCenters(const Mat_<Vec3f> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, vector<Vec2f> &centers);