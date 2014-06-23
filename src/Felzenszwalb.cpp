#include "Felzenszwalb.hpp"

static bool compareWeights(Edge edge1, Edge edge2) {
  return edge1.weight < edge2.weight;
}

DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize, Mat_<float> mask, ScaleType scaleType) {
	// sorts edge in increasing weight order
	vector<Edge> edges = graph.getEdges();
	sort(edges.begin(), edges.end(), compareWeights);

	// initializes the disjoint set forest to keep track of components, as
	// well as structures to keep track of component size, degree and internal
	// differences.
	DisjointSetForest segmentation(graph.numberOfVertices());
	vector<float> internalDifferences(graph.numberOfVertices(), 0);
	vector<float> volumes(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		volumes[i] = graph.degree(i);
	}

	// Goes through the edges, and fuses vertices if they pass a check,
	// updating internal differences.
	for (int i = 0; i < (int)edges.size(); i++) {
		Edge currentEdge = edges[i];
		int root1 = segmentation.find(currentEdge.source);
		int root2 = segmentation.find(currentEdge.destination);
		float thresh1 = 
			scaleType == CARDINALITY ? 
			(float)segmentation.getComponentSize(root1) :
			volumes[root1];
		float thresh2 =
			scaleType == CARDINALITY ?
			(float)segmentation.getComponentSize(root2) :
		    volumes[root2];
		float mInt = min(internalDifferences[root1] 
			+ ((float)k)/thresh1,
			internalDifferences[root2] 
		    + ((float)k)/thresh2);

		if (root1 != root2 && currentEdge.weight <= mInt) {
			int newRoot = segmentation.setUnion(root1,root2);
			volumes[newRoot] = volumes[root1] + volumes[root2];
			internalDifferences[newRoot] = currentEdge.weight;
		}
	}

	bool firstBgPixelFound = false;
	int bgSegment = -1;

	// Fuses background into a single, unconnected component
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i,j) <= 0) {
				if (!firstBgPixelFound) {
					bgSegment = toRowMajor(mask.cols, j, i);
					firstBgPixelFound = true;
				} else {
					segmentation.setUnion(bgSegment, toRowMajor(mask.cols, j, i));
				}
			}
		}
	}

	segmentation.fuseSmallComponents(graph, minCompSize, mask);

	return segmentation;
}

static double correlation_distance(const Mat *f1, const Mat *f2) {
  Mat zeromean1 = (*f1 - mean(*f1));
  Mat zeromean2 = (*f2 - mean(*f2)); 

  return 
    1. - (zeromean1.dot(zeromean2) / (norm(zeromean1)*norm(zeromean2)));
}

extern "C" {
  /**
   * Wrapper for the graph functions.
   */
  WeightedGraph *c_gridGraph(const float *image, int rows, int cols, 
			    int fdim, int bidirectional) {
    // convert the raw image data to the appropriate format
    int sizes[] = {rows, cols, fdim};
    Mat featmap(3, sizes, CV_32F, (void*)image);
    Mat_<float> mask = Mat_<float>::ones(rows, cols);
    WeightedGraph *outgraph = new WeightedGraph(rows * cols, 4);

    *outgraph = gridGraph(featmap, CONNECTIVITY_4, mask, 
			  correlation_distance,
			  bidirectional == 0 ? false : true);

    return outgraph;
  }

  void *free_graph(WeightedGraph *graph) {
    delete graph;
  }

  /**
   * Wrapper for the DisjointSetForest class.
   */
  DisjointSetForest new_DisjointSetForest(int nbelems) {
    return DisjointSetForest(nbelems);
  }

  int find(DisjointSetForest *forest, int elem) {
    return forest->find(elem);
  }

  int setUnion(DisjointSetForest *forest, int e1, int e2) {
    return forest->setUnion(e1, e2);
  }

  int getNumberOfComponents(DisjointSetForest *forest) {
    return forest->getNumberOfComponents();
  }

  int getComponentSize(DisjointSetForest *forest, int elem) {
    return forest->getComponentSize(elem);
  }

  /**
   * Wrapper for felzenszwalb's segmentation algorithm.
   */
  DisjointSetForest c_felzenszwalbSegment(int k, const WeightedGraph *graph,
					  int mincompsize,
					  int rows, int cols,
					  ScaleType st) {
    Mat_<float> mask = Mat_<float>::ones(rows, cols);

    return felzenszwalbSegment(k, *graph, mincompsize, mask, st);
  }
}
