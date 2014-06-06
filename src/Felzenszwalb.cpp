#include "Felzenszwalb.hpp"

static bool compareWeights(Edge edge1, Edge edge2) {
  return edge1.weight < edge2.weight;
}

static double constOne(const Mat& m1, const Mat& m2) {
	return 1;
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

DisjointSetForest combineSegmentations(const WeightedGraph &graph, vector<DisjointSetForest> &segmentations) {
  DisjointSetForest combination(graph.numberOfVertices());
  vector<Edge> edges = graph.getEdges();

  for (int i = 0; i < (int)edges.size(); i++) {
    Edge edge = edges[i];
    bool areInSameComponents = true;

    for (int j = 0; j < (int)segmentations.size(); j++) {
      int sourceRoot = segmentations[j].find(edge.source);
      int destinationRoot = segmentations[j].find(edge.destination);
      
      areInSameComponents = 
	areInSameComponents && (sourceRoot == destinationRoot);
    }

    if (areInSameComponents) {
      combination.setUnion(edge.source, edge.destination);
    }
  }

  return combination;
}

class EdgeCompare {
public:
	EdgeCompare();

	bool operator()(const Edge &e1, const Edge &e2) {
		return e1.weight > e2.weight;
	}
};
/*
typedef fibonacci_heap<Edge, boost::heap::compare<EdgeCompare> > PriorityQueue;
typedef typename PriorityQueue::handle_type handle_t;

static void updateWeights(PriorityQueue &queue, vector<list<pair<handle_t, int> > > &bidirAdjList, DisjointSetForest &segmentation, int vertex) {
	for (int i = 0; i < (int)bidirAdjList[vertex].size(); i++) {
		pair<handle_t, int> dst = bidirAdjList[vertex][i];
		int srcSize = segmentation.getComponentSize(vertex);
		int dstSize = segmentation.getComponentSize(dst.second);

		queue.increase(dst.first, srcSize + dstSize);
	}
}

void fuseComponentsDownTo(int nbComponents, DisjointSetForest &segmentation, const WeightedGraph& gridGraph) {
	// Compute the region adjacency graph of the segmentation, weighted by
	// sum of incident segment cardinality.
	WeightedGraph rag = segmentationGraph(segmentation, gridGraph);

	// store the edges, weighed by sum of incident segment size, into a min-heap
	// mutable priority queue data structure.
	
	// bidirectional adjacency list data structure for storing incident edges
	// handles
	vector<list<pair<handle_t, int> > > bidirAdjList(rag.numberOfVertices());
	EdgeCompare compare;
	PriorityQueue queue(compare);

	for (int i = 0; i < (int)rag.getEdges().size(); i++) {
		Edge edge = rag.getEdges()[i];
		Edge weighted;
		int srcSize = segmentation.getComponentSize(edge.source);
		int dstSize = segmentation.getComponentSize(edge.destination);

		weighted.source = edge.source;
		weighted.destination = edge.destination;
		weighted.weight = srcSize + dstSize;

		handle_t edgeHandle = queue.push(weighted);

		bidirAdjList[edge.source].push_back(pair<handle_t, int>(edgeHandle, edge.destination));
		bidirAdjList[edge.destination].push_back(pair<handle_t, int>(edgeHandle, edge.source));
	}

	// pop edges from the queue, fusing source and destination until the number of
	// components is small enough. Update weights accordingly.
	while (segmentation.getNumberOfComponents() > nbComponents) {
		Edge smallest = queue.top();
		queue.pop();

		segmentation.setUnion(smallest.source, smallest.destination);

		updateWeights(queue, bidirAdjList, segmentation, smallest.source);
		updateWeigths(queue, bidirAdjList, segmentation, smallest.destination);
	}
}
*/

static double euclidDistance(const Mat &m1, const Mat &m2) {
  return norm(m1 - m2);
}

DisjointSetForest felzenszwalbImageSegment(int k, const Mat &image, 
					   const Mat_<float> &mask, int minCompSize,
					   ScaleType ScaleType = CARDINALITY,
					   ) {
  assert(k >= 0);
  WeightedGraph graph = gridGraph(image, CONNECTIVITY_4, mask, euclidDistance, false);
  
  return felzenszwalbSegment(k, graph, minCompSize, mask, ScaleType);
}
