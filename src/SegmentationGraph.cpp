#include "SegmentationGraph.hpp"

WeightedGraph segmentationGraph(DisjointSetForest &segmentation, const WeightedGraph &grid) {
	int numberOfComponents = segmentation.getNumberOfComponents();
	WeightedGraph graph(numberOfComponents);
	vector<vector<bool> > adjMatrix(numberOfComponents, vector<bool>(numberOfComponents, false));
	map<int,int> rootIndexes = segmentation.getRootIndexes();
	//Mat_<int> borderLengths = computeBorderLengths(segmentation, grid);

	// for each pair of neighboring pixels
	for (int i = 0; i < (int)grid.getEdges().size(); i++) {
		Edge edge = grid.getEdges()[i];
		int srcRoot = rootIndexes[segmentation.find(edge.source)];
		int dstRoot = rootIndexes[segmentation.find(edge.destination)];

		// if they are not in the same segment and there isn't
		// already an edge between them, add one.
		if (srcRoot != dstRoot && 
			!adjMatrix[srcRoot][dstRoot] &&
			!adjMatrix[dstRoot][srcRoot]) {
				adjMatrix[srcRoot][dstRoot] = true;
				adjMatrix[dstRoot][srcRoot] = true;
				graph.addEdge(srcRoot, dstRoot, 1/*(float)borderLengths(srcRoot, dstRoot)*/);
		}
	}

	return graph;
}

Mat_<int> computeBorderLengths(DisjointSetForest &segmentation, WeightedGraph &gridGraph) {
	Mat_<int> borderLengths = Mat_<int>::zeros(segmentation.getNumberOfComponents(), segmentation.getNumberOfComponents());
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < (int)gridGraph.getEdges().size(); i++) {
		Edge edge = gridGraph.getEdges()[i];
		int srcRoot = segmentation.find(edge.source);
		int dstRoot = segmentation.find(edge.destination);

		if (srcRoot != dstRoot) {
			int src = rootIndexes[srcRoot];
			int dst = rootIndexes[dstRoot];

			borderLengths(src, dst) += 1;
			borderLengths(dst, src) += 1;
		}
	}

	return borderLengths;
}

vector<Vec<float,2> > segmentCenters(const Mat_<Vec<uchar,3> > &image, DisjointSetForest &segmentation) {
	int numberOfComponents = segmentation.getNumberOfComponents();
	vector<Vec<float, 2> > centers(numberOfComponents, Vec<int,2>(0,0));
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int root = segmentation.find(toRowMajor(image.cols, j, i));
			int rootIndex = rootIndexes[root];

			centers[rootIndex] += Vec<float,2>((float)i,(float)j)/((float)segmentation.getComponentSize(root));
		}
	}

	return centers;
}

LabeledGraph<Mat> groundGraph(const LabeledGraph<Mat> &unGrounded) {
	LabeledGraph<Mat> grounded(unGrounded.numberOfVertices() + 1);

	// copying edges
	for (int i = 0; i < (int)unGrounded.getEdges().size(); i++) {
		Edge edge = unGrounded.getEdges()[i];

		grounded.addEdge(edge.source, edge.destination, edge.weight);
	}

	// assumes all the labels have the same size
	int labelRows = unGrounded.getLabel(0).rows;
	int labelCols = unGrounded.getLabel(0).cols;
	int labelType = unGrounded.getLabel(0).type();

	// adding edges adjacent to the ground vertex, copying labels
	for (int i = 0; i < unGrounded.numberOfVertices(); i++) {
		assert(unGrounded.getLabel(i).rows == labelRows && unGrounded.getLabel(i).cols == labelCols);
		grounded.addLabel(i, unGrounded.getLabel(i));
		grounded.addEdge(i, unGrounded.numberOfVertices(), 1);
	}

	grounded.addLabel(unGrounded.numberOfVertices(), Mat::zeros(labelRows, labelCols, labelType));

	return grounded;
}

