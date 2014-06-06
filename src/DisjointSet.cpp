#include "DisjointSet.hpp"

DisjointSetForest::DisjointSetForest() {

}

DisjointSetForest::DisjointSetForest(int numberOfElements) 
	: forest(numberOfElements), 
	numberOfComponents(numberOfElements),
	componentSizes(numberOfElements, 1),
	isModified(true)
{
	for (int i = 0; i < numberOfElements; i++) {
		this->forest[i].parent = i;
		this->forest[i].rank = 0;
	}
}

int DisjointSetForest::constFind(int element) const {
	if (this->forest[element].parent == element) {
		return element;
	} else {
		return this->constFind(this->forest[element].parent);
	}
}

int DisjointSetForest::find(int element) {
	int currentParent = this->forest[element].parent;

	if (currentParent != element) {
		this->forest[element].parent = this->find(currentParent);
	}

	return this->forest[element].parent;
}

int DisjointSetForest::setUnion(int element1, int element2) {
	int root1 = this->find(element1);
	int root2 = this->find(element2);

	if (root1 == root2) {
		return root1;
	}

	// if the roots are different, then the union results in one
	// less component. We also indicate that the partition has been
	// modified so we must recompute root indexes.
	this->numberOfComponents--;
	this->isModified = true;

	if (this->forest[root1].rank < this->forest[root2].rank) {
		this->forest[root1].parent = root2;
		this->componentSizes[root2] += this->componentSizes[root1];
		return root2;
	} else if (this->forest[root1].rank > this->forest[root2].rank) {
		this->forest[root2].parent = root1;
		this->componentSizes[root1] += this->componentSizes[root2];
		return root1;
	} else {
		this->forest[root2].parent = root1;
		this->forest[root1].rank++;
		this->componentSizes[root1] += this->componentSizes[root2];
		return root1;
	}

}

Mat_<Vec<uchar, 3> > DisjointSetForest::toRegionImage(Mat_<Vec<uchar,3> > sourceImage, vector<Vec3b> colors_) {
	Mat_<Vec<uchar, 3> > regions(sourceImage.rows, sourceImage.cols);
	vector<Vec<uchar,3> > colors(sourceImage.rows * sourceImage.cols + 1);

	if (colors_.empty()) {
		for (int i = 0; i < sourceImage.rows * sourceImage.cols + 1; i++) {
			colors[i][0] = rand() % 255;
			colors[i][1] = rand() % 255;
			colors[i][2] = rand() % 255;
		}
	} else {
		colors = colors_;
	}

	map<int,int> rootIdx = this->getRootIndexes();

	for (int i = 0; i < sourceImage.rows; i++) {
		for (int j = 0; j < sourceImage.cols; j++) {
			int root = rootIdx[this->find(toRowMajor(sourceImage.cols, j, i))];

			regions(i, j) = colors[root];
		}
	}

	return regions;
}

int DisjointSetForest::getNumberOfComponents() const {
	return this->numberOfComponents;
}

map<int,int> DisjointSetForest::getRootIndexes() {
	if (!this->isModified) {
		return this->rootIndexes;
	}
	// if the forest has been modified, recompute the indexes.
	this->rootIndexes.clear();
	int currentIndex = 0;

	for (int i = 0; i < (int)this->forest.size(); i++) {
		int root = this->find(i);
		map<int,int>::iterator it = this->rootIndexes.find(root);

		if (it == this->rootIndexes.end()) {
			this->rootIndexes[root] = currentIndex;
			currentIndex++;
		}
	}

	this->isModified = false;

	return this->rootIndexes;
}

int DisjointSetForest::getComponentSize(int element) {
	int root = this->find(element);

	return this->componentSizes[root];
}

int DisjointSetForest::getNumberOfElements() const {
	return this->forest.size();
}

ostream &operator<<(ostream &os, DisjointSetForest &forest) {
	vector<vector<int>> comps(forest.getNumberOfComponents());

	for (int i = 0; i < forest.getNumberOfElements(); i++) {
		int root = forest.find(i);

		comps[forest.getRootIndexes()[root]].push_back(i);
	}

	os<<"(";
	for (int i = 0; i < (int)comps.size(); i++) {
		os<<"{";
		for (int j = 0; j < (int)comps[i].size(); j++) {
			os<<comps[i][j];
			if (j < (int)comps[i].size() - 1) {
				os<<", ";
			}
		}
		os<<"}";

		if (i < (int)comps.size() - 1) {
			os<<", ";
		}
	}
	os<<")"<<endl;

	return os;
}

void DisjointSetForest::fuseSmallComponents(WeightedGraph &segmentedGraph, int minSize, const Mat_<float> &mask) {
	for (int i = 0; i < (int)segmentedGraph.getEdges().size(); i++) {
		Edge edge = segmentedGraph.getEdges()[i];
		int srcRoot = this->find(edge.source);
		int dstRoot = this->find(edge.destination);
		pair<int,int> srcCoords = fromRowMajor(mask.cols, srcRoot);
		pair<int,int> dstCoords = fromRowMajor(mask.cols, dstRoot);

		if (mask(srcCoords.first, srcCoords.second) != 0
			&& mask(dstCoords.first, dstCoords.second) != 0
			&& srcRoot != dstRoot 
			&& (this->getComponentSize(srcRoot) < minSize || this->getComponentSize(dstRoot) < minSize)) {
			this->setUnion(srcRoot, dstRoot);
		}
	}
}

/*
void DisjointSetForest::fuseCloseComponents(const LabeledGraph<Mat> &segmentationGraph, double (*distFunc)(const Mat&, const Mat&), double threshold) {
	map<int,int> rootIndexes_ = this->getRootIndexes();
	vector<int> reverseIndexes(this->getNumberOfComponents());

	for (map<int,int>::iterator it = rootIndexes_.begin(); it != rootIndexes_.end(); it++) {
		reverseIndexes[it->second] = it->first;
	}

	for (int i = 0; i < (int)segmentationGraph.getEdges().size(); i++) {
		Edge edge = segmentationGraph.getEdges()[i];

		if (distFunc(segmentationGraph.getLabel(edge.source), segmentationGraph.getLabel(edge.destination)) < threshold) {
			int newRoot = this->setUnion(reverseIndexes[edge.source], reverseIndexes[edge.destination]);
			reverseIndexes[edge.source] = newRoot;
			reverseIndexes[edge.destination] = newRoot;
		}
	}
}*/

void gravityCenters(const Mat_<Vec3f> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, vector<Vec2f> &centers) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	centers.clear();
	centers.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		centers.push_back(Vec2f(0,0));
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = segmentation.find(toRowMajor(image.cols, j, i));
				int segmentIndex = rootIndexes[root];
				Vec2f position = Vec2f((float)i,(float)j);

				centers[segmentIndex] += position / (float)segmentation.getComponentSize(root);
			}
		}
	}
}
