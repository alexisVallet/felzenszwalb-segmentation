#include "Utils.hpp"

int toRowMajor(int width, int x, int y) {
  return x + width * y;
}

int toColumnMajor(int rows, int i, int j) {
	return i + j * rows;
}

pair<int,int> fromRowMajor(int width, int i) {
  pair<int,int> coords(i/width, i%width);

  return coords;
}

static bool isMask(const Mat_<float> &mask) {
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i,j) != 0 && mask(i,j) != 1) {
				cout<<"failed: mask("<<i<<","<<j<<") = "<<mask(i,j)<<endl;

				return false;
			}
		}
	}

	return true;
}

Mat_<double> sparseMul(SparseMat_<double> A, Mat_<double> b) {
	assert(A.size(1) == b.rows);
	assert(b.cols == 1);
	Mat_<double> c = Mat_<double>::zeros(b.rows, 1);

	SparseMatConstIterator_<double> it;

	// iterates over non zero elements
	for (it = A.begin(); it != A.end(); ++it) {
		const SparseMat_<double>::Node* n = it.node();
		int row = n->idx[0];
		int col = n->idx[1];

		c(row, 0) += it.value<double>() * b(col,0);
	}

	return c;
}

bool symmetric(Eigen::SparseMatrix<double> M) {
	for (int k = 0; k < M.outerSize(); k++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
			if (abs(it.value() - M.coeffRef(it.col(), it.row())) > 0) {
				return false;
			}
		}
	}

	return true;
}

bool positiveDefinite(Eigen::SparseMatrix<double> M) {
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > chol;

	chol.compute(M);

	return chol.info() == Eigen::Success;
}

// Remove a line and column with the same index in a sparse matrix.
void removeLineCol(const Eigen::SparseMatrix<double> &L, int v0, Eigen::SparseMatrix<double> &L0) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	tripletList.reserve(L.nonZeros());

	for (int k = 0; k < L.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
			if (it.row() != v0 && it.col() != v0) {
				int newRow = it.row() < v0 ? it.row() : it.row() - 1;
				int newCol = it.col() < v0 ? it.col() : it.col() - 1;

				tripletList.push_back(T(newRow, newCol, it.value()));
			}
		}
	}

	L0 = Eigen::SparseMatrix<double>(L.rows() - 1, L.cols() - 1);

	L0.setFromTriplets(tripletList.begin(), tripletList.end());
}

int toUpperTriangularPacked(int i, int j) {
	if (i > j) {
		return toUpperTriangularPacked(j, i);
	} else {
		int result = i + (j + 1) * j / 2;

		return result;
	}
}

Mat imHist(Mat hist, float scaleX, float scaleY){
	double maxVal=0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int rows = 64; //default height size
	int cols = hist.rows; //get the width size from the histogram
	Mat histImg = Mat::zeros((uchar)(rows*scaleX), (uchar)(cols*scaleY), CV_8UC3);
	//for each bin
	for(int i=0;i<cols-1;i++) {
		float histValue = hist.at<float>(i,0);
		float nextValue = hist.at<float>(i+1,0);
		Point pt1 = Point((int)(i*scaleX), (int)(rows*scaleY));
		Point pt2 = Point((int)(i*scaleX+scaleX), (int)(rows*scaleY));
		Point pt3 = Point((int)(i*scaleX+scaleX), (int)((rows-nextValue*rows/maxVal)*scaleY));
		Point pt4 = Point((int)(i*scaleX), (int)((rows-nextValue*rows/maxVal)*scaleY));

		int numPts = 5;
		Point pts[] = {pt1, pt2, pt3, pt4, pt1};

		fillConvexPoly(histImg, pts, numPts, Scalar(255,255,255));
	}
	return histImg;
}

void showHistograms(const Mat_<Vec3b> &image, const Mat_<float> &mask, int nbBins) {
	vector<Mat> channels;
	Mat newMask = Mat_<uchar>(mask);

	split(image, channels);

	for (int i = 0; i < (int)channels.size(); i++) {
		Mat histogram;
		int channelInd[1] = {0};
		int histSize[] = {nbBins};
		float hrange[] = {0, 256};
		const float *ranges[] = {hrange};

		calcHist(&channels[i], 1, channelInd, newMask, histogram, 1, histSize, ranges);
		Mat histogramDrawing = imHist(histogram);

		stringstream ss;

		ss<<"channel "<<i;

		imshow(ss.str(), histogramDrawing);
	}
}

static void equalizeGrayscaleHistogram(const Mat_<uchar> &image, const Mat_<float> &mask, Mat_<uchar> &equalized) {
	// first compute the histogram of the non masked elements
	Mat_<uchar> ucharMask = Mat_<uchar>(mask);

	int bins = 256;
	int histSize[] = {bins};
	float range[] = {0, 256};
	const float* ranges[] = {range};
	Mat_<float> histogram;
	int channels[] = {0};

	calcHist(&image, 1, channels, ucharMask, histogram, 1, histSize, ranges);

	// normalize the histogram
	Mat_<float> normalized;

	normalize(histogram, normalized, 255, 0, NORM_L1);

	// compute the accumulated normalized histogram
	Mat_<float> accumulated = Mat_<float>::zeros(histogram.rows, 1);
	accumulated(0,0) = normalized(0,0);

	for (int i = 1; i < histogram.rows; i++) {
		accumulated(i,0) = accumulated(i-1,0) + normalized(i,0);
	}
	// compute the equalized image from the accumulated histogram
	equalized = Mat_<uchar>(image.rows, image.cols);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int index = image(i,j);

			equalized(i,j) = (uchar)accumulated(index);
		}
	}
}

void equalizeColorHistogram(const Mat_<Vec3f> &image, const Mat_<float> &mask, Mat_<Vec3f> &equalized) {
	Mat_<Vec3f> rgbImage;
	Mat_<Vec3b> hsvImage;

	cvtColor(image, rgbImage, CV_Lab2BGR);
	imshow("bgr", rgbImage);
	cvtColor(Mat_<Vec3b>(rgbImage*255), hsvImage, CV_BGR2HSV);

	vector<Mat_<uchar> > channels(3);

	split(hsvImage, channels);

	Mat_<uchar> equalizedHue;

	equalizeGrayscaleHistogram(channels[0], mask, equalizedHue);

	channels[0] = equalizedHue;

	Mat_<Vec3b> equalizedRgb, equalizedHsv;
	
	for (int i = 1; i < 3; i++) {
		channels[i] = channels[i].mul(Mat_<uchar>(mask));
	}

	merge(channels, equalizedHsv);

	cvtColor(equalizedHsv, equalizedRgb, CV_HSV2BGR);
	imshow("bgr equalized", equalizedRgb);
	waitKey(0);
	cvtColor(Mat_<Vec3f>(equalizedRgb) / 255., equalized, CV_BGR2Lab);
}

void crop(const Mat_<Vec3b> &image, const Mat_<float> &mask, Mat_<Vec3b> &croppedImage, Mat_<float> &croppedMask) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	assert(countNonZero(mask) > 0);
	int minI = image.rows;
	int maxI = -1;
	int minJ = image.cols;
	int maxJ = -1;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				minI = min(i, minI);
				maxI = max(i, maxI);
				minJ = min(j, minJ);
				maxJ = max(j, maxJ);
			}
		}
	}

	image.rowRange(minI, maxI + 1).colRange(minJ, maxJ + 1).copyTo(croppedImage);
	mask.rowRange(minI, maxI + 1).colRange(minJ, maxJ + 1).copyTo(croppedMask);
}

void resizeImage(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, Mat_<Vec<uchar,3> > &resizedImage, Mat_<float> &resizedMask, int maxNbPixels, const Mat_<Vec3b> &manualSegmentation, Mat_<Vec3b> &resizedSegmentation) {
	assert(maxNbPixels >= 0);
	int nbPixels = countNonZero(mask);

	if (nbPixels > maxNbPixels) {
		double ratio = sqrt((double)maxNbPixels / (double)nbPixels);

		resize(image, resizedImage, Size(), ratio, ratio);
		resize(mask, resizedMask, Size(), ratio, ratio, INTER_NEAREST);
		if (manualSegmentation.rows != 0) {
			resize(manualSegmentation, resizedSegmentation, Size(), ratio, ratio);
		}
	} else {
		resizedImage = image;
		resizedMask = mask;
		if (manualSegmentation.rows != 0) {
			resizedSegmentation = manualSegmentation;
		}
	}
}

static double cvIndexing(const Mat_<double> &m, int i, int j) {
	return m(i,j);
}

static double eigenIndexing(const Eigen::MatrixXd &m, int i, int j) {
	return m(i,j);
}

void cvMatToCsv(const Mat_<double> &matrix, ofstream &out) {
	matToCsv<Mat_<double> >(matrix, matrix.rows, matrix.cols, cvIndexing, out);
}

void eigenMatToCsv(const Eigen::MatrixXd &matrix, ofstream &out) {
	matToCsv<Eigen::MatrixXd>(matrix, matrix.rows(), matrix.cols(), eigenIndexing, out);
}

void eigenToCv(const Eigen::MatrixXd &eigenMat, Mat_<double> &cvMat) {
	cvMat = Mat_<double>(eigenMat.rows(), eigenMat.cols(), (double*)eigenMat.data());

	cvMat = cvMat.t();
}

int approxNonzeros(const Eigen::MatrixXd m, double tol) {
	assert(tol >= 0);
	int nbNonzeros = 0;

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			if (abs(m(i,j)) >= tol) {
				nbNonzeros++;
			}
		}
	}

	return nbNonzeros;
}