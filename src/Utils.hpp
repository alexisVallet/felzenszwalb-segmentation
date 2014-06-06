/** @file */
#pragma once

#include <utility>
#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/**
 * Datatype indicating pixel connectivity.
 */
enum ConnectivityType {CONNECTIVITY_4 = 0, CONNECTIVITY_8 = 1};

/**
 * Converts coordinates in 2D array to row major format.
 */
int toRowMajor(int width, int x, int y);

/**
 * Converts coordinates in 2D array to column major format.
 */
int toColumnMajor(int rows, int i, int j);

/**
 * Converts a row major index to coordinates in a 2D array.
 */
pair<int,int> fromRowMajor(int width, int i);

/**
 * Multiplies a sparse n by n matrix by a dense n by 1 column vector.
 */
Mat_<double> sparseMul(SparseMat_<double> A, Mat_<double> b);

/**
 * Checks that a given sparse matrix is symmetric.
 */
bool symmetric(Eigen::SparseMatrix<double> M);

/**
 * Checks that a given sparse matrix is positive definite by attempting
 * to compute its Cholesky decomposition.
 */
bool positiveDefinite(Eigen::SparseMatrix<double> M);

/**
 * Removes the line and column of a specific index in a sparse matrix.
 */
void removeLineCol(const Eigen::SparseMatrix<double> &L, int v0, Eigen::SparseMatrix<double> &L0);

/**
 * Converts 2d coordinates to upper triangular packed storage column major format
 * intended for use with LAPACK.
 *
 * @param rows the number of rows in the matrix
 * @param i the row index in the matrix
 * @param j the column index in the matrix
 * @return the linear index of the element in upper triangular packed storage column major format.
 */
int toUpperTriangularPacked(int i, int j);

/**
* Draws an histogram. Taken from
* http://laconsigna.wordpress.com/2011/04/29/1d-histogram-on-opencv/
*
* @param hist histogram to draw
* @param scaleX horizontal scaling factor
* @param scaleY vertical scaling factor
* @return histogram image
*/
Mat imHist(Mat hist, float scaleX=1, float scaleY=1);

/**
 * Computes and displays histograms for each channel of a color image.
 *
 * @param image the image to compute histograms from.
 */
void showHistograms(const Mat_<Vec3b> &image, const Mat_<float> &mask, int nbBins);

/**
 * Equalize the histogram of a L*a*b* image. Equalizes by equalizing the hue of
 * the image.
 *
 * @param image the image to equalize the histogram of.
 * @param mask mask indicating pixels to take into account in the histogram.
 * @param equalized output image with normalized histogram.
 */
void equalizeColorHistogram(const Mat_<Vec3f> &image, const Mat_<float> &mask, Mat_<Vec3f> &equalized);

/**
 * Crops an image and its mask so it only contains the bounding box of the non 
 * masked elements.
 *
 * @param image image to crop.
 * @param mask mask of the image to crop.
 * @param croppedImage output cropped image.
 * @param croppedMask output cropped mask.
 */
void crop(const Mat_<Vec3b> &image, const Mat_<float> &mask, Mat_<Vec3b> &croppedImage, Mat_<float> &croppedMask);

/**
 * Resizes an image so the number of pixels is (roughly) lower or equal to a maximum.
 * Useful to limit running time of algorithms running in time super linear to the
 * the number of pixels in the image.
 *
 * @param image image to resize.
 * @param mask mask of the pixels to take into account in the source image.
 * @param resizedImage output resized image.
 * @param resizedMask output resized mask.
 */
void resizeImage(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, Mat_<Vec<uchar,3> > &resizedImage, Mat_<float> &resizedMask, int maxNbPixelsconst, const Mat_<Vec3b> &manualSegmentation = Mat_<Vec3b>(), Mat_<Vec3b> &resizedSegmentation = Mat_<Vec3b>());

/**
 * Vertical concatenation of matrices whose type and size is known at compile time.
 *
 * @param mat1 m1 by n matrix.
 * @param mat2 m2 by n matrix
 * @param res output (m1 + m2) by n matrix.
 */
template < typename _Tp, int m1, int m2, int n >
void vconcatX(const Matx<_Tp, m1, n> &mat1, const Matx<_Tp, m2, n> &mat2, Matx<_Tp, m1 + m2, n> &res) {
	Mat dm1(mat1);
	Mat dm2(mat2);
	Mat dres;

	vconcat(dm1, dm2, dres);

	res = Matx<_Tp, m1 + m2, n>(dres);
}

template < typename MatType >
void matToCsv(const MatType &matrix, int rows, int cols, double (*indexing)(const MatType &m, int i, int j), ofstream &out) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			out<<indexing(matrix, i, j);

			if (j < cols - 1) {
				out<<", ";
			}
		}
		out<<endl;
	}
}

/**
 * Outputs a csv file representation of an openCV matrix to an arbitrary
 * output stream.
 *
 * @param matrix matrix to output as csv file.
 * @param out output stream to send csv data to.
 */
void cvMatToCsv(const Mat_<double> &matrix, ofstream &out);

/**
 * Outputs a csv file representation of an Eigen matrix to an arbitrary
 * output stream.
 *
 * @param matrix matrix to output as csv file.
 * @param out output stream to send csv data to.
 */
void eigenMatToCsv(const Eigen::MatrixXd &matrix, ofstream &out);

/**
 * Create an OpenCV matrix header from 
 * an Eigen matrix without copying data.
 * This is kind of a hack, may result in unwanted behaviour when modifying
 * either matrix, and has not been tested outside of the win32 platform. Be 
 * careful using this, preferably only use it with matrices you know you won't
 * modify.
 *
 * @param eigenMat input Eigen matrix.
 * @param cvMat output OpenCV matrix.
 */
void eigenToCv(const Eigen::MatrixXd &eigenMat, Mat_<double> &cvMat);

/**
 * Computes the number of approximately non zero coefficient in a matrix,
 * up to some tolerance.
 *
 * @param m matrix to compute the number of nonzeros from.
 * @param tol nonegative tolerance, a coefficient will be considered
 * 0 iff it is absolutely lower than or equal than tol.
 */
int approxNonzeros(const Eigen::MatrixXd m, double tol = 10E-8);