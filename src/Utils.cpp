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
