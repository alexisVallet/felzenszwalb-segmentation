/** @file */
/**
 * Weighted graph data structure represented as adjacency
 * lists. Lists are stored as vectors for efficiency, as our
 * graphs in the use case of image segmentation have a limited
 * degree. Also keeps an edge list data structure for efficiently
 * listing edges in the graph.
 */
#pragma once

#include <cmath>
#include <assert.h>
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct Edge {
  int source;
  int destination;
  float weight;
};

struct HalfEdge {
  int destination;
  float weight;
};

class WeightedGraph {
private:
  vector<vector<HalfEdge> > adjacencyLists;
  vector<Edge> edges;
  vector<double> degrees;

public:
  WeightedGraph(); // should not be called
  /**
   * Initializes the graph with a given number of vertices and
   * an optional upper bound on the degree of vertices of the
   * graph.
   *
   * @param numberOfVertices the number of vertices of the graph
   * @param maxDegree an upper bound on the degree of vertices in
   * the graph.
   */
  WeightedGraph(int numberOfVertices, int maxDegree = -1);
  /**
   * Adds an edge to the graph. In the case of an undirected graph,
   * the order of source and destination does not matter.
   *
   * @param source the source vertex of the edge.
   * @param destination the destination vertex of the edge.
   * @param weight the weight of the edge.
   */
  void addEdge(int source, int destination, float weight);
  /**
   * Computes and returns a vector containing all the edges in the
   * graph.
   *
   * @return a vector containing all the edges in the graph.
   */
  const vector<Edge> &getEdges() const;
  /**
   * The number of vertices of the graph.
   *
   * @return the number of vertices of the graph.
   */
  int numberOfVertices() const;
  /**
   * Draws the graph on an image.
   *
   * @param verticesPositions vertices positions on the image
   * @param imageToDrawOn image the graph will be drawn over
   */
  void drawGraph(vector<Vec<float,2> > verticesPositions, Mat &imageToDrawOn);

  /**
   * Returns the adjacency list of a specific vertex. In the case of
   * unidirectional representation, this does not return all adjacent
   * vertices. Use a bidirectional representation for this.
   */
  const vector<HalfEdge> &getAdjacencyList(int vertex) const;

  friend ostream &operator<<(ostream &os, const WeightedGraph &graph);

  /**
   * Copy the edges of another graph into this graph.
   *
   * @param graph graph to copy edges from.
   */
  void copyEdges(const WeightedGraph &graph);

  /**
   * Returns the (undirected) weighted degree of a given vertex. The weighted
   * degree of vertex v is defined as the sum of the weights of all edges incident
   * to v. This method runs in constant time.
   *
   * @param vertex vertex to get the degree of.
   * @return the weighted degree of the vertex.
   */
  double degree(int vertex) const;
};

/**
 * Computes the connected components of a graph using a simple DFS procedure.
 *
 * @param graph the graph to compute connected components from.
 * @param inConnectedComponent output vector which associates to each vertex the
 * index of the connected component if belongs to.
 * @param nbCC output integer which will hold the number of connected components of
 * the graph.
 */
void connectedComponents(const WeightedGraph &graph, vector<int> &inConnectedComponent, int *nbCC);

/**
 * Checks that a graph is connected.
 */
bool connected(const WeightedGraph& graph);

/**
 * Checks that a graph contains no loops.
 */
bool noLoops(const WeightedGraph& graph);

/**
 * Checks that a graph has a bidirectional representation.
 */
bool bidirectional(const WeightedGraph& graph);

/** 
 * Computes the subgraphs induced by a specific partition.
 *
 * @param graph the graph to compute the subgraphs from
 * @param inSubgraph a graph.numberOfVertices() sized vector which associates to each vertex in the larger
 * graph the index of the subgraph it belongs to.
 * @param vertexIdx output vector containing a mapping from vertices in the graph to vertices in the corresponding
 * subgraph.
 * @param subgraphs output graphs which will be populated with the subgraphs.
 */ 
void inducedSubgraphs(const WeightedGraph &graph, const vector<int> &inSubgraph, int numberOfSubgraphs, vector<int> &vertexIdx, vector<WeightedGraph> &subgraphs);

/**
 * True iff g1 has strictly less vertices than g2. Useful for sorting graphs by size,
 * extracting the smallest or largest graph in a container using stl's sort,
 * min_element or max_element.
 *
 * @param g1 a graph.
 * @param g2 another graph.
 * @return True iff g1 has strictly less vertices than g2.
 */
bool compareGraphSize(const WeightedGraph& g1, const WeightedGraph& g2);

/**
 * Graph traversal using breadth first search. Returns the order of the traversal
 * in a vector v, e.g. v[0] is the first explored vertex, v[1] the second, etc.
 * Assumes a bidirectional graph data structure.
 *
 * @param graph graph to traverse with n vertices.
 * @param startingVertex vertex in the graph to start the traversal from.
 * @return the order of the traversal in a vector v, e.g. v[0] is the first 
 * explored vertex, v[1] the second, etc.
 */
vector<int> breadthFirstSearch(const WeightedGraph &graph, int startingVertex);

/**
 * Permutes the indices of vertices in the graph using some permutation.
 * The permutation indicates which vertex from the input graph corresponds to
 * each in the output, eg v[0] in the input will become 0 in the output.
 *
 * @param graph graph to permute vertices of.
 * @param permutation permutation of the vertices.
 */
WeightedGraph permuteVertices(const WeightedGraph &graph, const vector<int> &permutation);
