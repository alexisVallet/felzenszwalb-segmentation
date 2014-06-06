/** @file */
/**
 * Adjacency list implementation of a labeled weighted graph. Subclass
 * of WeightedGraph.
 */
#pragma once

#include <vector>

#include "WeightedGraph.hpp"

template < typename T >
class LabeledGraph : public WeightedGraph {
private:
  vector<T> labels;

public:
  LabeledGraph() : WeightedGraph() {
    
  }

  /**
   * See WeightedGraph::WeightedGraph(int,int). If no labels are
   * specified, initializes labels with the default value of the
   * label type (ie default constructor for classes).
   */
  LabeledGraph(int numberOfVertices, int maxDegree = -1, vector<T> labels = vector<T>()) 
    : WeightedGraph(numberOfVertices, maxDegree), labels(labels.empty() ? vector<T>(numberOfVertices) : labels) {
    
  }
  /**
   * Adds a label to a specific vertex.
   *
   * @param vertex the vertex to associate a label to.
   * @param label label to associate to the vertex.
   */
  void addLabel(int vertex, T label) {
    this->labels[vertex] = label;
  }
  /**
   * Returns the label of a vertex. NULL if vertex has no label.
   *
   * @param vertex a vertex in the graph.
   * @return the label of the vertex.
   */
  T getLabel(int vertex) const {
    return this->labels[vertex];
  }
};
