#include "WeightedGraph.hpp"

WeightedGraph::WeightedGraph() {
  
}

WeightedGraph::WeightedGraph(int numberOfVertices, int maxDegree) 
  : adjacencyLists(numberOfVertices), degrees(numberOfVertices, 0)
{
  if (maxDegree > 0) {
    for (int i = 0; i < numberOfVertices; i++) {
      this->adjacencyLists[i].reserve(maxDegree);
    }
    this->edges.reserve(numberOfVertices*maxDegree);
  }
}

void WeightedGraph::addEdge(int source, int destination, float weight = 1) {
  HalfEdge toAdd;

  assert(source >= 0 && source < this->numberOfVertices());
  assert(destination >= 0 && destination < this->numberOfVertices());

  // add the half edge to the adjacency list
  toAdd.destination = destination;
  toAdd.weight = weight;
  this->adjacencyLists[source].push_back(toAdd);

  // add the full edge to the edge list
  Edge fullEdge;

  fullEdge.source = source;
  fullEdge.destination = destination;
  fullEdge.weight = weight;
  this->edges.push_back(fullEdge);

  // updates degrees
  this->degrees[source] += weight;
  this->degrees[destination] += weight;
}

const vector<Edge> &WeightedGraph::getEdges() const {
  return this->edges;
}

int WeightedGraph::numberOfVertices() const {
  return this->adjacencyLists.size();
}

void WeightedGraph::drawGraph(vector<Vec<float,2> > verticesPositions, Mat &imageToDrawOn) {
	for (int i = 0; i < (int)this->edges.size(); i++) {
		Edge edge = this->edges[i];
		Vec<float,2> srcPos = verticesPositions[edge.source];
		Vec<float,2> dstPos = verticesPositions[edge.destination];

		line(imageToDrawOn, Point((int)floor(srcPos[1]), (int)floor(srcPos[0])), Point((int)floor(dstPos[1]), (int)floor(dstPos[0])), Scalar(0,0,0));
	}
}

const vector<HalfEdge> &WeightedGraph::getAdjacencyList(int vertex) const {
	return this->adjacencyLists[vertex];
}

ostream &operator<<(ostream &os, const WeightedGraph &graph) {
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		os<<i<<" : [";
		for (int j = 0; j < (int)graph.adjacencyLists[i].size(); j++) {
			os<<"("<<graph.adjacencyLists[i][j].destination<<","<<graph.adjacencyLists[i][j].weight<<")";
			if (j < (int)graph.adjacencyLists[i].size() - 1) {
				os<<", ";
			}
		}
		os<<"]"<<endl;
	}

	return os;
}

void connectedComponents(const WeightedGraph &graph, vector<int> &inConnectedComponent, int *nbCC) {
	*nbCC = 0;
	inConnectedComponent = vector<int>(graph.numberOfVertices(),-1);
	vector<bool> discovered(graph.numberOfVertices(), false);
	vector<int> stack;

	stack.reserve(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		if (!discovered[i]) {
			discovered[i] = true;

			inConnectedComponent[i] = *nbCC;

			stack.push_back(i);

			while (!stack.empty()) {
				int t = stack.back();
				stack.pop_back();

				for (int j = 0; j < (int)graph.getAdjacencyList(t).size(); j++) {
					HalfEdge edge = graph.getAdjacencyList(t)[j];

					if (!discovered[edge.destination]) {
						discovered[edge.destination] = true;

						inConnectedComponent[edge.destination] = *nbCC;

						stack.push_back(edge.destination);
					}
				}
			}
			(*nbCC)++;
		}
	}
}

bool connected(const WeightedGraph& graph) {
	int nbCC;
	vector<int> tmp;

	connectedComponents(graph, tmp, &nbCC);

	return nbCC == 1;
}

bool noLoops(const WeightedGraph& graph) {
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (edge.source == edge.destination) {
			return false;
		}
	}

	return true;
}

bool bidirectional(const WeightedGraph& graph) {
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		for (int j = 0; j < (int)graph.getAdjacencyList(i).size(); j++) {
			HalfEdge edge = graph.getAdjacencyList(i)[j];

			bool inOtherDir = false;

			for (int k = 0; k < (int)graph.getAdjacencyList(edge.destination).size(); k++) {
				HalfEdge other = graph.getAdjacencyList(edge.destination)[k];

				if (other.destination == i && abs(other.weight - edge.weight) <= 10E-8) {
					if (inOtherDir) {
						return false;
					} else {
						inOtherDir = true;
					}
				}
			}

			if (!inOtherDir) {
				return false;
			}
		}
	}

	return true;
}

void inducedSubgraphs(const WeightedGraph &graph, const vector<int> &inSubgraph, int numberOfSubgraphs, vector<int> &vertexIdx, vector<WeightedGraph> &subgraphs) {
	vector<int> subgraphSizes(numberOfSubgraphs,0);

	vertexIdx = vector<int>(graph.numberOfVertices(), -1);

	/*cout<<"vertexIdx: "<<vertexIdx.size()<<", subgraphSizes: "<<subgraphSizes.size()<<", inSubgraph: "<<inSubgraph.size()<<endl;

	cout<<"computing subgraph sizes and vertices indexes"<<endl;*/
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		//cout<<"inSubgraph["<<i<<"] = "<<inSubgraph[i]<<endl;
		vertexIdx[i] = subgraphSizes[inSubgraph[i]];
		subgraphSizes[inSubgraph[i]]++;
	}

	subgraphs = vector<WeightedGraph>(numberOfSubgraphs);

	for (int i = 0; i < numberOfSubgraphs; i++) {
		subgraphs[i] = WeightedGraph(subgraphSizes[i]);
	}
	
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (inSubgraph[edge.source] == inSubgraph[edge.destination]) {
			subgraphs[inSubgraph[edge.source]].addEdge(vertexIdx[edge.source], vertexIdx[edge.destination], edge.weight);
		}
	}
}

void WeightedGraph::copyEdges(const WeightedGraph &graph) {
	assert(graph.numberOfVertices() <= this->numberOfVertices());

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		this->addEdge(edge.source, edge.destination, edge.weight);
	}
}

double WeightedGraph::degree(int vertex) const {
	assert(vertex >= 0 && vertex < this->numberOfVertices());

	return this->degrees[vertex];
}

bool compareGraphSize(const WeightedGraph& g1, const WeightedGraph& g2) {
	return g1.numberOfVertices() < g2.numberOfVertices();
}

vector<int> breadthFirstSearch(const WeightedGraph &graph, int startingVertex) {
	vector<bool> marks(graph.numberOfVertices(), false);
	vector<int> bfsOrder;
	bfsOrder.reserve(graph.numberOfVertices());

	// slight variation on the usual BFS for possibly disconnected graphs: iterates 
	// on all vertices starting from a specific user-defined one. The ending 
	// condition looks a bit weird, but it just means we stop when we have gone full 
	// circle.
	int root = startingVertex;
	do {
		if (!marks[root]) {
			queue<int> toVisit;
			marks[root] = true;
			toVisit.push(root);

			while (!toVisit.empty()) {
				int current = toVisit.front();
				toVisit.pop();
				bfsOrder.push_back(current);

				for (int j = 0; j < (int)graph.getAdjacencyList(current).size(); j++) {
					HalfEdge edge = graph.getAdjacencyList(current)[j];
					int neighbor = edge.destination;

					if (!marks[neighbor]) {
						marks[neighbor] = true;
						toVisit.push(neighbor);
					}
				}
			}
		}
		root = (root + 1) % graph.numberOfVertices();
	} while (root != startingVertex);

	return bfsOrder;
}

WeightedGraph permuteVertices(const WeightedGraph &graph, const vector<int> &permutation) {
	assert(graph.numberOfVertices() == permutation.size());
	// first compute the inverse permutation
	vector<int> inverse(permutation.size());

	for (int i = 0; i < permutation.size(); i++) {
		inverse[permutation[i]] = i;
	}

	// then just add all the edges according to the inverse permutation
	WeightedGraph permutedGraph(graph.numberOfVertices());

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		int newSrc = inverse[edge.source];
		int newDst = inverse[edge.destination];

		permutedGraph.addEdge(newSrc, newDst, edge.weight);
	}

	return permutedGraph;
}
