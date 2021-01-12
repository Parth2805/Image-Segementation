#include <opencv2/opencv.hpp>
#include <queue>


using namespace cv;
using namespace std;



class Edge {
public:
	int toi;
	int toj;
	double weight;
	Edge() {
		weight = 0;
	}

	Edge(int toRowIndex, int toColIndex, double capacity) : toi(toRowIndex), toj(toColIndex),
		weight(capacity) {}

	bool isToSink() {
		if (toi == -2 && toj == -2)
			return true;
		return false;
	}
	bool isToSource() {
		if (toi == -1 && toj == -1)
			return true;
		return false;
	}
};

class Vertice {
public:
	int i;
	int j;
	int p_i;
	int p_j;
	bool isTraversed;
	vector< Edge > edgeList;
	Vertice() {
		i = 0;
		j = 0;
		isTraversed = false;
	}
	Vertice(int rowIndex, int columnIndex) : i(rowIndex), j(columnIndex) {}

	void setParent(int rowIndex, int colIndex) {
		p_i = rowIndex;
		p_j = colIndex;
	}
	void addEdge(int toRowIndex, int toColIndex, float weight) {
		edgeList.push_back(Edge(toRowIndex, toColIndex, weight));
	}

	Edge & getEdge(int rowIndex, int columnIndex) {
		for (int i = 0; i<edgeList.size(); i++) {
			Edge edge = edgeList.at(i);
			if (edge.toi == rowIndex && edge.toj == columnIndex)
				return edgeList.at(i);
		}
	}

	bool isSink() {
		if (i == -2 && j == -2)
			return true;
		return false;
	}
	bool isSource() {
		if (i == -1 && j == -1)
			return true;
		return false;
	}
};

class Pixel {
public:
	int i;
	int j;
	Pixel(int m, int n) {
		i = m;
		j = n;
	}
};

bool bfs(Vertice pixel, vector< vector< Vertice > > &adjacencyList, Vertice &sinkNode) {
	queue < Vertice > q;
	queue < Vertice > empty;
	q.push(pixel);
	while (!q.empty())
	{
		Vertice u = q.front();
		q.pop();
		for (int v = 0; v<u.edgeList.size(); v++)
		{
			Edge edge = u.edgeList.at(v);
			if (edge.toi >= 0 && edge.toj >= 0) {
				Vertice &nodePixel = adjacencyList.at(edge.toi).at(edge.toj);
				if (!nodePixel.isTraversed && edge.weight > 0)
				{
					nodePixel.isTraversed = true;
					nodePixel.setParent(u.i, u.j);
					q.push(nodePixel);
				}
			}
			else if (edge.isToSink() && edge.weight>0) {
				sinkNode.isTraversed = true;
				sinkNode.setParent(u.i, u.j);
				swap(q, empty);
				return true;
			}
		}
	}
	return sinkNode.isTraversed;
}

float fordFulkerson(vector< vector< Vertice > > &adjacencyList, Vertice &superSource, Vertice &superSink, int rows, int cols, Mat &out_image)
{
	float maxFlow = 0;
	int numOfPath = 0;
	while (bfs(superSource, adjacencyList, superSink)) {
		numOfPath++;
		for (int i = 0; i<rows; i++) {
			for (int j = 0; j<cols; j++) {
				superSink.isTraversed = false;
				adjacencyList.at(i).at(j).isTraversed = false;
			}
		}
		Vertice traversalNode = adjacencyList.at(superSink.p_i).at(superSink.p_j);
		double minFlow = LONG_MAX;
		
		while (!traversalNode.isSource()) {
			Vertice parentPixel;
			if (traversalNode.p_j == -1 && traversalNode.p_i == -1)
				parentPixel = superSource;
			else
				parentPixel = adjacencyList.at(traversalNode.p_i).at(traversalNode.p_j);
			
			minFlow = min(minFlow, parentPixel.getEdge(traversalNode.i, traversalNode.j).weight);
			traversalNode = parentPixel;
		}

		traversalNode = adjacencyList.at(superSink.p_i).at(superSink.p_j);

		while (true) {
			if (traversalNode.p_j == -1 && traversalNode.p_i == -1)
				break;

			Vertice copyParentPixel = adjacencyList.at(traversalNode.p_i).at(traversalNode.p_j);
			Vertice &parentPixel = adjacencyList.at(traversalNode.p_i).at(traversalNode.p_j);
			Edge &fromEdge = parentPixel.getEdge(traversalNode.i, traversalNode.j);
			Edge &toEdge = traversalNode.getEdge(parentPixel.i, parentPixel.j);
			fromEdge.weight -= minFlow;
			toEdge.weight += minFlow;

			traversalNode = copyParentPixel;
		}
		maxFlow += minFlow;

	}
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			Vec3b pixel = out_image.at<Vec3b>(i, j);
			if (adjacencyList.at(i).at(j).isTraversed) {
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255;
			}
			else {
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 0;
			}
			out_image.at<Vec3b>(i, j) = pixel;
		}
	}
	return maxFlow;
}




int main(int argc, char **argv) {
	
	if (argc != 4) {
		cout << "Usage: ../seg input_image initialization_file output_mask" << endl;
		return -1;
	}
	// Load the input image
	// the image should be a 3 channel image by default but we will double check that in teh seam_carving
	Mat image;
	image = imread(argv[1]);

	if (!image.data) {
		cout << "Could not load input image!!!" << endl;
		return -1;
	}

	if (image.channels() != 3) {
		cout << "Image does not have 3 channels!!! " << image.depth() << endl;
		return -1;
	}

	// the output image
	Mat out_image = image.clone();

	ifstream f(argv[2]);
	if (!f) {
		cout << "Could not load initial mask file!!!" << endl;
		return -1;
	}

//	int width = image.cols;
//	int height = image.rows;

	int n;
	f >> n;

	Mat smooth;
	Mat greyScale;
	Mat input_gray;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;


	//smoothening output
	GaussianBlur(image, smooth, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(smooth, smooth, CV_BGR2GRAY);
	input_gray = smooth.clone();
	normalize(smooth, smooth, 0, 255, NORM_MINMAX, CV_8UC1);

	Scharr(input_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);	
	convertScaleAbs(grad_x, abs_grad_x);
	Scharr(input_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);	
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, greyScale);
	Vertice superSource(-1, -1);
	Vertice superSink(-2, -2);

	std::vector< std::vector<Vertice> > adjacencyList(greyScale.rows, vector<Vertice>(greyScale.cols, Vertice()));

	for (int i = 0; i<greyScale.rows; i++) {
		for (int j = 0; j<greyScale.cols; j++) {
			adjacencyList.at(i).at(j).i = i;
			adjacencyList.at(i).at(j).j = j;
		}
	}

	float maxFlow = 0;
	float sumForSink = 0;
	int numSink = 0;
	float sumForSource = 0;
	float meanSink = 0;
	int numSource = 0;
	float meanSource = 0;
	// get the initil pixels
	for (int i = 0; i < n; ++i) {
		int x, y, t;
		f >> x >> y >> t;

		if (x < 0 || x >= image.cols || y < 0 || y >= image.rows) {
			cout << "I valid pixel mask!" << endl;

			return -1;
		}
		if (t == 0) {
			Vertice &nodePixel = adjacencyList.at(y).at(x);
			nodePixel.addEdge(-2, -2, LONG_MAX);
			

		}
		else {
			superSource.addEdge(y, x, LONG_MAX);
			
		}
	}

	double maxEdgeVal = 0;
	long edge_weight;
	for (int i = 0; i < greyScale.rows; i++) {
		for (int j = 0; j < greyScale.cols; j++) {

			Vertice &nodePixel = adjacencyList.at(i).at(j);

			vector<Pixel> pixelList;
			if (i>0) {
				Pixel pixelIndex(i - 1, j);
				pixelList.push_back(pixelIndex);
			}
			if (i<image.rows - 1) {
				Pixel pixelIndex(i + 1, j);
				pixelList.push_back(pixelIndex);
			}
			if (j>0) {
				Pixel pixelIndex(i, j - 1);
				pixelList.push_back(pixelIndex);
			}
			if (j<image.cols - 1) {
				Pixel pixelIndex(i, j + 1);
				pixelList.push_back(pixelIndex);
			}
			for (int pixelIndex = 0; pixelIndex<pixelList.size(); pixelIndex++) {
				double diff = (smooth.at<uchar>(i, j) - smooth.at<uchar>(pixelList.at(pixelIndex).i, pixelList.at(pixelIndex).j));
				if (diff<0.5) {
					edge_weight = LONG_MAX;
				}
				else {
					edge_weight = 1;
				}
				nodePixel.addEdge(pixelList.at(pixelIndex).i, pixelList.at(pixelIndex).j, edge_weight);
			}
		}
	}
	//cout << "Computing";
	maxFlow = fordFulkerson(adjacencyList, superSource, superSink, greyScale.rows, greyScale.cols, out_image);

	// write it on disk
	imwrite(argv[3], out_image);

	// also display them both

	namedWindow("Original image", WINDOW_AUTOSIZE);
	namedWindow("Show Marked Pixels", WINDOW_AUTOSIZE);
	imshow("Original image", image);
	imshow("Show Marked Pixels", out_image);
	waitKey(0);
	return 0;
};

