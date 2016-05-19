//# define EIGEN_USE_MKL_ALL		//uncomment if available

# include <chrono>
# include <iostream>
# include <exception>
# include <Eigen/Core>
# include <string>
# include <unordered_map>
    // User header
# include "cover_tree.h"
# include "parallel_cover_tree.h"

std::unordered_map<std::string, size_t> filename_map;
    
template<class InputIt, class UnaryFunction>
UnaryFunction parallel_for_each(InputIt first, InputIt last, UnaryFunction f)
{
    unsigned cores = std::thread::hardware_concurrency();
    
    auto task = [&f](InputIt start, InputIt end)->void{
        for (; start < end; ++start)
            f(*start);
    };
    
    const size_t total_length = std::distance(first, last);
    const size_t chunk_length = total_length / cores;
    InputIt chunk_start = first;
    std::vector<std::future<void>>  for_threads;
    for (unsigned i = 0; i < cores - 1; ++i)
    {
        const auto chunk_stop = std::next(chunk_start, chunk_length);
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
        chunk_start = chunk_stop;
    }
    for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));
    
    for (auto& thread : for_threads)
        thread.get();
    return f;
}

std::vector<std::string> read_lines(std::ifstream& in_file) {
    std::cout << "Reading filenames..." << std::endl;
    std::vector<std::string> res;
    std::string temp_line;
    size_t i = 0;
    while(std::getline(in_file, temp_line)) {
        res.push_back(temp_line);
        i++;
    }
    return res;
}

std::vector<point> readPointFile(std::string fileName)
{
    Eigen::initParallel();
    std::cout << "Number of OpenMP threads: " << Eigen::nbThreads( );
    for(int i=0; i<2048; ++i)
        powdict[i] = pow(base, i-1024);
    
    std::ifstream fin(fileName, std::ios::in|std::ios::binary);
    
    // Check for existance for file
    if (!fin)
        throw std::runtime_error("File not found : " + fileName);
    
    // Read the header for number of points, dimensions
    unsigned numPoints = 0;
    unsigned numDims = 0;
    fin.read((char *)&numPoints, sizeof(int));
    fin.read((char *)&numDims, sizeof(int));
    
    // Printing for debugging
    std::cout << "\nNumber of points: " << numPoints << "\nNumber of dims : " << numDims << std::endl;
    
    // List of points
    std::vector<point> pointList;
    pointList.reserve(numPoints);
    
    // Read the points, one by one
    double value;
    double *tmp_point = new double[numDims];
    for (size_t ptIter = 0; ptIter < numPoints; ptIter++)
    {
        // new point to be read
        fin.read((char *)tmp_point, sizeof(double)*numDims);
        point newPt;
        newPt.pt = Eigen::VectorXd(numDims);
        for (unsigned dim = 0; dim < numDims; dim++)
        {
            newPt.pt[dim] = tmp_point[dim];
        }
        
        newPt.ident = ptIter;
        // Add the point to the list
        pointList.push_back(newPt);
    }
    // Close the file
    fin.close();
    
    std::cout<<pointList[0].pt[0] << " " << pointList[0].pt[1] << " " << pointList[1].pt[0] << std::endl;
    
    return pointList;
}



int main(int argv, char** argc)
{
    if (argv < 2)
        throw std::runtime_error("Usage:\n./main <path to train filenames> <path_to_train_points>");
    
    std::cout << argc[1] << std::endl;
    std::cout << argc[2] << std::endl;
    
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::chrono::high_resolution_clock::time_point ts, tn;
    
    // Reading the file for points
    std::ifstream filenames_file(argc[1], std::ios::in);
    
    if(!filenames_file) throw std::runtime_error("Filenames file not found");
    
    std::vector<std::string> filenames = read_lines(filenames_file);

    std::vector<point> points = readPointFile(argc[2]);
        
    std::cout << "Files read" << std::endl;
    
    CoverTree* cTree;
    // Parallel Cover tree construction
    ts = std::chrono::high_resolution_clock::now();
    ParallelMake pct(0, points.size(), points);
    pct.compute();
    std::cout << "Computed!" << std::endl;
    cTree = pct.get_result();
    
    // Single core Cover tree construction
    //cTree = new CoverTree(pointList);
    
    tn = std::chrono::high_resolution_clock::now();
    std::cout << "Build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    //std::cout << *cTree << std::endl;
    cTree->calc_maxdist();
    
    std::cout << "Making map from filenames to features" << std::endl;
    
    filename_map.reserve(500000);
    for(size_t i = 0; i < points.size(); i++) {
        filename_map.insert(std::make_pair(filenames[i], i));
    }
    //std::cout << *cTree << std::endl;
    
    // find the nearest neighbor
    ts = std::chrono::high_resolution_clock::now();
    
    //Serial search
    std::string line;
    while(std::getline(std::cin, line)) {
        point feature = points[filename_map[line]];
        
        ts = std::chrono::high_resolution_clock::now();
        std::vector<point> nearest = cTree->nearNeighbors(feature, 5);
        tn = std::chrono::high_resolution_clock::now();
        for(auto _p : nearest)
            std::cout << filenames[_p.ident] << std::endl;
        
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    }
    /*
    std::cout << "Querying serially" << std::endl;
    for (const auto& queryPt : testPointList)
    {
        point& ct_nn = cTree->NearestNeighbour(queryPt);
    }
    
    // Parallel search (async)
    std::cout << "Quering parallely" << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    parallel_for_each(testPointList.begin(), testPointList.end(), [&](point& queryPt)->void{
        point& ct_nn = cTree->NearestNeighbourMulti(queryPt);
        //point bf_nn = bruteForceNeighbor(pointList, queryPt);
        //if (!ct_nn.isApprox(bf_nn))
        //{
        //	std::cout << "Something is wrong" << std::endl;
        //	std::cout << ct_nn.format(CommaInitFmt) << " " << bf_nn.format(CommaInitFmt) << " " << queryPt.format(CommaInitFmt) << std::endl;
        //	std::cout << (ct_nn - queryPt).norm() << " ";
        //	std::cout << (bf_nn - queryPt).norm() << std::endl;
        //}
    });
    tn = std::chrono::high_resolution_clock::now();
    std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    std::cout << "k-NN serially" << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    for (const auto& queryPt : testPointList)
    {
        std::vector<point> nnList = cTree->nearNeighbors(queryPt, 2);
        nearNeighborBruteForce(pointList, queryPt, 2, nnList);
    }
    tn = std::chrono::high_resolution_clock::now();
    std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    
    std::cout << "range serially" << std::endl;
    std::vector<point> nnList = cTree->rangeNeighbors(testPointList[0], 10);
    rangeBruteForce(pointList, testPointList[0], 10, nnList);
    
    */
    // Success
    return 0;
}
