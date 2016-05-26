//# define EIGEN_USE_MKL_ALL		//uncomment if available

# include <chrono>
# include <iostream>
# include <exception>
# include <Eigen/Core>
# include <Eigen/Eigenvalues> 
# include <string>
# include <array>
# include <map>
# include <unordered_map>
# include <picojson.h>

extern "C" {
    # include "mongoose.h"
}
    //# include <json/json.h>

// User header
# include "cover_tree.h"
# include "parallel_cover_tree.h"

#define MAX_QUERY_BYTE_LEN (200)

struct query {
    std::string filename;
    std::string region;
    int limit;
    int level;
    bool okay;
};

typedef Eigen::MatrixXf Mattype;
typedef Eigen::VectorXf Vectype;

// map from region to filenames, points & c-trees
std::map<std::string, std::unordered_map<std::string, size_t> > filename_reverse;
std::map<std::string, std::vector<point> > points;
std::map<std::string, std::vector<std::string> > filenames_map;
std::map<std::string, CoverTree*> cover_tree_map;

std::vector<std::string> regions;

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
    std::vector<std::future<void>> for_threads;
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
    std::vector<std::string> res;
    std::string temp_line;
    size_t i = 0;
    while(std::getline(in_file, temp_line)) {
        res.push_back(temp_line);
        i++;
    }
    return res;
}

std::vector<point> read_point_file(std::string fileName)
{
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
    std::cout << "Number of points: " << numPoints << std::endl << "Number of dims : " << numDims << std::endl;
    
    // List of points
    std::vector<point> pointList;
    pointList.reserve(numPoints);
    
    // Read the points, one by one
    double value;
    double tmp_point[numDims];
    for (size_t ptIter = 0; ptIter < numPoints; ptIter++)
    {
        fin.read((char *)tmp_point, sizeof(double)*numDims);
        // new point to be read
        point newPt;
        newPt.pt = Vectype(numDims);
        for (unsigned dim = 0; dim < numDims; dim++)
        {
            if(ptIter % 100000 == 0 && dim == 0) std::cout << tmp_point[dim] << " ";
            newPt.pt[dim] = (float)tmp_point[dim];
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

// PCA to 2 dims
std::vector<std::vector<float> > pca_2(std::vector<point> results) {
    Mattype mat(512, results.size());
    
    for(size_t i = 0; i < results.size(); ++i) {
        mat.col(i) = results[i].pt;//.transpose();
    }
    
    Mattype centered = mat.rowwise() - mat.colwise().mean();
    Mattype cov = centered.adjoint() * centered;
    
    Eigen::SelfAdjointEigenSolver<Mattype> eig(cov, Eigen::ComputeEigenvectors);
    
    std::vector<std::vector<float> > res;
    res.reserve(results.size());
    
    Vectype v_1 = eig.eigenvectors().col(0);
    Vectype v_2 = eig.eigenvectors().col(1);
        
    v_1 /= (fabs(v_1.maxCoeff()));
    v_2 /= (fabs(v_2.maxCoeff()));
    
    for(size_t i = 0; i < results.size(); ++i) {
        std::vector<float> temp;
        temp.push_back(v_1(i));
        temp.push_back(v_2(i));
        res.push_back(temp);
    }
    
    return res;
}

std::string format_res(std::string region, 
                       point& search_feature, 
                       std::vector<point> &similar_features,
                       std::vector<std::vector<float> > pca,
                       float duration) {
    
    std::vector<std::string> res_filenames(similar_features.size());
    std::vector<float> distances(similar_features.size());
    
    for(size_t i = 0; i < res_filenames.size(); ++i) {
        point& f = similar_features[i];
        res_filenames[i] = filenames_map[region][f.ident];
        distances[i] = (search_feature.pt - f.pt).norm();
    }
    
    // TODO: use a library for this!!
    
    std::ostringstream ss;
    ss << "{\"duration\":" << duration << ",\"matches\":[";
    size_t penultimate = res_filenames.size() - 1;
    for(size_t i = 0 ; i < res_filenames.size(); ++i) {
        std::vector<float>& curr_pca = pca[i];
        ss << "{";
        ss << "\"distance\":" << distances[i] << ",";
        ss << "\"filename\":\"" << res_filenames[i] << "\",";
        ss << "\"tsne_pos\":[" << curr_pca[0] << "," << curr_pca[1] << "]}";
        if(i != penultimate) ss << ",";
    }
    ss << "],";
    ss << "\"features_filename\":\"" << region << "_z19.dat\"";
    ss << "}";
    return ss.str();
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

std::map<std::string, std::string> query_to_map(const std::string query) {
    std::map<std::string, std::string> res;
    std::ostringstream curr_key;
    std::ostringstream curr_val;
    bool in_key = true;
    size_t i = 0;
    for(const auto& s : query) {
        if(s == '=') {
            in_key = false;
        } else if (s == '&') {
            in_key = true;
            res[curr_key.str()] = curr_val.str();
            curr_key.str("");
            curr_val.str("");
            curr_key.clear();
            curr_val.clear();
        } else {
            if(in_key) {
                curr_key << s;
            } else {
                curr_val << s;
            }
        }
        ++i;
        if(i > MAX_QUERY_BYTE_LEN) break;
    }
    res[curr_key.str()] = curr_val.str();
    return res;
}


bool is_valid_region(std::string region) {
    return std::find(regions.begin(), regions.end(), region) != regions.end();
}

struct query parse_query(std::string query_string) {
    std::map<std::string, std::string> query_map = query_to_map(query_string);
    struct query q;
    if(!(3 <= query_map.size() && query_map.size() <= 4) //only filename, limit & region required
       || !is_valid_region(query_map["region"])
       || query_map["filename"] == "") {
        q.okay = false;
        return q;
    }
    try {
        q.filename = query_map["filename"];
        q.limit = std::stoi(query_map["limit"]);
        if(q.limit > 100) q.limit = 100;
        q.region = query_map["region"];
        q.okay = true;
    } catch (std::invalid_argument& e) {
        q.okay = false;
    }       
    return q;
}

static void ev_handler(struct mg_connection *c, int ev, void *p) {
    
    if (ev == MG_EV_HTTP_REQUEST) {
  
        struct http_message *hm = (struct http_message *) p;
        struct mg_str query_buff = hm->query_string;
        
        if(query_buff.len != 0) {
            auto ts = std::chrono::high_resolution_clock::now();
            
            std::string res = "";
            
            char query_str[query_buff.len + 1];          
            std::copy(query_buff.p, query_buff.p + query_buff.len, query_str);         
            query_str[query_buff.len] = '\0';
            
            struct query q = parse_query(query_str);
            
            if(!q.okay) { 
                
                res = "{\"error\":\"malformed query\"}";
                
            } else {
                
                std::cout << q.filename;
                
                std::string region = q.region;
                size_t filename_idx = filename_reverse[region][q.filename];
                point feature = points[region][filename_idx];
                
                std::vector<point> nearest = cover_tree_map[region]->nearNeighborsMulti(feature, q.limit);
                
                std::vector<std::vector<float> > pca = pca_2(nearest);
                
                auto tn = std::chrono::high_resolution_clock::now();
                
                float time_taken = (float)(std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count())/1000.0f;
                res = format_res(q.region, feature, nearest, pca, time_taken);

            }
            auto tn = std::chrono::high_resolution_clock::now();
            std::cout << " : " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count());
            std::cout << "ms" << std::endl;
            
            mg_printf(c,  "HTTP/1.1 200 OK\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %d\r\n"
                          "\r\n"
                          "%s",
                          (int) res.length(), res.c_str());
            c->flags |= MG_F_SEND_AND_CLOSE;
            
        } else {
            
            mg_printf(c, "HTTP/1.1 200 OK\r\n"
                  "Content-Type: application/json\r\n"
                  "Content-Length: 0\r\n"
                  );
            c->flags |= MG_F_SEND_AND_CLOSE;
            
        }
    }
}

int main(int argv, char** argc)
{
    if (argv < 2)
        throw std::runtime_error("Usage:\n./main <path to config file> <port>");
    
    std::cout << "config path: " <<  argc[1] << std::endl;
    std::cout << "port number: " <<  argc[2] << std::endl;
    
    Eigen::initParallel();
    std::cout << "Number of OpenMP threads: " << Eigen::nbThreads() << std::endl;
    
    for(int i=0; i<2048; ++i)
        powdict[i] = pow(base, i-1024);
    
    std::chrono::high_resolution_clock::time_point ts, tn;
    
    // Read config file
    std::ifstream config_file(argc[1]);
    std::stringstream config_data;
    config_data << config_file.rdbuf();
    
    // Parse config file
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_data.str());

    if (! err.empty()) {
      std::cerr << err << std::endl;
    }
    
    picojson::array regions_config = config_json.get("19").get<picojson::array>();
    // Load data

    for(auto r : regions_config) {
        std::string region = r.get("region").get<std::string>();
        std::string filenames_path = r.get("filenames").get<std::string>();
        
        std::ifstream filenames_file(filenames_path, std::ios::in);
        if(!filenames_file) throw std::runtime_error("Filenames file not found: " + filenames_path);
        
        std::cout << "Building " << region << std::endl;
        
        filenames_map[region] = read_lines(filenames_file);
        
        std::string points_path = r.get("data").get<std::string>();
        
        points[region] = read_point_file(points_path);

        auto ts = std::chrono::high_resolution_clock::now();
       
        ParallelMake pct(0, points[region].size(), points[region]);
        pct.compute();
                
        cover_tree_map[region] = std::move(pct.get_result());
        cover_tree_map[region]->calc_maxdist();
       
        auto tn = std::chrono::high_resolution_clock::now();
        
        std::cout << "Build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
        std::cout << "Making map from filenames to features" << std::endl;

        filename_reverse[region].reserve(300000);
        for(size_t i = 0; i < points[region].size(); i++) {
            filename_reverse[region].insert(std::make_pair(filenames_map[region][i], i));
        }
        
        regions.push_back(region);
        std::cout << std::endl;
    }
    
    struct mg_mgr mgr;
    struct mg_connection *nc;
    
    mg_mgr_init(&mgr, NULL);
    nc = mg_bind(&mgr, argc[2], ev_handler);
    mg_set_protocol_http_websocket(nc);
    
    /* For each new connection, execute ev_handler in a separate thread */
    mg_enable_multithreading(nc);
    
    printf("Starting multi-threaded server on port %s\n", argc[2]);
    for (;;) {
        mg_mgr_poll(&mgr, 3000);
    }
    
    mg_mgr_free(&mgr);
    
    std::cout << "Ready" << std::endl;
    return 0;
}
