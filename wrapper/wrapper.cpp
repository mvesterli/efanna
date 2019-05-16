#include <cstdint>
#include <cstdlib>
#include <cstdbool>

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <malloc.h>

#include "efanna.hpp"

extern "C" {

// Number of neighbors
static unsigned int K = 10;

static int num_trees = 10;

static int merge_level = 8;

static int iterations = 8;

static int check = 25;

static int L = 30;

static int S = 10;

static std::string metric = "euclidean";

bool configure(const char* var, const char* val) {
  if (strcmp(var, "count") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      K = k;
      return true;
    }
  } else if (strcmp(var, "num_trees") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      num_trees = k;
      return true;
    }
  } else if (strcmp(var, "merge_level") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      merge_level = k;
      return true;
    }
  } else if (strcmp(var, "iterations") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      iterations = k;
      return true;
    }
  } else if (strcmp(var, "L") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      L = k;
      return true;
    }
  } else if (strcmp(var, "check") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
     check = k;
      return true;
    }
  } else if (strcmp(var, "S") == 0) {
    char* end;
    errno = 0;
    unsigned long long k = strtoull(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      S = k;
      return true;
    }
  } else if (strcmp(var, "metric") == 0) {
    metric = std::string(val);
    return true;
  } else return false;
}

bool end_configure(void) {
  return true;
}

static std::vector<std::vector<float>> pointset;

std::vector<float> parseEntry(const char* entry) {
  std::vector<float> e;
  std::string line(entry);
  float x;
  auto sstr = std::istringstream(line);
  while (sstr >> x) {
    e.push_back(x);
  }
  return e;
}

bool train(const char* entry) {
  auto parsed_entry = parseEntry(entry);
  pointset.push_back(parsed_entry);
  return true;
}

static efanna::FIndex<float>* knn_index;

void end_train(void) {
    // Normalize data
    if (metric == "angular") {
        for (auto& vec : pointset) {
            float square_sum = 0;
            for (auto v : vec) {
                square_sum += v*v;
            }
            float len = std::sqrt(square_sum);
            for (auto& v : vec) {
                v /= len;
            }
        }
    }
    
    size_t d = pointset[0].size();
    int padded_dimensions = (d+7)/8*8;
    float* raw_data = (float*) memalign(
        KGRAPH_MATRIX_ALIGN,
        padded_dimensions*pointset.size()*sizeof(float));
    for (size_t i=0; i < pointset.size(); i++) {
        for (size_t j=0; j < d; j++) {
            raw_data[i*padded_dimensions+j] = pointset[i][j];
        }
        for (size_t j=d; j < padded_dimensions; j++) {
            raw_data[i*padded_dimensions+j] = 0.0;
        }
    }

    efanna::Matrix<float> dataset(pointset.size(), padded_dimensions, raw_data);
    efanna::Distance<float>* distance_function;
    if (metric == "euclidean") {
        distance_function = new efanna::L2DistanceAVX<float>();
    } else if (metric == "angular") {
        distance_function = new efanna::CosineSimilarityAVX<float>();
//        distance_function = new efanna::CosineSimilarity<float>();
    } else {
        throw "Unsupported distance function";
    }
    efanna::KDTreeUbIndexParams params(
        true, num_trees, merge_level, iterations, check, K+L, K, num_trees, S);
    knn_index = new efanna::FIndex<float>(dataset, distance_function, params);
    knn_index->buildIndex();

    pointset.clear();
    pointset.shrink_to_fit();
}

bool prepare_query(const char* entry) {
    return false;
}

static int position;
static std::vector<unsigned int> result;

float dist(std::vector<float>& a, std::vector<float>& b) {
    float res = 0;
    for (size_t i=0; i < a.size(); i++) {
        res += a[i]*b[i];
    }
    return res;
}

size_t query(const char* entry, size_t k) {
  unsigned int query_id;
  position = 0;
  auto sstr = std::istringstream(std::string(entry));
  sstr >> query_id;
  result = knn_index->getGraphRow(query_id);
  return result.size();
}

size_t query_result(void) {
  if (position < result.size()) {
    auto res = result[position];
    position++;
    return res;
  } else return SIZE_MAX;
}

void end_query(void) {
}

}

