#pragma once
#include <ctime>
#include <map>
#include <string>
namespace CostProfile {
struct Entry {double seconds=0.; unsigned long count=0;};
inline std::map<std::string,Entry> entries;
struct Scope {
  const char*name; std::clock_t start;
  explicit Scope(const char*n):name(n),start(std::clock()){}
  ~Scope(){const auto elapsed=std::clock()-start;auto&e=entries[name];e.seconds+=double(elapsed)/CLOCKS_PER_SEC;++e.count;}
};
}
