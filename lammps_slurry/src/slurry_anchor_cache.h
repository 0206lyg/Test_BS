// Cache ONLY gap-independent, canonical-frame anchor residual data.
// The current gap-dependent film/singular resistance must be recomputed.
#ifndef SLURRY_ANCHOR_CACHE_H
#define SLURRY_ANCHOR_CACHE_H
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
namespace SlurryAnchorCache {
using PairKey=std::pair<std::uint64_t,std::uint64_t>;
struct Geometry {
  std::array<double,6> normalized_axes{};
  std::array<double,6> canonical_directors{};
  // Regularization, up to three anchor gaps, grid, bridge version and basis orders.
  std::array<double,8> settings{};
};
struct Statistics {
  std::uint64_t queries=0,exact_hits=0,approximate_hits=0,misses=0,stores=0,evictions=0;
  double largest_reused_director_change=0;
};
inline void validate(const Geometry&g){
  for(double x:g.normalized_axes)if(!(x>0)||!std::isfinite(x))throw std::invalid_argument("Invalid anchor-cache shape");
  for(double x:g.canonical_directors)if(!std::isfinite(x))throw std::invalid_argument("Invalid anchor-cache director");
  for(double x:g.settings)if(!std::isfinite(x))throw std::invalid_argument("Invalid anchor-cache setting");
}
// Payload is supplied by the physical provider. It must contain only residual
// anchor coefficients in the descriptor's canonical frame, never a complete
// pair matrix evaluated at an earlier gap or a world-frame response.
template<class Payload> class Cache {
  struct Entry {Geometry geometry;Payload payload;std::uint64_t seen=0;};
  std::map<PairKey,Entry> entries_;
  double director_tolerance_=0.;
  std::string revision_;
  std::uint64_t step_=0;
  Statistics statistics_{};
 public:
  // Positive tolerance explicitly changes the approximation; zero is exact.
  void configure(double director_tolerance,const std::string& model_revision){
    if(!(director_tolerance>=0)||!std::isfinite(director_tolerance))throw std::invalid_argument("Invalid anchor-cache orientation tolerance");
    if(model_revision.empty())throw std::invalid_argument("Anchor-cache model revision must be explicit");
    if(model_revision!=revision_||director_tolerance!=director_tolerance_)clear();
    revision_=model_revision;director_tolerance_=director_tolerance;
  }
  void begin_step(std::uint64_t step){
    if(step<step_)throw std::invalid_argument("Anchor-cache step must be monotonic; clear on restart");
    step_=step;
  }
  const Payload* find(const PairKey& key,const Geometry&geometry){
    validate(geometry);if(revision_.empty())throw std::logic_error("Configure anchor cache before lookup");
    ++statistics_.queries;
    const auto it=entries_.find(key);
    if(it==entries_.end()){++statistics_.misses;return nullptr;}
    auto&entry=it->second;
    // Changes to shape, regularization, anchor position or discretization are
    // never approximated, even when a director tolerance is requested.
    if(entry.geometry.normalized_axes!=geometry.normalized_axes||entry.geometry.settings!=geometry.settings){++statistics_.misses;return nullptr;}
    double delta=0;
    for(int k=0;k<6;++k)delta=std::max(delta,std::abs(entry.geometry.canonical_directors[k]-geometry.canonical_directors[k]));
    if(delta>director_tolerance_){++statistics_.misses;return nullptr;}
    entry.seen=step_;
    if(delta==0)++statistics_.exact_hits;else ++statistics_.approximate_hits;
    statistics_.largest_reused_director_change=std::max(statistics_.largest_reused_director_change,delta);
    return &entry.payload;
  }
  void store(const PairKey&key,const Geometry&geometry,const Payload&payload){
    validate(geometry);if(revision_.empty())throw std::logic_error("Configure anchor cache before store");
    if(key.first==key.second)throw std::invalid_argument("Anchor cache requires distinct particle identities");
    entries_[key]=Entry{geometry,payload,step_};++statistics_.stores;
  }
  void prune(std::uint64_t maximum_idle_steps){
    for(auto it=entries_.begin();it!=entries_.end();){
      if(step_-it->second.seen>maximum_idle_steps){it=entries_.erase(it);++statistics_.evictions;}else ++it;
    }
  }
  void clear(){statistics_.evictions+=entries_.size();entries_.clear();step_=0;}
  void reset_statistics(){statistics_=Statistics{};}
  const Statistics& statistics()const{return statistics_;}
  std::size_t size()const{return entries_.size();}
  double director_tolerance()const{return director_tolerance_;}
};
} // namespace SlurryAnchorCache
#endif
