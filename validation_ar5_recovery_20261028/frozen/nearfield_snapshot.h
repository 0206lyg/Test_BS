#ifndef SLURRY_NEARFIELD_SNAPSHOT_H
#define SLURRY_NEARFIELD_SNAPSHOT_H
#include "slurry_bridge_solver.h"
#include <fstream>
#include <iomanip>
namespace SlurryNearfieldSnapshot {
namespace bs=SlurryBridgeSolver;
template<class T> void put(std::ostream&o,const T&m){for(int r=0;r<m.rows();++r)for(int c=0;c<m.cols();++c)o<<m(r,c)<<' ';o<<'\n';}
template<class T> void get(std::istream&i,T&m){for(int r=0;r<m.rows();++r)for(int c=0;c<m.cols();++c)i>>m(r,c);}
inline void save(const char*path,const bs::Problem&p,const bs::Options&o,const bs::Result*w){
std::ofstream f(path);f<<std::setprecision(17)<<p.count<<' '<<p.dt<<' '<<o.relative_tolerance<<' '<<o.absolute_tolerance<<' '<<bool(w)<<'\n';put(f,p.prescribed);put(f,p.external);for(auto&m:p.self_mobility)put(f,m);
f<<p.near_resistance.nonZeros()<<'\n';for(int j=0;j<p.near_resistance.outerSize();++j)for(bs::Sparse::InnerIterator it(p.near_resistance,j);it;++it)f<<it.row()<<' '<<it.col()<<' '<<it.value()<<'\n';
f<<p.contacts.size()<<'\n';for(auto&c:p.contacts){f<<c.i<<' '<<c.j<<' '<<c.gap<<' '<<c.onset<<' '<<c.kn<<' '<<c.kt<<' '<<c.mu<<' '<<c.twisting_history<<' '<<c.kr<<' '<<c.kw<<' '<<c.mu_roll<<' '<<c.mu_twist<<' '<<c.roll_length<<' '<<c.twist_length<<'\n';put(f,c.Bi);put(f,c.Bj);put(f,c.affine);put(f,c.normal);put(f,c.history);put(f,c.rolling_history);}if(w)put(f,w->velocity);
}
inline void load(const char*path,bs::Problem&p,bs::Options&o,bs::Result&w,bool&haswarm){
std::ifstream f(path);if(!f)throw std::runtime_error("snapshot missing");f>>p.count>>p.dt>>o.relative_tolerance>>o.absolute_tolerance>>haswarm;p.prescribed.resize(11*p.count);p.external.resize(6*p.count);p.self_mobility.resize(p.count);get(f,p.prescribed);get(f,p.external);for(auto&m:p.self_mobility)get(f,m);
size_t n;f>>n;std::vector<Eigen::Triplet<double>>e;for(size_t k=0;k<n;++k){int i,j;double v;f>>i>>j>>v;e.emplace_back(i,j,v);}p.near_resistance.resize(11*p.count,11*p.count);p.near_resistance.setFromTriplets(e.begin(),e.end());f>>n;p.contacts.resize(n);for(auto&c:p.contacts){f>>c.i>>c.j>>c.gap>>c.onset>>c.kn>>c.kt>>c.mu>>c.twisting_history>>c.kr>>c.kw>>c.mu_roll>>c.mu_twist>>c.roll_length>>c.twist_length;get(f,c.Bi);get(f,c.Bj);get(f,c.affine);get(f,c.normal);get(f,c.history);get(f,c.rolling_history);}if(haswarm){w.velocity.resize(6*p.count);get(f,w.velocity);}if(!f)throw std::runtime_error("truncated snapshot");
}
}
#endif
