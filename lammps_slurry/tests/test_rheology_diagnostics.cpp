#include "slurry_rheology_diagnostics.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using Edge = std::pair<int, int>;
using Graph = std::vector<Edge>;

// Deliberately independent, slow definition: recompute every surviving degree
// from the edge list and remove all degree-0/1 vertices simultaneously.
double reference(int count, const Graph &edges) {
  std::vector<bool> present(count, true);
  std::vector<int> degree(count);
  bool changed;
  do {
    std::fill(degree.begin(), degree.end(), 0);
    for (const auto &edge : edges)
      if (present[edge.first] && present[edge.second]) {
        ++degree[edge.first];
        ++degree[edge.second];
      }
    changed = false;
    for (int i = 0; i < count; ++i)
      if (present[i] && degree[i] < 2) {
        present[i] = false;
        changed = true;
      }
  } while (changed);
  int survivors = 0, incidences = 0;
  for (int i = 0; i < count; ++i)
    if (present[i]) {
      ++survivors;
      incidences += degree[i];
    }
  return survivors ? static_cast<double>(incidences) / survivors
                   : std::numeric_limits<double>::quiet_NaN();
}

void equal(double actual, double expected) {
  if (std::isnan(expected) ? !std::isnan(actual) : actual != expected)
    throw std::runtime_error("Nonrattler coordination mismatch");
}

void check(SlurryRheologyDiagnostics::ContactNetwork &network,
           int count, const Graph &edges, double expected) {
  network.reset(count);
  for (const auto &edge : edges)
    network.add_contact(edge.first, edge.second);
  equal(reference(count, edges), expected);
  equal(network.nonrattler_coordination(), expected);
  // Repeated measurement must not prune the stored graph in place.
  equal(network.nonrattler_coordination(), expected);
}

int main() {
  using SlurryRheologyDiagnostics::normalized_normal_stress;
  // In xz shear, N1 = xx - zz and N2 = zz - yy. Physical rates below one
  // must not be replaced by the solver's internal nondimensionalizing rate.
  const auto normal = normalized_normal_stress(5., 2., 3., 2., .1);
  equal(normal[0], 10.);
  equal(normal[1], 5.);
  const auto pressure_shift = normalized_normal_stress(1005., 1002., 1003., 2., .1);
  equal(pressure_shift[0], normal[0]);
  equal(pressure_shift[1], normal[1]);
  const auto signed_normal = normalized_normal_stress(5., 4., 3., 1., 1.);
  equal(signed_normal[0], 2.);
  equal(signed_normal[1], -1.);
  const auto solvent = normalized_normal_stress(0., 0., 0., .001, 100.);
  equal(solvent[0], 0.);
  equal(solvent[1], 0.);
  const auto reversed = normalized_normal_stress(5., 2., 3., 2., -.1);
  equal(reversed[0], -normal[0]);
  equal(reversed[1], -normal[1]);
  const auto no_shear = normalized_normal_stress(5., 2., 3., 2., 0.);
  equal(no_shear[0], std::numeric_limits<double>::quiet_NaN());
  equal(no_shear[1], std::numeric_limits<double>::quiet_NaN());

  SlurryRheologyDiagnostics::ContactNetwork network;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  check(network, 0, {}, nan);
  check(network, 8, {}, nan);
  check(network, 2, {{0, 1}}, nan);
  check(network, 6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}, nan);
  check(network, 3, {{0, 1}, {1, 2}, {2, 0}}, 2.);
  check(network, 4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}, 3.);
  // Removing the last leaf must propagate down the entire branch and must
  // also remove its contact from the surviving ring's numerator.
  check(network, 9, {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4},
                     {4, 5}, {5, 6}, {6, 7}}, 2.);
  // Periodic images cannot be deduplicated solely by particle tags.
  check(network, 2, {{0, 1}, {0, 1}}, 2.);
  check(network, 1, {{0, 0}}, 2.);
  check(network, 3, {{0, 0}, {0, 1}, {1, 2}}, 2.);
  check(network, 2, {{0, 0}, {0, 1}, {1, 1}}, 3.);

  Graph long_branch{{0, 1}, {1, 2}, {2, 0}};
  for (int i = 2; i < 999; ++i)
    long_branch.emplace_back(i, i + 1);
  check(network, 1000, long_branch, 2.);

  std::mt19937 random(20261114);
  for (int trial = 0; trial < 1000; ++trial) {
    const int count = 1 + random() % 64;
    const int edge_count = random() % (count * 4 + 1);
    Graph edges;
    for (int e = 0; e < edge_count; ++e)
      edges.emplace_back(random() % count, random() % count);
    check(network, count, edges, reference(count, edges));
    std::shuffle(edges.begin(), edges.end(), random);
    check(network, count, edges, reference(count, edges));
  }
  // Existing capacities/scratch state must not leak between differently sized
  // accepted configurations, nor prevent appending after a query.
  check(network, 2, {{0, 1}}, nan);
  network.add_contact(0, 1);
  equal(network.nonrattler_coordination(), 2.);
  check(network, 0, {}, nan);

  std::cout << "Normal-stress axes/sign/normalization, nonrattler definition, "
               "periodic contacts, workspace reuse and "
               "2000 randomized graph comparisons passed\n";
}
