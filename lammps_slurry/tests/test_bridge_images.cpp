#include "slurry_bridge_images.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace SlurryBridgeImages;
using Key = std::tuple<int, int, int>;
static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
static Key key(const ImageIndex &n) { return {n[0], n[1], n[2]}; }

static void compare_brute(const Mat3 &H, const Vec3 &d, double R, int extent) {
  std::set<Key> expected, actual;
  for (int i = -extent; i <= extent; ++i)
    for (int j = -extent; j <= extent; ++j)
      for (int k = -extent; k <= extent; ++k) {
        ImageIndex n(i, j, k);
        const Vec3 r = d + H * n.cast<double>();
        if (r.norm() <= R + 1e-12)
          expected.insert(key(n));
      }
  for (const auto &image : images_within(d, H, R)) {
    actual.insert(key(image.image));
    check((image.displacement - d - H * image.image.cast<double>()).norm() < 1e-12,
          "image displacement");
  }
  check(expected == actual, "all image enumeration differs from independent fixed brute cube");
}

int main() {
  Mat3 H = Vec3(50., 80., 5.).asDiagonal();
  const Vec3 d(24., 0., 2. + 1. / 150.);
  compare_brute(H, d, 60., 20);
  const auto images = images_within(d, H, 60.);
  check(images.size() == 44, "small cell known 44 distinct images");
  int self_count = 0;
  for (const auto &image : images_within(Vec3::Zero(), H, 60.))
    if (positive_image(image.image))
      ++self_count;
  check(self_count == 25, "self images retain exactly one of each +/- pair");
  check(!positive_image(ImageIndex::Zero()), "self zero image excluded");

  Mat3 skew;
  skew << 140., 130., 0., 0., 40., 0., 0., 0., 140.;
  compare_brute(skew, Vec3(6., 14., 8.), 60., 5);
  check(images_within(Vec3(6., 14., 8.), skew, 60.).size() == 3, "skew known 3 images");
  Mat3 U;
  U << 1., -1., 0., 0., 1., 0., 0., 0., 1.;
  const auto a = images_within(Vec3(6., 14., 8.), skew, 60.);
  const auto b = images_within(Vec3(6., 14., 8.), skew * U, 60.);
  check(a.size() == b.size(), "integer lattice remap image count");
  for (const auto &image : a) {
    std::vector<Vec3> candidates;
    for (const auto &other : b)
      candidates.push_back(other.displacement);
    check(match_image(candidates, image.displacement, 1e-11).found,
          "integer lattice remap displacement invariance");
  }

  // One pair crosses the y boundary while its image separation remains smooth.
  Mat3 box = Vec3(10., 10., 10.).asDiagonal();
  Vec3 xi(2., 9.95, 1.), xj(2., 0.15, 1.);
  Vec3 ui(0., 1., 0.), uj(0., 0.5, 0.);
  const double dt = .1;
  const Vec3 old_d = xj + box * ImageIndex(0, 1, 0).cast<double>() - xi;
  const Vec3 predicted = old_d + dt * (uj - ui);
  xi += dt * ui;
  xj += dt * uj;
  xi[1] -= 10.;
  const auto wrapped = images_within(xj - xi, box, 1.);
  check(wrapped.size() == 1 && wrapped[0].image == ImageIndex::Zero(),
        "boundary crossing changes raw image integer");
  auto matched = match_image({predicted}, wrapped[0].displacement, 1e-12);
  check(matched.found && matched.sign == 1, "history retained through boundary crossing");

  // Shear cell at the flip: Hnew=Hphysical*U. The represented physical image
  // persists, even if the lexicographic canonical self sign changes.
  Mat3 pre;
  pre << 10., 5.1, 0., 0., 10., 0., 0., 0., 10.;
  Mat3 post = pre * U;
  const ImageIndex old_n(1, -1, 0), new_n(0, -1, 0);
  const Vec3 old_self = pre * old_n.cast<double>();
  check(positive_image(old_n) && !positive_image(new_n), "self representative actually flips sign");
  const Vec3 canonical_after = -post * new_n.cast<double>();
  check((canonical_after + old_self).norm() < 1e-12, "self remap physical displacement reverses");
  matched = match_image({old_self}, canonical_after, 1e-12, true);
  check(matched.found && matched.sign == -1, "self history reversed representative recognized");
  check(!match_image({old_self}, canonical_after, 1e-12, false).found,
        "nonself ordered tags cannot reverse");
  Vec3 normal = old_self.normalized(), spring = normal.cross(Vec3::UnitZ()) * .01,
       common(.1, .2, .3);
  const Vec3 reversed_normal = matched.sign * normal, reversed_spring = matched.sign * spring;
  check((reversed_normal + normal).norm() == 0. && (reversed_spring + spring).norm() == 0.,
        "reverse contact state sign");
  check((common - Vec3(.1, .2, .3)).norm() == 0., "common angular rotation unchanged");

  bool rejected = false;
  try {
    match_image({predicted, predicted}, predicted, 1e-12);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "ambiguous history must reject");
  check(!match_image({predicted}, predicted + Vec3(1., 0., 0.), 1e-12).found,
        "different image must not borrow history");
  std::cout << "bridge images: enumeration, wrapping, basis flip and self history passed\n";
}
