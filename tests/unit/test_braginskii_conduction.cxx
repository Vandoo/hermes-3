#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include <bout/bout_types.hxx>
#include <bout/constants.hxx>
#include <bout/coordinates.hxx>
#include <bout/fieldgroup.hxx>
#include <bout/paralleltransform.hxx>
#include <bout/region.hxx>
#include <bout/solver.hxx>
#include <bout/utils.hxx>
#include "gtest/gtest.h"

#include "../../include/braginskii_conduction.hxx"
#include "fake_mesh_fixture.hxx"
#include "test_extras.hxx" // FakeMesh

/// Global mesh
namespace bout {
namespace globals {
extern Mesh* mesh;
} // namespace globals
} // namespace bout

// The unit tests use the global mesh
using namespace bout::globals;

// Reuse the "standard" fixture for FakeMesh
class BraginskiiConductionTest : public FakeMeshFixture {
public:
  BraginskiiConductionTest()
      : FakeMeshFixture(), options({
                               {"units",
                                {{"eV", 1.0},
                                 {"meters", 1.0},
                                 {"seconds", 1.0},
                                 {"inv_meters_cubed", 1e19}}},
                               {"test+",
                                {{"type", "evolve_energy"},
                                 {"thermal_conduction", true},
                                 {"diagnose", true}}},
                           }),
        output_base({{"Nnorm", 1e19}, {"Tnorm", 1.}, {"Omega_ci", 1.}, {"rho_s0", 1.}}),
        state_base(
            {{"fastest_wave", 0.},
             {"species",
              {{"test+",
                {{"velocity", 0.}, {"density", 1.}, {"pressure", 1.}, {"AA", 2.}}}}}}) {
    Options::root()["Ptest+"]["function"] = "1.0";
    temp1 = linearGradient(1., 0., 0., 1., 1., 0., 0.);
    temp2 = linearGradient(0., 0., 0., 1., 2., 0., 0.);
    temp_perp_variation = linearGradient(0., 1., 2., 0., 0., 0., 0.);
  }
  Options options, output_base, state_base;
  Field3D temp1, temp2, temp_perp_variation;

  static Field3D linearGradient(BoutReal a, BoutReal b1_x, BoutReal b2_x, BoutReal b1_y,
                                BoutReal b2_y, BoutReal b1_z, BoutReal b2_z) {
    Field3D result(a);
    BOUT_FOR_SERIAL(i, result.getRegion("RGN_ALL")) {
      const int x = i.x();
      const int y = i.y();
      const int z = i.z();
      result[i] +=
          b1_x * x + b2_x * SQ(x) + b1_y * y + b2_y * SQ(y) + b1_z * z + b2_z * SQ(z);
    }
    return result;
  }

  BraginskiiConduction makeComponent(BoutReal kappa, std::string mode = "braginskii") {
    Options opts = options.copy();
    opts["test+"]["kappa_coefficient"] = kappa;
    opts["test+"]["conduction_collisions_mode"] = std::move(mode);
    return BraginskiiConduction("test+", opts, nullptr);
  }

  BraginskiiConduction makeLimitedComponent(BoutReal alpha, bool faces) {
    Options opts = options.copy();
    opts["test+"]["kappa_coefficient"] = 1.0;
    opts["test+"]["conduction_collisions_mode"] = "braginskii";
    opts["test+"]["kappa_limit_alpha"] = alpha;
    opts["test+"]["kappa_limit_faces"] = faces;
    return BraginskiiConduction("test+", opts, nullptr);
  }

  /// State with unit collision frequency, given temperature, and pressure = T
  /// (density 1)
  Options makeState(const Field3D& temperature) {
    Options state = this->state_base.copy();
    state["species"]["test+"]["charge"] = 1;
    state["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 1.;
    state["species"]["test+"]["collision_frequency"] = 1.;
    state["species"]["test+"]["temperature"] = temperature;
    state["species"]["test+"]["pressure"] = temperature;
    return state;
  }

  Field3D getDeriv(const Options& state) {
    return state["species"]["test+"]["energy_source"];
  }
};

TEST_F(BraginskiiConductionTest, ConductionGradientScaling) {
  BraginskiiConduction component = this->makeComponent(1.);
  Options state0;
  Options state1 = this->state_base.copy();
  Options state2;
  state1["species"]["test+"]["charge"] = 1;
  state1["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 1.;
  state1["species"]["test+"]["collision_frequency"] = 1.;

  state0 = state1.copy();
  state2 = state1.copy();

  state0["species"]["test+"]["temperature"] = this->temp_perp_variation;
  state1["species"]["test+"]["temperature"] = this->temp1;
  state2["species"]["test+"]["temperature"] = this->temp2;

  component.declareAllSpecies({"test+"});
  component.transform(state0);
  component.transform(state1);
  component.transform(state2);

  Field3D conduction0 = this->getDeriv(state0);
  Field3D conduction1 = this->getDeriv(state1);
  Field3D conduction2 = this->getDeriv(state2);
  BOUT_FOR_SERIAL(i, conduction1.getRegion("RGN_NOBNDRY")) {
    // Conduction is proportional to the parallel second derivative of temperature
    EXPECT_NE(conduction1[i], 0.);
    EXPECT_DOUBLE_EQ(conduction0[i], 0.);
    EXPECT_DOUBLE_EQ(conduction2[i], 2 * conduction1[i]);
  }
}

TEST_F(BraginskiiConductionTest, ConductionKappaScaling) {
  BraginskiiConduction component1 = this->makeComponent(1.);
  BraginskiiConduction component2 = this->makeComponent(2.);
  Options state1 = this->state_base.copy();
  state1["species"]["test+"]["charge"] = 1;
  state1["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 1.;
  state1["species"]["test+"]["collision_frequency"] = 1.;
  state1["species"]["test+"]["temperature"] = this->temp1;
  Options state2 = state1.copy();

  component1.declareAllSpecies({"test+"});
  component1.transform(state1);
  component2.declareAllSpecies({"test+"});
  component2.transform(state2);

  Field3D conduction1 = this->getDeriv(state1);
  Field3D conduction2 = this->getDeriv(state2);
  BOUT_FOR_SERIAL(i, conduction1.getRegion("RGN_NOBNDRY")) {
    // Conduction is proportional to the parallel diffusivity
    EXPECT_NE(conduction1[i], 0.);
    EXPECT_DOUBLE_EQ(conduction2[i], 2 * conduction1[i]);
  }
}

TEST_F(BraginskiiConductionTest, ConductionCollisionScaling) {
  BraginskiiConduction component = this->makeComponent(1.);
  Options state0;
  Options state1 = this->state_base.copy();
  Options state2;
  state1["species"]["test+"]["charge"] = 1;
  state1["species"]["test+"]["temperature"] = this->temp1;

  state0 = state1.copy();
  state2 = state1.copy();

  state0["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 0.5;
  state0["species"]["test+"]["collision_frequency"] = 0.5;
  state1["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 1.;
  state1["species"]["test+"]["collision_frequency"] = 1.;
  state2["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 2.;
  state2["species"]["test+"]["collision_frequency"] = 2.;

  component.declareAllSpecies({"test+"});
  component.transform(state0);
  component.transform(state1);
  component.transform(state2);

  Field3D conduction0 = this->getDeriv(state0);
  Field3D conduction1 = this->getDeriv(state1);
  Field3D conduction2 = this->getDeriv(state2);
  BOUT_FOR_SERIAL(i, conduction1.getRegion("RGN_NOBNDRY")) {
    // Conduction is inversely proportional to the collision frequency
    EXPECT_NE(conduction1[i], 0.);
    EXPECT_DOUBLE_EQ(conduction0[i], 2 * conduction1[i]);
    EXPECT_DOUBLE_EQ(2 * conduction2[i], conduction1[i]);
  }
}

TEST_F(BraginskiiConductionTest, ConductionCollisionsMode) {
  BraginskiiConduction component_braginskii = this->makeComponent(1., "braginskii");
  BraginskiiConduction component_multispecies = this->makeComponent(1., "multispecies");
  Options state_brag = this->state_base.copy();
  state_brag["species"]["test+"]["charge"] = 1;
  state_brag["species"]["test+"]["temperature"] = this->temp1;
  state_brag["species"]["test+"]["collision_frequencies"]["test+_test+_coll"] = 0.5;
  state_brag["species"]["test+"]["collision_frequencies"]["test+_e_coll"] = 0.5;
  state_brag["species"]["test+"]["collision_frequency"] = 1.0;
  Options state_multi = state_brag.copy();

  component_braginskii.declareAllSpecies({"test+"});
  component_braginskii.transform(state_brag);
  component_multispecies.declareAllSpecies({"test+", "e"});
  component_multispecies.transform(state_multi);

  Field3D conduction_brag = this->getDeriv(state_brag);
  Field3D conduction_multi = this->getDeriv(state_multi);
  BOUT_FOR_SERIAL(i, conduction_brag.getRegion("RGN_NOBNDRY")) {
    EXPECT_NE(conduction_brag[i], 0.);
    EXPECT_DOUBLE_EQ(conduction_brag[i], 2 * conduction_multi[i]);
  }
}

TEST_F(BraginskiiConductionTest, FaceLimiterLargeAlphaMatchesUnlimited) {
  // With a very weak limit the face-limited operator must reproduce the
  // unlimited Div_par_K_Grad_par_mod result
  BraginskiiConduction unlimited = this->makeComponent(1.);
  BraginskiiConduction face = this->makeLimitedComponent(1e12, true);
  Options state_unlimited = this->makeState(this->temp2);
  Options state_face = this->makeState(this->temp2);

  unlimited.declareAllSpecies({"test+"});
  unlimited.transform(state_unlimited);
  face.declareAllSpecies({"test+"});
  face.transform(state_face);

  Field3D conduction_unlimited = this->getDeriv(state_unlimited);
  Field3D conduction_face = this->getDeriv(state_face);
  Field3D kappa_unlimited = state_unlimited["species"]["test+"]["kappa_par"];
  Field3D kappa_face = state_face["species"]["test+"]["kappa_par"];
  BOUT_FOR_SERIAL(i, conduction_face.getRegion("RGN_NOBNDRY")) {
    EXPECT_NE(conduction_unlimited[i], 0.);
    EXPECT_NEAR(conduction_face[i], conduction_unlimited[i],
                1e-9 * std::abs(conduction_unlimited[i]));
    EXPECT_NEAR(kappa_face[i], kappa_unlimited[i], 1e-9 * std::abs(kappa_unlimited[i]));
  }
}

TEST_F(BraginskiiConductionTest, FaceLimiterReducesConduction) {
  // Check the flux through each face directly: it must equal the unlimited
  // face flux F times 1 / (1 + |F| / q_fl), with q_fl from the hotter cell
  // (unit metrics, so F is also the heat flux density), and so keep its sign
  // and be smaller in magnitude
  const BoutReal alpha = 0.2, AA = 2.0;
  BraginskiiConduction unlimited = this->makeComponent(1.);
  BraginskiiConduction face = this->makeLimitedComponent(alpha, true);
  Options state_unlimited = this->makeState(this->temp2);
  Options state_face = this->makeState(this->temp2);

  unlimited.declareAllSpecies({"test+"});
  unlimited.transform(state_unlimited);
  face.declareAllSpecies({"test+"});
  face.transform(state_face);

  const Field3D K = state_unlimited["species"]["test+"]["kappa_par"];
  const Field3D flow_unlimited = state_unlimited["species"]["test+"]["energy_flow_ylow"];
  const Field3D flow_face = state_face["species"]["test+"]["energy_flow_ylow"];
  const Field3D& T = this->temp2;
  BOUT_FOR_SERIAL(i, flow_face.getRegion("RGN_NOBNDRY")) {
    const auto im = i.ym();
    const BoutReal F = 0.5 * (K[im] + K[i]) * (T[i] - T[im]);
    const auto hot = (T[i] > T[im]) ? i : im;
    const BoutReal q_fl = alpha * T[hot] * std::sqrt(T[hot] / AA); // P = T
    const BoutReal expected = -F / (1. + std::abs(F) / q_fl);      // flow = -flux

    ASSERT_NE(flow_unlimited[i], 0.);
    EXPECT_NEAR(flow_unlimited[i], -F, 1e-12 * std::abs(F));
    EXPECT_NEAR(flow_face[i], expected, 1e-12 * std::abs(expected));
    EXPECT_GT(flow_face[i] * flow_unlimited[i], 0.);
    EXPECT_LT(std::abs(flow_face[i]), std::abs(flow_unlimited[i]));
  }
}

TEST_F(BraginskiiConductionTest, FaceLimiterKappaParBounded) {
  // The published kappa_par (preconditioner, flutter, diagnostics) must be
  // finite everywhere, including guard cells, lie between 0 and the
  // unlimited kappa, and copy the physical x guards from the domain
  BraginskiiConduction unlimited = this->makeComponent(1.);
  BraginskiiConduction face = this->makeLimitedComponent(0.2, true);
  Options state_unlimited = this->makeState(this->temp2);
  Options state_face = this->makeState(this->temp2);

  unlimited.declareAllSpecies({"test+"});
  unlimited.transform(state_unlimited);
  face.declareAllSpecies({"test+"});
  face.transform(state_face);

  const Field3D kappa_unlimited = state_unlimited["species"]["test+"]["kappa_par"];
  const Field3D kappa_face = state_face["species"]["test+"]["kappa_par"];
  BOUT_FOR_SERIAL(i, kappa_face.getRegion("RGN_ALL")) {
    EXPECT_TRUE(std::isfinite(kappa_face[i]));
    EXPECT_GE(kappa_face[i], 0.);
  }
  BOUT_FOR_SERIAL(i, kappa_face.getRegion("RGN_NOBNDRY")) {
    EXPECT_LE(kappa_face[i], kappa_unlimited[i]);
  }
  for (int y = 0; y < mesh->LocalNy; y++) {
    for (int z = 0; z < mesh->LocalNz; z++) {
      for (int x = 0; x < mesh->xstart; x++) {
        EXPECT_EQ(kappa_face(x, y, z), kappa_face(mesh->xstart, y, z));
      }
      for (int x = mesh->xend + 1; x < mesh->LocalNx; x++) {
        EXPECT_EQ(kappa_face(x, y, z), kappa_face(mesh->xend, y, z));
      }
    }
  }
}

TEST_F(BraginskiiConductionTest, FaceLimiterConservesEnergy) {
  // No periodic field lines: both y boundaries are targets, whose faces are
  // excluded, so the energy source must sum to zero along each field line
  static_cast<FakeMesh*>(mesh)->ix_separatrix = 0;
  BraginskiiConduction face = this->makeLimitedComponent(0.2, true);
  Options state = this->makeState(this->temp2);
  face.declareAllSpecies({"test+"});
  face.transform(state);

  Field3D conduction = this->getDeriv(state);
  const Coordinates* coord = mesh->getCoordinates();
  Field3D total{0.0}, scale{0.0}; // Sums along y, stored at y = ystart
  BOUT_FOR_SERIAL(i, conduction.getRegion("RGN_NOBNDRY")) {
    const BoutReal dV = coord->J()[i] * coord->dy()[i];
    total(i.x(), mesh->ystart, i.z()) += conduction[i] * dV;
    scale(i.x(), mesh->ystart, i.z()) += std::abs(conduction[i] * dV);
  }
  for (int x = mesh->xstart; x <= mesh->xend; x++) {
    for (int z = 0; z < mesh->LocalNz; z++) {
      EXPECT_GT(scale(x, mesh->ystart, z), 0.);
      EXPECT_NEAR(total(x, mesh->ystart, z), 0.0, 1e-12 * scale(x, mesh->ystart, z));
    }
  }
}

TEST_F(BraginskiiConductionTest, FaceLimiterUsesHotSideFreeStreaming) {
  // Temperature step: T = 1 for y <= y_hot, T = 0.01 above. Pressure = T,
  // so K = P / (nu AA) = T / 2 (unit metrics). Only the face between y_hot and
  // y_cold carries heat. Its free-streaming flux must be taken from the hot
  // cell: using the cold cell instead would make the flux ~600 times smaller.
  const int y_hot = mesh->ystart;
  const int y_cold = y_hot + 1;
  ASSERT_LE(y_cold + 1, mesh->yend);
  const BoutReal T_hot = 1.0, T_cold = 0.01, AA = 2.0, alpha = 0.2;

  Field3D temperature{T_cold};
  BOUT_FOR_SERIAL(i, temperature.getRegion("RGN_ALL")) {
    if (i.y() <= y_hot) {
      temperature[i] = T_hot;
    }
  }
  BraginskiiConduction face = this->makeLimitedComponent(alpha, true);
  Options state = this->makeState(temperature);
  face.declareAllSpecies({"test+"});
  face.transform(state);
  Field3D conduction = this->getDeriv(state);

  const BoutReal K_face = 0.5 * (T_hot + T_cold) / AA;
  const BoutReal q_SH = K_face * (T_cold - T_hot); // unit metrics: dT/dl = dT
  const BoutReal q_fl = alpha * T_hot * sqrt(T_hot / AA);
  const BoutReal flux = q_SH / (1. + std::abs(q_SH) / q_fl);

  for (int x = mesh->xstart; x <= mesh->xend; x++) {
    for (int z = 0; z < mesh->LocalNz; z++) {
      EXPECT_DOUBLE_EQ(conduction(x, y_hot, z), flux);
      EXPECT_DOUBLE_EQ(conduction(x, y_cold, z), -flux);
      EXPECT_DOUBLE_EQ(conduction(x, y_cold + 1, z), 0.0);
    }
  }
}

/// FakeMesh with closed field lines on one processor and a poloidal branch
/// cut between yend and ystart (one y guard cell): y communication copies the
/// guard cells periodically, as BoutMesh does across the cut with a shifted
/// metric (no twist-shift in the communication itself).
class BranchCutMesh : public FakeMesh {
public:
  BranchCutMesh(int nx, int ny, int nz, BoutReal shift_angle)
      : FakeMesh(nx, ny, nz), shift_angle(shift_angle) {}
  BoutReal shift_angle;

  std::pair<bool, BoutReal> hasBranchCutLower(int jx) const override {
    return std::make_pair(periodicY(jx), shift_angle);
  }
  std::pair<bool, BoutReal> hasBranchCutUpper(int jx) const override {
    return std::make_pair(periodicY(jx), shift_angle);
  }
  RangeIterator iterateBndryLowerY() const override { return RangeIterator(); }
  RangeIterator iterateBndryUpperY() const override { return RangeIterator(); }

  comm_handle send(FieldGroup& g) override { return sendY(g); }
  comm_handle sendY(FieldGroup& g, comm_handle UNUSED(handle) = nullptr) override {
    for (Field3D* f : g.field3d()) {
      for (int x = 0; x < LocalNx; x++) {
        for (int z = 0; z < LocalNz; z++) {
          (*f)(x, ystart - 1, z) = (*f)(x, yend, z);
          (*f)(x, yend + 1, z) = (*f)(x, ystart, z);
        }
      }
    }
    return nullptr;
  }
};

TEST_F(BraginskiiConductionTest, FaceLimiterConservesEnergyAcrossBranchCut) {
  // Closed field lines with a shifted metric and a branch cut whose
  // ShiftAngle is not a whole number of z cells. The guard cells at the cut
  // are then shifted by a fractional offset, so the two cells next to the cut
  // see the face at different toroidal positions; the limited face flux must
  // still be counted once, so that the energy source sums to zero over the
  // closed field lines.
  constexpr int nx = 3, ny = 7, nz = 7;
  const BoutReal shift_angle = 1.11; // In units of z cells (dz = 1)
  Mesh* const fixture_mesh = bout::globals::mesh;
  BranchCutMesh bc_mesh(nx, ny, nz, shift_angle);
  bout::globals::mesh = &bc_mesh;
  bc_mesh.createDefaultRegions();
  bc_mesh.setCoordinates(nullptr);
  auto coords = std::make_shared<Coordinates>(
      &bc_mesh, Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{1.0},
      Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{0.0}, Field2D{0.0},
      Field2D{0.0}, Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{0.0},
      Field2D{0.0}, Field2D{0.0}, Field2D{0.0}, Field2D{0.0});
  bc_mesh.setCoordinates(coords);
  bc_mesh.setGridDataSource(new FakeGridDataSource());

  // zShift grows along y; the guard cells at the cut are offset by
  // ShiftAngle, as Coordinates does for a grid file
  Field2D zShift{0.0};
  for (int x = 0; x < nx; x++) {
    for (int y = bc_mesh.ystart; y <= bc_mesh.yend; y++) {
      zShift(x, y) = 0.37 * y;
    }
    zShift(x, bc_mesh.ystart - 1) = zShift(x, bc_mesh.yend) - shift_angle;
    zShift(x, bc_mesh.yend + 1) = zShift(x, bc_mesh.ystart) + shift_angle;
  }
  coords->setParallelTransform(
      std::make_unique<ShiftedMetric>(bc_mesh, CELL_CENTRE, zShift, nz * 1.0));

  // Positive temperature varying in y and z, periodic across the cut
  Field3D temperature{1.0};
  const int ny_domain = bc_mesh.yend - bc_mesh.ystart + 1;
  BOUT_FOR_SERIAL(i, temperature.getRegion("RGN_ALL")) {
    temperature[i] = 1. + 0.5 * std::sin(2. * PI * (i.y() - bc_mesh.ystart) / ny_domain)
                     + 0.3 * std::cos(2. * PI * i.z() / nz + i.y());
  }
  bc_mesh.communicate(temperature);

  {
    BraginskiiConduction face = this->makeLimitedComponent(0.2, true);
    Options state = this->makeState(temperature);
    face.declareAllSpecies({"test+"});
    face.transform(state);

    const Field3D conduction = this->getDeriv(state);
    for (int x = bc_mesh.xstart; x <= bc_mesh.xend; x++) {
      BoutReal total = 0.0, scale = 0.0; // Unit metrics: cell volume 1
      for (int y = bc_mesh.ystart; y <= bc_mesh.yend; y++) {
        for (int z = 0; z < nz; z++) {
          total += conduction(x, y, z);
          scale += std::abs(conduction(x, y, z));
        }
      }
      EXPECT_GT(scale, 0.);
      EXPECT_NEAR(total, 0.0, 1e-12 * scale);
    }
  }
  bout::globals::mesh = fixture_mesh;
}

TEST_F(BraginskiiConductionTest, FaceLimiterRejectsOtherTransforms) {
  // The branch-cut treatment relies on a shift that keeps the sum over z, so
  // the face form accepts only the identity and FFT shifted-metric
  // transforms. MockParallelTransform (not FCI) stands for e.g. shiftedinterp.
  mesh->getCoordinates()->setParallelTransform(
      std::make_unique<MockParallelTransform>(*mesh, true));
  BraginskiiConduction face = this->makeLimitedComponent(0.2, true);
  Options state = this->makeState(this->temp2);
  face.declareAllSpecies({"test+"});
  bool thrown = false;
  try {
    face.transform(state);
  } catch (const BoutException& e) {
    thrown = true;
    EXPECT_NE(std::string(e.what()).find("paralleltransform"), std::string::npos)
        << e.what();
  }
  EXPECT_TRUE(thrown);
  mesh->getCoordinates()->setParallelTransform(
      std::make_unique<ParallelTransformIdentity>(*mesh));
}

/// FakeMesh that is periodic in x on one processor: x communication copies
/// the guard cells from the opposite edge of the domain
class PeriodicXMesh : public FakeMesh {
public:
  PeriodicXMesh(int nx, int ny, int nz) : FakeMesh(nx, ny, nz) { periodicX = true; }

  comm_handle send(FieldGroup& g) override { return sendX(g); }
  comm_handle sendX(FieldGroup& g, comm_handle UNUSED(handle) = nullptr,
                    bool UNUSED(disable_corners) = false) override {
    const int nx_domain = xend - xstart + 1;
    for (Field3D* f : g.field3d()) {
      for (int y = 0; y < LocalNy; y++) {
        for (int z = 0; z < LocalNz; z++) {
          for (int x = 0; x < xstart; x++) {
            (*f)(x, y, z) = (*f)(x + nx_domain, y, z);
          }
          for (int x = xend + 1; x < LocalNx; x++) {
            (*f)(x, y, z) = (*f)(x - nx_domain, y, z);
          }
        }
      }
    }
    return nullptr;
  }
};

TEST_F(BraginskiiConductionTest, FaceLimiterKappaParPeriodicX) {
  // With periodicX, firstX() and lastX() are still true on the edge
  // processors, but the x guards of kappa_par come from the communication
  // (the opposite edge), and must not be overwritten by the local edge
  constexpr int nx = 5, ny = 5, nz = 3;
  Mesh* const fixture_mesh = bout::globals::mesh;
  PeriodicXMesh px_mesh(nx, ny, nz);
  bout::globals::mesh = &px_mesh;
  px_mesh.createDefaultRegions();
  px_mesh.setCoordinates(nullptr);
  px_mesh.setGridDataSource(new FakeGridDataSource());
  auto coords = std::make_shared<Coordinates>(
      &px_mesh, Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{1.0},
      Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{0.0}, Field2D{0.0},
      Field2D{0.0}, Field2D{1.0}, Field2D{1.0}, Field2D{1.0}, Field2D{0.0},
      Field2D{0.0}, Field2D{0.0}, Field2D{0.0}, Field2D{0.0});
  px_mesh.setCoordinates(coords);
  coords->setParallelTransform(std::make_unique<ParallelTransformIdentity>(px_mesh));
  ASSERT_GT(px_mesh.xend, px_mesh.xstart); // The two edges must differ

  // Positive temperature, different at the two x edges
  Field3D temperature{1.0};
  BOUT_FOR_SERIAL(i, temperature.getRegion("RGN_ALL")) {
    temperature[i] = 1. + 0.4 * i.x() + 0.1 * SQ(i.y()) + 0.05 * i.z();
  }

  {
    BraginskiiConduction face = this->makeLimitedComponent(0.2, true);
    Options state = this->makeState(temperature);
    face.declareAllSpecies({"test+"});
    face.transform(state);

    const Field3D kappa_face = state["species"]["test+"]["kappa_par"];
    for (int y = px_mesh.ystart; y <= px_mesh.yend; y++) {
      for (int z = 0; z < nz; z++) {
        EXPECT_NE(kappa_face(px_mesh.xstart, y, z), kappa_face(px_mesh.xend, y, z));
        EXPECT_EQ(kappa_face(px_mesh.xstart - 1, y, z), kappa_face(px_mesh.xend, y, z));
        EXPECT_EQ(kappa_face(px_mesh.xend + 1, y, z), kappa_face(px_mesh.xstart, y, z));
      }
    }
  }
  bout::globals::mesh = fixture_mesh;
}

// TODO: Add tests for more types of collisions (e.g., electron-ion, charge-exchante, ion
// neutral...)
