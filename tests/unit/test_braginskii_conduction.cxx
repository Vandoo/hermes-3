#include <cmath>
#include <string>
#include <utility>

#include <bout/bout_types.hxx>
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
  BraginskiiConduction unlimited = this->makeComponent(1.);
  BraginskiiConduction face = this->makeLimitedComponent(0.2, true);
  Options state_unlimited = this->makeState(this->temp2);
  Options state_face = this->makeState(this->temp2);

  unlimited.declareAllSpecies({"test+"});
  unlimited.transform(state_unlimited);
  face.declareAllSpecies({"test+"});
  face.transform(state_face);

  Field3D conduction_unlimited = this->getDeriv(state_unlimited);
  Field3D conduction_face = this->getDeriv(state_face);
  BOUT_FOR_SERIAL(i, conduction_face.getRegion("RGN_NOBNDRY")) {
    // Each face flux is reduced but keeps its sign
    EXPECT_LT(std::abs(conduction_face[i]), std::abs(conduction_unlimited[i]));
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

// TODO: Add tests for more types of collisions (e.g., electron-ion, charge-exchante, ion
// neutral...)
