#include <string>
#include <utility>

#include <cmath>

#include <bout/bout_types.hxx>
#include <bout/constants.hxx>
#include <bout/region.hxx>
#include <bout/solver.hxx>
#include <bout/utils.hxx>
#include "gtest/gtest.h"

#include "../../include/braginskii_collisions.hxx"
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

namespace {
/// Options for conduction of electrons and one singly charged ion species,
/// with default kappa coefficients, in the given collisions mode
Options electronIonOptions(const std::string& mode) {
  Options opts{{"units",
                {{"eV", 1.0},
                 {"meters", 1.0},
                 {"seconds", 1.0},
                 {"inv_meters_cubed", 1e19}}}};
  for (const std::string sp : {"e", "d+"}) {
    opts[sp]["type"] = "evolve_pressure";
    opts[sp]["thermal_conduction"] = true;
    opts[sp]["conduction_collisions_mode"] = mode;
  }
  return opts;
}

/// Electron and deuterium ion state, n = 1e19 m^-3, T = 50 eV
Options electronIonState() {
  Options state{{"fastest_wave", 0.}};
  state["species"]["e"]["AA"] = SI::Me / SI::Mp;
  state["species"]["e"]["charge"] = -1.;
  state["species"]["d+"]["AA"] = 2.;
  state["species"]["d+"]["charge"] = 1.;
  for (const std::string sp : {"e", "d+"}) {
    state["species"][sp]["density"] = 1.;
    state["species"][sp]["temperature"] = 50.;
    state["species"][sp]["pressure"] = 50.;
    state["species"][sp]["velocity"] = 0.;
  }
  return state;
}
} // namespace

TEST_F(BraginskiiConductionTest, ElectronModesAgreeForFitzpatrickRates) {
  // Hermes-3's e-e rate is nu_ei / sqrt(2) for a singly charged ion (same Coulomb
  // logarithm, m_e/m_i -> 0). Braginskii mode uses 3.16/sqrt(2) with nu_ee,
  // multispecies uses 3.16 with nu_ei (e-e is already inside the coefficient), so the
  // electron conductivities must be identical.
  const BoutReal nu_ee = 0.7;
  Options state = electronIonState();
  state["species"]["e"]["collision_frequencies"]["e_e_coll"] = nu_ee;
  state["species"]["e"]["collision_frequencies"]["e_d+_coll"] = sqrt(2.) * nu_ee;
  state["species"]["d+"]["collision_frequencies"]["d+_d+_coll"] = 1.;
  Options state_brag = state.copy();
  Options state_multi = state.copy();

  Options opts_brag = electronIonOptions("braginskii");
  Options opts_multi = electronIonOptions("multispecies");
  BraginskiiConduction brag("conduction", opts_brag, nullptr);
  BraginskiiConduction multi("conduction", opts_multi, nullptr);
  brag.declareAllSpecies({"e", "d+"});
  multi.declareAllSpecies({"e", "d+"});
  brag.transform(state_brag);
  multi.transform(state_multi);

  const Field3D kappa_brag = state_brag["species"]["e"]["kappa_par"];
  const Field3D kappa_multi = state_multi["species"]["e"]["kappa_par"];
  BOUT_FOR_SERIAL(i, kappa_brag.getRegion("RGN_NOBNDRY")) {
    EXPECT_GT(kappa_brag[i], 0.);
    EXPECT_NEAR(kappa_multi[i], kappa_brag[i], 1e-12 * kappa_brag[i]);
  }
}

TEST_F(BraginskiiConductionTest, SingleIonModesAgreeWithBraginskiiCollisions) {
  // With the rates of the collisions component, braginskii and multispecies modes give
  // the same conductivity for one singly charged ion species, up to:
  //  - electrons: the e-e and e-i Coulomb logarithms differ (NRL formulary), here by
  //    0.8 %
  //  - ions: multispecies also counts ion-electron collisions, which are slower than
  //    ion-ion collisions by about sqrt(2 m_e / m_i) (2.2 % for deuterium, Te = Ti)
  Options collision_opts{{"units",
                          {{"eV", 1.0},
                           {"meters", 1.0},
                           {"seconds", 1.0},
                           {"inv_meters_cubed", 1e19}}}};
  BraginskiiCollisions collisions("collisions", collision_opts, nullptr);
  Options state = electronIonState();
  collisions.declareAllSpecies(SpeciesInformation({"e"}, {}, {"d+"}, {}));
  collisions.transform(state);
  Options state_brag = state.copy();
  Options state_multi = state.copy();

  Options opts_brag = electronIonOptions("braginskii");
  Options opts_multi = electronIonOptions("multispecies");
  BraginskiiConduction brag("conduction", opts_brag, nullptr);
  BraginskiiConduction multi("conduction", opts_multi, nullptr);
  brag.declareAllSpecies({"e", "d+"});
  multi.declareAllSpecies({"e", "d+"});
  brag.transform(state_brag);
  multi.transform(state_multi);

  const Field3D ke_brag = state_brag["species"]["e"]["kappa_par"];
  const Field3D ke_multi = state_multi["species"]["e"]["kappa_par"];
  const Field3D ki_brag = state_brag["species"]["d+"]["kappa_par"];
  const Field3D ki_multi = state_multi["species"]["d+"]["kappa_par"];
  BOUT_FOR_SERIAL(i, ke_brag.getRegion("RGN_NOBNDRY")) {
    EXPECT_GT(ke_brag[i], 0.);
    EXPECT_NEAR(ke_multi[i] / ke_brag[i], 1., 0.015);
    EXPECT_GT(ki_brag[i], 0.);
    EXPECT_NEAR(ki_multi[i] / ki_brag[i], 1., 0.04);
  }
}

// TODO: Add tests for more types of collisions (e.g., electron-ion, charge-exchante, ion
// neutral...)
