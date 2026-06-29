
#include <bout/bout_types.hxx>
#include <bout/boutexception.hxx>
#include <bout/constants.hxx>
#include <bout/derivs.hxx>
#include <bout/difops.hxx>
#include <bout/field.hxx>
#include <bout/field3d.hxx>
#include <bout/field_factory.hxx>
#include <bout/fv_ops.hxx>
#include <bout/globals.hxx>
#include <bout/initialprofiles.hxx>
#include <bout/invert_pardiv.hxx>
#include <bout/options.hxx>
#include <bout/output.hxx>
#include <bout/output_bout_types.hxx>
#include <bout/solver.hxx>

#include "../include/component.hxx"
#include "../include/div_ops.hxx"
#include "../include/evolve_pressure.hxx"
#include "../include/guarded_options.hxx"
#include "../include/hermes_build_config.hxx"
#include "../include/hermes_utils.hxx"
#include "../include/permissions.hxx"

#include <string>

using bout::globals::mesh;

namespace {
  /// Adaptive source term to prevent variable dropping below a floor value.
  ///
  /// Smooth (differentiable) turn-on. Far from the floor this equals the
  /// original ramp  max(lowvalue - f, 0) / scalefactor  (full push for
  /// f << lowvalue, zero for f >> lowvalue), but it transitions through
  /// f = lowvalue with a softplus so the source has a CONTINUOUS derivative.
  /// The original min()/ramp form has a kink at the floor whose derivative
  /// jump corrupts CVODE's difference-quotient Newton-Krylov Jacobian when
  /// many cells sit at the floor -> nonlinear-convergence-failure collapse.
  /// Transition width is a fraction of the floor value.
  void add_low_sourceterm(Field3D& result, const Field3D& f,
                        const BoutReal lowvalue, const BoutReal scalefactor) {
    const BoutReal width = 0.3 * lowvalue; // smoothing width (fraction of floor)
    BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
      const BoutReal x = (lowvalue - f[i]) / width;
      // numerically-stable softplus: ln(1 + e^x) = max(x,0) + ln(1 + e^{-|x|})
      const BoutReal sp = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
      result[i] += (width / scalefactor) * sp;
    }
  }
}

EvolvePressure::EvolvePressure(std::string name, Options& alloptions, Solver* solver)
    : Component({readOnly("species:{name}:{inputs}", Regions::Interior),
                 readWrite("species:{name}:{outputs}")}),
      name(name) {

  auto& options = alloptions[name];

  evolve_log = options["evolve_log"]
                   .doc("Evolve the logarithm of pressure?")
                   .withDefault<bool>(false);

  density_floor = options["density_floor"].doc("Minimum density floor").withDefault(1e-7);

  low_n_diffuse_perp = options["low_n_diffuse_perp"]
                           .doc("Perpendicular diffusion at low density")
                           .withDefault<bool>(false);

  temperature_floor = options["temperature_floor"]
                          .doc("Low temperature scale for low_T_diffuse_perp")
                          .withDefault<BoutReal>(0.1)
                      / get<BoutReal>(alloptions["units"]["eV"]);

  low_T_diffuse_perp = options["low_T_diffuse_perp"]
                           .doc("Add cross-field diffusion at low temperature?")
                           .withDefault<bool>(false);

  pressure_floor = density_floor * temperature_floor;

  low_p_diffuse_perp = options["low_p_diffuse_perp"]
                           .doc("Perpendicular diffusion at low pressure")
                           .withDefault<bool>(false);

  damp_p_nt = options["damp_p_nt"]
                  .doc("Damp P - N*T? Active when P < 0 or N < density_floor")
                  .withDefault<bool>(false);

  low_p_source = options["low_p_source"]
                     .doc("Add adaptive source to ddt(P) when P falls below pressure_floor")
                     .withDefault<bool>(false);

  low_p_source_scale = options["low_p_source_scale"]
                           .doc("Timescale for low_p_source term [normalised]. Smaller = more aggressive.")
                           .withDefault<BoutReal>(1e-1);

  conduction_collisions_mode = options["conduction_collisions_mode"]
      .doc("Can be multispecies: all collisions, or braginskii: self collisions and ie")
      .withDefault<std::string>("multispecies");

  if (evolve_log) {
    // Evolve logarithm of pressure
    solver->add(logP, std::string("logP") + name);
    // Save the pressure to the restart file
    // so the simulation can be restarted evolving pressure
    // get_restart_datafile()->addOnce(P, std::string("P") + name);

    if (!alloptions["hermes"]["restarting"]) {
      // Set logN from N input options
      initial_profile(std::string("P") + name, P);
      logP = log(P);
    } else {
      // Ignore these settings
      Options::root()[std::string("P") + name].setConditionallyUsed();
    }
  } else {
    // Evolve the pressure in time
    solver->add(P, std::string("P") + name);
  }

  bndry_flux = options["bndry_flux"]
                   .doc("Allow flows through radial boundaries")
                   .withDefault<bool>(true);

  poloidal_flows =
      options["poloidal_flows"].doc("Include poloidal ExB flow").withDefault<bool>(true);

  p_div_v = options["p_div_v"]
                .doc("Use p*Div(v) form? Default, false => v * Grad(p) form")
                .withDefault<bool>(false);

  hyper_z = options["hyper_z"].doc("Hyper-diffusion in Z").withDefault(-1.0);

  hyper_z_T = options["hyper_z_T"]
                  .doc("4th-order dissipation of temperature")
                  .withDefault<BoutReal>(-1.0);

  diagnose = options["diagnose"]
                 .doc("Save additional output diagnostics")
                 .withDefault<bool>(false);

  diagnose_terms = options["diagnose_terms"]
    .doc("Save detailed per-term diagnostics for ddt(P)? Useful for diagnosing negative P.")
    .withDefault<bool>(false);

  enable_precon = options["precondition"]
                      .doc("Enable preconditioner? (Note: solver may not use it)")
                      .withDefault<bool>(true);

  const Options& units = alloptions["units"];
  const BoutReal Nnorm = units["inv_meters_cubed"];
  const BoutReal Tnorm = units["eV"];
  const BoutReal Omega_ci = 1. / units["seconds"].as<BoutReal>();

  auto& p_options = alloptions[std::string("P") + name];
  source_normalisation =
      SI::qe * Nnorm * Tnorm * Omega_ci; // [Pa/s] or [W/m^3] if converted to energy
  time_normalisation = 1. / Omega_ci;    // [s]

  // Try to read the pressure source from the mesh
  // Units of Pascals per second
  source = 0.0;
  mesh->get(source, std::string("P") + name + "_src");
  // Allow the user to override the source
  source = p_options["source"]
               .doc(std::string("Source term in ddt(P") + name
                    + std::string("). Units [Pa/s], note P = 2/3 E"))
               .withDefault(source)
           / (source_normalisation);

  source_time_dependent = p_options["source_time_dependent"]
                              .doc("Use a time-dependent source?")
                              .withDefault<bool>(false);

  // If time dependent, parse the function with respect to time from the input file
  if (source_time_dependent) {
    auto str = p_options["source_prefactor"]
                   .doc("Time-dependent function of multiplier on ddt(P" + name
                        + std::string(") source."))
                   .as<std::string>();
    source_prefactor_function = FieldFactory::get()->parse(str, &p_options);
  }

  if (p_options["source_only_in_core"]
          .doc("Zero the source outside the closed field-line region?")
          .withDefault<bool>(false)) {
    for (int x = mesh->xstart; x <= mesh->xend; x++) {
      if (!mesh->periodicY(x)) {
        // Not periodic, so not in core
        for (int y = mesh->ystart; y <= mesh->yend; y++) {
          for (int z = mesh->zstart; z <= mesh->zend; z++) {
            source(x, y, z) = 0.0;
          }
        }
      }
    }
  }

  neumann_boundary_average_z = p_options["neumann_boundary_average_z"]
                                   .doc("Apply neumann boundary with Z average?")
                                   .withDefault<bool>(false);

  numerical_viscous_heating = options["numerical_viscous_heating"]
                                  .doc("Include heating due to numerical viscosity?")
                                  .withDefault<bool>(false);

  if (numerical_viscous_heating) {
    fix_momentum_boundary_flux =
        options["fix_momentum_boundary_flux"]
            .doc("Fix Y boundary momentum flux to boundary midpoint value?")
            .withDefault<bool>(false);
  }

  thermal_conduction = options["thermal_conduction"]
                           .doc("Include parallel heat conduction?")
                           .withDefault<bool>(true);

  if (source_time_dependent) {
    setPermissions(readOnly("time"));
  }
  substitutePermissions("name", {name});
  substitutePermissions("inputs", {"density"});
  substitutePermissions("outputs", {"pressure", "temperature"});
}

void EvolvePressure::transform_impl(GuardedOptions& state) {

  if (evolve_log) {
    // Evolving logP, but most calculations use P
    P = exp(logP);
  }

  mesh->communicate(P);

  if (neumann_boundary_average_z) {
    // Take Z (usually toroidal) average and apply as X (radial) boundary condition
    if (mesh->firstX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
        BoutReal Pavg = 0.0; // Average P in Z
        for (int k = 0; k < mesh->LocalNz; k++) {
          Pavg += P(mesh->xstart, j, k);
        }
        Pavg /= mesh->LocalNz;

        // Apply boundary condition
        for (int k = 0; k < mesh->LocalNz; k++) {
          P(mesh->xstart - 1, j, k) = 2. * Pavg - P(mesh->xstart, j, k);
          P(mesh->xstart - 2, j, k) = P(mesh->xstart - 1, j, k);
        }
      }
    }

    if (mesh->lastX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
        BoutReal Pavg = 0.0; // Average P in Z
        for (int k = 0; k < mesh->LocalNz; k++) {
          Pavg += P(mesh->xend, j, k);
        }
        Pavg /= mesh->LocalNz;

        for (int k = 0; k < mesh->LocalNz; k++) {
          P(mesh->xend + 1, j, k) = 2. * Pavg - P(mesh->xend, j, k);
          P(mesh->xend + 2, j, k) = P(mesh->xend + 1, j, k);
        }
      }
    }
  }

  auto species = state["species"][name];

  // Calculate temperature
  // Not using density boundary condition
  N = getNoBoundary<Field3D>(species["density"]);

  Field3D Pfloor = softFloor(P, pressure_floor);
  T = Pfloor / softFloor(N, density_floor);
  Pfloor = N * T; // Ensure consistency

  // re initialise T guard cells
  T.applyBoundary("neumann");
  // Ion-only diagnostic: 
  //if (identifySpeciesType(name) == SpeciesType::ion) {
  //  BoutReal Pmin = min(P, true);  // true = include all processors
  //  if (Pmin < 0.0) {
  //    // Derive Tmin from raw P so it also reflects the true negative value
  //    BoutReal Tmin = min(P / softFloor(N, density_floor), true);
  //    output.write("\n[evolve_pressure] WARNING: {:s} P_min = {:e}, T_min = {:e} < 0 at t = {:e}\n",
  //                 name, Pmin, Tmin, get<BoutReal>(state["time"]));
  //  }
  //}

  set(species["pressure"], Pfloor);
  set(species["temperature"], T);
}

void EvolvePressure::finally(const Options& state) {

  // Get the section containing this species
  const auto& species = state["species"][name];

  // Get updated pressure and temperature with boundary conditions
  P = get<Field3D>(species["pressure"]);
  P.clearParallelSlices();
  const Field3D Pfloor = softFloor(P, pressure_floor);

  T = get<Field3D>(species["temperature"]);
  N = get<Field3D>(species["density"]);

  if (species.isSet("charge") and (fabs(get<BoutReal>(species["charge"])) > 1e-5)
      and state.isSection("fields") and state["fields"].isSet("phi")) {
    // Electrostatic potential set and species is charged -> include ExB flow

    const Field3D phi = get<Field3D>(state["fields"]["phi"]);

    ddtP_ExB = -Div_n_bxGrad_f_B_XPPM(P, phi, bndry_flux, poloidal_flows, true);
    // ExB is the first contribution, so assign (=) to initialise ddt(P).
    ddt(P) = ddtP_ExB;
  } else {
    ddt(P) = 0.0;
  }

  if (species.isSet("velocity")) {
    const Field3D V = get<Field3D>(species["velocity"]);

    // Typical wave speed used for numerical diffusion
    Field3D fastest_wave;
    if (state.isSet("fastest_wave")) {
      fastest_wave = get<Field3D>(state["fastest_wave"]);
    } else {
      const BoutReal AA = get<BoutReal>(species["AA"]);
      fastest_wave = sqrt(floor(T, 0.0) / AA); // guards can be <0 -> protect the sqrt
    }

    if (p_div_v) {
      // Use the P * Div(V) form
      //ddt(P) -= FV::Div_par_mod<hermes::Limiter>(P, V, fastest_wave, flow_ylow_advection);
      ddtP_advection = -FV::Div_par_mod<hermes::Limiter>(P, V, fastest_wave, flow_ylow_advection);
      ddt(P) += ddtP_advection;

      // Work done. This balances energetically a term in the momentum equation
      E_PdivV = -Pfloor * Div_par(V);
      //ddt(P) += (2. / 3) * E_PdivV;
      ddtP_PdivV = (2. / 3) * E_PdivV;
      ddt(P) += ddtP_PdivV;

    } else {
      // Use V * Grad(P) form
      // Note: A mixed form has been tried (on 1D neon example)
      //       -(4/3)*FV::Div_par(P,V) + (1/3)*(V * Grad_par(P) - P * Div_par(V))
      //       Caused heating of charged species near sheath like p_div_v
      ddtP_advection = -(5. / 3) * FV::Div_par_mod<hermes::Limiter>(P, V, fastest_wave, flow_ylow_advection);
      ddt(P) += ddtP_advection;

      E_VgradP = V * Grad_par(P);
      ddtP_VgradP = (2. / 3) * E_VgradP;
      ddt(P) += ddtP_VgradP;
    }
    flow_ylow_advection *= 5. / 2; // Energy flow
    flow_ylow = flow_ylow_advection;

    if (state.isSection("fields") and state["fields"].isSet("Apar_flutter")) {
      // Magnetic flutter term
      const Field3D Apar_flutter = get<Field3D>(state["fields"]["Apar_flutter"]);
      ddt(P) -= (5. / 3) * Div_n_g_bxGrad_f_B_XZ(P, V, -Apar_flutter);
      ddt(P) += (2. / 3) * V * bracket(P, Apar_flutter, BRACKET_ARAKAWA);
    }

    if (numerical_viscous_heating || diagnose) {
      // Viscous heating coming from numerical viscosity from the Lax flux
      const Field3D Nlim = softFloor(N, density_floor);
      const BoutReal AA = get<BoutReal>(species["AA"]); // Atomic mass
      Sp_nvh = (2. / 3) * AA
               * FV::Div_par_fvv_heating(Nlim, V, fastest_wave, flow_ylow_viscous_heating,
                                         fix_momentum_boundary_flux);
      flow_ylow_viscous_heating *= AA;
      flow_ylow += flow_ylow_viscous_heating;
      if (numerical_viscous_heating) {
        ddt(P) += Sp_nvh;
      }
    }
  } else if (diagnose) {
    flow_ylow = 0;
  }

  if (species.isSet("low_n_coeff")) {
    // Low density parallel diffusion
    const Field3D low_n_coeff = get<Field3D>(species["low_n_coeff"]);
    ddt(P) += FV::Div_par_K_Grad_par(low_n_coeff * T, N)
              + FV::Div_par_K_Grad_par(low_n_coeff, P);
  }

  if (low_n_diffuse_perp) {
    //ddt(P) +=
    //    Div_Perp_Lap_FV_Index(density_floor / softFloor(N, 1e-3 * density_floor), P);
	ddtP_low_n_diff_perp = Div_Perp_Lap_FV_Index(density_floor / softFloor(N, 1e-3 * density_floor), P);
	ddt(P) += ddtP_low_n_diff_perp;
  }

  if (low_T_diffuse_perp) {
    ddt(P) +=
        1e-4
        * Div_Perp_Lap_FV_Index(
            floor(temperature_floor / softFloor(T, 1e-3 * temperature_floor) - 1.0, 0.0),
            T);
  }

  if (low_p_diffuse_perp) {
    Field3D Plim = softFloor(P, 1e-3 * pressure_floor);
    ddtP_low_p_diff_perp = Div_Perp_Lap_FV_Index(pressure_floor / Plim, P);
    ddt(P) += ddtP_low_p_diff_perp;
  }


  if (hyper_z > 0.) {
    ddt(P) -= hyper_z * D4DZ4_Index(P);
  }

  if (hyper_z_T > 0.) {
    ddt(P) -= hyper_z_T * D4DZ4_Index(T);
  }

  //////////////////////
  // Other sources

  if (source_time_dependent) {
    // Evaluate the source_prefactor function at the current time in seconds and scale
    // source with it
    const BoutReal time = get<BoutReal>(state["time"]);
    const BoutReal source_prefactor =
        source_prefactor_function->generate(bout::generator::Context().set(
            "x", 0, "y", 0, "z", 0, "t", time * time_normalisation));
    final_source = source * source_prefactor;
  } else {
    final_source = source;
  }

  Sp = final_source;
  if (species.isSet("energy_source")) {
    Sp += (2. / 3) * get<Field3D>(species["energy_source"]); // For diagnostic output
  }
#if CHECKLEVEL >= 1
  if (species.isSet("pressure_source")) {
    throw BoutException(
        "Components must evolve `energy_source` rather then `pressure_source`");
  }
#endif
  ddt(P) += Sp;

  if (damp_p_nt) {
    // Term to force evolved P towards N * T
    // This is active when P < 0 or when N < density_floor
    //ddt(P) += N * T - P;
    ddtP_damp_NT = N * T - P;
    ddt(P) += ddtP_damp_NT;
  }

  if (low_p_source) {
    //add_low_sourceterm(ddt(P), get<Field3D>(species["pressure"]), pressure_floor, low_p_source_scale);
    ddtP_low_p_source = 0.0;
    add_low_sourceterm(ddtP_low_p_source, get<Field3D>(species["pressure"]), pressure_floor, low_p_source_scale);
    ddt(P) += ddtP_low_p_source;
  }

  // Scale time derivatives
  if (state.isSet("scale_timederivs")) {
    ddt(P) *= get<Field3D>(state["scale_timederivs"]);
  }

  if (evolve_log) {
    ddt(logP) = ddt(P) / P;
  }

#if CHECKLEVEL >= 1
  for (const auto& i : P.getRegion("RGN_NOBNDRY")) {
    if (!std::isfinite(ddt(P)[i])) {
      throw BoutException("ddt(P{}) non-finite at {}. Sp={}\n", name, i, Sp[i]);
    }
  }
#endif

  if (diagnose) {
    // Save flows of energy if they are set

    if (species.isSet("energy_flow_xlow")) {
      flow_xlow = get<Field3D>(species["energy_flow_xlow"]);
    }
    if (species.isSet("energy_flow_ylow")) {
      flow_ylow += get<Field3D>(species["energy_flow_ylow"]);
    }
  }
}

void EvolvePressure::outputVars(Options& state) {
  // Normalisations
  auto Nnorm = get<BoutReal>(state["Nnorm"]);
  auto Tnorm = get<BoutReal>(state["Tnorm"]);
  auto Omega_ci = get<BoutReal>(state["Omega_ci"]);
  auto rho_s0 = get<BoutReal>(state["rho_s0"]);

  BoutReal Pnorm = SI::qe * Tnorm * Nnorm; // Pressure normalisation

  if (evolve_log) {
    state[std::string("P") + name].force(P);
  }

  state[std::string("P") + name].setAttributes({{"time_dimension", "t"},
                                                {"units", "Pa"},
                                                {"conversion", Pnorm},
                                                {"standard_name", "pressure"},
                                                {"long_name", name + " pressure"},
                                                {"species", name},
                                                {"source", "evolve_pressure"}});

  if (diagnose) {
    set_with_attrs(state[std::string("T") + name], T,
                   {{"time_dimension", "t"},
                    {"units", "eV"},
                    {"conversion", Tnorm},
                    {"standard_name", "temperature"},
                    {"long_name", name + " temperature"},
                    {"species", name},
                    {"source", "evolve_pressure"}});

    set_with_attrs(state[std::string("ddt(P") + name + std::string(")")], ddt(P),
                   {{"time_dimension", "t"},
                    {"units", "Pa s^-1"},
                    {"conversion", Pnorm * Omega_ci},
                    {"long_name", std::string("Rate of change of ") + name + " pressure"},
                    {"species", name},
                    {"source", "evolve_pressure"}});

    set_with_attrs(state[std::string("SP") + name], Sp,
                   {{"time_dimension", "t"},
                    {"units", "Pa s^-1"},
                    {"conversion", Pnorm * Omega_ci},
                    {"standard_name", "pressure source"},
                    {"long_name", name + " pressure source"},
                    {"species", name},
                    {"source", "evolve_pressure"}});

    set_with_attrs(state[std::string("P") + name + std::string("_src")], final_source,
                   {{"time_dimension", "t"},
                    {"units", "Pa s^-1"},
                    {"conversion", Pnorm * Omega_ci},
                    {"standard_name", "pressure source"},
                    {"long_name", name + " pressure source"},
                    {"species", name},
                    {"source", "evolve_pressure"}});

    if (p_div_v) {
      if (E_PdivV.isAllocated()) {
        set_with_attrs(state["E" + name + "_PdivV"], E_PdivV,
                       {{"time_dimension", "t"},
                        {"units", "W / m^-3"},
                        {"conversion", Pnorm * Omega_ci},
                        {"standard_name", "energy source"},
                        {"long_name", name + " energy source due to pressure gradient"},
                        {"species", name},
                        {"source", "evolve_pressure"}});
      }
    } else {
      if (E_VgradP.isAllocated()) {
        set_with_attrs(state["E" + name + "_VgradP"], E_VgradP,
                       {{"time_dimension", "t"},
                        {"units", "W / m^-3"},
                        {"conversion", Pnorm * Omega_ci},
                        {"standard_name", "energy source"},
                        {"long_name", name + " energy source due to pressure gradient"},
                        {"species", name},
                        {"source", "evolve_pressure"}});
      }
    }

    if (flow_xlow.isAllocated()) {
      set_with_attrs(
          state[fmt::format("ef{}_tot_xlow", name)], flow_xlow,
          {{"time_dimension", "t"},
           {"units", "W"},
           {"conversion", rho_s0 * SQ(rho_s0) * Pnorm * Omega_ci},
           {"standard_name", "power"},
           {"long_name", name + " power through X cell face. Note: May be incomplete."},
           {"species", name},
           {"source", "evolve_pressure"}});
    }
    if (flow_ylow.isAllocated()) {
      set_with_attrs(
          state[fmt::format("ef{}_tot_ylow", name)], flow_ylow,
          {{"time_dimension", "t"},
           {"units", "W"},
           {"conversion", rho_s0 * SQ(rho_s0) * Pnorm * Omega_ci},
           {"standard_name", "power"},
           {"long_name", name + " power through Y cell face. Note: May be incomplete."},
           {"species", name},
           {"source", "evolve_pressure"}});
    }

    if (flow_ylow_advection.isAllocated()) {
      set_with_attrs(state[fmt::format("ef{}_adv_ylow", name)], flow_ylow_advection,
                     {{"time_dimension", "t"},
                      {"units", "W"},
                      {"conversion", rho_s0 * SQ(rho_s0) * Pnorm * Omega_ci},
                      {"standard_name", "power"},
                      {"long_name", name + " advected energy flow through Y cell face."},
                      {"species", name},
                      {"source", "evolve_pressure"}});
    }

    if (flow_ylow_viscous_heating.isAllocated()) {
      set_with_attrs(
          state[fmt::format("ef{}_visc_heat_ylow", name)], flow_ylow_viscous_heating,
          {{"time_dimension", "t"},
           {"units", "W"},
           {"conversion", rho_s0 * SQ(rho_s0) * Pnorm * Omega_ci},
           {"standard_name", "power"},
           {"long_name",
            name
                + " energy flow due to Lax flux numerical viscosity through Y cell face"},
           {"species", name},
           {"source", "evolve_pressure"}});
    }

    if (numerical_viscous_heating) {
      set_with_attrs(
          state[std::string("E") + name + std::string("_nvh")], Sp_nvh * 3 / .2,
          {{"time_dimension", "t"},
           {"units", "W"},
           {"conversion", Pnorm * Omega_ci},
           {"standard_name", "energy source"},
           {"long_name", name + " energy source from numerical viscous heating"},
           {"species", name},
           {"source", "evolve_pressure"}});
    }
    if (diagnose_terms) {
      // Per-term diagnostics for ddt(P). Useful for identifying which term
      // drives P negative in e.g. PFR regions.
      if (ddtP_ExB.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_ExB_") + name], ddtP_ExB,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " ExB pressure term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_advection.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_advection_") + name], ddtP_advection,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " parallel advection pressure term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_PdivV.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_PdivV_") + name], ddtP_PdivV,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " PdivV pressure term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_VgradP.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_VgradP_") + name], ddtP_VgradP,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " VgradP pressure term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_cond.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_cond_") + name], ddtP_cond,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " parallel conduction pressure term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_damp_NT.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_damp_NT_") + name], ddtP_damp_NT,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " NT damping pressure term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_low_p_source.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_low_p_source_") + name], ddtP_low_p_source,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " adaptive low-P source term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_low_n_diff_perp.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_low_n_diff_perp_") + name], ddtP_low_n_diff_perp,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " adaptive low-n diffuse term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
      if (ddtP_low_p_diff_perp.isAllocated()) {
        set_with_attrs(state[std::string("ddtP_low_p_diff_perp_") + name], ddtP_low_p_diff_perp,
                       {{"time_dimension", "t"}, {"units", "Pa s^-1"},
                        {"conversion", Pnorm * Omega_ci},
                        {"long_name", name + " adaptive low-P diffuse term"},
                        {"species", name}, {"source", "evolve_pressure"}});
      }
	}
  }
}

void EvolvePressure::precon(const Options& state, BoutReal gamma) {
  // Note: This preconditioner handles the conduction term in the
  // equation. That term is actually calculated elsewhere (e.g.,
  // BraginskiiConduction), so doing the preconditioning here breaks
  // encapsulation to some extent. However, it is not expected that
  // there will be any need to change the preconditioner in the near
  // future and the current preconditioner should work well-enough for
  // any implementation of conduction. Therefore, we are just leaving
  // this as is for now.
  if (!(enable_precon and thermal_conduction)) {
    return; // Disabled
  }

  static std::unique_ptr<InvertParDiv> inv;
  if (!inv) {
    // Initialise parallel inversion class
    inv = InvertParDiv::create();
    inv->setCoefA(1.0);
  }
  const auto& species = state["species"][name];
  const Field3D N = get<Field3D>(species["density"]);

  // Set the coefficient in Div_par( B * Grad_par )
  Field3D coef = -(2. / 3) * gamma * get<Field3D>(species["kappa_par"])
                 / softFloor(N, density_floor);

  if (state.isSet("scale_timederivs")) {
    coef *= get<Field3D>(state["scale_timederivs"]);
  }

  inv->setCoefB(coef);
  Field3D dT = ddt(P);
  dT.applyBoundary("neumann");
  ddt(P) = inv->solve(dT);
}
