#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <bout/bout_types.hxx>
#include <bout/boutexception.hxx>
#include <bout/constants.hxx>
#include <bout/coordinates.hxx>
#include <bout/difops.hxx>
#include <bout/field2d.hxx>
#include <bout/field3d.hxx>
#include <bout/field_factory.hxx>
#include <bout/fv_ops.hxx>
#include <bout/globals.hxx>
#include <bout/mesh.hxx>
#include <bout/options.hxx>
#include <bout/output.hxx>
#include <bout/paralleltransform.hxx>
#include <bout/region.hxx>
#include <bout/solver.hxx>
#include <bout/sys/range.hxx>
#include <bout/utils.hxx>
#include <fmt/format.h>

#include "../include/braginskii_conduction.hxx"
#include "../include/component.hxx"
#include "../include/div_ops.hxx"
#include "../include/hermes_utils.hxx"
#include "../include/permissions.hxx"

using bout::globals::mesh;

namespace {
/// Unlimited conductive flux through the y face between cells a (lower y) and b
/// (upper y), in field-aligned coordinates. Identical to the face fluxes of
/// Div_par_K_Grad_par_mod for each conduction_method.
BoutReal face_flux_unlimited(const Field3D& K, const Field3D& f, const Coordinates* coord,
                             const Ind3D& a, const Ind3D& b,
                             bout::ConductionMethod method) {
  const auto& J = coord->J();
  const auto& g_22 = coord->g_22();
  const auto& dy = coord->dy();
  switch (method) {
  case bout::ConductionMethod::Original: {
    const BoutReal c = 0.5 * (K[a] + K[b]);
    const BoutReal Jf = 0.5 * (J[a] + J[b]);
    const BoutReal g_22f = 0.5 * (g_22[a] + g_22[b]);
    return c * Jf * 2. * (f[b] - f[a]) / ((dy[a] + dy[b]) * g_22f);
  }
  case bout::ConductionMethod::ProductJK: {
    const BoutReal cJ = 0.5 * (K[a] * J[a] + K[b] * J[b]);
    const BoutReal g_22f = 0.5 * (g_22[a] + g_22[b]);
    return cJ * 2. * (f[b] - f[a]) / ((dy[a] + dy[b]) * g_22f);
  }
  case bout::ConductionMethod::Harmonic: {
    const BoutReal cond_a = K[a] * J[a] / (g_22[a] * dy[a]);
    const BoutReal cond_b = K[b] * J[b] / (g_22[b] * dy[b]);
    const BoutReal denom = cond_a + cond_b;
    const BoutReal C_edge = (std::abs(denom) > std::numeric_limits<BoutReal>::epsilon())
                                ? 2.0 * cond_a * cond_b / denom
                                : 0.0;
    return C_edge * (f[b] - f[a]);
  }
  }
  throw BoutException("Unknown conduction_method");
}

/// Parallel heat conduction with the free-streaming flux limit applied on each
/// y face (option kappa_limit_faces), rather than to the cell-centred kappa.
///
/// On the face between cells a and b the unlimited flux F is computed with the
/// same stencil as Div_par_K_Grad_par_mod (selected conduction_method), and
/// converted to a parallel heat flux density q_SH = F sqrt(g_22) / J (face
/// averages of J and g_22). The free-streaming flux is taken from the hotter
/// of the two cells,
///   q_fl = alpha * P_hot * sqrt(T_hot / AA),
/// and the face flux is limited as F / (1 + |q_SH| / q_fl), the same harmonic
/// combination as the cell-centred limiter.
///
/// Why: the cell-centred limiter makes kappa vary strongly from cell to cell,
/// because each cell's limiting factor depends on its own gradient and
/// temperature. With a shifted metric that kappa is shifted to field-aligned
/// coordinates (an FFT in z) before the face fluxes are formed, and a spiky
/// kappa can ring to negative values on some faces; a negative face
/// conductivity drives heat from the colder cell into the hotter one. Here
/// only the unlimited K, T and P are shifted, and the limiting factor lies in
/// (0, 1], so each face flux keeps the sign of the unlimited face flux. This
/// removes the ringing caused by shifting an already limited kappa; the shift
/// of the unlimited kappa is the same as without a limiter. The cap
/// uses the face's own gradient, and q_fl is that of the side the heat comes
/// from (cf. the upwind saturated flux of Mignone et al. 2012, ApJS 198, 7,
/// Appendix A).
///
/// Fluxes through target faces are not included (the sheath removes that
/// heat separately). As alpha -> infinity, for positive hot-side pressure and
/// temperature, the face fluxes tend to those of
/// Div_par_K_Grad_par_mod(K, T, flow_ylow, false, method).
///
/// Each face flux is applied to both cells with the same value. Interior
/// faces are evaluated by both cells from identical field-aligned samples.
/// At a poloidal branch cut the guard cells are shifted with a zShift offset
/// by ShiftAngle, so the two cells see the same rows at different toroidal
/// positions; with a fractional offset the FFT interpolates, and because the
/// limiter is nonlinear the two evaluations would not have the same sum over
/// z. There the cell below the cut takes the flux computed by the cell above
/// it (its lower face), shifted back to standard coordinates, communicated
/// and shifted into its own frame. Shifts preserve the sum over z, so energy
/// is conserved.
///
/// @param face_factor  Output: mean of the limiting factors 1/(1 + |q_SH|/q_fl)
///                     of each cell's faces, in standard (not field-aligned)
///                     coordinates, clipped to [0, 1] (the inverse shift of a
///                     cell-to-cell varying factor can ring outside it). Valid
///                     in RGN_NOBNDRY. Used only for the kappa_par output.
Field3D Div_par_K_Grad_par_facelimited(const Field3D& Kin, const Field3D& Tin,
                                       const Field3D& Pin, BoutReal AA, BoutReal alpha,
                                       bout::ConductionMethod method,
                                       Field3D& flow_ylow, Field3D& face_factor) {
  Mesh* fmesh = Kin.getMesh();
  const Coordinates* coord = Tin.getCoordinates();

  const Field3D K = toFieldAligned(Kin, "RGN_NOX");
  const Field3D T = toFieldAligned(Tin, "RGN_NOX");
  const Field3D P = toFieldAligned(Pin, "RGN_NOX");

  Field3D result{zeroFrom(T)};
  flow_ylow = zeroFrom(T);
  Field3D flux_lower{zeroFrom(T)}; // Limited flux through each cell's lower face
  Field3D flux_upper{zeroFrom(T)}; // Limited flux through each cell's upper face
  Field3D factor{zeroFrom(T)};     // Mean limiting factor of each cell's faces
  BOUT_FOR(i, factor.getRegion("RGN_ALL")) { factor[i] = 1.0; }

  // Limited flux through the face between a (lower y) and b (upper y).
  // limit is set to the limiting factor 1/(1 + |q_SH|/q_fl)
  auto face_flux = [&](const Ind3D& a, const Ind3D& b, BoutReal& limit) {
    const BoutReal flux = face_flux_unlimited(K, T, coord, a, b, method);
    const BoutReal Jf = 0.5 * (coord->J()[a] + coord->J()[b]);
    const BoutReal g_22f = 0.5 * (coord->g_22()[a] + coord->g_22()[b]);
    const BoutReal q_SH = flux * sqrt(g_22f) / Jf; // Parallel heat flux density

    const Ind3D hot = (T[b] > T[a]) ? b : a;
    const BoutReal T_hot = std::max(T[hot], 0.0);
    const BoutReal q_fl = alpha * std::max(P[hot], 0.0) * sqrt(T_hot / AA);

    limit = 1. / (1. + std::abs(q_SH) / std::max(q_fl, 1e-10));
    return flux * limit;
  };

  BOUT_FOR(i, result.getRegion("RGN_NOBNDRY")) {
    BoutReal factor_sum = 0.0;
    int nface = 0;

    // Upper face, unless it is a target
    if (fmesh->periodicY(i.x()) || !fmesh->lastY(i.x()) || (i.y() != fmesh->yend)) {
      BoutReal f = 1.0;
      flux_upper[i] = face_flux(i, i.yp(), f);
      result[i] += flux_upper[i] / (coord->dy()[i] * coord->J()[i]);
      factor_sum += f;
      ++nface;
    }
    // Lower face, unless it is a target
    if (fmesh->periodicY(i.x()) || !fmesh->firstY(i.x()) || (i.y() != fmesh->ystart)) {
      BoutReal f = 1.0;
      flux_lower[i] = face_flux(i.ym(), i, f);
      result[i] -= flux_lower[i] / (coord->dy()[i] * coord->J()[i]);
      flow_ylow[i] = -flux_lower[i] * coord->dx()[i] * coord->dz()[i];
      factor_sum += f;
      ++nface;
    }
    if (nface > 0) {
      factor[i] = factor_sum / nface;
    }
  }

  // Branch cuts: replace the upper-face flux of the last cell before the cut
  // by the lower-face flux of the first cell after it. All processors take
  // part in the communication; only cells with a branch cut use the result.
  if (dynamic_cast<const ShiftedMetric*>(&coord->getParallelTransform()) != nullptr) {
    Field3D flux_lower_std = fromFieldAligned(flux_lower);
    fmesh->communicate(flux_lower_std);
    const Field3D flux_from_above = toFieldAligned(flux_lower_std, "RGN_NOX");
    for (int x = fmesh->xstart; x <= fmesh->xend; x++) {
      if (!fmesh->hasBranchCutUpper(x).first) {
        continue;
      }
      for (int z = 0; z < fmesh->LocalNz; z++) {
        const auto i = indexAt(result, x, fmesh->yend, z);
        result[i] += (flux_from_above[i.yp()] - flux_upper[i])
                     / (coord->dy()[i] * coord->J()[i]);
      }
    }
  }

  result = fromFieldAligned(result, "RGN_NOBNDRY");
  flow_ylow = fromFieldAligned(flow_ylow);
  face_factor = fromFieldAligned(factor);
  BOUT_FOR(i, face_factor.getRegion("RGN_NOBNDRY")) {
    face_factor[i] = std::clamp(face_factor[i], 0.0, 1.0);
  }

#if CHECK > 2
  bout::checkFinite(result);
#endif

  return result;
}
} // namespace

BraginskiiConduction::BraginskiiConduction(const std::string& name, Options& alloptions,
                                           Solver*)
    : NamedComponent(name, {readOnly("species:{sp}:{input_vars}"),
                            readIfSet("fields:Apar_flutter"),
                            readWrite("species:{sp}:{output_vars}")}) {

  // Get settings for each species
  for (const auto& kv : alloptions.getChildren()) {
    auto& options = alloptions[kv.first];
    // Check if the component is a species which undergoes energy/pressure evolution
    if (options.isValue() || !options.isSet("type")
        || (options["type"].as<std::string>().find("evolve_pressure") == std::string::npos
            && options["type"].as<std::string>().find("evolve_energy")
                   == std::string::npos)) {
      continue;
    }
    if (!options["thermal_conduction"]
             .doc("Include parallel heat conduction?")
             .withDefault<bool>(true)) {
      continue;
    }
    const std::string name = kv.first;

    all_diagnose[name] = options["diagnose"]
                             .doc("Save additional output diagnostics")
                             .withDefault<bool>(false);

    BoutReal default_kappa; // default conductivity, changes depending on species
    switch (identifySpeciesType(name)) {
    case SpeciesType::ion:
      default_kappa = 3.9;
      break;
    case SpeciesType::electron:
      // Hermes-3 electron collision time is in Fitzpatrick form (3.187 in
      // https://farside.ph.utexas.edu/teaching/plasma/Plasma/node41.html) This means that
      // the Braginskii prefactor of 3.16 needs to be divided by sqrt(2) to be consistent.
      default_kappa = 3.16 / sqrt(2);
      break;
    case SpeciesType::neutral:
      default_kappa = 2.5;
      break;
    default:
      throw BoutException("Unhandled species type in default_kappa switch");
    }

    all_kappa_coefficient[name] =
        options["kappa_coefficient"]
            .doc("Numerical coefficient in parallel heat conduction. Default is "
                 "3.16/sqrt(2) for electrons, 2.5 for neutrals and 3.9 otherwise")
            .withDefault(default_kappa);

    all_kappa_limit_alpha[name] =
        options["kappa_limit_alpha"]
            .doc("Flux limiter factor. < 0 means no limit. Typical is 0.2 "
                 "for electrons, 1 for ions.")
            .withDefault(-1.0);
    all_kappa_limit_faces[name] =
        options["kappa_limit_faces"]
            .doc("Apply the flux limiter to the heat flux on each parallel cell face, "
                 "with the free-streaming flux of the hotter neighbouring cell, "
                 "instead of to the cell-centred kappa. Not available for FCI or "
                 "use_petsc_operator.")
            .withDefault<bool>(false);
    all_conduction_collisions_mode[name] =
        options["conduction_collisions_mode"]
            .doc("Can be multispecies: all collisions, or "
                 "braginskii: self collisions and ie")
            .withDefault<std::string>("multispecies");
  }

  std::vector<std::string> coll_types;

  substitutePermissions("input_vars", {"AA", "density", "pressure", "temperature"});
  substitutePermissions("output_vars",
                        {"energy_source", "kappa_par", "energy_flow_ylow"});
  std::vector<std::string> species;
  for (const auto& [sp, mode] : all_conduction_collisions_mode) {
    species.push_back(sp);
    if (mode == "braginskii" and identifySpeciesType(sp) != SpeciesType::neutral) {
      setPermissions(readIfSet(
          fmt::format("species:{}:collision_frequencies:{}_{}_coll", sp, sp, sp)));
    } else if (mode == "multispecies") {
      setPermissions(readIfSet(fmt::format("species:{}:collision_frequencies:{}_{}_coll",
                                           sp, sp, "{all_species}")));
      setPermissions(readIfSet(fmt::format("species:{}:collision_frequencies:{}_{}_cx",
                                           sp, sp, "{all_species}")));
    } else if (mode == "AFN" and identifySpeciesType(sp) == SpeciesType::neutral) {
      setPermissions(readIfSet(fmt::format("species:{}:collision_frequencies:{}_{}_cx",
                                           sp, sp, "{positive_ions}")));
      setPermissions(readIfSet(fmt::format("species:{}:collision_frequencies:{}_{}_iz",
                                           sp, sp, "{positive_ions}")));
    }
  }
  substitutePermissions("sp", species);

  conduction_method = alloptions["conduction_method"].withDefault(conduction_method);

  use_petsc_operator =
      alloptions["use_petsc_operator"]
          .doc("Use Petsc for parallel heat conduction. Grids need the weights.")
          .withDefault<bool>(false);

  if (use_petsc_operator) {

    if (!mesh->isFci()) {
      throw BoutException("Used Petsc operator for Braginskii conduction without Fci!");
    }
#if BOUT_HAS_PETSC
    PetscOperators ops(mesh);
    petsc_parallel = ops.getParallel();
#else
    throw BoutException("petsc_operator requires PETSc");
#endif
  }
}

void BraginskiiConduction::transform_impl(GuardedOptions& state) {

  for (auto& kv : state["species"].getChildren()) {
    const std::string& name = kv.first;
    // Skip any species for which energy is not being evolved
    if (all_conduction_collisions_mode.count(name) == 0) {
      continue;
    }
    /// Get the section containing this species
    auto species = state["species"][name];
    const std::string conduction_collisions_mode = all_conduction_collisions_mode[name];

    // Braginskii mode: plasma - self collisions and ei, neutrals - CX, IZ
    if (all_collision_names.count(name) == 0) { /// Calculate only once - at the beginning
      std::vector<std::string> collision_names;
      const auto species_type = identifySpeciesType(species.name());

      if (conduction_collisions_mode == "braginskii") {
        for (const auto& collision : species["collision_frequencies"].getChildren()) {

          std::string collision_name = collision.second.name();

          if (species_type == SpeciesType::neutral) {
            throw BoutException("\tBraginskii conduction collisions mode not available "
                                "for neutrals, choose multispecies or afn");
          } else if (species_type == SpeciesType::electron) {
            if ( /// Electron-electron collisions
                (collisionSpeciesMatch(collision_name, species.name(), "e", "coll",
                                       "exact"))) {
              collision_names.push_back(collision_name);
            }

          } else if (species_type == SpeciesType::ion) {
            if ( /// Self-collisions
                (collisionSpeciesMatch(collision_name, species.name(), species.name(),
                                       "coll", "exact"))) {
              collision_names.push_back(collision_name);
            }
          }
        }

        // Multispecies mode: all collisions and CX are included
      } else if (conduction_collisions_mode == "multispecies") {
        for (const auto& collision : species["collision_frequencies"].getChildren()) {

          const std::string collision_name = collision.second.name();

          if ( /// Charge exchange
              (collisionSpeciesMatch(collision_name, species.name(), "", "cx", "partial"))
              or
              /// Any collision (en, in, ee, ii, nn)
              (collisionSpeciesMatch(collision_name, species.name(), "", "coll",
                                     "partial"))) {
            collision_names.push_back(collision_name);
          }
        }

      } else if (conduction_collisions_mode == "afn") {
        for (const auto& collision : species["collision_frequencies"].getChildren()) {

          const std::string collision_name = collision.second.name();

          if (species_type != SpeciesType::neutral) {
            throw BoutException("\tAFN conduction collisions mode not available for ions "
                                "or electrons, choose braginskii or multispecies");
          }
          if ( /// Charge exchange
              (collisionSpeciesMatch(collision_name, species.name(), "+", "cx",
                                     "partial"))
              or
              /// Ionisation
              (collisionSpeciesMatch(collision_name, species.name(), "+", "iz",
                                     "partial"))) {
            collision_names.push_back(collision_name);
          }
        }

      } else {
        throw BoutException("\tConduction_collisions_mode incorrect", species.name());
      }

      if (collision_names.empty()) {
        throw BoutException("\tNo collisions found for {:s} in evolve_pressure for "
                            "selected collisions mode",
                            species.name());
      }

      /// Write chosen collisions to log file
      output_info.write("\t{:s} conduction collisionality mode: '{:s}' using ",
                        species.name(), conduction_collisions_mode);
      for (const auto& collision : collision_names) {
        output_info.write("{:s} ", collision);
      }

      output_info.write("\n");
      all_collision_names[name] = collision_names;
    }

    const auto& collision_names = all_collision_names[name];
    Field3D& nu = all_nu[name];
    Field3D& kappa_par = all_kappa_par[name];
    Field3D& flow_ylow_conduction = all_flow_ylow_conduction[name];
    const BoutReal kappa_coefficient = all_kappa_coefficient[name];
    const BoutReal kappa_limit_alpha = all_kappa_limit_alpha[name];
    const bool face_limited = (kappa_limit_alpha > 0.0) and all_kappa_limit_faces[name];

    /// Collect the collisionalities based on list of names
    nu = 0;
    for (const auto& collision_name : collision_names) {
      nu += GET_VALUE(Field3D, species["collision_frequencies"][collision_name]);
    }

    // Get updated pressure and temperature with boundary conditions
    // Note: Retain pressures which fall below zero
    // FIXME: Can I be sure this is the same as the value used in EvolvePressure::finally?
    // FIXME: We end up applying these operations twice: here and in
    // EvolvePressure::finally

    Field3D P = GET_VALUE(Field3D, species["pressure"]);
    // Only clear parallel slices when not Fci
    if (!P.isFci()) {
      P.clearParallelSlices();
    }
    const Field3D Pfloor = floor(P, 0.0); // Restricted to never go below zero
    const Field3D T = get<Field3D>(species["temperature"]);
    const Field3D N = get<Field3D>(species["density"]);

    // Calculate ion collision times
    const Field3D tau = 1. / softFloor(nu, 1e-10);
    const BoutReal AA = get<BoutReal>(species["AA"]); // Atomic mass

    // Parallel heat conduction
    // Braginskii expression for parallel conduction
    // kappa ~ n * v_th^2 * tau
    //
    // Note: Coefficient is slightly different for electrons (3.16) and ions (3.9)
    kappa_par = kappa_coefficient * Pfloor * tau / AA;

    if (face_limited) {
      // Limit is applied on the cell faces below
      if (P.isFci() or use_petsc_operator) {
        throw BoutException("{:s}: kappa_limit_faces is not implemented for FCI or "
                            "use_petsc_operator",
                            name);
      }
    } else if (kappa_limit_alpha > 0.0) {
      /*
       * Flux limiter, as used in SOLPS.
       *
       * Calculate the heat flux from Spitzer-Harm and flux limit
       *
       * Typical value of alpha ~ 0.2 for electrons
       *
       * R.Schneider et al. Contrib. Plasma Phys. 46, No. 1-2, 3 – 191 (2006)
       * DOI 10.1002/ctpp.200610001
       */

      // Spitzer-Harm heat flux
      const Field3D q_SH = kappa_par * Grad_par(T);
      // Free-streaming flux
      const Field3D q_fl = kappa_limit_alpha * N * T * sqrt(T / AA);

      // This results in a harmonic average of the heat fluxes
      kappa_par /= (1. + abs(q_SH / softFloor(q_fl, 1e-10)));
    }

    // Values of kappa on cell boundaries are needed for fluxes. Not all collision
    // frequencies are valid in the guard cells (those set by reactions are only
    // calculated in the domain), so take the guard cells from the neighbouring
    // processors rather than from nu.
    mesh->communicate(kappa_par);

    // Fci does not work with mesh->iterateBndryLowerY(), so set the boundaries differently
    if (P.isFci()) {

      kappa_par.applyParallelBoundary("parallel_neumann_o1");

    } else {
      for (RangeIterator r = mesh->iterateBndryLowerY(); !r.isDone(); r++) {
        for (int jz = 0; jz < mesh->LocalNz; jz++) {
          auto i = indexAt(kappa_par, r.ind, mesh->ystart, jz);
          auto im = i.ym();
          kappa_par[im] = kappa_par[i];
        }
      }
      for (RangeIterator r = mesh->iterateBndryUpperY(); !r.isDone(); r++) {
        for (int jz = 0; jz < mesh->LocalNz; jz++) {
          auto i = indexAt(kappa_par, r.ind, mesh->yend, jz);
          auto ip = i.yp();
          kappa_par[ip] = kappa_par[i];
        }
      }
    }

    // Note: Flux through boundary turned off, because sheath heat flux
    // is calculated and removed separately
    if (face_limited) {
      Field3D face_factor;
      add(species["energy_source"],
          Div_par_K_Grad_par_facelimited(kappa_par, T, Pfloor, AA, kappa_limit_alpha,
                                         conduction_method, flow_ylow_conduction,
                                         face_factor));
      add(species["energy_flow_ylow"], flow_ylow_conduction);

      // Output kappa (preconditioner, flutter, diagnostics): the unlimited
      // kappa times the mean limiting factor of the cell's faces, so
      // 0 <= kappa_par <= unlimited kappa. Only the domain cells are changed;
      // guard cells are then set as for the unlimited kappa, and the physical
      // x guards are copied from the neighbouring domain cell.
      BOUT_FOR(i, kappa_par.getRegion("RGN_NOBNDRY")) { kappa_par[i] *= face_factor[i]; }
      mesh->communicate(kappa_par);
      for (RangeIterator r = mesh->iterateBndryLowerY(); !r.isDone(); r++) {
        for (int jz = 0; jz < mesh->LocalNz; jz++) {
          auto i = indexAt(kappa_par, r.ind, mesh->ystart, jz);
          kappa_par[i.ym()] = kappa_par[i];
        }
      }
      for (RangeIterator r = mesh->iterateBndryUpperY(); !r.isDone(); r++) {
        for (int jz = 0; jz < mesh->LocalNz; jz++) {
          auto i = indexAt(kappa_par, r.ind, mesh->yend, jz);
          kappa_par[i.yp()] = kappa_par[i];
        }
      }
      for (int y = 0; y < mesh->LocalNy; y++) {
        for (int z = 0; z < mesh->LocalNz; z++) {
          if (mesh->firstX()) {
            for (int x = 0; x < mesh->xstart; x++) {
              kappa_par(x, y, z) = kappa_par(mesh->xstart, y, z);
            }
          }
          if (mesh->lastX()) {
            for (int x = mesh->xend + 1; x < mesh->LocalNx; x++) {
              kappa_par(x, y, z) = kappa_par(mesh->xend, y, z);
            }
          }
        }
      }
    }
    set(species["kappa_par"], kappa_par);

    if (face_limited) {
      // Conduction already added above
    } else if (use_petsc_operator) {
      // Petsc can be used to calculate the divergence of the heat flux
#if BOUT_HAS_PETSC
      add(species["energy_source"], petsc_parallel.Div_par_K_Grad_par(kappa_par, T));
#endif
    } else {
      add(species["energy_source"],
          Div_par_K_Grad_par_mod(kappa_par, T, flow_ylow_conduction, false,
                                 conduction_method));
      add(species["energy_flow_ylow"], flow_ylow_conduction);
    }
    if (state.isSection("fields") and IS_SET(state["fields:Apar_flutter"])) {
      // Magnetic flutter term. The operator splits into 4 pieces:
      // Div(k b b.Grad(T)) = Div(k b0 b0.Grad(T)) + Div(k d0 db.Grad(T))
      //                    + Div(k db b0.Grad(T)) + Div(k db db.Grad(T))
      // The first term is already calculated above.
      // Here we add the terms containing db
      const Field3D Apar_flutter = get<Field3D>(state["fields"]["Apar_flutter"]);
      Field3D db_dot_T = bracket(T, Apar_flutter, BRACKET_ARAKAWA);
      Field3D b0_dot_T = Grad_par(T);
      mesh->communicate(db_dot_T, b0_dot_T);
      db_dot_T.applyBoundary("neumann");
      b0_dot_T.applyBoundary("neumann");
      add(species["energy_source"],
          Div_par(kappa_par * db_dot_T)
              - Div_n_g_bxGrad_f_B_XZ(kappa_par, db_dot_T + b0_dot_T, Apar_flutter));
    }
  }
}

void BraginskiiConduction::outputVars(Options& state) {
  // Normalisations
  auto Nnorm = get<BoutReal>(state["Nnorm"]);
  auto Tnorm = get<BoutReal>(state["Tnorm"]);
  auto Omega_ci = get<BoutReal>(state["Omega_ci"]);
  auto rho_s0 = get<BoutReal>(state["rho_s0"]);

  const BoutReal Pnorm = SI::qe * Tnorm * Nnorm; // Pressure normalisation

  for (const auto& [name, diagnose] : all_diagnose) {
    if (!diagnose) {
      continue;
    }
    const Field3D& kappa_par = all_kappa_par[name];
    const Field3D& nu = all_nu[name];
    const Field3D& flow_ylow_conduction = all_flow_ylow_conduction[name];
    set_with_attrs(state[std::string("kappa_par_") + name], kappa_par,
                   {{"time_dimension", "t"},
                    {"units", "W / m / eV"},
                    {"conversion", (Pnorm * Omega_ci * SQ(rho_s0)) / Tnorm},
                    {"long_name", name + " heat conduction coefficient"},
                    {"species", name},
                    {"source", "evolve_pressure"}});

    set_with_attrs(state[std::string("K") + name + std::string("_cond")], nu,
                   {{"time_dimension", "t"},
                    {"units", "s^-1"},
                    {"conversion", Omega_ci},
                    {"long_name", "collision frequency for conduction"},
                    {"species", name},
                    {"source", "evolve_pressure"}});

    set_with_attrs(state[fmt::format("ef{}_cond_ylow", name)], flow_ylow_conduction,
                   {{"time_dimension", "t"},
                    {"units", "W"},
                    {"conversion", rho_s0 * SQ(rho_s0) * Pnorm * Omega_ci},
                    {"standard_name", "power"},
                    {"long_name", name + " conduction through Y cell face."},
                    {"species", name},
                    {"source", "evolve_pressure"}});
  }
}
