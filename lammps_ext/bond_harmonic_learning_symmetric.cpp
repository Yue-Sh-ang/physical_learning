/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "bond_harmonic_learning_symmetric.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondHarmonicLearningSymmetric::BondHarmonicLearningSymmetric(LAMMPS *_lmp) : Bond(_lmp)
{
  born_matrix_enable = 1;
}

/* ---------------------------------------------------------------------- */

BondHarmonicLearningSymmetric::~BondHarmonicLearningSymmetric()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(e_t);
    memory->destroy(eta);
    memory->destroy(alpha);
    memory->destroy(vmin);
    memory->destroy(train);
    memory->destroy(mode);
    memory->destroy(phase);
    memory->destroy(target);
  }
}

/* ---------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::compute(int eflag, int vflag)
{
  // "_c" refers to clamped state; "_f" to free state;
  // "_sc" refers to symmetric clamped state; "_sf" to symmetric free state.
  int i1_c, i2_c, i1_f, i2_f, i1_sc, i2_sc, i1_sf, i2_sf, n, type;
  double delx_c, dely_c, delz_c,
         delx_f, dely_f, delz_f,
         delx_sc, dely_sc, delz_sc,
         delx_sf, dely_sf, delz_sf,
         ebond_c, fbond_c, ebond_f, fbond_f,
         ebond_sc, fbond_sc, ebond_sf, fbond_sf;
  double rsq_c, r_c, dr_c, rk_c,
         rsq_f, r_f, dr_f, rk_f,
         rsq_sc, r_sc, dr_sc, rk_sc,
         rsq_sf, r_sf, dr_sf, rk_sf,
         dk, dl;
  double e_f, e_c, lfac = 1, lfac_s = 1;

  ebond_c = 0.0; ebond_f = 0.0;
  ebond_sc = 0.0; ebond_sf = 0.0;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1_c = bondlist[n][0];
    i2_c = bondlist[n][1];
    type = bondlist[n][2];
    i1_f = i1_c + 1;
    i2_f = i2_c + 1;
    i1_sc = i1_c + 2;
    i2_sc = i2_c + 2;
    i1_sf = i1_c + 3;
    i2_sf = i2_c + 3;

    delx_c = x[i1_c][0] - x[i2_c][0];
    dely_c = x[i1_c][1] - x[i2_c][1];
    delz_c = x[i1_c][2] - x[i2_c][2];
    rsq_c = delx_c * delx_c + dely_c * dely_c + delz_c * delz_c;
    r_c = sqrt(rsq_c);

    delx_f = x[i1_f][0] - x[i2_f][0];
    dely_f = x[i1_f][1] - x[i2_f][1];
    delz_f = x[i1_f][2] - x[i2_f][2];
    rsq_f = delx_f * delx_f + dely_f * dely_f + delz_f * delz_f;
    r_f = sqrt(rsq_f);

    delx_sc = x[i1_sc][0] - x[i2_sc][0];
    dely_sc = x[i1_sc][1] - x[i2_sc][1];
    delz_sc = x[i1_sc][2] - x[i2_sc][2];
    rsq_sc = delx_sc * delx_sc + dely_sc * dely_sc + delz_sc * delz_sc;
    r_sc = sqrt(rsq_sc);

    delx_sf = x[i1_sf][0] - x[i2_sf][0];
    dely_sf = x[i1_sf][1] - x[i2_sf][1];
    delz_sf = x[i1_sf][2] - x[i2_sf][2];
    rsq_sf = delx_sf * delx_sf + dely_sf * dely_sf + delz_sf * delz_sf;
    r_sf = sqrt(rsq_sf);

    if (target[type] == 1) { // target

      // compute applied strain for original state
      e_f = (r_f - r0[type]) / r0[type];
      e_c = e_f + eta[type] * (e_t[type] - e_f);

      // compute length scale factor
      lfac = 1 + phase[type] * e_c;
      lfac_s = 1 + phase[type] * e_t[type];
      
    }

    else if (target[type] == 2) { // source (target for symmetric state)

      // compute applied strain for symmetric state
      e_f = (r_sf - r0[type]) / r0[type];
      e_c = e_f + eta[type] * (e_t[type] - e_f);

      // compute length scale factor
      lfac_s = 1 + phase[type] * e_c;
      lfac = 1 + phase[type] * e_t[type];

    }

    else { // regular bond
      lfac = 1;
      lfac_s = 1;
    }

    dr_c = r_c - r0[type] * lfac;
    rk_c = k[type] * dr_c;

    dr_sc = r_sc - r0[type] * lfac_s;
    rk_sc = k[type] * dr_sc;
    
    dr_f = r_f - r0[type] * lfac;
    rk_f = k[type] * dr_f;

    dr_sf = r_sf - r0[type] * lfac_s;
    rk_sf = k[type] * dr_sf;

    if (target[type] == 1) rk_f = 0.0;
    else if (target[type] == 2) rk_sf = 0.0;

    // force & energy

    if (r_c > 0.0) {
      fbond_c = -2.0 * rk_c / r_c;
    }
    else {
      fbond_c = 0.0;
    }
    if (r_f > 0.0) {
      fbond_f = -2.0 * rk_f / r_f;
    }
    else {
      fbond_f = 0.0;
    }
    if (r_sc > 0.0) {
      fbond_sc = -2.0 * rk_sc / r_sc;
    }
    else {
      fbond_sc = 0.0;
    }
    if (r_sf > 0.0) {
      fbond_sf = -2.0 * rk_sf / r_sf;
    }
    else {
      fbond_sf = 0.0;
    }

    if (eflag) {

      ebond_c = rk_c * dr_c;
      ebond_f = rk_f * dr_f;
      ebond_sc = rk_sc * dr_sc;
      ebond_sf = rk_sf * dr_sf;

    }

    // apply force to each atom

    if (newton_bond || i1_c < nlocal) {

      f[i1_c][0] += delx_c * fbond_c;
      f[i1_c][1] += dely_c * fbond_c;
      f[i1_c][2] += delz_c * fbond_c;

      f[i1_f][0] += delx_f * fbond_f;
      f[i1_f][1] += dely_f * fbond_f;
      f[i1_f][2] += delz_f * fbond_f;

      f[i1_sc][0] += delx_sc * fbond_sc;
      f[i1_sc][1] += dely_sc * fbond_sc;
      f[i1_sc][2] += delz_sc * fbond_sc;

      f[i1_sf][0] += delx_sf * fbond_sf;
      f[i1_sf][1] += dely_sf * fbond_sf;
      f[i1_sf][2] += delz_sf * fbond_sf;

    }

    if (newton_bond || i2_c < nlocal) {

      f[i2_c][0] -= delx_c * fbond_c;
      f[i2_c][1] -= dely_c * fbond_c;
      f[i2_c][2] -= delz_c * fbond_c;

      f[i2_f][0] -= delx_f * fbond_f;
      f[i2_f][1] -= dely_f * fbond_f;
      f[i2_f][2] -= delz_f * fbond_f;

      f[i2_sc][0] -= delx_sc * fbond_sc;
      f[i2_sc][1] -= dely_sc * fbond_sc;
      f[i2_sc][2] -= delz_sc * fbond_sc;

      f[i2_sf][0] -= delx_sf * fbond_sf;
      f[i2_sf][1] -= dely_sf * fbond_sf;
      f[i2_sf][2] -= delz_sf * fbond_sf;

    }

    if (evflag) {
      ev_tally(i1_c, i2_c, nlocal, newton_bond, ebond_c, fbond_c, delx_c, dely_c, delz_c);
      ev_tally(i1_f, i2_f, nlocal, newton_bond, ebond_f, fbond_f, delx_f, dely_f, delz_f);
      ev_tally(i1_sc, i2_sc, nlocal, newton_bond, ebond_sc, fbond_sc, delx_sc, dely_sc, delz_sc);
      ev_tally(i1_sf, i2_sf, nlocal, newton_bond, ebond_sf, fbond_sf, delx_sf, dely_sf, delz_sf);
    }

    // Update stiffness for next integration step
    if (train[type] == 2) {

      if (mode[type] == 1) { // directed aging
        dk = -2.0 * alpha[type]/eta[type] * (rk_c * dr_c + rk_sc * dr_sc);
        k[type] += dk;
      }

      else { // coupled learning
        dk = alpha[type]/eta[type] * (dr_f * dr_f - dr_c * dr_c + dr_sf * dr_sf - dr_sc * dr_sc);
        k[type] += dk;
      }

      if (k[type] < vmin[type]) k[type] = vmin[type];

    }
  }
}

/* ---------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(k, np1, "bond:k");
  memory->create(r0, np1, "bond:r0");
  memory->create(e_t, np1, "bond:e_t");
  memory->create(eta, np1, "bond:eta");
  memory->create(alpha, np1, "bond:alpha");
  memory->create(vmin, np1, "bond:vmin");
  memory->create(train, np1, "bond:train");
  memory->create(mode, np1, "bond:mode");
  memory->create(phase, np1, "bond:phase");
  memory->create(target, np1, "bond:target");

  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::coeff(int narg, char **arg)
{
  if (narg != 11) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  double k_one = utils::numeric(FLERR, arg[1], false, lmp);
  double r0_one = utils::numeric(FLERR, arg[2], false, lmp);
  double e_t_one = utils::numeric(FLERR, arg[3], false, lmp);
  double eta_one = utils::numeric(FLERR, arg[4], false, lmp);
  double alpha_one = utils::numeric(FLERR, arg[5], false, lmp);
  double vmin_one = utils::numeric(FLERR, arg[6], false, lmp);

  int train_one = utils::inumeric(FLERR, arg[7], false, lmp);
  int mode_one = utils::inumeric(FLERR, arg[8], false, lmp);
  int phase_one = utils::inumeric(FLERR, arg[9], false, lmp);
  int target_one = utils::inumeric(FLERR, arg[10], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    r0[i] = r0_one;
    e_t[i] = e_t_one;
    eta[i] = eta_one;
    alpha[i] = alpha_one;
    vmin[i] = vmin_one;
    train[i] = train_one;
    mode[i] = mode_one;
    phase[i] = phase_one;
    target[i] = target_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondHarmonicLearningSymmetric::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::write_restart(FILE *fp)
{
  fwrite(&k[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&r0[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&e_t[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&eta[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&alpha[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&vmin[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&train[1], sizeof(int), atom->nbondtypes, fp);
  fwrite(&mode[1], sizeof(int), atom->nbondtypes, fp);
  fwrite(&phase[1], sizeof(int), atom->nbondtypes, fp);
  fwrite(&target[1], sizeof(int), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &k[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &r0[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &e_t[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &eta[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &alpha[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &vmin[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &train[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &mode[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &phase[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &target[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&k[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&r0[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&e_t[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&eta[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&alpha[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&vmin[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&train[1], atom->nbondtypes, MPI_INT, 0, world);
  MPI_Bcast(&mode[1], atom->nbondtypes, MPI_INT, 0, world);
  MPI_Bcast(&phase[1], atom->nbondtypes, MPI_INT, 0, world);
  MPI_Bcast(&target[1], atom->nbondtypes, MPI_INT, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++) fprintf(fp, "%d %.15g %.15g %.15g %.15g %.15g %.15g %d %d %d %d\n", i, k[i], r0[i], e_t[i], eta[i], alpha[i], vmin[i], train[i], mode[i], phase[i], target[i]);
}

/* ---------------------------------------------------------------------- */

double BondHarmonicLearningSymmetric::single(int type, double rsq, int /*i*/, int /*j*/, double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -2.0 * rk / r;
  return 4.0 * rk * dr; // scale by 4 for roughly 4 times atoms
}

/* ---------------------------------------------------------------------- */

void BondHarmonicLearningSymmetric::born_matrix(int type, double rsq, int /*i*/, int /*j*/, double &du, double &du2)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  du2 = 0.0;
  du = 0.0;
  du2 = 2 * k[type];
  if (r > 0.0) du = du2 * dr;
}

/* ----------------------------------------------------------------------
   return ptr to internal members upon request
------------------------------------------------------------------------ */

void *BondHarmonicLearningSymmetric::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "k") == 0) return (void *) k;
  if (strcmp(str, "r0") == 0) return (void *) r0;
  if (strcmp(str, "e_t") == 0) return (void *) e_t;
  if (strcmp(str, "eta") == 0) return (void *) eta;
  if (strcmp(str, "alpha") == 0) return (void *) alpha;
  if (strcmp(str, "vmin") == 0) return (void *) vmin;
  if (strcmp(str, "train") == 0) return (void *) train;
  if (strcmp(str, "mode") == 0) return (void *) mode;
  if (strcmp(str, "phase") == 0) return (void *) phase;
  if (strcmp(str, "target") == 0) return (void *) target;
  return nullptr;
}
