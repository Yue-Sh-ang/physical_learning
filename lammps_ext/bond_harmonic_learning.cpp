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

#include "bond_harmonic_learning.h"

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
inline double sign(double x) {
    return (x > 0) - (x < 0);
}
/* ---------------------------------------------------------------------- */

BondHarmonicLearning::BondHarmonicLearning(LAMMPS *_lmp) : Bond(_lmp)
{
  born_matrix_enable = 1;
}

/* ---------------------------------------------------------------------- */

BondHarmonicLearning::~BondHarmonicLearning()
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
    memory->destroy(v);
    memory->destroy(m);
    memory->destroy(beta1);
    memory->destroy(beta2);
    memory->destroy(t);
  }
}

/* ---------------------------------------------------------------------- */

void BondHarmonicLearning::compute(int eflag, int vflag)
{
  // "_c" refers to clamped state; "_f" to free state
  int i1_c, i2_c, i1_f, i2_f, n, type;
  double delx_c, dely_c, delz_c,
         delx_f, dely_f, delz_f,
         ebond_c, fbond_c, ebond_f, fbond_f;
  double rsq_c, r_c, dr_c, rk_c,
         rsq_f, r_f, dr_f, rk_f,
         dk, dl;
  double e_f, e_c, lfac = 1;
  double alpha_t;

  ebond_c = 0.0;
  ebond_f = 0.0;
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

    // clampled setting
    if (target[type]) {
      e_f = (r_f - r0[type]) / r0[type];
      
      if (eta[type]<0) {// overclamped
        e_c = e_f - eta[type] * sign(e_t[type] - e_f);
      }
      else{
        e_c = e_f + eta[type] * (e_t[type] - e_f);
      }

      // compute length scale factor
      lfac = 1 + phase[type] * e_c;
    }

    else {
      lfac = 1;
    }

    dr_c = r_c - r0[type] * lfac;
    rk_c = k[type] * dr_c;
    
    dr_f = r_f - r0[type];
    rk_f = k[type] * dr_f * (1 - target[type]);

    // force & energy

    if (r_c > 0.0) {
      fbond_c = -2.0 * rk_c / r_c;
    }
    else {
      fbond_c = 0.0;
    }

    if (eflag) ebond_c = rk_c * dr_c;

    if (r_f > 0.0) {
      fbond_f = -2.0 * rk_f / r_f;
    }
    else {
      fbond_f = 0.0;
    }

    if (eflag) ebond_f = rk_f * dr_f;

    // apply force to each of 2 atoms

    if (newton_bond || i1_c < nlocal) {
      f[i1_c][0] += delx_c * fbond_c;
      f[i1_c][1] += dely_c * fbond_c;
      f[i1_c][2] += delz_c * fbond_c;

      f[i1_f][0] += delx_f * fbond_f;
      f[i1_f][1] += dely_f * fbond_f;
      f[i1_f][2] += delz_f * fbond_f;

    }

    if (newton_bond || i2_c < nlocal) {
      f[i2_c][0] -= delx_c * fbond_c;
      f[i2_c][1] -= dely_c * fbond_c;
      f[i2_c][2] -= delz_c * fbond_c;

      f[i2_f][0] -= delx_f * fbond_f;
      f[i2_f][1] -= dely_f * fbond_f;
      f[i2_f][2] -= delz_f * fbond_f;
    }

    if (evflag) {
      ev_tally(i1_c, i2_c, nlocal, newton_bond, ebond_c, fbond_c, delx_c, dely_c, delz_c);
      ev_tally(i1_f, i2_f, nlocal, newton_bond, ebond_f, fbond_f, delx_f, dely_f, delz_f);
    }

    //set learning rate
    if (phase[type] == 2) { //lr decay according to loss, same as the original overclamping setting
      alpha_t = alpha[type] * abs((e_t[type] - e_f)/e_t[type]);
    }
    else {// constant learning rate
      alpha_t = alpha[type];
    }
    
    // Update stiffness for next integration step
    if (train[type] == 2) {

      if (mode[type] == 1) { // directed aging
        dk = -2.0 * alpha_t/eta[type] * rk_c * dr_c;
        k[type] += dk;
        t[type] += 1;
      }

      if (mode[type] == 3) { // adaptive learning adam
        double G = (dr_f * dr_f - dr_c * dr_c);
        m[type] = beta1[type] * m[type] + (1 - beta1[type]) * G;
        v[type] = beta2[type] * v[type] + (1 - beta2[type]) * G * G;
        double mhat = m[type] / (1 - pow(beta1[type], t[type]));
        double vhat = v[type] / (1 - pow(beta2[type], t[type]));
        dk = alpha[type]/eta[type] * mhat / (sqrt(vhat) + 1e-8);
        k[type] += dk;
      	t[type] += 1;
      }

      if (mode[type] == 4) { // SGD with momentum
        double G = (dr_f * dr_f - dr_c * dr_c);
        m[type] = beta1[type] * m[type] + (1 - beta1[type]) * G;
        dk = alpha_t/eta[type] * m[type];
        k[type] += dk;
        t[type] += 1;
      }
      
      if (mode[type] == 2) { // coupled learning SGD
        dk = alpha_t/eta[type] * (dr_f * dr_f - dr_c * dr_c);
        k[type] += dk;
        t[type] += 1;
      }

      if (k[type] < vmin[type]) k[type] = vmin[type];

    }
    else if (train[type] == 1) { // update relaxed length r0
      error->all(FLERR,"relaxed length update not implemented yet");
    }

    

    }

}


/* ---------------------------------------------------------------------- */

void BondHarmonicLearning::allocate()
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
  memory->create(v, np1, "bond:v");
  memory->create(m, np1, "bond:m");
  memory->create(beta1, np1, "bond:beta1");
  memory->create(beta2, np1, "bond:beta2");
  memory->create(t, np1, "bond:t");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondHarmonicLearning::coeff(int narg, char **arg)
{
  if (narg != 11 && narg != 16) error->all(FLERR, "Incorrect args for bond coefficients");

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
  double v_one = 0.0;
  double m_one = 0.0;
  double beta1_one = 0.0;
  double beta2_one = 0.0;
  double t_one = 0.0;
  if (narg == 16) {
    m_one = utils::numeric(FLERR, arg[11], false, lmp);
    v_one = utils::numeric(FLERR, arg[12], false, lmp);
    beta1_one = utils::numeric(FLERR, arg[13], false, lmp);
    beta2_one = utils::numeric(FLERR, arg[14], false, lmp);
    t_one = utils::numeric(FLERR, arg[15], false, lmp);
  }

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
    m[i] = m_one;
    v[i] = v_one;
    t[i] = t_one;
    beta1[i] = beta1_one;
    beta2[i] = beta2_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondHarmonicLearning::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondHarmonicLearning::write_restart(FILE *fp)
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
  if (mode[1] == 3 or mode[1] == 4) {
    fwrite(&m[1], sizeof(double), atom->nbondtypes, fp);
    fwrite(&v[1], sizeof(double), atom->nbondtypes, fp);
    fwrite(&beta1[1], sizeof(double), atom->nbondtypes, fp);
    fwrite(&beta2[1], sizeof(double), atom->nbondtypes, fp);
    fwrite(&t[1], sizeof(double), atom->nbondtypes, fp);
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondHarmonicLearning::read_restart(FILE *fp)
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
     // only read v if mode == 3
    if (mode[1] == 3 or mode[1] == 4) {
      utils::sfread(FLERR, &m[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
      utils::sfread(FLERR, &v[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
      utils::sfread(FLERR, &beta1[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
      utils::sfread(FLERR, &beta2[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
      utils::sfread(FLERR, &t[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    }
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
    // only bcast v if mode == 3  or 4
  if (mode[1] == 3 or mode[1] == 4) {
    MPI_Bcast(&m[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
    MPI_Bcast(&v[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
    MPI_Bcast(&beta1[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
    MPI_Bcast(&beta2[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
    MPI_Bcast(&t[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  }

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondHarmonicLearning::write_data(FILE *fp)
{
  if (mode[1] == 3 or mode[1] == 4) {
    for (int i = 1; i <= atom->nbondtypes; i++) fprintf(fp, "%d %.15g %.15g %.15g %.15g %.15g %.15g %d %d %d %d %.15g %.15g %.15g %.15g %d\n", i, k[i], r0[i], e_t[i], eta[i], alpha[i], vmin[i], train[i], mode[i], phase[i], target[i], m[i], v[i], beta1[i], beta2[i], t[i]);
  }
  else {
    for (int i = 1; i <= atom->nbondtypes; i++) fprintf(fp, "%d %.15g %.15g %.15g %.15g %.15g %.15g %d %d %d %d\n", i, k[i], r0[i], e_t[i], eta[i], alpha[i], vmin[i], train[i], mode[i], phase[i], target[i]);
  }
}

/* ---------------------------------------------------------------------- */

double BondHarmonicLearning::single(int type, double rsq, int /*i*/, int /*j*/, double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr * (1 - target[type]);
  fforce = 0;
  if (r > 0.0) fforce = -2.0 * rk / r;
  return 2.0 * rk * dr; // double for 2 times atoms
}

/* ---------------------------------------------------------------------- */

void BondHarmonicLearning::born_matrix(int type, double rsq, int /*i*/, int /*j*/, double &du, double &du2)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  du2 = 0.0;
  du = 0.0;
  du2 = 2 * k[type] * (1 - target[type]);
  if (r > 0.0) du = du2 * dr;
}

/* ----------------------------------------------------------------------
   return ptr to internal members upon request
------------------------------------------------------------------------ */

void *BondHarmonicLearning::extract(const char *str, int &dim)
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
  if (strcmp(str, "m") == 0 && (mode[1] == 3 or mode[1] == 4)) return (void *) m;
  if (strcmp(str, "v") == 0 && (mode[1] == 3 or mode[1] == 4)) return (void *) v;
  if (strcmp(str, "beta1") == 0 && (mode[1] == 3 or mode[1] == 4)) return (void *) beta1;
  if (strcmp(str, "beta2") == 0 && (mode[1] == 3 or mode[1] == 4)) return (void *) beta2;
  if (strcmp(str, "t") == 0 && (mode[1] == 3 or mode[1] == 4)) return (void *) t;
  return nullptr;
}
