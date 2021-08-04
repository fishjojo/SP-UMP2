#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "fblas.h"

void contract_o2v3(double* out, double* mata, double* matb, 
                   int nocc, int nvir, int contract_idx)
{
    //contract_idx==0: ijca,ijcd->ijad
    //contract_idx==1: ijab,ijdb->ijad
    size_t no2 = nocc * nocc;
    size_t nv2 = nvir * nvir;

    const int m = nvir;
    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    switch(contract_idx){
        case 0:
            #pragma omp parallel for schedule(static)
            for(size_t ij=0; ij<no2; ij++){
                double* a = mata + ij * nv2;
                double* b = matb + ij * nv2; 
                double* c = out + ij * nv2;
                dgemm_(&TRANS_N, &TRANS_T, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        case 1:
            #pragma omp parallel for schedule(static)
            for(size_t ij=0; ij<no2; ij++){
                double* a = mata + ij * nv2;
                double* b = matb + ij * nv2;
                double* c = out + ij * nv2;
                dgemm_(&TRANS_T, &TRANS_N, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        default:
            exit(1);
    }
    return;
}

void contract_o2v3_cmplx(complex double* out, complex double* mata, complex double* matb, 
                         int nocc, int nvir, int contract_idx)
{
    //contract_idx==0: ijca,ijcd->ijad
    //contract_idx==1: ijab,ijdb->ijad
    size_t no2 = nocc * nocc;
    size_t nv2 = nvir * nvir;

    const int m = nvir;
    const complex double D0 = 0;
    const complex double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    switch(contract_idx){
        case 0:
            #pragma omp parallel for schedule(static)
            for(size_t ij=0; ij<no2; ij++){
                complex double* a = mata + ij * nv2;
                complex double* b = matb + ij * nv2; 
                complex double* c = out + ij * nv2;
                zgemm_(&TRANS_N, &TRANS_T, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        case 1:
            #pragma omp parallel for schedule(static)
            for(size_t ij=0; ij<no2; ij++){
                complex double* a = mata + ij * nv2;
                complex double* b = matb + ij * nv2;
                complex double* c = out + ij * nv2;
                zgemm_(&TRANS_T, &TRANS_N, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        default:
            exit(1);
    }
    return;
}


void contract_o2v3_i(double* out, double* mata, double* matb, 
                     int nocc, int nvir, int contract_idx)
{
    //contract_idx==0: jca,jcd->jad
    //contract_idx==1: jab,jdb->jad
    size_t nv2 = nvir * nvir;

    const int m = nvir;
    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    switch(contract_idx){
        case 0:
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<nocc; j++){
                double* a = mata + j * nv2;
                double* b = matb + j * nv2; 
                double* c = out + j * nv2;
                dgemm_(&TRANS_N, &TRANS_T, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        case 1:
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<nocc; j++){
                double* a = mata + j * nv2;
                double* b = matb + j * nv2;
                double* c = out + j * nv2;
                dgemm_(&TRANS_T, &TRANS_N, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        default:
            exit(1);
    }
    return;
}

void contract_o2v3_i_cmplx(complex double* out, complex double* mata, complex double* matb,
                         int nocc, int nvir, int contract_idx)
{
    //contract_idx==0: jca,jcd->jad
    //contract_idx==1: jab,jdb->jad
    size_t nv2 = nvir * nvir;

    const int m = nvir;
    const complex double D0 = 0;
    const complex double D1 = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    switch(contract_idx){
        case 0:
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<nocc; j++){
                complex double* a = mata + j * nv2;
                complex double* b = matb + j * nv2;
                complex double* c = out + j * nv2;
                zgemm_(&TRANS_N, &TRANS_T, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        case 1:
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<nocc; j++){
                complex double* a = mata + j * nv2;
                complex double* b = matb + j * nv2;
                complex double* c = out + j * nv2;
                zgemm_(&TRANS_T, &TRANS_N, &m, &m, &m,
                       &D1, b, &m, a, &m, &D0, c, &m);
            }
            break;
        default:
            exit(1);
    }
    return;
}
