#pragma once

double ggubfs(double* dseed);
int randomi(int n, double* dseed);
int random_int_0n(int n, double* dseed);
void random_permutation(int n_s, int* s, double* dseed);
void random_permutation2(int n_s, unsigned char* s, double* dseed);