// t^4
// The `fourth_elevation` kernel function is performing the fourth power elevation of each element in the input array `t` 
// and storing the result in the output array `t_4`.
__kernel void fourth_elevation(__global float *t, __global float *t_4,
                               int n) {
  int i = get_global_id(0);
  if (i < n) {
    float t_i = t[i];
    float t_i2 = t_i * t_i;
    t_4[i] = t_i2 * t_i2;
  }
}
