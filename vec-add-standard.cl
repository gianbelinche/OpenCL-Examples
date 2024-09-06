__kernel void add_vec_gpu(__global float *vector_vector, const int vector_len) {
	const int idx = get_global_id(0);
	if (idx < vector_len){
		float j = vector_vector[idx]*vector_vector[idx];
		vector_vector[idx] = j*j;
	}
}
