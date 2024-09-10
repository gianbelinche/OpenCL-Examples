__kernel void fourth_elevation(__global int *vector_vector, const int vector_len) {
	const int idx = get_global_id(0);
	if (idx < vector_len){
		int j = vector_vector[idx]*vector_vector[idx];
		vector_vector[idx] = j*j;
	}
}
